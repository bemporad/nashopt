""" Operator Extrapolation (OE) method for monotone generalized Nash equilibrium problems.

Unlike golden_ratio.py, this module does NOT lift the GNEP() to a primal-dual
VI: it applies Algorithm 1 directly to the *original* variational GNE problem

    find x* in X such that <M(x*), x-x*> >= 0  for all x in X,       (1.1)

X = {x : g(x) <= 0, lb <= x <= ub} the game's actual feasible set (shared
nonlinear inequalities included) and M(x) the pseudogradient of the game (the
stack of grad_{x_i} f_i(x) over all agents i). Consequently each iteration
requires a genuine projection onto X -- a small convex NLP (since g is
nonlinear), solved here with IPOPT (via cyipopt), warm-started from the
previous projection.

Reference:

    G. Kotsalis, G. Lan, T. Li, "Simple and Optimal Methods for Stochastic
    Variational Inequalities, I: Operator Extrapolation," arXiv:2011.02987v5
    (2023).

Algorithm 1 (page 4), with the Euclidean prox-function V(x,y) = 1/2||x-y||^2
(so the argmin defining each step reduces to a single projection):

    x_0 = x_1 in X
    for t = 1,...,k:
        x_{t+1} = argmin_{x in X} gamma_t <M(x_t) + beta_t(M(x_t)-M(x_{t-1})), x> + V(x_t,x)
                = P_X( x_t - gamma_t*(M(x_t) + beta_t*(M(x_t)-M(x_{t-1}))) )

(the paper's extrapolation weight lambda_t is renamed beta throughout this
module to avoid clashing with the game's own dual variables/Lagrange
multipliers, which do not otherwise appear here since the problem is not
lifted). Parameters gamma_t, beta_t are set constant, per Theorem 2.3 (page 6)
for generalized strongly monotone VIs (GSMVI, mu > 0 in (1.7)):

    gamma_t = 1/(2L),   beta_t = (mu/L+1)^{-1} = L/(L+mu),   t = 1,...,k,

which gives the linear rate V(x_{k+1},x*) <= (L/mu) (L/(L+mu))^{k-1} V(x_1,x*).

Because the game's costs are quadratic, M(x) = G x + c is affine, so L and mu
are exact (not heuristic, since the problem is not lifted and X is the true
feasible set) and computed as

    L  = ||G||_2                        (spectral norm, Lipschitz constant of M)
    mu = lambda_min(0.5*(G+G^T))        (monotonicity constant of M)

with G the (constant) Jacobian of M, evaluated once via automatic
differentiation.

Only shared inequality constraints (g, ng) and box constraints (lb, ub) are
supported: a GNEP() with linear/nonlinear equality constraints (Aeq, h) is not
covered here.

(C) 2026 A. Bemporad
"""

import numpy as np
import jax
import jax.numpy as jnp
import cyipopt
import time
from types import SimpleNamespace

jax.config.update("jax_enable_x64", True)

_INF = 2e19  # cyipopt sentinel for "no bound"

_WARM_OPTS = [
    ('warm_start_init_point',       'yes'),
    ('warm_start_bound_push',       1e-6),
    ('warm_start_mult_bound_push',  1e-6),
    ('warm_start_bound_frac',       1e-6),
    ('warm_start_slack_bound_frac', 1e-6),
    ('warm_start_slack_bound_push', 1e-6),
]


def _check_supported(gnep):
    if gnep.neq > 0 or gnep.nh > 0:
        raise NotImplementedError(
            "op_extrapolation only supports GNEP() instances with shared "
            "inequality constraints (g, ng) and box constraints (lb, ub); "
            "this GNEP has linear and/or nonlinear equality constraints "
            "(Aeq/h), which are not covered here."
        )


def _pseudogradient(gnep):
    """M(x) = stack of grad_{x_i} f_i(x) over all agents i (jitted)."""
    i1, i2, N = gnep.i1, gnep.i2, gnep.N

    @jax.jit
    def M(x):
        return jnp.concatenate([gnep.df[i](x[i1[i]:i2[i]], x) for i in range(N)])
    return M


def compute_L_mu(gnep, x0=None):
    """Lipschitz constant L and monotonicity modulus mu of the game's pseudogradient
    M(x), computed from the quadratic costs: M(x) = G x + c is affine, so its Jacobian
    equals the constant matrix G everywhere (evaluated here at an arbitrary point x0).

        L  = ||G||_2                 (spectral norm)
        mu = lambda_min(0.5*(G+G^T))

    Returns (L, mu, G).
    """
    M = _pseudogradient(gnep)
    if x0 is None:
        x0 = np.zeros(gnep.nvar)
    G = np.asarray(jax.jacobian(M)(jnp.asarray(x0, dtype=float)))
    L = float(np.linalg.norm(G, 2))
    mu = float(np.linalg.eigvalsh(0.5 * (G + G.T)).min())
    return L, mu, G


class _ProjProblem:
    """cyipopt callbacks: project point v onto the feasible set X.

        min  (1/2) ||x - v||^2
        s.t. g(x) <= 0,  lb <= x <= ub
    """

    def __init__(self, gnep):
        self.gnep = gnep
        self.v = np.zeros(gnep.nvar)

    def objective(self, x):
        d = x - self.v
        return 0.5 * float(np.dot(d, d))

    def gradient(self, x):
        return (x - self.v).astype(np.float64)

    def constraints(self, x):
        return np.asarray(self.gnep.g(jnp.asarray(x)), dtype=np.float64).ravel()

    def jacobian(self, x):
        return np.asarray(self.gnep.dg(jnp.asarray(x)), dtype=np.float64).ravel()


class _Projector:
    """Projection onto X = {g(x)<=0, lb<=x<=ub}, warm-started from the previous
    solution. Uses IPOPT (cyipopt) when g is present; a plain elementwise clip
    onto the box otherwise (X reduces to a box, no NLP needed).
    """

    def __init__(self, gnep):
        lb_np = np.asarray(gnep.lb, dtype=np.float64)
        ub_np = np.asarray(gnep.ub, dtype=np.float64)

        if gnep.ng > 0:
            self._prob = _ProjProblem(gnep)
            lb_x = np.where(np.isfinite(lb_np), lb_np, -_INF)
            ub_x = np.where(np.isfinite(ub_np), ub_np, _INF)
            self._nlp = cyipopt.Problem(
                n=gnep.nvar, m=gnep.ng,
                lb=lb_x, ub=ub_x,
                cl=-_INF * np.ones(gnep.ng), cu=np.zeros(gnep.ng),
                problem_obj=self._prob,
            )
            self._nlp.add_option('print_level', 0)
            self._nlp.add_option('sb', 'yes')
            self._nlp.add_option('hessian_approximation', 'limited-memory')
            self._nlp.add_option('max_iter', 300)
            self._nlp.add_option('tol', 1e-10)
            for key, val in _WARM_OPTS:
                self._nlp.add_option(key, val)
        else:
            self._nlp = None
            self._lb = np.where(np.isfinite(lb_np), lb_np, -np.inf)
            self._ub = np.where(np.isfinite(ub_np), ub_np, np.inf)

        self._x_prev = None
        self._mult_g = None
        self._mult_x_L = None
        self._mult_x_U = None

    def project(self, v):
        v = np.asarray(v, dtype=np.float64)
        if self._nlp is None:
            return np.clip(v, self._lb, self._ub)

        self._prob.v = v
        x0 = self._x_prev if self._x_prev is not None else v.copy()
        if self._mult_g is None:
            # First call: no dual information yet, cold-start the multipliers.
            x_sol, info = self._nlp.solve(x0)
        else:
            # Warm start both primal (x0) and dual (lagrange/zl/zu) variables
            # from the previous projection, as intended by warm_start_init_point.
            x_sol, info = self._nlp.solve(
                x0, lagrange=self._mult_g, zl=self._mult_x_L, zu=self._mult_x_U)

        self._x_prev = np.asarray(x_sol, dtype=np.float64).copy()
        self._mult_g = np.asarray(info["mult_g"], dtype=np.float64).copy()
        self._mult_x_L = np.asarray(info["mult_x_L"], dtype=np.float64).copy()
        self._mult_x_U = np.asarray(info["mult_x_U"], dtype=np.float64).copy()
        return self._x_prev


def op_extrapolation(gnep, x1, L=None, mu=None, safety=1.0,
                      max_iter=1000, tol=1e-10, stopping="step", check_every=1, verbose=1):
    """Solve a monotone GNEP() with the Operator Extrapolation (OE) method of
    Kotsalis, Lan and Li (2023), Algorithm 1, applied directly to the original
    (non-lifted) variational GNE problem: find x* in X = {g(x)<=0, lb<=x<=ub}
    such that <M(x*), x-x*> >= 0 for all x in X. Parameters are set according
    to Theorem 2.3 (GSMVI, constant stepsizes) from L and mu computed on the
    game's quadratic costs (see compute_L_mu).

    Parameters
    ----------
    gnep : nashopt.GNEP
        The nonlinear GNEP to be solved. Must have no equality constraints
        (Aeq, h); only shared inequality constraints (g, ng) and box
        constraints (lb, ub) are supported.
    x1 : array-like, shape (nvar,)
        Initial iterate x_0 = x_1 (need not be feasible; it is projected onto
        X before the first iteration).
    L, mu : float, optional
        Lipschitz constant / monotonicity modulus of the pseudogradient M.
        If None, computed from the quadratic costs via compute_L_mu(gnep, x1).
    safety : float
        Safety factor in (0,1] applied to the theoretical stepsize gamma_t =
        1/(2L) (Theorem 2.3 uses safety=1; since M is exactly affine and X is
        the true feasible set, safety=1 is theoretically justified here).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Stopping tolerance (meaning depends on `stopping`).
    stopping : str
        "step" (default): stop when ||x_{t+1}-x_t|| <= tol -- cheap (no extra
        projection), but only an indirect proxy for optimality.
        "residual": stop when the natural-map residual
            r_gamma(x) := (x - P_X(x - gamma*M(x))) / gamma
        satisfies ||r_gamma(x_t)|| <= tol. r_gamma(x) = 0 if and only if x
        solves the VI, so this is a genuine first-order optimality measure.
        Unlike golden_ratio.py, P_X requires solving an NLP (IPOPT) here, so
        this option roughly doubles the per-check cost; see `check_every`.
    check_every : int
        Only used when stopping="residual": evaluate the (expensive) natural
        residual every `check_every` iterations rather than every iteration,
        to amortize its NLP-solve cost. The step-based ||x_{t+1}-x_t|| is
        still tracked every iteration regardless of `stopping`.
    verbose : int
        0 = silent, 1 = final report, 2 = per-iteration report.

    Returns
    -------
    sol : SimpleNamespace
        x : ndarray, primal solution x*
        iters : int, number of iterations performed
        residual : float, ||x_{t+1}-x_t|| at termination
        nat_residual : float, last computed ||r_gamma(x_t)|| (nan if never
            computed, i.e. stopping="step")
        converged : bool
        L, mu : float, values used to set gamma, beta
        gamma, beta : float, stepsize and extrapolation weight used (Eq. (2.15))
        history : dict with 'residual' (every iteration) and 'nat_residual'
            (every `check_every` iterations, empty if stopping="step")

    (C) 2026 A. Bemporad
    """
    _check_supported(gnep)
    if stopping not in ("step", "residual"):
        raise ValueError("stopping must be 'step' or 'residual'.")

    nvar = gnep.nvar
    x1 = np.asarray(x1, dtype=float).reshape(nvar)

    if L is None or mu is None:
        L_est, mu_est, _ = compute_L_mu(gnep, x1)
        L = L_est if L is None else L
        mu = mu_est if mu is None else mu

    if mu <= 0.0 and verbose > 0:
        print("\033[1;33mWarning: mu <= 0 (merely monotone quadratic costs); "
              "Theorem 2.3's linear-rate guarantee requires mu > 0.\033[0m")

    gamma = safety / (2.0 * L)
    beta = L / (L + max(mu, 0.0))  # extrapolation weight (paper's lambda_t)

    M = _pseudogradient(gnep)
    projector = _Projector(gnep)
    # A separate, independently warm-started projector for the natural-residual
    # check, so its iterates (projections of x - gamma*M(x)) don't disturb the
    # warm start of the main extrapolated-step projector above.
    res_projector = _Projector(gnep) if stopping == "residual" else None

    if verbose > 0:
        print(f"Operator Extrapolation (not lifted): nvar={nvar}, ng={gnep.ng}, "
              f"L={L:.4e}, mu={mu:.4e}, gamma={gamma:.4e}, beta={beta:.4e}")

    x_curr = projector.project(x1)  # x_0 = x_1, projected onto X
    x_prev = x_curr.copy()
    Mx_prev = np.asarray(M(jnp.asarray(x_prev)))

    history_res, history_nat_res = [], []
    converged = False
    nat_res_norm = np.nan
    k = 0
    for k in range(1, max_iter + 1):
        Mx_curr = np.asarray(M(jnp.asarray(x_curr)))
        dM = Mx_curr - Mx_prev

        w = x_curr - gamma * (Mx_curr + beta * dM)
        x_next = projector.project(w)

        step_norm = float(np.linalg.norm(x_next - x_curr))
        history_res.append(step_norm)

        check_norm = step_norm
        if stopping == "residual" and k % check_every == 0:
            # Natural-map residual r_gamma(x^t) = (x^t - P_X(x^t - gamma*M(x^t)))/gamma:
            # zero iff x^t solves the VI. Costs one extra IPOPT projection.
            x_nat = res_projector.project(x_curr - gamma * Mx_curr)
            nat_res_norm = float(np.linalg.norm(x_curr - x_nat)) / gamma
            history_nat_res.append(nat_res_norm)
            check_norm = nat_res_norm

        if verbose > 1:
            msg = f"  iter {k:4d}: ||x^(k+1)-x^k|| = {step_norm:.6e}"
            if stopping == "residual" and k % check_every == 0:
                msg += f", ||r_gamma(x^k)|| = {nat_res_norm:.6e}"
            print(msg)

        x_prev, Mx_prev = x_curr, Mx_curr
        x_curr = x_next

        if check_norm <= tol:
            converged = True
            break

    if verbose > 0:
        status = "converged" if converged else "did not converge (max_iter reached)"
        msg = (f"Operator Extrapolation {status}: {k} iterations, "
               f"||x^(k+1)-x^k|| = {history_res[-1]:.4e}")
        if stopping == "residual":
            msg += f", ||r_gamma(x^k)|| = {nat_res_norm:.4e}"
        print(msg)

    sol = SimpleNamespace(
        x=x_curr,
        iters=k,
        residual=history_res[-1] if history_res else np.nan,
        nat_residual=nat_res_norm,
        converged=converged,
        L=L,
        mu=mu,
        gamma=gamma,
        beta=beta,
        history={"residual": history_res, "nat_residual": history_nat_res},
    )
    return sol


def solve_op_extrapolation(gnep, x0=None, solver_opts=None, verbose=1):
    """
    Solve a monotone GNEP() via the Operator Extrapolation method (op_extrapolation
    above), returning a solution object with the same layout as GNEP.solve().

    Parameters:
    -----------
    gnep : GNEP
        Nonlinear GNEP object (nashopt.nonlinear.gnep_base.GNEP), with only
        shared inequality constraints (g, ng) and box constraints (lb, ub).
    x0 : array-like or None
        Initial guess for the Nash equilibrium x, used as the initial iterate
        x1. Used only if solver_opts does not itself provide 'x1'.
    solver_opts : dict or None
        Keyword arguments forwarded to op_extrapolation(gnep, ...): x1, L, mu,
        safety, max_iter, tol, stopping, check_every, verbose. See
        op_extrapolation's docstring for details. If None, op_extrapolation's
        own defaults are used.
    verbose : int, optional
        Verbosity level. 0: silent. >0: termination report. Used only if
        solver_opts does not itself specify 'verbose' (in which case that
        value is passed through to op_extrapolation for per-iteration
        reporting).

    Returns:
    --------
    sol : SimpleNamespace
        Solution object with fields:
        x : ndarray
            Computed GNE solution.
        res : ndarray
            1-element array holding the final step norm ||x^{k+1}-x^k||.
        lam : list
            Empty list: the method is not lifted, so no multiplier estimate
            is produced.
        stats : Statistics about the optimization result.
    """
    opts = dict(solver_opts) if solver_opts is not None else {}
    opts.setdefault("x1", x0 if x0 is not None else np.zeros(gnep.nvar))
    opts.setdefault("verbose", 2 if verbose > 1 else 0)

    t0 = time.perf_counter()
    result = op_extrapolation(gnep, **opts)
    t0 = time.perf_counter() - t0

    converged = result.converged

    if verbose > 0:
        color = "\033[1;32m" if converged else "\033[1;31m"
        status = "converged" if converged else "reached the maximum number of iterations"
        print(f"{color}Operator Extrapolation {status} after {result.iters} iterations: "
              f"||x^(k+1)-x^k|| = {result.residual:.3e}, time = {t0:.3f} seconds.\033[0m")
        if not converged:
            print("\033[1;33mWarning: maximum number of iterations reached; "
                  "an equilibrium may not have been found.\033[0m")

    stats = SimpleNamespace()
    stats.solver = "op_extrapolation"
    stats.kkt_evals = result.iters
    stats.elapsed_time = t0
    stats.status_str = "converged" if converged else "max_iterations_reached"
    stats.info = {"converged": converged, "residual": result.residual,
                  "nat_residual": result.nat_residual, "L": result.L, "mu": result.mu}

    sol = SimpleNamespace()
    sol.x = np.asarray(result.x)
    sol.res = np.atleast_1d(np.asarray(result.residual))
    sol.lam = []
    sol.stats = stats
    sol.norm_residual = float(result.residual)
    return sol
