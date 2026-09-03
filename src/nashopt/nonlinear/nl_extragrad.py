# NashOpt: A Python package for computing Generalized Nash Equilibria (GNE) in noncooperative games.
#
# Korpelevich's extragradient method for variational GNE of nonlinear GNEPs,
# plugged into GNEP.solve() as solver="extragrad".
#
# Variational GNE: find x* in X such that
#     F(x*)^T (y - x*) >= 0  for all y in X,
# where F(x)[si:ei] = nabla_{x_i} f_i(x) is the pseudogradient and
#     X = {x : g(x) <= 0, h(x) = 0, Aeq x = beq, lb <= x <= ub}.
#
# Algorithm (Korpelevich 1976):
#     y^k     = P_X(x^k - alpha * F(x^k))
#     x^{k+1} = P_X(x^k - alpha * F(y^k))
#
# Step-size condition for convergence: alpha < 1 / L,
# where L is the Lipschitz constant of F.
#
# IPOPT (via cyipopt) solves each projection subproblem by default, with
# warm-starting from the previous solution; a penalized least-squares (TRF)
# projector is available as a lighter-weight alternative not requiring cyipopt.
#
# (C) 2026 Alberto Bemporad

import numpy as np
import time
import jax.numpy as jnp
from scipy.optimize import least_squares
from types import SimpleNamespace


_INF = 2e19  # cyipopt sentinel for "no bound"

_WARM_OPTS = [
    ('warm_start_init_point',     'yes'),
    ('warm_start_bound_push',      1e-6),
    ('warm_start_mult_bound_push', 1e-6),
    ('warm_start_bound_frac',      1e-6),
    ('warm_start_slack_bound_frac',1e-6),
    ('warm_start_slack_bound_push',1e-6),
]


class _ProjProblem:
    """cyipopt callbacks: project point v onto the feasible set X.

        min  (1/2) ||x - v||^2
        s.t. g(x) <= 0,  h(x) = 0,  Aeq x = beq,  lb <= x <= ub
    """

    def __init__(self, gnep):
        self.gnep = gnep
        self.v = np.zeros(gnep.nvar)
        if gnep.Aeq is not None:
            self._Aeq_np = np.asarray(gnep.Aeq, dtype=np.float64)
            self._beq_np = np.asarray(gnep.beq, dtype=np.float64)
        else:
            self._Aeq_np = None

    def objective(self, x):
        d = x - self.v
        return 0.5 * float(np.dot(d, d))

    def gradient(self, x):
        return (x - self.v).astype(np.float64)

    def constraints(self, x):
        g = self.gnep
        xj = jnp.asarray(x)
        parts = []
        if g.ng > 0:
            parts.append(np.asarray(g.g(xj), dtype=np.float64).ravel())
        if g.nh > 0:
            parts.append(np.asarray(g.h(xj), dtype=np.float64).ravel())
        if g.neq > 0:
            parts.append((self._Aeq_np @ x - self._beq_np).astype(np.float64))
        return np.concatenate(parts) if parts else np.zeros(0)

    def jacobian(self, x):
        # Dense row-major Jacobian as a flat 1-D array (m * n,).
        g = self.gnep
        xj = jnp.asarray(x)
        rows = []
        if g.ng > 0:
            rows.append(np.asarray(g.dg(xj), dtype=np.float64))
        if g.nh > 0:
            rows.append(np.asarray(g.dh(xj), dtype=np.float64))
        if g.neq > 0:
            rows.append(self._Aeq_np)
        return np.vstack(rows).ravel() if rows else np.zeros(0)


class _Projector:
    """IPOPT-based projection onto X, warm-started from the previous solution."""

    def __init__(self, gnep):
        import cyipopt
        prob = _ProjProblem(gnep)
        self._prob = prob
        nvar = gnep.nvar

        lb_np = np.asarray(gnep.lb, dtype=np.float64)
        ub_np = np.asarray(gnep.ub, dtype=np.float64)
        lb_x = np.where(np.isfinite(lb_np), lb_np, -_INF)
        ub_x = np.where(np.isfinite(ub_np), ub_np,  _INF)

        ng, nh, neq = gnep.ng, gnep.nh, gnep.neq
        m = ng + nh + neq
        if m > 0:
            cl = np.concatenate([
                -_INF * np.ones(ng),  # g(x) <= 0
                np.zeros(nh),         # h(x) = 0
                np.zeros(neq),        # Aeq x - beq = 0
            ])
            cu = np.zeros(m)
        else:
            cl = np.zeros(0)
            cu = np.zeros(0)

        self._nlp = cyipopt.Problem(
            n=nvar, m=m,
            lb=lb_x, ub=ub_x,
            cl=cl, cu=cu,
            problem_obj=prob,
        )
        nlp = self._nlp
        nlp.add_option('print_level', 0)
        nlp.add_option('sb', 'yes')
        nlp.add_option('hessian_approximation', 'limited-memory')
        nlp.add_option('max_iter', 300)
        nlp.add_option('tol', 1e-10)
        for key, val in _WARM_OPTS:
            nlp.add_option(key, val)

        self._x_prev = None

    def project(self, v):
        self._prob.v = np.asarray(v, dtype=np.float64)
        x0 = self._x_prev if self._x_prev is not None else np.asarray(v, dtype=np.float64)
        x_sol, _ = self._nlp.solve(x0)
        self._x_prev = np.asarray(x_sol, dtype=np.float64).copy()
        return self._x_prev


class _ProjectorTRF:
    """TRF least-squares projection onto X (box constraints hard, others penalized).

        min  (1/2) ||x - v||^2 + (rho/2)(||max(g(x),0)||^2 + ||h(x)||^2 + ||Aeq x - beq||^2)
        s.t. lb <= x <= ub

    Solved as a nonlinear least-squares problem with residuals
        [x - v,  rho*max(g(x),0),  rho*h(x),  rho*(Aeq x - beq)]
    using scipy.optimize.least_squares with method='trf'.
    """

    def __init__(self, gnep, rho=1e5):
        self._gnep = gnep
        self._rho = rho

        lb_np = np.asarray(gnep.lb, dtype=np.float64)
        ub_np = np.asarray(gnep.ub, dtype=np.float64)
        self._lb = np.where(np.isfinite(lb_np), lb_np, -np.inf)
        self._ub = np.where(np.isfinite(ub_np), ub_np,  np.inf)

        if gnep.Aeq is not None:
            self._Aeq_np = np.asarray(gnep.Aeq, dtype=np.float64)
            self._beq_np = np.asarray(gnep.beq, dtype=np.float64)
        else:
            self._Aeq_np = None

        self._v = np.zeros(gnep.nvar)
        self._x_prev = None

    def _residual(self, x):
        gnep = self._gnep
        rho = self._rho
        xj = jnp.asarray(x)
        parts = [x - self._v]
        if gnep.ng > 0:
            parts.append(rho * np.maximum(np.asarray(gnep.g(xj), dtype=np.float64), 0.0))
        if gnep.nh > 0:
            parts.append(rho * np.asarray(gnep.h(xj), dtype=np.float64))
        if gnep.neq > 0:
            parts.append(rho * (self._Aeq_np @ x - self._beq_np))
        return np.concatenate(parts)

    def _jacobian(self, x):
        gnep = self._gnep
        rho = self._rho
        xj = jnp.asarray(x)
        nvar = gnep.nvar
        rows = [np.eye(nvar)]
        if gnep.ng > 0:
            gx = np.asarray(gnep.g(xj), dtype=np.float64)
            dgx = np.asarray(gnep.dg(xj), dtype=np.float64)  # (ng, nvar)
            mask = (gx > 0.0).astype(np.float64)[:, None]
            rows.append(rho * mask * dgx)
        if gnep.nh > 0:
            dhx = np.asarray(gnep.dh(xj), dtype=np.float64)  # (nh, nvar)
            rows.append(rho * dhx)
        if gnep.neq > 0:
            rows.append(rho * self._Aeq_np)
        return np.vstack(rows)

    def project(self, v):
        self._v = np.asarray(v, dtype=np.float64)
        x0 = self._x_prev if self._x_prev is not None else self._v.copy()
        sol = least_squares(
            self._residual, x0,
            jac=self._jacobian,
            bounds=(self._lb, self._ub),
            method='trf',
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
        )
        self._x_prev = sol.x.copy()
        return self._x_prev


def _pseudogradient(gnep, x):
    """Pseudogradient F(x)[si:ei] = nabla_{x_i} f_i(x)."""
    xj = jnp.asarray(x)
    Fx = np.empty(gnep.nvar)
    for i in range(gnep.N):
        si, ei = int(gnep.i1[i]), int(gnep.i2[i])
        Fx[si:ei] = np.asarray(gnep.df[i](xj[si:ei], xj))
    return Fx


def extragrad_nlgnep(
    gnep,
    tol=1e-8,
    maxiter=1000,
    alpha=None,
    x0=None,
    verbose=False,
    projection_solver="ipopt",
    rho=1e5,
):
    """
    Korpelevich extragradient method for variational GNE of nonlinear GNEPs.

    Parameters
    ----------
    gnep : GNEP
        Nonlinear GNEP object (nashopt.nonlinear.gnep_base.GNEP), constructed
        with variational=True.
    tol : float
        Stop when ||x^{k+1} - x^k||_inf < tol.
    maxiter : int
        Maximum iterations.
    alpha : float, optional
        Step size. If None, estimated as 0.99 / L where L is the Lipschitz
        constant of F approximated by a single finite-difference ratio.
    x0 : ndarray (nvar,), optional
        Initial point. Default: zeros.
    verbose : bool
        Print per-iteration residuals.
    projection_solver : str
        Projection subproblem solver: "ipopt" (default) or "trf".
        "ipopt" requires cyipopt to be installed. "trf" uses
        scipy.optimize.least_squares with method='trf'; inequality and
        equality constraints are penalized with weight rho, and only box
        bounds are enforced as hard constraints.
    rho : float
        Penalty weight for constraint violations used by the "trf" projector.
        Ignored when projection_solver="ipopt".

    Returns
    -------
    SimpleNamespace with:
        x            : ndarray -- approximate variational GNE
        elapsed_time : float   -- wall-clock seconds (excluding setup)
        status_str   : str     -- 'converged' or 'max_iterations_reached'
        num_iters    : int     -- iterations performed
        info         : dict    -- {'converged': bool, 'final_err': float}
    """
    nvar = gnep.nvar

    if x0 is None:
        x0 = np.zeros(nvar)
    else:
        x0 = np.asarray(x0, dtype=np.float64).copy()

    if alpha is None:
        rng = np.random.default_rng(0)
        eps = 1e-4 * max(np.linalg.norm(x0), 1.0)
        xb = x0 + eps * rng.standard_normal(nvar)
        Fa = _pseudogradient(gnep, x0)
        Fb = _pseudogradient(gnep, xb)
        L = np.linalg.norm(Fa - Fb) / max(np.linalg.norm(x0 - xb), 1e-15)
        alpha = 0.99 / max(L, 1e-12)

    # Two independent projectors: y-step and x-step warm-start separately.
    projection_solver = projection_solver.lower()
    if projection_solver == "ipopt":
        proj_y = _Projector(gnep)
        proj_x = _Projector(gnep)
    elif projection_solver == "trf":
        proj_y = _ProjectorTRF(gnep, rho=rho)
        proj_x = _ProjectorTRF(gnep, rho=rho)
    else:
        raise ValueError(f"Unknown projection_solver '{projection_solver}'. Use 'ipopt' or 'trf'.")

    # Project x0 to feasibility before timing.
    x = proj_x.project(x0)

    t_start = time.perf_counter()
    status_str = "max_iterations_reached"
    err = np.nan
    k = -1

    for k in range(maxiter):
        Fx = _pseudogradient(gnep, x)
        y = proj_y.project(x - alpha * Fx)
        Fy = _pseudogradient(gnep, y)
        x_new = proj_x.project(x - alpha * Fy)

        err = np.linalg.norm(x_new - x, np.inf)
        x = x_new

        if verbose:
            print(f"  extragrad_nl iter {k+1}: ||dx||_inf={err:.3e}")

        if err < tol:
            status_str = "converged"
            break

    elapsed = time.perf_counter() - t_start

    return SimpleNamespace(
        x=x,
        elapsed_time=elapsed,
        status_str=status_str,
        num_iters=k + 1,
        info={"converged": status_str == "converged", "final_err": float(err)},
    )


def solve_extragrad(gnep, x0=None, solver_opts=None, verbose=1):
    """
    Solve the variational GNE of a nonlinear GNEP via Korpelevich's extragradient
    method (extragrad_nlgnep above), returning a solution object with the same
    layout as GNEP.solve().

    Parameters:
    -----------
    gnep : GNEP
        Nonlinear GNEP object (nashopt.nonlinear.gnep_base.GNEP). The
        extragradient method targets the variational GNE, so gnep should
        normally be constructed with variational=True.
    x0 : array-like or None
        Initial guess for the Nash equilibrium x. Used only if solver_opts
        does not itself provide 'x0'.
    solver_opts : dict or None
        Keyword arguments forwarded to extragrad_nlgnep(gnep, ...): tol,
        maxiter, alpha, x0, verbose, projection_solver, rho. See
        extragrad_nlgnep's docstring for details. If None, extragrad_nlgnep's
        own defaults are used.
    verbose : int, optional
        Verbosity level. 0: silent. >0: termination report. Used only if
        solver_opts does not itself specify 'verbose' (in which case that
        value is passed through to extragrad_nlgnep for per-iteration
        reporting).

    Returns:
    --------
    sol : SimpleNamespace
        Solution object with fields:
        x : ndarray
            Computed GNE solution.
        res : ndarray
            1-element array holding the final step norm ||x^{k+1}-x^k||_inf.
        lam : list
            Empty list: the extragradient method is multiplier-free (feasibility
            is enforced via projections, not Lagrange multipliers).
        stats : Statistics about the optimization result.
    """

    if (gnep.ng > 0 or gnep.has_eq) and not gnep.variational:
        print("\033[1;33mWarning: solver='extragrad' targets the variational GNE, but "
              "the GNEP was not constructed with variational=True.\033[0m")

    opts = dict(solver_opts) if solver_opts is not None else {}
    opts.setdefault("x0", x0)
    opts.setdefault("verbose", verbose > 1)

    t0 = time.perf_counter()
    result = extragrad_nlgnep(gnep, **opts)
    t0 = time.perf_counter() - t0

    converged = result.info["converged"]
    final_err = float(result.info["final_err"])

    if verbose > 0:
        color = "\033[1;32m" if converged else "\033[1;31m"
        status = "converged" if converged else "reached the maximum number of iterations"
        print(f"{color}Extragradient method {status} after {result.num_iters} iterations: "
              f"||x^(k+1) - x^k||_inf = {final_err:.3e}, time = {t0:.3f} seconds.\033[0m")
        if not converged:
            print("\033[1;33mWarning: maximum number of iterations reached; "
                  "an equilibrium may not have been found.\033[0m")

    stats = SimpleNamespace()
    stats.solver = "extragrad"
    stats.kkt_evals = result.num_iters
    stats.elapsed_time = t0
    stats.status_str = result.status_str
    stats.info = result.info

    sol = SimpleNamespace()
    sol.x = np.asarray(result.x)
    sol.res = np.atleast_1d(np.asarray(final_err))
    sol.lam = []
    sol.stats = stats
    sol.norm_residual = final_err
    return sol
