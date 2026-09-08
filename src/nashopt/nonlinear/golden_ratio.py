""" Golden Ratio Algorithm (GRAAL) for monotone generalized Nash equilibrium problems.

A GNEP() with shared inequality constraints g(x) <= 0 and box constraints
lb <= x <= ub is solved by lifting it to the primal-dual monotone variational
inequality (1) of

    Y. Malitsky, "Golden Ratio Algorithms for Variational Inequalities,"
    Mathematical Programming, 2019 (arXiv:1803.08832),

with z = (x, lambda_g), lambda_g >= 0 the multipliers of the shared (nonlinear)
inequalities g(x) <= 0,

    F(z) = F(x, lambda_g) = [ M(x) + Jg(x)^T lambda_g ;  -g(x) ],

where M(x) is the pseudogradient of the game (the stack of grad_{x_i} f_i(x)
over all agents i) and Jg(x) is the Jacobian of g. The shared inequalities are
entirely handled through the dual variable lambda_g inside F and are NOT
re-imposed on the primal projection: only the box constraints lb <= x <= ub
define the primal feasible set A = {x : lb <= x <= ub}, so that

    C = A x R_+^{ng}

is a Cartesian product and g(z) := indicator function of C in (1). (Enforcing
g(x) <= 0 a second time on the primal projection would short-circuit the dual
dynamics: x would always be forced feasible on its own, so lambda_g would never
receive the ascent/descent signal it needs to reach the correct KKT
multiplier.) Since the resolvent (proximal operator) of an indicator function
is the Euclidean projection (independent of the stepsize), the recursion only
requires an elementwise clip onto A and an elementwise clip onto R_+^{ng} --
no NLP solve is needed at any iteration.

Because g is nonlinear, F is bilinear in (x, lambda_g) and hence generally not
globally Lipschitz, so the fixed-stepsize Golden Ratio Algorithm of Theorem 1
(Sect. 2, Eq. (10), which requires a known global Lipschitz constant L) is not
directly applicable. Instead this module implements the fully adaptive,
line-search-free instance of the algorithm given on page 3 (just before
Sect. 2), which estimates a local Lipschitz constant from consecutive
iterates and requires no knowledge of L:

    theta_k = min{ (10/9) theta_{k-1},
                   9/(16 theta_{k-2}) * ||z^k-z^{k-1}||^2 / ||F(z^k)-F(z^{k-1})||^2,
                   theta_bar }
    zbar^k  = (z^k + 2 zbar^{k-1}) / 3
    z^{k+1} = prox_{theta_k g}(zbar^k - theta_k F(z^k)) = P_C(zbar^k - theta_k F(z^k))

started from arbitrary z^0, z^1 in V (zbar^0 := z^1), theta_0 = theta_{-1} > 0,
theta_bar > 0. (Here the paper's stepsize, denoted lambda_k, is renamed
theta_k throughout this module to avoid clashing with the game's own dual
variable lambda_g.) (z^k), (zbar^k) converge to a solution of (1), i.e. to a
pair (x*, lambda_g*) satisfying the KKT conditions of the game.

Only shared inequality constraints (g, ng) and box constraints (lb, ub) are
supported: a GNEP() with linear/nonlinear equality constraints (Aeq, h) is not
covered by this lifting.

(C) 2026 A. Bemporad
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
from types import SimpleNamespace

jax.config.update("jax_enable_x64", True)


def _check_supported(gnep):
    if gnep.neq > 0 or gnep.nh > 0:
        raise NotImplementedError(
            "golden_ratio only supports GNEP() instances with shared inequality "
            "constraints (g, ng) and box constraints (lb, ub); this GNEP has "
            "linear and/or nonlinear equality constraints (Aeq/h), which are "
            "not covered by the primal-dual lifting used here."
        )


def _pseudogradient(gnep):
    """M(x) = stack of grad_{x_i} f_i(x) over all agents i (jitted)."""
    i1, i2, N = gnep.i1, gnep.i2, gnep.N

    @jax.jit
    def M(x):
        return jnp.concatenate([gnep.df[i](x[i1[i]:i2[i]], x) for i in range(N)])
    return M


def _lifted_F(gnep):
    """F(x,lambda_g) = [M(x) + Jg(x)^T lambda_g ; -g(x)], as a function of z=(x,lambda_g)."""
    nvar, ng = gnep.nvar, gnep.ng
    M = _pseudogradient(gnep)

    if ng > 0:
        @jax.jit
        def F(z):
            x, lam = z[:nvar], z[nvar:]
            return jnp.concatenate([M(x) + gnep.dg(x).T @ lam, -gnep.g(x)])
    else:
        @jax.jit
        def F(z):
            return M(z[:nvar])
    return F


def _project_onto_C(gnep, w):
    """Euclidean projection of w=(x,lambda_g) onto C = A x R_+^{ng}, A={lb<=x<=ub}.
    Both parts are plain elementwise clips (the resolvent/prox of an indicator
    function of a box, or of the nonnegative orthant, is just a projection).
    """
    nvar = gnep.nvar
    x_new = np.clip(w[:nvar], np.asarray(gnep.lb), np.asarray(gnep.ub))
    lam_new = np.maximum(w[nvar:], 0.0)
    return np.concatenate([x_new, lam_new])


def golden_ratio(gnep, x1, lam1=None, x0=None, lam0=None, theta0=1.0, theta_bar=1e6,
                  max_iter=1000, tol=1e-8, stopping="step", verbose=1):
    """Solve a monotone GNEP() with the adaptive Golden Ratio Algorithm (page 3
    of the reference), applied to the lifted primal-dual VI F(x,lambda_g) =
    [M(x) + Jg(x)^T lambda_g ; -g(x)].

    Parameters
    ----------
    gnep : nashopt.GNEP
        The nonlinear GNEP to be solved. Must have no equality constraints
        (Aeq, h); only shared inequality constraints (g, ng) and box
        constraints (lb, ub) are supported.
    x1 : array-like, shape (nvar,)
        Primal iterate x^1 (need not be feasible).
    lam1 : array-like, shape (ng,), optional
        Dual iterate lambda_g^1 >= 0 (default: zeros).
    x0, lam0 : array-like, optional
        Primal/dual iterate z^0 = (x^0, lambda_g^0) used, together with z^1,
        to bootstrap the first adaptive stepsize. Default: equal to x1, lam1
        (in which case the ||z^1-z^0||/||F(z^1)-F(z^0)|| ratio at k=1 is
        undefined and simply skipped, see `theta0`).
    theta0 : float
        Common initial value of theta_0 = theta_{-1} > 0 used at k=1 (before
        any curvature information is available).
    theta_bar : float
        Upper bound theta_bar > 0 on the adaptive stepsize theta_k.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Stopping tolerance (meaning depends on `stopping`).
    stopping : str
        "step" (default): stop when ||z^{k+1}-z^k|| <= tol -- cheap, but only
        an indirect proxy for optimality (small steps do not by themselves
        certify near-optimality).
        "residual": stop when the natural-map residual
            r_theta(z) := (z - P_C(z - theta*F(z))) / theta
        satisfies ||r_theta(z^k)|| <= tol, evaluated at theta = theta_k (the
        current adaptive stepsize). r_theta(z) = 0 for some theta > 0 if and
        only if z solves the VI (equivalently 0 in F(z)+N_C(z)), for ANY
        theta > 0, so this is a genuine first-order optimality measure. Since
        P_C is a cheap elementwise clip here, computing it costs only one
        extra clip per iteration (no extra evaluation of F).
    verbose : int
        0 = silent, 1 = final report, 2 = per-iteration report.

    Returns
    -------
    sol : SimpleNamespace
        x : ndarray, primal solution x*
        lam : ndarray, dual solution lambda_g* (empty if gnep.ng == 0)
        iters : int, number of iterations performed
        residual : float, ||z^{k+1}-z^k|| at termination
        nat_residual : float, ||r_theta(z^k)|| at termination
        converged : bool
        theta : float, last adaptive stepsize theta_k used
        history : dict with 'residual', 'nat_residual' and 'theta' trajectories
        jax_jit_time : float, wall-clock seconds spent jax jit-compiling F
            before the main loop starts

    (C) 2026 A. Bemporad
    """
    _check_supported(gnep)
    if stopping not in ("step", "residual"):
        raise ValueError("stopping must be 'step' or 'residual'.")

    nvar, ng = gnep.nvar, gnep.ng

    x1 = np.asarray(x1, dtype=float).reshape(nvar)
    lam1 = np.zeros(ng) if lam1 is None else np.asarray(lam1, dtype=float).reshape(ng)
    lam1 = np.maximum(lam1, 0.0)
    z1 = np.concatenate([x1, lam1])

    if x0 is None and lam0 is None:
        z0 = z1.copy()
    else:
        x0 = x1 if x0 is None else np.asarray(x0, dtype=float).reshape(nvar)
        lam0 = lam1 if lam0 is None else np.maximum(np.asarray(lam0, dtype=float).reshape(ng), 0.0)
        z0 = np.concatenate([x0, lam0])

    F = _lifted_F(gnep)

    # Trigger and time the jax jit-compilation of F on the actual problem
    # shapes, before the main loop calls it.
    t_jit0 = time.perf_counter()
    F(jnp.asarray(z0)).block_until_ready()
    jax_jit_time = time.perf_counter() - t_jit0

    if verbose > 0:
        print(f"Golden Ratio Algorithm (adaptive): nvar={nvar}, ng={ng}, "
              f"theta0={theta0:.4e}, theta_bar={theta_bar:.4e}")

    z_prev, z_curr = z0, z1
    Fz_prev = np.asarray(F(jnp.asarray(z_prev)))
    zbar_prev = z1.copy()  # zbar^0 := z^1
    theta_prev, theta_prev2 = theta0, theta0  # theta_0 = theta_{-1} = theta0

    history_res, history_nat_res, history_theta = [], [], []
    converged = False
    k = 0
    for k in range(1, max_iter + 1):
        Fz_curr = np.asarray(F(jnp.asarray(z_curr)))

        dz = z_curr - z_prev
        dF = Fz_curr - Fz_prev
        ndF2 = float(dF @ dF)
        ratio_term = (9.0 / (16.0 * theta_prev2)) * (float(dz @ dz) / ndF2) if ndF2 > 1e-300 else np.inf

        theta_k = min((10.0 / 9.0) * theta_prev, ratio_term, theta_bar)

        # Natural-map residual r_theta(z^k) = (z^k - P_C(z^k - theta_k*F(z^k)))/theta_k:
        # zero iff z^k solves the VI. Reuses Fz_curr (already computed above), so this
        # only costs one extra elementwise clip.
        nat_res = (z_curr - _project_onto_C(gnep, z_curr - theta_k * Fz_curr)) / theta_k
        nat_res_norm = float(np.linalg.norm(nat_res))

        zbar_k = (z_curr + 2.0 * zbar_prev) / 3.0

        w = zbar_k - theta_k * Fz_curr
        z_next = _project_onto_C(gnep, w)

        step_norm = float(np.linalg.norm(z_next - z_curr))
        history_res.append(step_norm)
        history_nat_res.append(nat_res_norm)
        history_theta.append(theta_k)

        check_norm = nat_res_norm if stopping == "residual" else step_norm

        if verbose > 1:
            print(f"  iter {k:4d}: theta_k = {theta_k:.4e}, "
                  f"||z^(k+1)-z^k|| = {step_norm:.6e}, ||r_theta(z^k)|| = {nat_res_norm:.6e}")

        z_prev, Fz_prev = z_curr, Fz_curr
        z_curr = z_next
        zbar_prev = zbar_k
        theta_prev2 = theta_prev
        theta_prev = theta_k

        if check_norm <= tol:
            converged = True
            break

    x_star, lam_star = z_curr[:nvar], z_curr[nvar:]

    if verbose > 0:
        status = "converged" if converged else "did not converge (max_iter reached)"
        print(f"Golden Ratio Algorithm {status}: {k} iterations, "
              f"||z^(k+1)-z^k|| = {history_res[-1]:.4e}, "
              f"||r_theta(z^k)|| = {history_nat_res[-1]:.4e}, theta_k = {history_theta[-1]:.4e}")

    sol = SimpleNamespace(
        x=x_star,
        lam=lam_star,
        iters=k,
        residual=history_res[-1] if history_res else np.nan,
        nat_residual=history_nat_res[-1] if history_nat_res else np.nan,
        converged=converged,
        theta=history_theta[-1] if history_theta else theta0,
        history={"residual": history_res, "nat_residual": history_nat_res, "theta": history_theta},
        jax_jit_time=jax_jit_time,
    )
    return sol


def solve_golden_ratio(gnep, x0=None, solver_opts=None, verbose=1):
    """
    Solve a monotone GNEP() via the adaptive Golden Ratio Algorithm (golden_ratio
    above), returning a solution object with the same layout as GNEP.solve().

    Parameters:
    -----------
    gnep : GNEP
        Nonlinear GNEP object (nashopt.nonlinear.gnep_base.GNEP), with only
        shared inequality constraints (g, ng) and box constraints (lb, ub).
    x0 : array-like or None
        Initial guess for the Nash equilibrium x, used as the primal iterate
        x1. Used only if solver_opts does not itself provide 'x1'.
    solver_opts : dict or None
        Keyword arguments forwarded to golden_ratio(gnep, ...): x1, lam1, x0,
        lam0, theta0, theta_bar, max_iter, tol, stopping, verbose. See
        golden_ratio's docstring for details. If None, golden_ratio's own
        defaults are used.
    verbose : int, optional
        Verbosity level. 0: silent. >0: termination report. Used only if
        solver_opts does not itself specify 'verbose' (in which case that
        value is passed through to golden_ratio for per-iteration reporting).

    Returns:
    --------
    sol : SimpleNamespace
        Solution object with fields:
        x : ndarray
            Computed GNE solution.
        res : ndarray
            1-element array holding the final step norm ||z^{k+1}-z^k||.
        lam : ndarray
            Shared inequality multiplier lambda_g* (empty if gnep.ng == 0).
        stats : Statistics about the optimization result.
    """
    opts = dict(solver_opts) if solver_opts is not None else {}
    opts.setdefault("x1", x0 if x0 is not None else np.zeros(gnep.nvar))
    opts.setdefault("verbose", 2 if verbose > 1 else 0)

    t0 = time.perf_counter()
    result = golden_ratio(gnep, **opts)
    t0 = time.perf_counter() - t0

    converged = result.converged

    if verbose > 0:
        color = "\033[1;32m" if converged else "\033[1;31m"
        status = "converged" if converged else "reached the maximum number of iterations"
        print(f"{color}Golden Ratio Algorithm {status} after {result.iters} iterations: "
              f"||z^(k+1)-z^k|| = {result.residual:.3e}, ||r_theta(z)|| = {result.nat_residual:.3e}, "
              f"time = {t0:.3f} seconds.\033[0m")
        if not converged:
            print("\033[1;33mWarning: maximum number of iterations reached; "
                  "an equilibrium may not have been found.\033[0m")

    stats = SimpleNamespace()
    stats.solver = "golden_ratio"
    stats.kkt_evals = result.iters
    stats.elapsed_time = t0
    stats.jax_jit_time = result.jax_jit_time
    stats.status_str = "converged" if converged else "max_iterations_reached"
    stats.info = {"converged": converged, "residual": result.residual,
                  "nat_residual": result.nat_residual, "theta": result.theta}

    sol = SimpleNamespace()
    sol.x = np.asarray(result.x)
    sol.res = np.atleast_1d(np.asarray(result.residual))
    sol.lam = np.asarray(result.lam)
    sol.stats = stats
    sol.norm_residual = float(result.residual)
    return sol
