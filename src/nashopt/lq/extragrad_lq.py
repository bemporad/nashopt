"""
Korpelevich's Extragradient method for the variational GNE of LQ games.

Player i solves:
    min_{x_i}  (1/2) x^T Q_i x + c_i^T x
    s.t.   A x <= b,   Aeq x = beq,   lb <= x <= ub

The variational GNE is a solution to the VI: find x* in X s.t.
    F(x*)^T (y - x*) >= 0  for all y in X
where F(x) = G x + r is the (affine) pseudogradient,
    G[si:ei, :] = Q[i][si:ei, :]
    r[si:ei]    = c[i][si:ei]

Algorithm (Korpelevich 1976):
    y^k     = P_X(x^k - alpha * F(x^k))
    x^{k+1} = P_X(x^k - alpha * F(y^k))

where P_X is the projection onto the feasible set X.
Step-size condition for convergence: alpha < 1 / spectral_norm(G).
DAQP is used for each projection, with active-set warm-starting between steps.

This module mirrors operator_extrapolation.py's `stopping`/`check_every` options:
"step" (default) stops on the cheap ||x^{k+1}-x^k|| proxy; "residual" stops on the
natural-map residual r_alpha(x) = (x - y^k)/alpha -- for extragradient this comes
for FREE, since y^k = P_X(x^k - alpha*F(x^k)) is already computed as the method's
own first half-step, unlike operator_extrapolation's own "residual" mode which
needs one extra projection; "gap" stops on gap(x) = max_{z in X} F(x)^T (x-z),
computed by solving an LP each check.

[1] G.M. Korpelevich, "The extragradient method for finding saddle points and other problems," Matecon, vol. 12, pp. 747-756, 1976.

(C) 2026 A. Bemporad
"""

import numpy as np
from types import SimpleNamespace
import daqp


def extragradient_gnep(
    dim, Q, c, A=None, b=None, lb=None, ub=None, Aeq=None, beq=None,
    tol=1e-8, maxiter=1000, alpha=None, x0=None, stopping="step", check_every=1,
    verbose=False, get_lambda=False
):
    """
    Korpelevich extragradient method for variational GNE of a LQ game.

    Parameters
    ----------
    dim : list of int
        Decision-variable count per agent; nvar = sum(dim).
    Q : list of ndarray (nvar x nvar)
        Symmetrized full quadratic cost matrices.
    c : list of ndarray (nvar,)
        Linear cost vectors; only the i-th block c[i][si:ei] enters F.
    A : ndarray (m x nvar), optional
        Shared inequality constraint matrix.
    b : ndarray (m,), optional
        Shared inequality RHS.
    lb : ndarray (nvar,), optional
        Variable lower bounds.
    ub : ndarray (nvar,), optional
        Variable upper bounds.
    Aeq : ndarray (q x nvar), optional
        Shared equality constraint matrix.
    beq : ndarray (q,), optional
        Shared equality RHS.
    tol : float
        Stopping tolerance (meaning depends on `stopping`).
    maxiter : int
        Maximum iterations.
    alpha : float, optional
        Step size. Default: 0.99 / spectral_norm(G).
    x0 : ndarray (nvar,), optional
        Initial point (projected to X if infeasible). Default: zeros.
    stopping : str
        "step" (default): stop when ||x^{k+1}-x^k|| <= tol -- cheap (no extra
        projection or LP solve), but only an indirect proxy for optimality.
        "residual": stop when the natural-map residual
            r_alpha(x^k) := (x^k - y^k) / alpha
        satisfies ||r_alpha(x^k)|| <= tol, where y^k = P_X(x^k - alpha*F(x^k))
        is already computed as the method's own first half-step -- so, unlike
        operator_extrapolation_gnep's "residual" mode, this costs NOTHING
        extra. r_alpha(x) = 0 if and only if x solves the VI, so this is a
        genuine first-order optimality measure.
        "gap": stop when gap(x) = max_{z in X} F(x)^T (x - z) <= tol, computed
        by solving an LP each check. NOT recommended when X is
        unbounded: the LP can then be unbounded.
    check_every : int
        Only used when stopping in ("residual", "gap"): evaluate the (extra,
        for "gap") check every `check_every` iterations rather than every
        iteration, to amortize its cost. The step-based ||x^{k+1}-x^k|| is
        still tracked every iteration regardless of `stopping`.
    verbose : bool
        Print per-iteration residuals.
    get_lambda : bool
        If True, return the dual variables associated with the solution found by solving an LP.


    Returns
    -------
    SimpleNamespace with fields:
        x              : ndarray -- approximate variational GNE
        elapsed_time   : float   -- wall-clock seconds (excluding setup)
        status_str     : str     -- 'converged' or 'max_iterations_reached'
        num_iters      : int     -- iterations performed
        num_daqp_iters : int     -- total lower-level DAQP active-set iterations
                         spent across every DAQP solve() call of this run (the
                         two projections per outer iteration, any "gap" checks,
                         and the final get_lambda LP if requested), including
                         any cold-start retries
        info           : dict    -- {'converged': bool, 'final_gap': float
                         (final "step"/"residual"/"gap" check value, whichever
                         `stopping` used), 'alpha': float, 'stopping': str}
        history        : dict    -- {'step': [...] (every iteration), 'check':
                         [...] (every `check_every` iterations when stopping in
                         ("residual", "gap"), empty when stopping="step")}
    """
    import time

    if stopping not in ("step", "residual", "gap"):
        raise ValueError("stopping must be 'step', 'residual', or 'gap'.")

    N = len(dim)
    nvar = sum(dim)

    # Pseudogradient F(x) = G_mat @ x + r_vec
    G_mat = np.zeros((nvar, nvar))
    r_vec = np.zeros(nvar)
    offset = 0
    for i, ni in enumerate(dim):
        si, ei = offset, offset + ni
        G_mat[si:ei, :] = Q[i][si:ei, :]
        r_vec[si:ei] = c[i][si:ei]
        offset += ni

    if alpha is None:
        L = np.linalg.norm(G_mat, 2)
        alpha = 0.99 / max(L, 1e-12)

    # ---- Build DAQP constraint data for the projection QP ----
    # Projection of y: min 1/2 ||x - y||^2  s.t. bl_d <= AA_d x <= bu_d
    # DAQP sense codes: 0=inactive ineq, 1=active at bu, 3=active at bl, 5=equality
    m    = A.shape[0]   if A   is not None else 0
    q_eq = Aeq.shape[0] if Aeq is not None else 0
    has_box = (lb is not None) or (ub is not None)

    rows, bu_parts, bl_parts, sense_parts = [], [], [], []

    if m > 0:
        rows.append(np.asarray(A,   dtype=np.float64))
        bu_parts.append(np.asarray(b, dtype=np.float64))
        bl_parts.append(np.full(m, -np.inf))
        sense_parts.append(np.zeros(m, dtype=np.int32))

    if q_eq > 0:
        rows.append(np.asarray(Aeq,  dtype=np.float64))
        bu_parts.append(np.asarray(beq, dtype=np.float64))
        bl_parts.append(np.asarray(beq, dtype=np.float64))
        sense_parts.append(5 * np.ones(q_eq, dtype=np.int32))

    if has_box:
        rows.append(np.eye(nvar))
        _ub = np.asarray(ub, dtype=np.float64) if ub is not None else np.full(nvar, np.inf)
        _lb = np.asarray(lb, dtype=np.float64) if lb is not None else np.full(nvar, -np.inf)
        bu_parts.append(_ub)
        bl_parts.append(_lb)
        sense_parts.append(np.zeros(nvar, dtype=np.int32))

    if rows:
        AA_d       = np.vstack(rows)
        bu_d       = np.concatenate(bu_parts)
        bl_d       = np.concatenate(bl_parts)
        sense_base = np.concatenate(sense_parts).astype(np.int32)
    else:
        AA_d       = np.zeros((0, nvar), dtype=np.float64)
        bu_d       = np.zeros(0, dtype=np.float64)
        bl_d       = np.zeros(0, dtype=np.float64)
        sense_base = np.zeros(0, dtype=np.int32)

    ncon_d    = AA_d.shape[0]
    eq_mask   = (sense_base == 5)             # equality rows -- sense never changes
    finite_bl = np.isfinite(bl_d)             # rows with a finite lower bound (box rows)
    Q_proj    = np.eye(nvar, dtype=np.float64)

    Q_lp = np.zeros((nvar, nvar), dtype=np.float64)

    total_daqp_iters = 0

    def _daqp_solve(*args, **kwargs):
        """daqp.solve wrapper that accumulates DAQP's own reported active-set
        iteration count (report["iterations"]) across every call this run
        makes -- i.e. the actual lower-level active-set work done, as opposed
        to the number of outer iterations of this function."""
        nonlocal total_daqp_iters
        x_sol, y_sol, flag, report = daqp.solve(*args, **kwargs)
        if isinstance(report, dict):
            total_daqp_iters += int(report.get("iterations", 0))
        return x_sol, y_sol, flag, report

    def compute_gap(x, Fx):
        # gap(x) = max_{z in X} F(x)^T (x - z) = F(x)^T x - min_{z in X} F(x)^T z
        z, _, flag, _ = _daqp_solve(Q_lp, np.asarray(Fx, dtype=np.float64),
                                AA_d, bu_d, bl_d, sense=sense_base.copy())
        return float(Fx @ x) - float(Fx @ z) if flag == 1 else np.inf

    def project(y, sense_ws):
        """Project y onto X; return (projected_point, updated_sense)."""
        c_proj = -np.asarray(y, dtype=np.float64)
        x_p, _, flag, report = _daqp_solve(
            Q_proj, c_proj, AA_d, bu_d, bl_d, sense=sense_ws.copy()
        )
        if flag != 1:
            # warm-start failed -- retry cold
            x_p, _, flag, report = _daqp_solve(
                Q_proj, c_proj, AA_d, bu_d, bl_d, sense=sense_base.copy()
            )
        lam = report["lam"] if (flag == 1 and "lam" in report) else np.zeros(ncon_d)
        # Rebuild warm-start sense from dual variables
        new_sense = sense_base.copy()         # preserves equality senses (5)
        ineq = ~eq_mask
        new_sense[ineq & (lam > 1e-8)]              = 1  # active at upper bound
        new_sense[ineq & finite_bl & (lam < -1e-8)] = 3  # active at lower bound (box only)
        return x_p, new_sense

    # ---- Initialise at a feasible point ----
    x = np.asarray(x0, dtype=np.float64).copy() if x0 is not None else np.zeros(nvar)
    x, sense_x = project(x, sense_base.copy())
    sense_y = sense_base.copy()

    t_start = time.perf_counter()
    status_str = "max_iterations_reached"
    check_norm = np.nan

    history_step, history_check = [], []
    k = -1

    for k in range(maxiter):
        Fx = G_mat @ x + r_vec

        y, sense_y = project(x - alpha * Fx, sense_y)
        Fy = G_mat @ y + r_vec
        x_new, sense_x = project(x - alpha * Fy, sense_x)

        step_norm = float(np.linalg.norm(x_new - x))
        history_step.append(step_norm)
        check_norm = step_norm

        do_check = stopping in ("residual", "gap") and ((k + 1) % check_every == 0)
        if do_check:
            if stopping == "residual":
                # y = P_X(x - alpha*F(x)) is already the method's own first
                # half-step above, so this natural-map residual is free.
                check_norm = float(np.linalg.norm(x - y)) / alpha
            else:  # "gap"
                check_norm = compute_gap(x, Fx)
            history_check.append(check_norm)

        if verbose:
            msg = f"  extragradient iter {k+1}: ||x^(k+1)-x^k|| = {step_norm:.6e}"
            if do_check:
                msg += f", {stopping} check = {check_norm:.6e}"
            print(msg)

        x = x_new

        if check_norm <= tol:
            status_str = "converged"
            break

    elapsed = time.perf_counter() - t_start

    if get_lambda:
        # Solve QP to get dual variables associated with the solution, which is assume feasible
        #
        # min_{lambda,nu} (b-Ax)'lambda + .5*rho*||F(x)+A'lambda+E'\nu||^2 + .5*gamma*(||lambda||^2 + ||nu||^2)
        # s.t. lambda >=0
        Fx = G_mat @ x + r_vec
        rho=1.e5 # high penalty on violation of stationarity condition
        gamma = 1.e-5 # small regularization to ensure positive definiteness
        Q_qp = gamma*np.eye(m + q_eq, dtype=np.float64)
        if m>0:
            Q_qp[:m, :m] = rho*A@A.T
        if q_eq>0:
            Q_qp[m:, m:] = rho*Aeq@Aeq.T
        if m>0 and q_eq>0:
            Q_qp[:m, m:] = rho*A@Aeq.T
            Q_qp[m:, :m] = rho*Aeq@A.T
        c_qp = np.zeros(m + q_eq, dtype=np.float64)
        if m>0:
            c_qp[:m] = np.maximum(b - A @ x,0.) + rho*A@Fx
        if q_eq>0:
            c_qp[m:] = rho*Aeq@Fx
        if m>0:
            A_qp = np.zeros((m, m + q_eq), dtype=np.float64)
            A_qp[:, :m] = np.eye(m)
            bl_qp = np.zeros(m, dtype=np.float64)
            bu_qp = np.full(m, np.inf, dtype=np.float64)

        # Call DAQP to solve the LP
        sense = np.zeros(m, dtype=np.int32)
        #sense[:nvar] = 5  # equality constraints
        lam_nu, _, flag, report = _daqp_solve(Q_qp, c_qp, A_qp, bu_qp, bl_qp, sense)
        lam = lam_nu[:m] if m>0 else np.zeros(0)
        nu = lam_nu[m:] if q_eq>0 else np.zeros(0)
    else:
        lam = None
        nu = None

    return SimpleNamespace(
        x=x,
        lam=lam,
        nu=nu,
        elapsed_time=elapsed,
        status_str=status_str,
        num_iters=k + 1,
        num_daqp_iters=total_daqp_iters,
        info={"converged": status_str == "converged", "final_gap": float(check_norm),
              "alpha": float(alpha), "stopping": stopping},
        history={"step": history_step, "check": history_check},
    )
