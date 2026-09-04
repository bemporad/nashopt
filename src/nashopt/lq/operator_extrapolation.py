"""
Kotsalis-Lan-Li (2023) Operator Extrapolation method for strongly-monotone variational GNE of LQ games.

Player i solves:
    min_{x_i}  (1/2) x^T Q_i x + c_i^T x
    s.t.   A x <= b,   Aeq x = beq,   lb <= x <= ub

The variational GNE is a solution to the VI: find x* in X s.t.
    F(x*)^T (y - x*) >= 0  for all y in X
where F(x) = G x + r is the (affine) pseudogradient,
    G[si:ei, :] = Q[i][si:ei, :]
    r[si:ei]    = c[i][si:ei]

Algorithm:

    x^{k+1} = P_X(gamma^k ( F(x^k)+lambda^k(F(x^k)-F(x^{k-1})) )' x^k +0.5||x-x^k||_2^2

where P_X is the projection onto the feasible set X, and

    gamma^k = safety/(2 * L)
    lambda^k = L/(L+mu)

where L is the Lipschitz constant of F and mu is the strong monotonicity constant of F.
In particular, since the problem is quadratic, ||F(x)-F(y)||_2 <= ||G||_2 ||x-y||_2, so

L = spectral_norm(G)

and (F(x)-F(y))'(x-y) = (x-y)'G'(x-y) = (x-y)'(G+G')/2 (x-y) >= mu ||x-y||_2^2 for

mu = min_eig(G+G')/2

DAQP is used for each projection, with active-set warm-starting between steps.

Because F is exactly affine and X exactly polyhedral, once the DAQP projection's
active-constraint pattern (already tracked for warm-starting) has stopped changing
across outer iterations, the remaining problem is a linear system, not an
optimization problem: option `active_set_finish` (default True) monitors this
pattern and, once stable, attempts one exact finish via the reduced KKT system,
terminating in finite time instead of continuing the asymptotic linear-rate
iterations, the finite-termination idea of "DR-DAQP" in (Arnstrom, Benenati, Belgioioso, 2026), applied to the operator-extrapolation outer loop.

[1] G. Kotsalis, G. Lan, T. Li, "Simple and Optimal Methods for Stochastic Variational Inequalities, I: Operator Extrapolation", SIAM Journal on Optimization, vol. 32, no. 3, pp. 2041-2073, 2022.

(C) 2026 A. Bemporad
"""

import numpy as np
from types import SimpleNamespace
import daqp

_MU_TOL = 1e-8               # below this, G is not trusted as strongly monotone
_ACTIVE_SET_PATIENCE = 3     # consecutive outer iterations the DAQP active-set
                             # pattern must stay unchanged before it is trusted
                             # enough to attempt the exact finish below
_ACTIVE_SET_FEAS_TOL = 1e-7  # primal/dual feasibility tolerance used to
                             # VALIDATE (not just propose) a guessed active set
                             # before accepting its exact finish


def _exact_finish(G_mat, r_vec, AA_d, bu_d, bl_d, sense, nvar, feas_tol=_ACTIVE_SET_FEAS_TOL):
    """Attempt to recover the EXACT variational GNE from a guessed constraint
    active set `sense` (DAQP convention: 0 = inactive inequality, 1 = active
    at its upper bound, 3 = active at its lower bound, 5 = equality, always
    active), by solving the reduced KKT linear system

        G x + r + A_active^T lam_active = 0
        A_active x = target_active   (bu_d where active-at-upper, bl_d where active-at-lower)
        A_eq x = b_eq

    exactly (a single linear solve), then VALIDATING the guess -- dual
    feasibility of the active rows (lam_active >= 0 where active at the upper
    bound, lam_active <= 0 where active at the lower bound -- DAQP's own signed
    multiplier convention, see project()/solve_qp() above) and primal
    feasibility of the rows guessed inactive. Stationarity and the
    active/equality rows hold to machine precision by construction of the
    solve itself and need no separate check.

    This is the finite-termination idea of "DR-DAQP" (Arnstrom, Benenati,
    Belgioioso, 2026), applied here to operator
    extrapolation's own outer iteration.

    Returns (x, valid): valid=False if the reduced system is singular/
    rank-deficient (the guessed active set is redundant, so cannot be
    trusted) or fails either feasibility check; x is still returned (possibly
    infeasible) for diagnostic purposes even when valid=False.
    """
    active = (sense == 1) | (sense == 3)
    idx_active = np.where(active)[0]
    idx_eq = np.where(sense == 5)[0]
    idx_inactive = np.where(sense == 0)[0]

    n_act = idx_active.size
    n_eq = idx_eq.size
    A_act = AA_d[idx_active, :] if n_act > 0 else np.zeros((0, nvar))
    # target = upper bound where active-at-upper (sense==1), lower bound where active-at-lower (sense==3)
    target_act = np.where(sense[idx_active] == 1, bu_d[idx_active], bl_d[idx_active]) if n_act > 0 else np.zeros(0)
    A_eq = AA_d[idx_eq, :] if n_eq > 0 else np.zeros((0, nvar))
    b_eq = bu_d[idx_eq] if n_eq > 0 else np.zeros(0)  # bu_d == bl_d on equality rows

    n_tot = nvar + n_act + n_eq
    KKT = np.zeros((n_tot, n_tot))
    rhs = np.zeros(n_tot)
    KKT[:nvar, :nvar] = G_mat
    rhs[:nvar] = -r_vec
    if n_act > 0:
        KKT[:nvar, nvar:nvar + n_act] = A_act.T
        KKT[nvar:nvar + n_act, :nvar] = A_act
        rhs[nvar:nvar + n_act] = target_act
    if n_eq > 0:
        KKT[:nvar, nvar + n_act:] = A_eq.T
        KKT[nvar + n_act:, :nvar] = A_eq
        rhs[nvar + n_act:] = b_eq

    try:
        sol, _, rank, _ = np.linalg.lstsq(KKT, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None, False
    if rank < n_tot:
        return None, False  # redundant/degenerate active set: cannot trust this guess

    x = sol[:nvar]

    if n_act > 0:
        lam_active = sol[nvar:nvar + n_act]
        active_sense = sense[idx_active]
        if np.any(lam_active[active_sense == 1] < -feas_tol):
            return x, False
        if np.any(lam_active[active_sense == 3] > feas_tol):
            return x, False

    if idx_inactive.size > 0:
        vals = AA_d[idx_inactive, :] @ x
        if np.any(vals > bu_d[idx_inactive] + feas_tol):
            return x, False
        finite_lo = np.isfinite(bl_d[idx_inactive])
        if np.any(finite_lo & (vals < bl_d[idx_inactive] - feas_tol)):
            return x, False

    return x, True


def operator_extrapolation_gnep(
    dim, Q, c, A=None, b=None, lb=None, ub=None, Aeq=None, beq=None,
    x0=None, L=None, mu=None, safety=1.0,
    tol=1e-8, maxiter=1000, stopping="step", check_every=1,
    check_monotonicity=True, active_set_finish=True,
    verbose=False, get_lambda=False
):
    """
    Kotsalis-Lan-Li (2023) Operator Extrapolation method for strongly-monotone variational GNE of LQ games.

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
    x0 : ndarray (nvar,), optional
        Initial iterate x_0 = x_1 (need not be feasible; it is projected onto X before the
        first iteration). Default: zeros.
    L, mu : float, optional
        Lipschitz constant / monotonicity modulus of F. If None (default), computed exactly
        from G_mat (F is affine, so its Jacobian equals G_mat everywhere):
            L  = ||G_mat||_2                 (spectral norm)
            mu = lambda_min(0.5*(G_mat+G_mat.T))
    safety : float
        Safety factor in (0,1] applied to the theoretical step size gamma = safety/(2*L)
        (safety=1, the default, is theoretically justified since F is exactly affine and X
        is the true feasible set).
    tol : float
        Stopping tolerance (meaning depends on `stopping`).
    maxiter : int
        Maximum number of outer iterations.
    stopping : str
        "step" (default): stop when ||x_{k+1}-x_k|| <= tol -- cheap (no extra projection or
        LP solve), but only an indirect proxy for optimality.
        "residual": stop when the natural-map residual
            r_gamma(x) := (x - P_X(x - gamma*F(x))) / gamma
        satisfies ||r_gamma(x)|| <= tol. r_gamma(x) = 0 if and only if x solves the VI, so this
        is a genuine first-order optimality measure, at the cost of one extra DAQP projection
        per check.
        "gap": stop when gap(x) = max_{z in X} F(x)^T (x - z) <= tol, computed by solving an LP
        each check -- another genuine optimality measure.
    check_every : int
        Only used when stopping in ("residual", "gap"): evaluate the (extra) check every
        `check_every` iterations rather than every iteration, to amortize its cost. The
        step-based ||x_{k+1}-x_k|| is still tracked every iteration regardless of `stopping`.
    check_monotonicity : bool
        The extrapolation weight eta = L/(L+mu) needs mu's actual numeric value, so
        mu = lambda_min(sym(G_mat)) is always computed here when not supplied via `mu`. This
        flag only controls whether the result is validated: if True (default) and mu <=
        _MU_TOL, raise ValueError (this solver has no proximal-point fallback for the
        merely-monotone case -- use solver='qp_gnep' or 'extragradient' instead); if False, a
        low/negative mu is used as-is (informational only -- it typically shows up as eta
        close to 1 and/or excessive iterations, with no hard error).
    active_set_finish : bool
        If True (default), monitor the active-constraint pattern DAQP returns at each outer
        iteration's projection (already computed for warm-starting); once it has stayed
        unchanged for _ACTIVE_SET_PATIENCE consecutive iterations, attempt one exact finish
        (_exact_finish): solve the reduced KKT linear system implied by that pattern and, if
        it VALIDATES (dual-feasible active multipliers, primal-feasible inactive rows), accept
        it as the exact variational GNE and stop -- rather than continuing operator
        extrapolation's asymptotic linear-rate iterations. This is the finite-termination idea
        of "DR-DAQP" (Arnstrom, Benenati, Belgioioso, 2026), applied to this
        method's own outer loop. Set False to recover plain operator extrapolation with no exact-finish
        attempts.
    verbose : bool or int
        False/0 = silent, True/1 = final report, 2 = per-iteration report.
    get_lambda : bool
        If True, return the dual variables associated with the solution found by solving an LP.

    Returns
    -------
    SimpleNamespace with fields:
        x              : ndarray -- approximate variational GNE
        elapsed_time   : float   -- wall-clock seconds (excluding setup)
        status_str     : str     -- 'converged', 'max_iterations_reached', or
                         'active_set_identified' (exact finish accepted; only possible when
                         active_set_finish=True)
        num_iters      : int     -- outer iterations performed (cut short by the exact finish
                         when status_str is 'active_set_identified')
        num_daqp_iters : int     -- total lower-level DAQP active-set iterations spent across
                         every DAQP solve() call of this run (projections, the residual/gap
                         checks, and the final get_lambda LP if requested), including any
                         cold-start retries
        info           : dict    -- {'converged': bool, 'final_gap': float (final "step"/
                         "residual"/"gap" check value, whichever `stopping` used), 'L': float,
                         'mu': float, 'gamma': float, 'eta': float, 'stopping': str,
                         'active_set_finish': bool, 'active_set_hit': bool -- True iff the
                         exact finish fired}
        history        : dict    -- {'step': [...] (every iteration), 'check': [...] (every
                         `check_every` iterations when stopping in ("residual", "gap"), empty
                         when stopping="step")}
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

    if L is None or mu is None:
        L_est = float(np.linalg.norm(G_mat, 2))
        mu_est = float(np.min(np.linalg.eigvalsh((G_mat + G_mat.T) / 2)))
        L = L_est if L is None else L
        mu = mu_est if mu is None else mu

    if check_monotonicity and mu <= _MU_TOL:
        raise ValueError(
            f"operator_extrapolation_gnep: mu = lambda_min(sym(G)) = {mu:.3e} <= {_MU_TOL:.0e}; "
            "G is not (numerically) strongly monotone. This solver has no proximal-point "
            "fallback for the merely-monotone case -- use solver='qp_gnep' or 'extragradient' "
            "instead, or pass check_monotonicity=False to bypass this check at your own risk."
        )

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
        """daqp.solve wrapper that accumulates the reported active-set iteration count."""
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

    def solve_qp(xt, F_xt, F_xt1, gamma, eta, sense_ws):
        """Operator extrapolation step: proximal QP with extrapolated pseudogradient.

        Solves min_{x in X} (1/2)||x - xt||^2 + gamma * F_hat^T x

        where F_hat = F(xt) + eta*(F(xt) - F(xt1)).
        """
        F_extrap = F_xt + eta * (F_xt - F_xt1)
        c_proj = np.asarray(gamma * F_extrap - xt, dtype=np.float64)
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
    sense_res = sense_base.copy()  # independent warm start for the "residual" mode's own projection

    t_start = time.perf_counter()
    status_str = "max_iterations_reached"
    check_norm = np.nan

    gamma = safety / (2 * L)
    eta = L / (L + mu)  # extrapolation weight (Kotsalis-Lan-Li's lambda_t)

    xt = x.copy() # current iterate
    F_xt  = G_mat @ x  + r_vec
    F_xt1 = F_xt.copy()

    # Active-set finish (DR-DAQP idea, see _exact_finish): only meaningful
    # when there is a constraint pattern to identify at all.
    active_set_track = active_set_finish and ncon_d > 0
    prev_sense = sense_y.copy() if active_set_track else None
    stable_count = 1 if active_set_track else 0
    attempted_this_run = False
    active_set_hit = False

    history_step, history_check = [], []
    k = -1

    for k in range(maxiter):
        x_new, sense_y = solve_qp(xt, F_xt, F_xt1, gamma, eta, sense_y)

        step_norm = float(np.linalg.norm(x_new - xt))
        history_step.append(step_norm)
        check_norm = step_norm

        do_check = stopping in ("residual", "gap") and ((k + 1) % check_every == 0)
        if do_check:
            if stopping == "residual":
                x_nat, sense_res = project(xt - gamma * F_xt, sense_res)
                check_norm = float(np.linalg.norm(xt - x_nat)) / gamma
            else:  # "gap"
                check_norm = compute_gap(xt, F_xt)
            history_check.append(check_norm)

        if verbose > 1:
            msg = f"  op-extrap. iter {k+1}: ||x^(k+1)-x^k|| = {step_norm:.6e}"
            if do_check:
                msg += f", {stopping} check = {check_norm:.6e}"
            print(msg)

        if active_set_track:
            if np.array_equal(sense_y, prev_sense):
                stable_count += 1
            else:
                prev_sense = sense_y.copy()
                stable_count = 1
                attempted_this_run = False
            if stable_count >= _ACTIVE_SET_PATIENCE and not attempted_this_run:
                attempted_this_run = True
                x_ex, ok = _exact_finish(G_mat, r_vec, AA_d, bu_d, bl_d, sense_y, nvar)
                if ok:
                    x_new = x_ex
                    active_set_hit = True
                    if verbose > 1:
                        print(f"  op-extrap. iter {k+1}: active set stable for "
                              f"{stable_count} iterations -- exact finish accepted")

        F_xt1 = F_xt.copy()
        F_xt = G_mat @ x_new + r_vec

        xt = x_new

        if active_set_hit:
            status_str = "active_set_identified"
            if stopping == "gap":
                check_norm = compute_gap(xt, F_xt)  # refresh at the exact-finish point for reporting
            break

        if check_norm <= tol:
            status_str = "converged"
            break

    x = xt  # returned solution is the last iterate, not the initial point

    elapsed = time.perf_counter() - t_start
    step_gamma = gamma  # save the step size before `gamma` is reused below

    if get_lambda:
        # min_{lambda,nu} (b-Ax)'lambda + .5*rho*||F(x)+A'lambda+E'\nu||^2 + .5*gamma*(||lambda||^2 + ||nu||^2)
        # s.t. lambda >=0
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
            c_qp[:m] = np.maximum(b - A @ x,0.) + rho*A@F_xt
        if q_eq>0:
            c_qp[m:] = rho*Aeq@F_xt
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

    if verbose:
        if status_str == "active_set_identified":
            status_msg = "converged (active set identified, exact finish)"
        elif status_str == "converged":
            status_msg = "converged"
        else:
            status_msg = "did not converge (max_iter reached)"
        msg = (f"Operator Extrapolation {status_msg}: {k+1} outer iterations, "
               f"{total_daqp_iters} DAQP active-set iterations, "
               f"||x^(k+1)-x^k|| = {history_step[-1] if history_step else float('nan'):.4e}")
        if stopping != "step":
            msg += f", {stopping} check = {check_norm:.4e}"
        print(msg)

    return SimpleNamespace(
        x=x,
        lam=lam,
        nu=nu,
        elapsed_time=elapsed,
        status_str=status_str,
        num_iters=k + 1,
        num_daqp_iters=total_daqp_iters,
        info={"converged": status_str in ("converged", "active_set_identified"),
              "final_gap": float(check_norm),
              "L": float(L),
              "mu": float(mu),
              "gamma": float(step_gamma),
              "eta": float(eta),
              "stopping": stopping,
              "active_set_finish": active_set_finish,
              "active_set_hit": active_set_hit},
        history={"step": history_step, "check": history_check}
    )
