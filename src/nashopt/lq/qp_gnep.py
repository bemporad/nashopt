"""
QP-based solver for the variational GNE of LQ games via KKT conditions.

Player i solves:
    min_{x_i}  (1/2) x^T Q_i x + c_i^T x_i
    s.t.   A x <= b,   Aeq x = beq,   lb <= x <= ub

The variational GNE satisfies the KKT system:
    F x + f + A^T lambda + Aeq^T mu = 0
    A x <= b,   lambda >= 0,   lambda^T (A x - b) = 0
    Aeq x = beq

where F[si:ei, :] = Q[i][si:ei, :] and f[si:ei] = c[i][si:ei].

The KKT system is solved as a QP, either directly or via proximal point iterations,
each one solving a QP. The only assumption is mere monotonicity of the pseudogradient matrix F.

[1] A. Bemporad, T. Tatarenko, "Solving Monotone Linear-Quadratic Generalized Nash Equilibrium Problems via Quadratic Programming," arXiv preprint 2608.07336, 2026

(C) 2026 A. Bemporad
"""

import numpy as np
from scipy.linalg import block_diag, lu_factor, lu_solve
from scipy.sparse import csc_matrix as sp, eye as sp_eye, vstack as sp_vstack
from qpsolvers import solve_qp
import daqp
import time
from types import SimpleNamespace


def qp_gnep(
    dim, Q, c, A=None, b=None, lb=None, ub=None, Aeq=None, beq=None,
    proximal=True, solver="daqp", rho=1.e-4, tol=1e-4, maxiter=1000, hessian_reg=1e-8, verbose=False,
    reduced=True, anderson=False, guler=True
):
    """
    KKT-based QP solver for the variational GNE of a LQ game,
    with proximal point regularization for mere monotonicity.

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
    proximal : bool
        If True, use proximal point iterations (this can use QP solvers that require strong convexity).
        If False, call solve_qp once directly (only works with QP solvers that do not require strong convexity).
    solver : str
        qpsolvers solver name (default: "clarabel" when proximal=False, "daqp" when proximal=True).
        All solvers supported by qpsolvers library should work, in principle, when the GNEP is strongly monotone.
    rho : float
        Proximal regularization parameter (only used when proximal=True).
    tol : float
        Stop when violation of complementarity slackness condition (\lambda^k)'(A x^k - b) < tol (only used when proximal=True).
    maxiter : int
        Maximum proximal point iterations (only used when proximal=True).
        Note: when maxiter = 1, this is equivalent to adding a regularization term rho*I to the QP Hessian and solving a single QP, without iterating.
    hessian_reg : float
        Regularization added to Q1 as hessian_reg*I (only used when proximal=False, generic solver).
    verbose : bool
        Print per-iteration residuals (only used when proximal=True).
    reduced : bool
        If True, eliminate x (and nu) via the stationarity and equality-constraint
        conditions and solve the condensed dual QP in lambda only (eq:QP_dual),
        then recover x from eq:elim_x.  Requires F to be invertible and, when
        equality constraints are present, Aeq to have full row rank.
        Implemented for solver="daqp" and solver="clarabel".
    anderson : bool
        If True, use Anderson acceleration instead of proximal-point iterations.
        Only implemented for solver="daqp" with reduced=True.
    guler : bool
        If True, use accelerated proximal-point iterations with constant rho (not Anderson acceleration), 
        as described in (Güler, 1992). Only implemented for solver="daqp".
        
    Returns
    -------
    SimpleNamespace with fields:
        x            : ndarray -- variational GNE
        lam          : ndarray -- Lagrange multipliers for inequality constraints
        mu           : ndarray -- Lagrange multipliers for equality constraints
        elapsed_time : float   -- wall-clock seconds (excluding setup)
        status_str   : str     -- 'converged', 'max_iterations_reached', or 'solver_failed'
        num_iters    : int     -- for solver="daqp", total number of QP-solver (active-set)
                                  iterations summed over all QPs solved; for other solvers,
                                  number of proximal-point outer iterations
        info         : dict    -- {'converged': bool, 'final_gap': float}
    """

    if solver is None:
        solver = "daqp" if proximal else "clarabel"

    total_qp_iters = None  # set by the daqp branches to the summed QP-solver iteration count

    N = len(dim)
    nvar = sum(dim)
    i2 = np.cumsum(dim)
    i1 = np.concatenate(([0], i2[:-1]))

    # Pseudogradient matrix and linear term
    F = np.zeros((nvar, nvar))
    f = np.zeros(nvar)
    for i in range(N):
        s, e = i1[i], i2[i]
        F[s:e, :] = Q[i][s:e, :]
        f[s:e] = c[i][s:e]

    # Fold box constraints into AA, bb
    AA = A.copy() if A is not None else np.zeros((0, nvar))
    bb = b.copy() if b is not None else np.zeros(0)
    if lb is not None or ub is not None:
        _lb = np.asarray(lb) if lb is not None else np.full(nvar, -np.inf)
        _ub = np.asarray(ub) if ub is not None else np.full(nvar, np.inf)
        idx_lb = np.where(np.isfinite(_lb))[0]
        idx_ub = np.where(np.isfinite(_ub))[0]
        if len(idx_lb) > 0:
            rows = np.zeros((len(idx_lb), nvar))
            rows[np.arange(len(idx_lb)), idx_lb] = 1.0
            AA = np.vstack((AA, -rows))
            bb = np.concatenate((bb, -_lb[idx_lb]))
        if len(idx_ub) > 0:
            rows = np.zeros((len(idx_ub), nvar))
            rows[np.arange(len(idx_ub)), idx_ub] = 1.0
            AA = np.vstack((AA, rows))
            bb = np.concatenate((bb, _ub[idx_ub]))

    nAA = AA.shape[0]
    nAeq = Aeq.shape[0] if Aeq is not None else 0
    _Aeq = Aeq if Aeq is not None else np.zeros((0, nvar))
    _beq = beq if beq is not None else np.zeros(0)
    nx = nvar + nAA + nAeq

    # Build KKT QP in z = [x; lambda; mu]:
    #   min  (1/2) z^T Q1 z + c1^T z
    #   s.t. A1 z <= b1,  Aeq1 z = beq1
    # solve_qp uses (1/2) z^T P z, so pass P = F+F.T for the x block.
    _reg = 0.0 if proximal else hessian_reg
    Q1 = sp(block_diag(F + F.T + _reg * np.eye(nvar), _reg * np.eye(nAA + nAeq)))
    c1 = np.concatenate((f, bb, _beq))
    A1 = sp(np.vstack((
        np.hstack((AA,  np.zeros((nAA, nAA + nAeq)))),   # A x <= b
        np.hstack((np.zeros((nAA, nvar)), -np.eye(nAA), np.zeros((nAA, nAeq))))  # lambda >= 0
    )))
    b1 = np.concatenate((bb, np.zeros(nAA)))
    Aeq1 = sp(np.vstack((
        np.hstack((_Aeq, np.zeros((nAeq, nAA + nAeq)))),     # Aeq x = beq
        np.hstack((F, AA.T, _Aeq.T))                          # stationarity: F x + f + A^T lam + Aeq^T mu = 0
    )))
    beq1 = np.concatenate((_beq, -f))

    if reduced:
        # Compute P and x0 via LU decompositions (Algorithm 1).
        # P = F^{-1} - F^{-1}E^T M^{-1} E F^{-1},  E = _Aeq,  M = E F^{-1} E^T
        lu_F, piv_F = lu_factor(F)
        U_diag = np.diag(lu_F[:nvar, :nvar])
        is_singular = np.any(U_diag == 0)
        if is_singular:
            raise ValueError("F is singular; cannot use reduced formulation. Set reduced=False")
        if nAeq == 0:
            P = lu_solve((lu_F, piv_F), np.eye(nvar))
            x0 = lu_solve((lu_F, piv_F), -f)
        else:
            Y = lu_solve((lu_F, piv_F), _Aeq.T)          # F^{-1} E^T,  n x q
            M_eq = _Aeq @ Y                               # E F^{-1} E^T,  q x q
            lu_M, piv_M = lu_factor(M_eq)
            Z = lu_solve((lu_F, piv_F), _Aeq.T, trans=1).T  # E F^{-1},  q x n
            C = lu_solve((lu_M, piv_M), Z)                # M^{-1} E F^{-1},  q x n
            Q_mat = np.eye(nvar) - _Aeq.T @ C            # I - E^T M^{-1} E F^{-1},  n x n
            P = lu_solve((lu_F, piv_F), Q_mat)            # F^{-1} Q,  n x n
            x0 = -P @ f + Y @ lu_solve((lu_M, piv_M), _beq)

        # Recover mu = mu0 + mu_lam_coeff @ lambda from the stationarity condition
        # Aeq x = beq combined with x = -F^{-1}(f + AA^T lambda + Aeq^T mu):
        #   mu = -M_eq^{-1} (Z f + beq) - M_eq^{-1} Z AA^T lambda,  Z = Aeq F^{-1}
        if nAeq > 0:
            mu0 = -lu_solve((lu_M, piv_M), Z @ f + _beq)
            mu_lam_coeff = -lu_solve((lu_M, piv_M), Z @ AA.T)
        else:
            mu0 = np.zeros(0)
            mu_lam_coeff = np.zeros((0, nAA))

        if nAA == 0:
            t_start = time.perf_counter()
            elapsed = time.perf_counter() - t_start
            return SimpleNamespace(
                x=x0,
                lam = np.zeros(0),
                mu = mu0,
                elapsed_time=elapsed,
                status_str="converged",
                num_iters=0,
                info={"converged": True, "final_gap": 0.0}
            )

        # Condensed dual QP in lambda (eq:QP_dual):
        #   min  lambda^T M_s lambda + q_d^T lambda
        #   s.t. M lambda + q_d >= 0,  lambda >= 0
        # M = A P A^T,  M_s = (M + M^T)/2,  q_d = b - A x0
        M_dual = AA @ P @ AA.T
        M_s_dual = (M_dual + M_dual.T) / 2
        q_d = bb - AA @ x0

    # -----------------------------------------------------------------------
    # DAQP solver
    # -----------------------------------------------------------------------
    if solver == "daqp":

        if reduced:
            # Reduced constraints in lambda: -M_dual lam <= q_d, -lam <= 0
            AA_d_red = np.vstack([-M_dual, -np.eye(nAA)])
            bu_d_red = np.concatenate([q_d, np.zeros(nAA)])
            bl_d_red = np.full(2 * nAA, -np.inf)
            sense_red0 = np.zeros(2 * nAA, dtype=np.int32)
            ineq_d_red = np.ones(2 * nAA, dtype=bool)
            fin_bl_d_red = np.isfinite(bl_d_red)  # all False
            total_qp_iters = 0

            t_start = time.perf_counter()

            if anderson:
                # Anderson acceleration: proximal QP at each step, then mix iterates
                H_daqp_red = 2 * M_s_dual + rho * np.eye(nAA)
                model = daqp.Model()
                flag_setup, _ = model.setup(H_daqp_red, q_d, AA_d_red, bu_d_red, bl_d_red, sense_red0)
                if flag_setup != 1:
                    raise ValueError(
                        f"daqp setup failed (flag={flag_setup}); H min eigenvalue = "
                        f"{np.min(np.linalg.eigvalsh(H_daqp_red)):.3e}. Increase rho."
                    )
                lam = np.zeros(nAA)
                sense = sense_red0.copy()
                m = 5
                Y_hist = []
                R_hist = []
                status_str = "max_iterations_reached"
                gap = np.nan
                j = 0
                for j in range(1, maxiter + 1):
                    lam_y, _, flag, info = model.solve()
                    total_qp_iters += info.get("iterations", 0)
                    if flag != 1:
                        model.update(sense=sense_red0)
                        lam_y, _, flag, info = model.solve()
                        total_qp_iters += info.get("iterations", 0)
                    if lam_y is None or flag != 1:
                        status_str = "solver_failed"
                        break
                    lam_dual = info.get("lam", np.zeros(2 * nAA))
                    new_sense = sense_red0.copy()
                    new_sense[ineq_d_red & (lam_dual > 1e-8)] = 1
                    new_sense[ineq_d_red & fin_bl_d_red & (lam_dual < -1e-8)] = 3
                    sense = new_sense
                    rk = lam_y - lam
                    
                    # Evaluate complimentarity slackness violation gap = lambda'(b-Ax)
                    x_sol = x0 - P @ AA.T @ lam_y
                    gap = lam_y * (bb - AA @ x_sol)
                    if verbose:
                        print(f"  qp_gnep iter {j}: compl. slackness violation = {gap:.2e}")
                    if gap < tol:
                        lam = lam_y
                        status_str = "converged"
                        break

                    Y_hist.append(lam_y.copy())
                    R_hist.append(rk.copy())
                    if len(Y_hist) > m:
                        Y_hist.pop(0)
                        R_hist.pop(0)
                    mk = len(Y_hist)
                    R_mat = np.column_stack(R_hist)
                    e = np.ones(mk)
                    v = np.linalg.solve(R_mat.T @ R_mat + 1e-14 * np.eye(mk), e)
                    alpha = v / (e @ v)
                    lam = np.column_stack(Y_hist) @ alpha
                    model.update(f=q_d - rho * lam, sense=sense)

            elif proximal:
                # Case 1: reduced + proximal
                H_daqp_red = 2 * M_s_dual + rho * np.eye(nAA)
                model = daqp.Model()
                flag_setup, _ = model.setup(H_daqp_red, q_d, AA_d_red, bu_d_red, bl_d_red, sense_red0)
                if flag_setup != 1:
                    raise ValueError(
                        f"daqp setup failed (flag={flag_setup}); H min eigenvalue = "
                        f"{np.min(np.linalg.eigvalsh(H_daqp_red)):.3e}. Increase rho."
                    )
                lam = np.zeros(nAA)
                if guler:
                    lam_prev = lam.copy()
                    y = lam.copy()
                    theta = 1.0 # theta_0
                sense = sense_red0.copy()
                status_str = "max_iterations_reached"
                gap = np.nan

                for j in range(1, maxiter + 1):
                    lam_new, cost_new, flag, info = model.solve()
                    total_qp_iters += info.get("iterations", 0)
                    if flag != 1:
                        model.update(sense=sense_red0)  # cold restart
                        lam_new, cost_new, flag, info = model.solve()
                        total_qp_iters += info.get("iterations", 0)
                    if lam_new is None or flag != 1:
                        status_str = "solver_failed"
                        break

                    # Evaluate complimentarity slackness violation gap = lambda'(b-Ax) from QP's optimal cost
                    if not guler:
                        gap = np.maximum(cost_new -.5*rho*np.sum((lam_new - lam)**2),0.)
                    else:
                        gap = np.maximum(cost_new -.5*rho*np.sum((lam_new - y)**2),0.)
                        lam_prev = lam.copy()
                    lam = lam_new
                    x_sol = x0 - P @ AA.T @ lam
                    #gap = lam @ (bb - AA @ x_sol) # <- same value, but more expensive to compute
                    
                    if verbose:
                        print(f"  qp_gnep iter {j}: compl. slackness violation = {gap:.2e}")
                    if gap < tol:
                        status_str = "converged"
                        break

                    lam_dual = info.get("lam", np.zeros(2 * nAA))
                    new_sense = sense_red0.copy()
                    new_sense[ineq_d_red & (lam_dual > 1e-8)] = 1
                    new_sense[ineq_d_red & fin_bl_d_red & (lam_dual < -1e-8)] = 3
                    sense = new_sense
                    
                    if guler:
                        theta_prev = theta
                        theta = theta_prev * (np.sqrt(theta_prev**2 + 4.0) - theta_prev) / 2.0
                        y = lam + theta * (1.0 / theta_prev - 1.0) * (lam - lam_prev)
                        model.update(f=q_d - rho * y, sense=sense)
                    else:
                        model.update(f=q_d - rho * lam, sense=sense)
            else:
                # Case 2: reduced, no proximal
                H_daqp_red = 2 * M_s_dual
                lam, _, flag, info = daqp.solve(
                    H_daqp_red, q_d, AA_d_red, bu_d_red, bl_d_red, sense=sense_red0.copy()
                )
                j = 1
                total_qp_iters = info.get("iterations", 0)
                if lam is None or flag != 1:
                    status_str = "solver_failed"
                    lam = np.zeros(nAA)
                else:
                    status_str = "converged"
                gap = np.nan
                x_sol = x0 - P @ AA.T @ lam

            elapsed = time.perf_counter() - t_start
            mu = mu0 + mu_lam_coeff @ lam

            return SimpleNamespace(
                x=x_sol,
                lam=lam,
                mu=mu,
                elapsed_time=elapsed,
                status_str=status_str,
                num_iters=total_qp_iters,
                info={"converged": status_str == "converged", "final_gap": float(gap)}
            )

        # Non-reduced: KKT QP in z = [x; lambda; mu]
        # Constraint format: bl_d <= AA_d z <= bu_d, with sense encoding active set
        rows_d, bu_parts, bl_parts, s0_parts = [], [], [], []
        if nAA > 0:
            rows_d.append(np.hstack((AA, np.zeros((nAA, nAA + nAeq)))))
            bu_parts.append(bb)
            bl_parts.append(np.full(nAA, -np.inf))
            s0_parts.append(np.zeros(nAA, dtype=np.int32))
            rows_d.append(np.hstack((np.zeros((nAA, nvar)), -np.eye(nAA), np.zeros((nAA, nAeq)))))
            bu_parts.append(np.zeros(nAA))
            bl_parts.append(np.full(nAA, -np.inf))
            s0_parts.append(np.zeros(nAA, dtype=np.int32))
        if nAeq > 0:
            rows_d.append(np.hstack((_Aeq, np.zeros((nAeq, nAA + nAeq)))))
            bu_parts.append(_beq)
            bl_parts.append(_beq.copy())
            s0_parts.append(5 * np.ones(nAeq, dtype=np.int32))
        rows_d.append(np.hstack((F, AA.T, _Aeq.T)))  # stationarity
        bu_parts.append(-f)
        bl_parts.append(-f.copy())
        s0_parts.append(5 * np.ones(nvar, dtype=np.int32))
        AA_d     = np.vstack(rows_d)
        bu_d     = np.concatenate(bu_parts)
        bl_d     = np.concatenate(bl_parts)
        sense0   = np.concatenate(s0_parts).astype(np.int32)
        ncon_d   = AA_d.shape[0]
        ineq_d   = (sense0 != 5)
        fin_bl_d = np.isfinite(bl_d)
        total_qp_iters = 0

        t_start = time.perf_counter()

        if proximal:
            # Case 3: non-reduced + proximal
            H_daqp = block_diag(F + F.T + rho * np.eye(nvar), rho * np.eye(nAA + nAeq))
            model = daqp.Model()
            flag_setup, _ = model.setup(H_daqp, c1, AA_d, bu_d, bl_d, sense0)
            if flag_setup != 1:
                raise ValueError(
                    f"daqp setup failed (flag={flag_setup}); H min eigenvalue = "
                    f"{np.min(np.linalg.eigvalsh(H_daqp)):.3e}. Increase rho."
                )
            z = np.zeros(nx)
            if guler:
                z_prev = z.copy()
                y = z.copy()
                theta = 1.0 # theta_0
            sense = sense0.copy()
            status_str = "max_iterations_reached"
            gap = np.nan
            j = 0
            for j in range(1, maxiter + 1):
                z_new, _, flag, info = model.solve()
                total_qp_iters += info.get("iterations", 0)
                if flag != 1:
                    model.update(sense=sense0)  # cold restart
                    z_new, _, flag, info = model.solve()
                    total_qp_iters += info.get("iterations", 0)
                if z_new is None or flag != 1:
                    status_str = "solver_failed"
                    break
                
                z = z_new
                gap = z[nvar:nvar+nAA] @ (bb - AA @ z[:nvar])
                if  verbose:
                    print(f"  qp_gnep iter {j}: compl. slackness violation = {gap:.2e}")
                if gap < tol:
                    status_str = "converged"
                    break
                lam_d = info.get("lam", np.zeros(ncon_d))
                new_sense = sense0.copy()
                new_sense[ineq_d & (lam_d > 1e-8)] = 1
                new_sense[ineq_d & fin_bl_d & (lam_d < -1e-8)] = 3
                sense = new_sense

                if guler:
                    theta_prev = theta
                    theta = theta_prev * (np.sqrt(theta_prev**2 + 4.0) - theta_prev) / 2.0
                    y = z + theta * (1.0 / theta_prev - 1.0) * (z - z_prev)
                    z_prev = z.copy()
                    model.update(f=c1 - rho * y, sense=sense)
                else:
                    model.update(f=c1 - rho * z, sense=sense)
        else:
            # Case 4: non-reduced, no proximal
            H_daqp = block_diag(F + F.T, np.zeros((nAA + nAeq, nAA + nAeq)))
            z_new, _, flag, info = daqp.solve(H_daqp, c1, AA_d, bu_d, bl_d, sense=sense0.copy())
            j = 1
            total_qp_iters = info.get("iterations", 0)
            if z_new is None or flag != 1:
                status_str = "solver_failed"
                z = np.zeros(nx)
            else:
                status_str = "converged"
                z = z_new
            gap = np.nan

    # -----------------------------------------------------------------------
    # Clarabel solver
    # -----------------------------------------------------------------------
    elif solver == "clarabel":
        import clarabel
        cl_settings = clarabel.DefaultSettings()
        cl_settings.verbose = False
        cl_settings.presolve_enable = False            # required for solver.update()
        cl_settings.chordal_decomposition_enable = False

        if reduced:
            # Reduced constraints in lambda via NonnegativeCone: b_cl - A_cl lam >= 0
            #   -M_dual lam <= q_d  =>  M_dual lam + q_d >= 0
            #   -I lam <= 0         =>  lam >= 0
            A_cl_red = sp_vstack([sp(-M_dual), sp(-np.eye(nAA))], format="csc")
            b_cl_red = np.concatenate([q_d, np.zeros(nAA)])
            cones_cl_red = [clarabel.NonnegativeConeT(2 * nAA)]

            t_start = time.perf_counter()

            if proximal:
                # Case 1: reduced + proximal
                P_cl_red = sp(np.triu(2 * M_s_dual + rho * np.eye(nAA)))
                solver_cl_red = clarabel.DefaultSolver(
                    P_cl_red, q_d, A_cl_red, b_cl_red, cones_cl_red, cl_settings
                )
                lam = np.zeros(nAA)
                status_str = "max_iterations_reached"
                gap = np.nan
                j = 0
                for j in range(1, maxiter + 1):
                    sol = solver_cl_red.solve()
                    if sol.status not in (clarabel.SolverStatus.Solved, clarabel.SolverStatus.AlmostSolved):
                        status_str = "solver_failed"
                        break
                    lam_new = np.array(sol.x)
                    lam = lam_new
                    gap = lam @ (M_dual @ lam + q_d)
                    if verbose:
                        print(f"  qp_gnep iter {j}: compl. slackness violation = {gap:.2e}")
                    if gap < tol:
                        status_str = "converged"
                        break
                    solver_cl_red.update(q=q_d - rho * lam)
            else:
                # Case 2: reduced, no proximal
                P_cl_red = sp(np.triu(2 * M_s_dual))
                solver_cl_red = clarabel.DefaultSolver(
                    P_cl_red, q_d, A_cl_red, b_cl_red, cones_cl_red, cl_settings
                )
                sol = solver_cl_red.solve()
                j = 1
                if sol.status not in (clarabel.SolverStatus.Solved, clarabel.SolverStatus.AlmostSolved):
                    status_str = "solver_failed"
                    lam = np.zeros(nAA)
                else:
                    status_str = "converged"
                    lam = np.array(sol.x)
                gap = np.nan

            elapsed = time.perf_counter() - t_start
            x_sol = x0 - P @ AA.T @ lam
            mu = mu0 + mu_lam_coeff @ lam
            return SimpleNamespace(
                x=x_sol,
                lam=lam,
                mu=mu,
                elapsed_time=elapsed,
                status_str=status_str,
                num_iters=j,
                info={"converged": status_str == "converged", "final_gap": float(gap)}
            )

        # Non-reduced: build full KKT constraint matrix and cones
        if nAA > 0:
            A_cl   = sp_vstack([A1, Aeq1], format="csc")
            b_cl   = np.concatenate([b1, beq1])
            cones_cl = [clarabel.NonnegativeConeT(2 * nAA), clarabel.ZeroConeT(nAeq + nvar)]
        else:
            A_cl   = Aeq1
            b_cl   = beq1
            cones_cl = [clarabel.ZeroConeT(nAeq + nvar)]

        t_start = time.perf_counter()

        if proximal:
            # Case 3: non-reduced + proximal
            P_cl = sp(np.triu(block_diag(F + F.T + rho * np.eye(nvar), rho * np.eye(nAA + nAeq))))
            solver_cl = clarabel.DefaultSolver(P_cl, c1, A_cl, b_cl, cones_cl, cl_settings)
            z = np.zeros(nx)
            status_str = "max_iterations_reached"
            gap = np.nan
            j = 0
            for j in range(1, maxiter + 1):
                sol = solver_cl.solve()
                if sol.status not in (clarabel.SolverStatus.Solved, clarabel.SolverStatus.AlmostSolved):
                    status_str = "solver_failed"
                    break
                z_new = np.array(sol.x)
                z = z_new
                gap = z[nvar:nvar+nAA] @ (bb - AA @ z[:nvar])
                if verbose:
                    print(f"  qp_gnep iter {j}: compl. slackness violation = {gap:.2e}")
                if gap < tol:
                    status_str = "converged"
                    break
                solver_cl.update(q=c1 - rho * z)
        else:
            # Case 4: non-reduced, no proximal
            P_cl = sp(np.triu(block_diag(F + F.T, np.zeros((nAA + nAeq, nAA + nAeq)))))
            solver_cl = clarabel.DefaultSolver(P_cl, c1, A_cl, b_cl, cones_cl, cl_settings)
            sol = solver_cl.solve()
            j = 1
            if sol.status not in (clarabel.SolverStatus.Solved, clarabel.SolverStatus.AlmostSolved):
                status_str = "solver_failed"
                z = np.zeros(nx)
            else:
                status_str = "converged"
                z = np.array(sol.x)
            gap = np.nan

    # -----------------------------------------------------------------------
    # Generic solver via qpsolvers
    # -----------------------------------------------------------------------
    else:
        t_start = time.perf_counter()
        if proximal:
            Qrho = Q1 + rho * sp_eye(nx, format="csc")
            z = np.zeros(nx)
            status_str = "max_iterations_reached"
            gap = np.nan
            j = 0
            for j in range(1, maxiter + 1):
                z_new = solve_qp(Qrho, c1 - rho * z, A1, b1, Aeq1, beq1, solver=solver)
                if z_new is None:
                    status_str = "solver_failed"
                    break
                z = z_new
                gap = z[nvar:nvar+nAA] @ (bb - AA @ z[:nvar])
                if verbose:
                    print(f"  qp_gnep iter {j}: compl. slackness violation = {gap:.2e}")
                if gap < tol:
                    status_str = "converged"
                    break
        else:
            z = solve_qp(Q1, c1, A1, b1, Aeq1, beq1, solver=solver)
            j = 1
            if z is None:
                status_str = "solver_failed"
                z = np.zeros(nx)
            else:
                status_str = "converged"
            gap = np.nan

    elapsed = time.perf_counter() - t_start

    return SimpleNamespace(
        x=z[:nvar],
        lam=z[nvar:nvar+nAA],
        mu=z[nvar+nAA:nx],
        elapsed_time=elapsed,
        status_str=status_str,
        num_iters=total_qp_iters if total_qp_iters is not None else j,
        info={"converged": status_str == "converged", "final_gap": float(gap)}
    )
