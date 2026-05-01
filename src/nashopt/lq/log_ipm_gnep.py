# NashOpt: A Python package for computing Generalized Nash Equilibria (GNE) in noncooperative games.
#
# Solve Linear-Quadratic GNEPs via log-domain interior point method.
#
# Note: this code was partially auto-generated using AI.
#
# (C) 2025-2026 Alberto Bemporad

import warnings
import numpy as np
from scipy import linalg
from types import SimpleNamespace
import time

def solve(N, n, Q, S, p, A, b, G=None, h=None, lb=None, ub=None, 
                 eps =1.e-8, tau =1.e-10, eta1 =50, eta2 =100, gamma =1.e-9, beta =1.0, max_outer =500):
    """
    Find a variational GNE solution of a LQ-GNE problem using the
    log-domain interior point method (Algorithm 1) from [1].

    Parameters
    ----------
    N : int
        Number of players.
    n : int or list of int
        Number of variables per player. If int, all players have the same number of variables.
    Q : list of ndarray
        List of local Hessians Q_i for each player.
    S : list of list of ndarray
        List of cross-player coupling matrices S[i][j].
    p : list of ndarray
        List of linear cost vectors p_i for each player.
    A : ndarray
        Inequality constraint matrix in A x + b >= 0 form.
    b : ndarray
        Inequality constraint vector in A x + b >= 0 form.
    G : ndarray, optional
        Equality constraint matrix in G x - h = 0 form.
    h : ndarray, optional
        Equality constraint vector in G x - h = 0 form.
    lb : (nN,) array or None
        Lower bounds on x. When given, rows I x - lb >= 0 are prepended to
        (A, b) automatically. Default: None.
    ub : (nN,) array or None
        Upper bounds on x. When given, rows -I x + ub >= 0 are prepended to
        (A, b) automatically (after any lb rows). Default: None.

    eps      : float   barrier-parameter convergence threshold  (mu <= eps)
    tau      : float   Newton-step norm threshold for inner termination
    eta1     : int     max Newton steps per outer iteration (centering)
    eta2     : int     max Newton steps in the final refinement pass
    gamma    : float   diagonal regularisation for the Newton system
    beta     : float   step-size damping  (alpha = 1/max(1, (1/(2*beta))||Delta_v||^2_inf))
    max_outer: int     hard cap on outer iterations

    Returns
    -------
    sol : SimpleNamespace with fields
        x   : ndarray (nN,)   variational GNE strategy profile
        nu  : ndarray (m,)    multipliers for the inequality constraints
                            satisfies  W x + p_vec - A_aug.T nu ~= 0 where A_aug is the
                            internally augmented matrix (lb rows, ub rows, then
                            user-supplied rows)
        lam : ndarray (q,)    multipliers for the equality constraints
        info : dict
            'mu', 'outer_iters', 'converged',
            'res_stationarity', 'res_complementarity'
        elapsed_time : float   total time taken by the algorithm in seconds
        
    [1] B. Liu, D. Liao-McPherson, "A Log-domain Interior Point Method for Convex Quadratic Games," 
        American Control Conference, 2025.
    """
    t0 = time.perf_counter()
    
    # 1. Build W and linear cost f
    N = int(N)

    if np.isscalar(n):
        dims = [int(n)] * N
    else:
        dims = [int(d) for d in n]
    nN  = sum(dims)
    idx = np.cumsum([0] + dims)

    W = np.zeros((nN, nN))
    for i in range(N):
        si, ei = idx[i], idx[i + 1]
        W[si:ei, si:ei] = np.asarray(Q[i], dtype=float)
        for j in range(N):
            if i != j:
                sj, ej = idx[j], idx[j + 1]
                W[si:ei, sj:ej] = np.asarray(S[i][j], dtype=float)

    f = np.concatenate(
        [np.asarray(p[i], dtype=float).ravel() for i in range(N)]
    )

    # 2. Extract constraint data
    A = np.asarray(A, dtype=float)           # (m, nN)
    b = np.asarray(b, dtype=float).ravel()   # (m,)
    m = A.shape[0]
    if A.shape[1] != nN:
        raise ValueError(
            f"A has {A.shape[1]} columns but nN={nN}."
        )

    # Prepend box-constraint rows (lb then ub)
    aug_A, aug_b = [], []
    if lb is not None:
        finite_lb = np.isfinite(lb)
        if finite_lb.any():
            idx = np.where(finite_lb)[0]
            aug_A.append( np.eye(nN, dtype=float)[idx])
            aug_b.append(-np.asarray(lb, dtype=float)[idx])
    if ub is not None:
        finite_ub = np.isfinite(ub)
        if finite_ub.any():
            idx = np.where(finite_ub)[0]
            aug_A.append(-np.eye(nN, dtype=float)[idx])
            aug_b.append( np.asarray(ub, dtype=float)[idx])
    if aug_A:
        A = np.vstack(aug_A + [A])
        b = np.concatenate(aug_b + [b])
        m = A.shape[0]

    has_eq = (
        G is not None
        and np.asarray(G).size > 0
    )
    if has_eq:
        G = np.asarray(G, dtype=float)
        h = np.asarray(h, dtype=float).ravel()
        q = G.shape[0]
    else:
        G, h, q = np.zeros((0, nN)), np.zeros(0), 0

    n_sys = nN + q

    # -- 3. Problem scaling ---------------------------------------------------
    sf = np.linalg.norm(f, np.inf) + 1.0
    Ws = W / sf
    fs = f / sf

    sA = np.abs(b) + 1.0
    As = A / sA[:, None]
    bs = b / sA

    if has_eq:
        sG = np.abs(h) + 1.0
        Gs = G / sG[:, None]  
        hs = h / sG
    else:
        Gs, hs, sG = G, h, np.ones(1)

    # Core subroutines
    def _newton_direction(mu, v):
        """Solve the reduced Newton system and return (x_new, dv, lam_new, alpha)."""
        sqrt_mu = np.sqrt(mu)
        # clip so that e2v = exp(2*v) stays within a safe range
        v_clp   = np.clip(v, -100.0, 100.0)
        ev      = np.exp(v_clp)
        e2v     = ev * ev

        K       = Ws + As.T @ (e2v[:, None] * As)
        rhs_x   = 2.0 * sqrt_mu * (As.T @ ev) - fs - As.T @ (e2v * bs)

        if q > 0:
            M_sys       = np.zeros((n_sys, n_sys))
            M_sys[:nN, :nN] =  K
            M_sys[:nN, nN:] = -Gs.T
            M_sys[nN:, :nN] =  Gs
            M_sys[nN:, nN:] =  gamma * np.eye(q)
            rhs         = np.concatenate([rhs_x, -hs])
            sol         = linalg.solve(M_sys, rhs, assume_a="gen",
                                       check_finite=False)
            x_new   = sol[:nN];  lam_new = sol[nN:]
        else:
            K_reg   = K + gamma * np.eye(nN)
            x_new   = linalg.solve(K_reg, rhs_x, assume_a="gen",
                                    check_finite=False)
            lam_new = np.zeros(0)

        dv      = 1.0 - (bs + As @ x_new) * ev / sqrt_mu
        alpha   = 1.0 / max(1.0, (0.5 / beta) * np.max(np.abs(dv)) ** 2)
        return x_new, dv, lam_new, alpha

    def _newton(mu, v, eta):
        """Run up to eta damped Newton steps at fixed mu."""
        x_cur = np.zeros(nN);  lam_cur = np.zeros(q)
        for _ in range(eta):
            x_cur, dv, lam_cur, alpha = _newton_direction(mu, v)
            v = v + alpha * dv
            if np.linalg.norm(dv) <= tau:
                break
        x_cur, _, lam_cur, _ = _newton_direction(mu, v)
        return v, x_cur, lam_cur

    def _line_search(mu, v):
        """Find mu* = inf{mu' > 0 : ||Delta_v(v,mu')||_inf <= 1}."""
        _, dv, _, _ = _newton_direction(mu, v)
        if np.max(np.abs(dv)) > 1.0:
            return mu
        lo, hi = 1e-14, mu
        for _ in range(60):
            mid = np.sqrt(lo * hi)
            _, dv_mid, _, _ = _newton_direction(mid, v)
            if np.max(np.abs(dv_mid)) <= 1.0:
                hi = mid
            else:
                lo = mid
            if hi / max(lo, 1e-300) < 1.0 + 1e-4:
                break
        return hi

    # 5. Outer loop (Algorithm 1)
    mu = 1.0
    v = np.ones(m)
    x_sol = np.zeros(nN)
    lam_sol = np.zeros(q)

    n_outer = 0
    for n_outer in range(max_outer):
        _, dv_check, _, _ = _newton_direction(mu, v)
        if np.max(np.abs(dv_check)) <= 1.0 and mu <= eps:
            break
        mu = _line_search(mu, v)
        v, x_sol, lam_sol = _newton(mu, v, eta1)
    else:
        warnings.warn(
            f"log_ipm_gnep: max_outer={max_outer} reached without "
            f"convergence (mu={mu:.2e}).",
            stacklevel=2,
        )

    # 6. Final refinement
    v, x_sol, lam_sol = _newton(mu, v, eta2)

    # 7. Recover duals and undo scaling
    v_clp   = np.clip(v, -500.0, 500.0)
    sqrt_mu = np.sqrt(mu)
    nu_s    = sqrt_mu * np.exp( v_clp)
    s_s     = sqrt_mu * np.exp(-v_clp)

    nu_sol  = (sf / sA) * nu_s
    if q > 0:
        lam_sol = (sf / sG) * lam_sol

    # 8. Residuals
    lam_full = lam_sol * sG / sf if q > 0 else np.zeros(0)
    res_stat = float(np.linalg.norm(
        Ws @ x_sol + fs - As.T @ nu_s
        - (Gs.T @ lam_full if q > 0 else 0.0)
    ))
    res_comp = float(np.linalg.norm(np.minimum(nu_s, s_s)))

    elapsed_time = time.perf_counter() - t0
    
    info = {
        "mu":                   mu,
        "outer_iters":          n_outer + 1,
        "converged":            mu <= eps,
        "res_stationarity":     res_stat,
        "res_complementarity":  res_comp,
    }
    
    sol = SimpleNamespace()
    sol.x = x_sol
    sol.lam = nu_sol # Lagrange multipliers for the inequality constraints, not split by agent
    sol.mu = lam_sol # Lagrange multipliers for the equality constraints, not split by agent
    sol.info = info
    sol.num_iters = n_outer + 1
    sol.elapsed_time = elapsed_time
    
    return sol
