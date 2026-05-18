"""
GoldNash - A variant of Goldfarb-Idnani dual active-set algorithm adapted for computing the
variational Generalized Nash Equilibrium (v-GNE) of an N-player
linear-quadratic game with shared affine constraints (inequalities and equalities).

The algorithm assumes an N-player game with decision vector  x = (x_1, ..., x_N) in R^{nvar},
where x_i in R^{n_i} is player i's decision variable and sum(n_i) = nvar. Each player i minimizes

    J_i(x) = 1/2 x' Q[i] x  +  c[i]' x

subject to the joint constraints  A x <= b  and  E x = f. Possible local constraints on x_i, including lower and upper bounds, are considered encoded by appropriate rows in A, E and corresponding entries in b, f.

[1] A. Bemporad, “GoldNash: A Goldfarb-Idnani variant for strongly monotone linear-quadratic games,” 2026. Available on arXiv at https://arxiv.org/abs/2605.16002. 

(C) 2026, A. Bemporad
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# S_bar^{-1} update kernels
# ---------------------------------------------------------------------------

def _s_inv_add_np(S_inv, c, d, e):
    """Bordered inverse update (NumPy): extend (ell×ell) S_inv by one row/col. O(ell^2)."""
    ell   = S_inv.shape[0]
    alpha = S_inv @ c
    beta  = S_inv.T @ d
    inv_s = 1.0 / (e - float(d @ alpha))
    S_new = np.empty((ell + 1, ell + 1))
    S_new[:ell, :ell] = S_inv + inv_s * np.outer(alpha, beta)
    S_new[:ell, ell]  = -inv_s * alpha
    S_new[ell, :ell]  = -inv_s * beta
    S_new[ell, ell]   = inv_s
    return S_new


def _s_inv_drop_np(S_inv, k):
    """Rank-1 inverse update (NumPy): remove row/col k from (ell×ell) S_inv. O(ell^2)."""
    u    = S_inv[:, k]
    v    = S_inv[k, :]
    mkk  = float(S_inv[k, k])
    mask = np.ones(S_inv.shape[0], dtype=bool)
    mask[k] = False
    return S_inv[np.ix_(mask, mask)] - np.outer(u[mask], v[mask]) / mkk


def _refresh_S_inv(W, E, A, Y_A, Z_A, EY_E, has_eq, neq):
    """Rebuild S_bar^{-1} from the current working set W to reset numerical drift."""
    W_arr = np.array(W, dtype=int) if W else np.empty(0, dtype=int)
    ell   = neq + len(W)
    S     = np.empty((ell, ell))
    if has_eq:
        S[:neq, :neq] = EY_E
        if len(W) > 0:
            S[:neq, neq:] = E @ Y_A[:, W_arr]
            S[neq:, :neq] = (E @ Z_A[:, W_arr]).T
    if len(W) > 0:
        S[neq:, neq:] = A[W_arr] @ Y_A[:, W_arr]
    return np.linalg.inv(S)


# -----------------------------------------------------------------------------
# Main solver
# -----------------------------------------------------------------------------

def goldnash(
    Q: List[np.ndarray],
    c: List[np.ndarray],
    dim: List[int],
    *,
    A: np.ndarray,
    b: np.ndarray,
    E: np.ndarray = None,
    f: np.ndarray = None,
    max_iter: int = 10000,
    tol: float = 1e-10,
    verbose: bool = False,
    check_monotone: bool = False,
    refresh_freq: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Solve the v-GNE of an LQ game with shared affine constraints.

    Parameters
    ----------
    Q        : list of N arrays, shape (nvar, nvar)
               Q[i] is player i's quadratic cost matrix.
    c        : list of N arrays, shape (nvar,)
               c[i] is player i's linear cost vector.
    dim      : list [n1, n2, ..., nN]
               Dimension of each player's decision variable.
               sum(dim) == nvar.
    A        : array, shape (ncon, nvar), optional
               Shared inequality constraint matrix  (A x <= b).
    b        : array, shape (ncon,), optional
               Shared inequality RHS.
    E        : array, shape (neq, nvar), optional
               Shared equality constraint matrix  (E x = f).
               Must have full row rank.  If None, no equalities are imposed.
    f        : array, shape (neq,), optional
               Shared equality RHS.  Must be provided together with E.
    max_iter : int, optional
               Maximum outer (and inner) iterations (default 1000).
    tol      : float, optional
               Feasibility / optimality tolerance (default 1e-10).
    verbose  : bool, optional
               Print iteration log if True.
    check_monotone : bool, optional
                Check positive definiteness of the symmetric part of the pseudogradient matrix before starting solving the problem (default True).
    refresh_freq : int, optional
               Recompute S_bar^{-1} from scratch every this many ADD/DROP steps
               to limit numerical drift (set 0 to disable).
    Returns
    -------
    x_star   : array (nvar,)
               v-GNE primal solution.
    info     : dict
               'status'           - 'optimal' | 'infeasible' | 'max_iterations'
               'outer_iterations' - number of constraint-addition attempts
               'total_steps'      - total primal / dual update steps taken
               'primal_residual'  - max(A x* - b)  at termination
               'eq_residual'      - ||E x* - f||  at termination (0.0 if E is None)
               'kkt_residual'     - ||G x* + r + A' lam* + E' nu*||  at termination
               'lam'              - inequality dual vector lambda* (empty array if A is None)
               'nu'               - equality dual vector nu* (empty array if E is None)

    Raises
    ------
    ValueError
        If input dimensions are inconsistent.

    (C) 2026, A. Bemporad
    """

    # #######################################
    # Check input arguments
    # #######################################

    N    = len(dim)
    nvar = int(sum(dim))

    if (A is None) != (b is None):
        raise ValueError("A and b must both be provided or both be None.")
    has_ineq = A is not None
    if has_ineq:
        A    = np.asarray(A, dtype=float)
        b    = np.asarray(b, dtype=float).ravel()
        ncon = A.shape[0]
        if A.shape[1] != nvar:
            raise ValueError(
                f"A has shape {A.shape} but sum(dim) = {nvar}: column mismatch."
            )
        if b.shape[0] != ncon:
            raise ValueError(
            f"b has length {b.shape[0]} but A has {ncon} rows."
        )
    else:
        ncon = 0

    if (E is None) != (f is None):
        raise ValueError("E and f must both be provided or both be None.")
    has_eq = E is not None
    if has_eq:
        E    = np.asarray(E, dtype=float)
        f    = np.asarray(f, dtype=float).ravel()
        neq  = E.shape[0]
        if E.ndim != 2 or E.shape[1] != nvar:
            raise ValueError(
                f"E must have shape (neq, {nvar}), got {E.shape}."
            )
        if f.shape[0] != neq:
            raise ValueError(
                f"f has length {f.shape[0]} but E has {neq} rows."
            )
    else:
        neq = 0

    # Assemble pseudogradient  F(x) = G x + r
    starts = [0] + list(np.cumsum(dim))
    G      = np.zeros((nvar, nvar))
    r      = np.zeros(nvar)

    for i in range(N):
        ri = slice(starts[i], starts[i + 1])
        Qi = np.asarray(Q[i], dtype=float)
        ci = np.asarray(c[i], dtype=float).ravel()
        if Qi.shape != (nvar, nvar):
            raise ValueError(f"Q[{i}] must be ({nvar},{nvar}), got {Qi.shape}.")
        if ci.size != nvar:
            raise ValueError(f"c[{i}] must have length {nvar}, got {ci.size}.")
        G[ri, :] = Qi[ri, :]
        r[ri]    = ci[ri]

    # Verify strong monotonicity of the game
    if check_monotone:
        Gs = 0.5 * (G + G.T)          # symmetric part of pseudogradient
        eigs = np.linalg.eigvalsh(Gs)
        if eigs.min() <= 0.0:
            raise ValueError("The symmetric part of the pseudogradient matrix is not positive definite "
                f"(min eigenvalue = {eigs.min():.3e}). "
                "Game is not strongly monotone."
            )

    # Factor G. Note: if Gs > 0  ->  G is invertible and w'G^{-1}w > 0 for w != 0.
    try:
        G_lu, G_piv = lu_factor(G)
    except Exception as exc:
        raise ValueError("LU factorization of pseudogradient matrix failed, it may be singular.") from exc

    if has_eq:
        Y_E  = lu_solve((G_lu, G_piv), E.T)            # (nvar, neq)
        EY_E = E @ Y_E                                  # (neq, neq); = E G^{-1} E^T

    # Initialize solution
    #
    # Without equalities: x_0 = -G^{-1} r, nu = [].
    # With equalities: solve the saddle-point system
    #     [G  E^T] [x]   [-r]
    #     [E   0] [nu] = [ f]
    # via Schur elimination: nu = (E G^{-1} E^T)^{-1} (E x_0 - f),  x = x_0 - Y_E nu,
    # where x_0 = -G^{-1} r is the unconstrained point.
    #
    x   = lu_solve((G_lu, G_piv), -r)    # x_0 = -G^{-1} r

    if has_eq:
        # Precompute Y_E = G^{-1}E^T, Z = G^{-T}E^T
        nu  = np.linalg.solve(EY_E, E @ x - f)  # (neq,)
        x   = x - Y_E @ nu                       # project onto Ex = f
    else:
        nu  = np.empty(0)
    lam = np.zeros(ncon)                 # (ncon,) dual iterate (shared price)

    if not has_ineq:
        # No inequalities, we're done
        status = "optimal"
        prim_res = 0.0

    else:
        # ###################################
        # MAIN ALGORITHM: active-set method on the inequalities, maintaining feasibility and stationarity at each step.
        # ###################################

        # Precompute Y = G^{-1}A^T, Z = G^{-T}A^T
        # Y_A[:,k] = G^{-1} a_k  (primal step ingredients)
        # Z_A[:,k] = G^{-T} a_k  (needed for the d-row in the bordered S_bar update)
        Y_A = lu_solve((G_lu, G_piv), A.T)            # (nvar, ncon)
        Z_A = lu_solve((G_lu, G_piv), A.T, trans=1)   # (nvar, ncon); G^{-T} A^T

        W: List[int] = []                    # working set: active inequality indices

        # S_bar = A_bar_W G^{-1} A_bar_W^T; we maintain S_inv = S_bar^{-1} directly.
        # Row/column ordering: equality rows first [0..neq-1], then inequality rows
        # in insertion order [neq..neq+|W|-1].
        if has_eq:
            S_inv = np.linalg.inv(E @ Y_E)  # (neq, neq)
        else:
            S_inv = None
        update_count = 0

        outer_iter  = 0
        total_steps = 0
        status      = "running"

        # ***Outer loop***: add most violated constraint until optimal or infeasible.
        while outer_iter < max_iter and status == "running":
            outer_iter += 1

            # Find most violated constraint
            rho_all = A @ x - b             # rho_k = a_k'x - b_k  (<= 0 if feasible)
            p       = int(np.argmax(rho_all)) # p = first index where the maximum value occurs.
            #                                   (= lowest-index among most-violated constraints)
            rho_p   = float(rho_all[p])

            if rho_p <= tol:
                status = "optimal"
                break

            if verbose:
                _W_str = str(sorted(W)) if W else "{}"
                print(f"[outer {outer_iter:3d}]  p={p}  violation={rho_p:.3e}"
                    f"  |W|={len(W)}  W={_W_str}")

            # Tentative multiplier for p (builds up across inner steps).
            # Stationarity invariant:  G x + r + E' nu + Sum_{kinW} lam_k a_k + mu_p a_p = 0.
            mu_p = 0.0

            # ***Inner loop***: step until p is added or infeasibility found
            inner_status = "running"
            for h in range(max_iter):
                total_steps += 1

                a_p   = A[p, :]
                y_p   = Y_A[:, p]           # G^{-1} a_p
                z_p   = Z_A[:, p]           # G^{-T} a_p  (needed on ADD)
                W_arr = np.array(W, dtype=int) if W else np.empty(0, dtype=int)
                ell   = neq + len(W)

                # Solve S_bar r_bar = rhs_perm; compute primal step z
                #
                # S_bar (maintained incrementally) uses eq-first ordering:
                #   rows/cols 0..neq-1  <->  equality constraints
                #   rows/cols neq..ell-1 <->  inequality constraints in W (insertion order)
                # rhs_perm = [E y_p ; A_W y_p]  matches that ordering.

                if ell == 0:
                    # W={}, no equalities: unconstrained step.
                    r_W      = np.empty(0)
                    s        = np.empty(0)
                    z        = -y_p
                    rhs_perm = np.empty(0)
                else:
                    parts = []
                    if has_eq:
                        parts.append(E @ y_p)
                    if len(W) > 0:
                        parts.append(A[W_arr] @ y_p)
                    rhs_perm   = np.concatenate(parts)
                    r_bar_perm = S_inv @ rhs_perm
                    s          = r_bar_perm[:neq]
                    r_W        = r_bar_perm[neq:]
                    z          = -y_p
                    if has_eq:
                        z += Y_E @ s
                    if len(W) > 0:
                        z += Y_A[:, W_arr] @ r_W

                # Compute step lengths
                rho_p_cur = float(a_p @ x) - b[p]
                denom     = float(a_p @ z) # expected < 0

                t2 = (-rho_p_cur / denom) if denom < -tol else np.inf
                t1     = np.inf

                drop_j = -1 # local W-index to drop
                for j, k in enumerate(W):
                    if r_W[j] > tol:
                        cand = lam[k] / r_W[j]
                        if cand < t1:
                            t1     = cand
                            drop_j = j

                if verbose:
                    print(f"   [inner {h:3d}]  t_1={t1:.3e}  t_2={t2:.3e}"
                        f"  denom={denom:.3e}")

                # Infeasibility: cannot proceed in either direction.
                if t1 == np.inf and t2 == np.inf:
                    inner_status = "infeasible"
                    break

                t = min(t1, t2)

                # Update primal and dual variables
                x += t * z
                if len(W) > 0:
                    lam[W_arr] -= t * r_W
                if has_eq:
                    nu -= t * s
                mu_p   += t
                lam[p]  = mu_p

                # Update working set and S_inv = S_bar^{-1}
                if t2 <= t1:
                    # ADD constraint p: bordered inverse update.
                    # New S_bar = [[S_bar, c], [d^T, e]] where
                    #   c = rhs_perm (already computed), d = A_bar @ z_p, e = a_p' y_p
                    d_parts = []
                    if has_eq:
                        d_parts.append(E @ z_p)
                    if len(W) > 0:
                        d_parts.append(A[W_arr] @ z_p)
                    d   = np.concatenate(d_parts) if d_parts else np.empty(0)
                    e   = float(a_p @ y_p)
                    if ell == 0:
                        S_inv = np.array([[1.0 / e]])
                    else:
                        S_inv = _s_inv_add_np(S_inv, rhs_perm, d, e)
                    W.append(p)
                    update_count += 1
                    if refresh_freq > 0 and update_count % refresh_freq == 0:
                        S_inv = _refresh_S_inv(
                            W, E, A, Y_A, Z_A,
                            EY_E if has_eq else None, has_eq, neq,
                        )
                    inner_status = "added"
                    if verbose:
                        print(f"   -> constraint {p} added.  W={sorted(W)}")
                    break

                else:
                    # DROP constraint j* at working-set position drop_j.
                    # In eq-first S_bar its index is neq + drop_j.
                    k_drop    = neq + drop_j
                    dropped_k = W[drop_j]
                    W.pop(drop_j)
                    lam[dropped_k] = 0.0
                    ell_new = neq + len(W)
                    if ell_new == 0:
                        S_inv = None
                    else:
                        S_inv = _s_inv_drop_np(S_inv, k_drop)
                    update_count += 1
                    if refresh_freq > 0 and update_count % refresh_freq == 0 and ell_new > 0:
                        S_inv = _refresh_S_inv(
                            W, E, A, Y_A, Z_A,
                            EY_E if has_eq else None, has_eq, neq,
                        )
                    if verbose:
                        print(f"   -> constraint {dropped_k} dropped. W={sorted(W)}")

            else:
                inner_status = "max_inner"

            # Propagate terminal inner state.
            if inner_status == "infeasible":
                status = "infeasible"
            elif inner_status == "max_inner":
                status = "max_iterations"
            # else "added": outer loop continues normally.

        if status == "running":
            status = "max_iterations"

        prim_res = float(np.max(A @ x - b))

    if has_eq:
        eq_res  = float(np.linalg.norm(E @ x - f))
        kkt_res = float(np.linalg.norm(G @ x + r + A.T @ lam + E.T @ nu))
    else:
        eq_res  = 0.0
        kkt_res = float(np.linalg.norm(G @ x + r + A.T @ lam))

    info = {
        "status":           status,
        "outer_iterations": outer_iter,
        "total_steps":      total_steps,
        "primal_residual":  prim_res,
        "eq_residual":      eq_res,
        "kkt_residual":     kkt_res,
        "lam":              lam,
        "nu":               nu,
    }

    return x, info
