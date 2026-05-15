# NashOpt: A Python package for computing Generalized Nash Equilibria (GNE) in noncooperative games.
#
# Solve Linear-Quadratic GNEPs via Lemke's method (quantecon backend).
#
# (C) 2025-2026 Alberto Bemporad

import numpy as np
import scipy as sp
from types import SimpleNamespace
import time
from typing import Dict, Tuple

from quantecon.optimize.lcp_lemke import lcp_lemke
from quantecon.optimize.linprog_simplex import PivOptions
from .._common.optional_deps import add_box_constraints

_STATUS_MAP = {0: 'optimal', 1: 'max_iterations', 2: 'ray_termination'}


def vgne_lemke_dual(
    G: np.ndarray,
    r: np.ndarray,
    A: np.ndarray = None,
    b: np.ndarray = None,
    E: np.ndarray = None,
    f: np.ndarray = None,
    max_iter: int = 10**6,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Solve the v-GNE KKT system via Lemke's algorithm on the reduced LCP in lam* only.

    Given pseudogradient F(x) = Gx + r, shared inequality Ax <= b (optional),
    and shared equality Ex = f (optional), eliminate (x*, nu*) from the
    stationarity and equality conditions to obtain the LCP(q, M) in lam* alone:

        M = A P A^T,   q = b - A x_0,

    where P = G^{-1} - G^{-1}E^T(EG^{-1}E^T)^{-1}EG^{-1} and x_0 is the
    equality-constrained solution at lam* = 0. When E is absent, P = G^{-1} and
    x_0 = -G^{-1}r. When A is absent, the solution is x_0 with no LCP step.

    Parameters
    ----------
    G, r : pseudogradient data (G nvar x nvar, r nvar).
    A, b : shared inequality constraint data (A ncon x nvar, b ncon). Optional.
    E, f : shared equality constraint data (E neq x nvar, f neq). Optional.
    max_iter : maximum number of iterations for Lemke's method (default: 10**6)
    tol : tolerance for pivoting and ratio tests in Lemke's method (default: 1e-12)


    Returns
    -------
    x_star : (nvar,) primal v-GNE solution.
    info   : SimpleNamespace with fields
               status       - 'optimal' | 'trivial' | 'max_iterations' | 'ray_termination'
               iterations   - number of Lemke pivot steps
               kkt_residual - ||Gx* + r + A^T lam* + E^T nu*||
               lam          - (ncon,) dual variable for inequalities (empty if A is None)
               nu           - (neq,) dual variable for equalities (empty if E is None)
    """
    G = np.asarray(G, dtype=float)
    r = np.asarray(r, dtype=float).ravel()

    has_ineq = A is not None
    has_eq   = E is not None

    if has_ineq:
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).ravel()
    if has_eq:
        E = np.asarray(E, dtype=float)
        f = np.asarray(f, dtype=float).ravel()

    G_lu = sp.linalg.lu_factor(G)

    # Base solution x_0: equality-constrained Nash equilibrium at lam* = 0.
    # Solve [G E^T; E 0][x_0; nu_0] = [-r; f] via Schur elimination.
    x_free = sp.linalg.lu_solve(G_lu, -r)            # -G^{-1}r
    if has_eq:
        Y_E     = sp.linalg.lu_solve(G_lu, E.T)          # G^{-1}E^T,  (nvar, neq)
        EY_E    = E @ Y_E                                  # EG^{-1}E^T, (neq, neq)
        EY_E_lu = sp.linalg.lu_factor(EY_E)
        nu_0    = sp.linalg.lu_solve(EY_E_lu, E @ x_free - f)  # ν at λ*=0
        x_0     = x_free - Y_E @ nu_0                     # project onto Ex = f
    else:
        x_0  = x_free
        nu_0 = np.empty(0)

    if not has_ineq:
        # No inequalities: solution is the equality-constrained point.
        x_star     = x_0
        lam        = np.empty(0)
        nu         = nu_0
        num_iter   = 0
        status_str = 'optimal'
    else:
        # Build LCP(q, M) in λ* only.
        # With equalities: M = A P A^T, where P = G^{-1} - G^{-1}E^T(EG^{-1}E^T)^{-1}EG^{-1}.
        # PA^T = Y_A - Y_E @ (EY_E)^{-1} @ (E Y_A).
        Y_A = sp.linalg.lu_solve(G_lu, A.T)              # G^{-1}A^T, (nvar, ncon)

        if has_eq:
            EY_A = E @ Y_A                                            # (neq, ncon)
            PAt  = Y_A - Y_E @ sp.linalg.lu_solve(EY_E_lu, EY_A)    # PA^T, (nvar, ncon)
        else:
            EY_A = None
            PAt  = Y_A                                                # G^{-1}A^T

        M_lcp = np.ascontiguousarray(A @ PAt)          # (ncon, ncon)
        q_lcp = np.ascontiguousarray(b - A @ x_0)      # (ncon,)

        piv_opts = PivOptions(tol_piv=tol, tol_ratio_diff=tol)
        res = lcp_lemke(M_lcp, q_lcp, max_iter=max_iter, piv_options=piv_opts)

        lam    = res.z                                  # λ*, (ncon,)
        x_star = x_0 - PAt @ lam

        if has_eq:
            nu = nu_0 - sp.linalg.lu_solve(EY_E_lu, EY_A @ lam)
        else:
            nu = np.empty(0)

        num_iter   = res.num_iter
        status_str = ('trivial' if (res.status == 0 and num_iter == 0)
                      else _STATUS_MAP[res.status])

    # KKT stationarity residual: Gx* + r + A^T λ* + E^T ν*
    kkt = G @ x_star + r
    if has_ineq:
        kkt = kkt + A.T @ lam
    if has_eq:
        kkt = kkt + E.T @ nu

    info = SimpleNamespace(
        status       = status_str,
        iterations   = num_iter,
        kkt_residual = float(np.linalg.norm(kkt)),
        lam          = lam,
        nu           = nu,
    )

    return x_star, info


def solve(Q, c, dim, A=None, b=None, lb=None, ub=None, E=None, f=None,
          max_iter=10**6, tol=1e-12):
    """
    Attempts to compute a variational GNE of the LQ-GNEP via Lemke's method applied on the dual
    reformulation of the KKT conditions of the game (quantecon backend).

    Each player i minimizes a convex quadratic objective over the shared
    feasible set  {x : A x <= b, E x = f}:

        min_{xi}  0.5 x' Q[i] x  +  c[i]' x
        s.t.      A x <= b   (shared inequality constraints, optional)
                  E x  = f   (shared equality constraints, optional)
                  lb <= x <= ub (finite lower and upper bounds, optional)

    Parameters
    ----------
    Q   : list of N arrays             cost Hessians; Q[i] is (nvar, nvar), should be symmetrized
    c   : list of N arrays             linear cost vectors; c[i] is (nvar,)
    dim : list of N ints               number of variables per player
    A   : (m, nvar) array or None      inequality constraint matrix (A x <= b form)
    b   : (m,) array or None           inequality constraint offset
    lb  : (nvar,) array or None        lower bounds on variables
    ub  : (nvar,) array or None        upper bounds on variables
    E   : (q, nvar) array or None      equality constraint matrix (E x = f)
    f   : (q,) array or None           equality constraint RHS
    max_iter : int                     maximum number of iterations for Lemke's method
    tol      : float                   pivot tolerance for Lemke's method

    Returns
    -------
    sol : SimpleNamespace with fields
        x = solution
        lam = Lagrange multipliers for inequalities + bounds (empty if A is None and lb/ub are None)
        nu  = Lagrange multipliers for equalities (empty if E is None)
        status = info dict returned by vgne_lemke_dual
        num_iters = number of iterations performed
        success = True if the LCP was solved (or no LCP needed)
        elapsed_time = elapsed time in seconds

    [1] D.A. Schiro, J.-S. Pang, U.V. Shanbhag, "On the solution of affine generalized
        Nash equilibrium problems with shared constraints by Lemke's method,"
        Mathematical Programming, Ser. A, 142:1–46, 2013.
    """

    t0 = time.perf_counter()

    N    = len(Q)
    nvar   = sum(dim)
    idx  = np.cumsum([0] + list(dim))

    if (A is None) != (b is None):
        raise ValueError("A and b must both be provided or both be None.")
    if (E is None) != (f is None):
        raise ValueError("E and f must both be provided or both be None.")

    W     = np.zeros((nvar, nvar))
    p_vec = np.zeros(nvar)
    for i in range(N):
        si, ei = idx[i], idx[i + 1]
        W[si:ei, :]  = Q[i][si:ei, :]
        p_vec[si:ei] = c[i][si:ei]

    # Handle variable bounds by adding rows to A and b.
    if lb is None and ub is None:
        AA = A
        bb = b
    else:
        AA, bb = add_box_constraints(nvar, A=A, b=b, lb=lb, ub=ub)
    
    x_star, info = vgne_lemke_dual(
        G=W, r=p_vec, A=AA, b=bb, E=E, f=f, max_iter=max_iter, tol=tol
    )

    elapsed_time = time.perf_counter() - t0

    sol = SimpleNamespace()
    sol.x            = x_star
    sol.lam          = info.lam
    sol.nu           = info.nu
    sol.status       = info
    sol.num_iters    = info.iterations
    sol.success      = info.status in ('trivial', 'optimal')
    sol.elapsed_time = elapsed_time

    return sol
