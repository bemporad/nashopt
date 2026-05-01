# NashOpt: A Python package for computing Generalized Nash Equilibria (GNE) in noncooperative games.
#
# Solve Linear-Quadratic GNEPs via Lemke's method.
#
# (C) 2025-2026 Alberto Bemporad

import numpy as np
from quantecon.optimize import lcp_lemke
from types import SimpleNamespace
import time

def solve(Q, S, p, A, b, lb, ub=None):
    """
    Attempts to compute a variational GNE of the LQ-GNEP via Lemke's method.

    Each player i minimizes a convex quadratic objective over the shared
    feasible set  {x : A x + b >= 0}:

        min_{xi}  0.5 xi' Qi xi  +  xi' sum(j\neq i) S[i][j] xj  +  pi' xi
        s.t.      A x + b >= 0   (shared constraints,  A x + b >= 0 form)
                  x >= lb        (finite lower bounds, used for the shift y = x-lb)

    A VE is a GNE where all players share the same Lagrange multiplier for the
    shared constraints, i.e. the solution of the aggregated VI  F(x)'(z-x) >= 0
    with mapping  F(x) = W x + p_vec.

    Parameters
    ----------
    Q  : list of N arrays              local PSD Hessians; Q[i] is (n_i, n_i)
    S  : NxN list; S[i][j] \in R^{n_i x n_j} or None when i == j
    p  : list of N arrays              linear cost vectors; p[i] is (n_i,)
    A  : (m, nN) array                 constraint matrix (A x + b >= 0 form)
    b  : (m,) array                    constraint offset
    lb : (nN,) array                   lower bounds on x; used for the variable
                                       shift y = x - lb >= 0 (must be finite).
    ub : (nN,) array or None           upper bounds on x. When given, the rows
                                       -I x + ub >= 0 are prepended to (A, b)
                                       automatically. Default: None.

    Returns
    -------
    GNEResult namedtuple:
        x        (nN,)  equilibrium strategies  (NaN if Lemke fails)
        lam      (m,)   common VE multipliers   (NaN if Lemke fails)
                        satisfies  W x + p_vec = A_aug.T lam  where A_aug is the
                        internally augmented constraint matrix (ub rows, if any,
                        prepended before the user-supplied rows)
        status   int    0 = solution, 1 = iter-limit, 2 = ray-termination
        num_iter int    number of Lemke pivots performed
        success  bool   True iff a VE was found

    LCP formulation (variable shift  y = x - lb >= 0)
    --------------------------------------------------
    Rewrite  A x + b >= 0  as  -(-A x - b) >= 0  (i.e., in A_ub x <= b_ub
    form with A_ub = -A, b_ub = b).  After the shift y = x - lb the VE
    KKT conditions become the standard LCP  Mz + q >= 0, z >= 0, z*(Mz+q)=0
    with z = (y, lam) and

         [  W    -A.T ]         [    p_tilde     ]
    M =  [            ],  q =  [                 ],   p_tilde = W lb + p_vec
         [  A     0   ]         [  b + A lb      ]

    Sign of q's second block: row j of  M z + q >= 0  is
        (A y)_j + (b + A lb)_j  =  (A x)_j + b_j  >= 0   (ok, since x = y + lb)

    KKT at solution x* = y* + lb  (interior points y*_k > 0):
      - Stationarity:    W x* + p_vec - A.T lam* = 0
      - Feasibility:     A x* + b >= 0
      - Complementarity: lam*_j (A x* + b)_j = 0,   lam* >= 0
      
    [1] D.A. Schiro, J.-S. Pang, U.V. Shanbhag, "On the solution of affine generalized 
        Nash equilibrium problems with shared constraints by Lemke’s method," 
        Mathematical Programming, Ser. A, 142:1–46, 2013.
    """
    
    t0 = time.perf_counter()
    
    # -- dimensions ------------------------------------------------------------
    N    = len(Q)
    dims = [Q[i].shape[0] for i in range(N)]
    nN   = sum(dims)
    idx = np.cumsum([0] + dims)
    A   = np.asarray(A, float)
    b   = np.asarray(b, float)
    lb  = np.asarray(lb, float)

    # Prepend upper-bound rows  -I x + ub >= 0  (lb is handled by the shift)
    if ub is not None:
        ub = np.asarray(ub, float)
        A  = np.vstack([-np.eye(nN), A])
        b  = np.concatenate([ub, b])

    m = A.shape[0]

    # Form aggregated pseudo-gradient Jacobian W
    W = np.zeros((nN, nN))
    for i in range(N):
        si, ei = idx[i], idx[i + 1]
        W[si:ei, si:ei] = Q[i]
        for j in range(N):
            if i != j:
                sj, ej = idx[j], idx[j + 1]
                W[si:ei, sj:ej] = S[i][j]

    p_vec = np.concatenate(p)

    # variable shift:  y = x - lb >= 0
    p_tilde = W @ lb + p_vec        # adjusted linear cost
    b_shift = b + A @ lb            # slack at lb: (A lb + b) = (Ax+b)|_{x=lb}

    # LCP problem for finding vGNE solution
    M_lcp = np.block([
        [W,   -A.T                  ],   # stationarity
        [A,    np.zeros((m, m))     ],   # feasibility
    ])
    q_lcp = np.concatenate([p_tilde, b_shift])

    # Call Lemke's method to solve the LCP
    res = lcp_lemke(M_lcp.astype(float), q_lcp.astype(float))

    if res.success:
        y_star   = res.z[:nN]
        lam_star = res.z[nN:]
        x_star   = y_star + lb
    else:
        x_star   = np.full(nN, np.nan)
        lam_star = np.full(m,  np.nan)

    elapsed_time = time.perf_counter() - t0
    
    sol = SimpleNamespace()
    sol.x = x_star
    sol.lam = lam_star # Lagrange multipliers, not split by agent
    sol.status = res.status
    sol.num_iters = res.num_iter
    sol.success = res.success
    sol.elapsed_time = elapsed_time

    return sol

