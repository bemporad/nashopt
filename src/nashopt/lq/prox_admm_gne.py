# NashOpt: A Python package for computing Generalized Nash Equilibria (GNE) in noncooperative games.
#
# Solve Linear-Quadratic GNEPs via proximal ADMM.
#
# (C) 2025-2026 Alberto Bemporad

import numpy as np
from scipy.linalg import solve_triangular
import time
from types import SimpleNamespace

def solve(sizes, Q, c, A, b, C=None, d=None, lb=None, ub=None, x0=None, rho=1.0, gamma=1.0, maxiter=1000, tol=1e-6, verbose=True):
    """
    Solve LQ-GNEP problem using the proximal ADMM Algorithm 4.1 in [1].
    
    Linear quadratic GNEP with shared constraints:
    
    x_i^* = argmin_{x_i} 0.5 x^T Q_i x + c_i^T x
    s.t. A x <= b, C x = d, lb <= x_i <= ub, i=1,...,N
         x_{-i} = x_{-i}^*
         
    where x = (x_1, ..., x_N) is the concatenation of all players' decision variables, and Q_i, c_i are player i's cost parameters. As the algorithm in [1] is only formulated for equality constraints, we consider the following reformulation with N+1 players:

    x_i^* = argmin_{x_i} 0.5 x^T Q_i x + c_i^T x
    s.t. [A;I;-I] x + s = [b;ub;-lb], C x = d, i=1,...,N
         x_{-i} = x_{-i}^*, s = s^*

    s^* = argmin_{s} 0.5*||[A;I;-I] x + s - [b;ub;-lb]||^2
    s.t. [A;I;-I] x + s - [b;ub;-lb], s>=0
         x = x*

    Each prox-ADMM subproblem is solved in closed-form. 
    
    [1] E. Borgens and C. Kanzow, "ADMM-type Methods for Generalized Nash Equilibrium Problems in Hilbert Spaces," Siam Journal of Optimization, vol. 31, n.1, pp. 377-403, 2021.
        
    (C) 2026 A. Bemporad, February 16, 2026    
    """
    
    t0 = time.perf_counter()
    
    N = len(sizes)  # number of players
    nvar = sum(sizes) # total number of optimization variables
    m = A.shape[0]  # number of inequality constraints
    m_orig = m
    
    if C is not None:
        p = C.shape[0]  # number of equality constraints
    else:
        p = 0
        C = np.zeros((0, nvar))
        d = np.zeros(0)
    
    sum_i = np.cumsum(sizes)
    not_i = [list(range(sizes[0], nvar))]
    for i in range(1, N):
        not_i.append(list(range(sum_i[i-1])) + list(range(sum_i[i], nvar)))
    ii = [list(range(sum_i[i]-sizes[i], sum_i[i])) for i in range(N)]

    if x0 is None:
        x0 = np.zeros(nvar) # initial guess for GNE
    if lb is None:
        lb = -np.inf*np.ones(nvar)  # lower bounds on variables
    if ub is None:
        ub = np.inf*np.ones(nvar)  # upper bounds on variables
    
    has_bounds = any(ub < np.inf) or any(lb > -np.inf) 
    if has_bounds:
        # Embed box constraints into inequality constraints if needed, and update A, b, S accordingly
        AA = []
        bb = []
        is_bound = np.zeros(2*nvar, dtype=bool)
        for i in range(nvar):
            ei = np.zeros(nvar)
            ei[i] = 1.0
            if ub[i] < np.inf:
                AA.append(ei.reshape(1, -1))
                bb.append(ub[i])
                is_bound[2*i] = True
            if lb[i] > -np.inf:
                AA.append(-ei.reshape(1, -1))
                bb.append(-lb[i])
                is_bound[2*i+1] = True
        if m>0:
            A = np.vstack((A, np.vstack(AA)))
            b = np.hstack((b, np.hstack(bb)))
            m += len(AA)
        else:
            A = np.vstack(AA)
            b = np.hstack(bb)
            m = len(AA)
    
    # Initialization
    xk = x0.copy()  # (n,)
    muk = np.zeros(m)       # (m,)
    sk = b-A@xk      # (m,)
    if p>0:
        nuk = np.zeros(p)  

    # Note: we do not include bound constraints on xi as they are already embedded in the reformulated problem via A, b. Therefore, each agent's problem can be solved explicitly as an unconstrained QP.
    L = [np.linalg.cholesky(Q[i][ii[i],:][:,ii[i]] 
                            + gamma*np.eye(sizes[i]) 
                            + rho*(A[:,ii[i]].T @ A[:,ii[i]] 
                                   + (C[:,ii[i]].T @ C[:,ii[i]] if p>0 else 0))
                            ) for i in range(N)] # Cholesky factorization L@L.T for quadratic terms in players' cost functions
        
    go = True
    iter = 0
    while go:
        iter += 1

        # Primal updates
        res_x = 0.
        
        for i in range(N):
            # Solve (L@L.T)x = -ell via forward/backward substitution using the lower Cholesky factorization L of Q_i + gamma*I
            const_A = A[:, not_i[i]] @ xk[not_i[i]] + sk - b
            ell = (
                c[i][ii[i]]
                + Q[i][ii[i],:][:,not_i[i]] @ xk[not_i[i]]
                - gamma * xk[ii[i]]
                + A[:, ii[i]].T @ muk
                + rho * A[:, ii[i]].T @ const_A
            )
            if p > 0:
                const_C = C[:, not_i[i]] @ xk[not_i[i]] - d
                ell += C[:, ii[i]].T @ nuk + rho * C[:, ii[i]].T @ const_C            
            y = solve_triangular(L[i], -ell, lower=True) 
            xi = solve_triangular(L[i].T, y, lower=False)

            dxi = xi - xk[ii[i]] # change in player i's decision variables
            xk[ii[i]] = xi # update player i's decision variables during the agent's loop, according with (4.1a) in [1]
            res_x = max(res_x, np.linalg.norm(dxi))

        si = np.maximum(((1.+rho)*(b-A@xk)+gamma*sk-muk)/(1.+rho+gamma), 0) # closed-form solution for slack var
        dsk = si - sk # change in slack variable
        sk = si # update slack variable during the agent's loop, according with (4.1a) in [1]
        res_x = max(res_x, np.linalg.norm(dsk))
        
        # Dual updates
        dmu = (A@xk+sk-b)
        muk += rho*dmu
        res_con = np.linalg.norm(dmu)
        if p>0:
            dnu = (C@xk-d)
            nuk += rho*dnu
            res_con = max(res_con, np.linalg.norm(dnu))
            
        if verbose:
            print(f"iter {iter}: res_x={res_x:.4e}, res_con={res_con:.4e}")
        if max(res_x, res_con) < tol:
            if verbose:
                print("Convergence achieved")
            status_str = 'Converged'
            go = False
        if iter == maxiter:
            if verbose:
                print("Max iterations reached")
            status_str = 'Max iters reached'
            go = False

    lambda_k = muk
    if p>0:
        eta_k = nuk
       
    elapsed_time = time.perf_counter() - t0
    if verbose:
        print(f"Prox-ADMM finished in {iter} iterations and {elapsed_time:.4f} seconds.")

    sol = SimpleNamespace()
    sol.x = xk
    sol.lam = [[]]*N
    h=0
    for i in range(N):
        local_lam_lb = [] # local multipliers for player i's decision variable lower-bounds (if any)
        local_lam_ub = [] # local multipliers for player i's decision variable upper-bounds (if any)
        if has_bounds:
            for j in ii[i]:
                if is_bound[2*j]:
                    local_lam_ub.append(lambda_k[m_orig + h]) # multiplier for upper bound constraint on variable j
                    h += 1
                if is_bound[2*j+1]:
                    local_lam_lb.append(lambda_k[m_orig + h]) # multiplier for lower bound constraint on variable j
                    h += 1
        sol.lam[i] = np.hstack((lambda_k[:m_orig],np.array(local_lam_lb), np.array(local_lam_ub))) # concatenate the original multipliers for the shared constraints with the local multipliers for player i's decision variables 
    sol.mu = np.tile(eta_k, (N, 1)) if p>0 else None
    sol.elapsed_time = elapsed_time
    sol.num_iters = iter
    sol.status_str = status_str
    return sol

