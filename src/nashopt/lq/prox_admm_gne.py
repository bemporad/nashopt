# NashOpt: A Python package for computing Generalized Nash Equilibria (GNE) in noncooperative games.
#
# Solve Linear-Quadratic GNEPs via proximal ADMM.
#
# (C) 2025-2026 Alberto Bemporad

import numpy as np
import cvxpy as cp
import time
from types import SimpleNamespace

def solve(sizes, Q, c, A, b, C=None, d=None, lb=None, ub=None, x0=None, rho=1.0, gamma=1.0, maxiter=1000, tol=1e-6, verbose=True, cvx_solver=cp.OSQP):
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

    s^* = argmin_{s} ||[A;I;-I] x + s - [b;ub;-lb]||^2
    s.t. [A;I;-I] x + s - [b;ub;-lb], s>=0
         x = x*

    Each subproblem is solved via CVXPY with the specified solver "cvx_solver" (default is OSQP, but MOSEK, XPRESS, CLARABEL, or other solvers supported by CVXPY can be used as well). 
    
    [1] E. Borgens and C. Kanzow, "ADMM-type Methods for Generalized Nash Equilibrium Problems in Hilbert Spaces," Siam Journal of Optimization, vol. 31, n.1, pp. 377-403, 2021.
        
    (C) 2026 A. Bemporad, February 16, 2026    
    """
    
    t0 = time.time()
    
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
    
    if cvx_solver == "OSQP":
        solver = cp.OSQP
    elif cvx_solver == "MOSEK":
        solver = cp.MOSEK
    elif cvx_solver == "XPRESS":
        solver = cp.XPRESS
    elif cvx_solver == "CLARABEL":
        solver = cp.CLARABEL
    else:
        raise ValueError(f"Unsupported cvx_solver '{cvx_solver}'. Supported solvers are: 'OSQP', 'MOSEK', 'XPRESS', 'CLARABEL'.")

    # Initialization
    xk = x0.copy()  # (n,)
    muk = np.zeros(m)       # (m,)
    sk = b-A@xk      # (m,)
    if p>0:
        nuk = np.zeros(p)  

    # Setup CVXPY problems for proximal player updates
    x_var = [cp.Variable(sizes[i]) for i in range(N)] # player i's decision variables
    s_var = cp.Variable(m) # slack variable vector for inequality constraints
    p_x = cp.Parameter(nvar)
    p_s = cp.Parameter(m)
    p_mu = cp.Parameter(m) 
    if p>0:
        p_nu = cp.Parameter(p) # local copy of nu (for player i)
    
    cvx_constraints = [[]]*(N+1) # do not include bound constraints on x_var[i] as they are already embedded in the reformulated problem via A, b
    cvx_constraints[N] += [s_var >= 0] # slack variable constraints for player N (slack variable player)
   
    cvx_obj_br = [0.5*cp.quad_form(x_var[i], Q[i][ii[i],:][:,ii[i]]) + (c[i][ii[i]] + Q[i][ii[i]][:,not_i[i]] @ p_x[not_i[i]]).T @ x_var[i]  for i in range(N)]
    cvx_obj_br += [0.5*cp.sum_squares(A@p_x + s_var - b)] # objective for slack variable player
    for i in range(N):
        cvx_obj_br[i] += p_mu@(A[:, ii[i]] @ x_var[i]) + \
                        0.5*gamma * cp.sum_squares(x_var[i] - p_x[ii[i]]) + \
                        0.5*rho*cp.sum_squares(A[:, ii[i]] @ x_var[i] +\
                                                A[:, not_i[i]] @ p_x[not_i[i]] + p_s - b)
        if p>0:
            cvx_obj_br[i] += p_nu@(C[:, ii[i]] @ x_var[i]) +\
                0.5*rho * cp.sum_squares(C[:, ii[i]] @ x_var[i] + C[:, not_i[i]] @ p_x[not_i[i]] - d)
    cvx_obj_br[N] += 0.5*cp.sum_squares(A@p_x + s_var - b) + p_mu@s_var + 0.5*gamma*cp.sum_squares(s_var - p_s) + 0.5*rho*cp.sum_squares(A@p_x + s_var - b)
                            
    br_prob = [cp.Problem(cp.Minimize(cvx_obj_br[i]), cvx_constraints[i]) for i in range(N+1)]
    
    p_s.value = sk.copy() # initial value of slack variable for player i
        
    go = True
    iter = 0
    while go:
        iter += 1

        # Primal updates
        res_x = 0.
        p_s.value = sk.copy() # current value of slack variable for player i           
        p_mu.value = muk.copy() # current value of mu for player i
        if p>0:
            p_nu.value = nuk.copy() # current value of nu for player i
        for i in range(N):
            p_x.value = xk.copy() # copy most updated version of xk for player i's problem
            x_var[i].value = xk[ii[i]].copy() # warm start with current value of player i's decision variables
            br_prob[i].solve(solver=solver, warm_start=True, polish=True, verbose=False)                
            dxi = x_var[i].value - xk[ii[i]] # change in player i's decision variables
            xk[ii[i]] = x_var[i].value # update player i's decision variables during the agent's loop, according with (4.1a) in [1]
            res_x = max(res_x, np.linalg.norm(dxi))
        p_x.value = xk.copy()
        s_var.value = sk.copy() # warm start with current value of slack variable
        br_prob[N].solve(solver=solver, warm_start=True, polish=True, verbose=False)
        dsk = s_var.value - sk # change in slack variable
        sk = s_var.value # update slack variable during the agent's loop, according with (4.1a) in [1]
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
       
    elapsed_time = time.time() - t0
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
    sol.f = [0.5 * xk.T @ Q[i] @ xk + c[i].T @ xk for i in range(N)]
    sol.status_str = status_str
    return sol

