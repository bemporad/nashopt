# NashOpt: A Python package for computing Generalized Nash Equilibria (GNE) in noncooperative games.
#
# Reporting functions. 
#
# (C) 2025-2026 Alberto Bemporad

import numpy as np

def eval_residual(res, verbose, f_evals, elapsed_time):
    warn_tol = 1.e-4
    norm_res = np.sqrt(np.sum(res**2))
    if verbose > 0:
        print(
            f"GNEP solved: ||KKT residual||_2 = {norm_res:.3e} found in {f_evals} function evaluations, time = {elapsed_time:.3f} seconds.")  
        if norm_res > warn_tol:
            print(
                f"\033[1;33mWarning: the KKT residual norm > {warn_tol}, an equilibrium may not have been found.\033[0m")
    return norm_res

def check_equilibrium_common(gnep, x, p=None, verbose=True, **kwargs):
    """ Check if x is a GNE by evaluating the best response of each agent at x and comparing it with x, as well as comparing the associated objective values.
    
    Parameters:
    -----------
    x : array-like
        Joint strategy to check for equilibrium.
    p : array-like, optional
        Game parameters to check for equilibrium. Only used for parametric GNEPs.
    verbose : bool, optional
        If True, print the distance between x and the best response for each agent, as well as the difference in objective values.
    kwargs : dict, optional
        Additional keyword arguments to pass to the best response computation.
        
    Returns:
    -----------
    dx : ndarray
        Difference between the current strategy and the collection of best responses for each agent.
    df : ndarray
        Difference between the current objective values and the optimal objective values for each agent.
    """

    dx = np.empty(gnep.nvar)
    df = np.empty(gnep.N)
    for i in range(int(gnep.N)):
        xi = x[gnep.i1[i]:gnep.i2[i]]
        if p is None:
            sol_i = gnep.best_response(i, x, **kwargs)
        else:
            sol_i = gnep.best_response(i, x, p, **kwargs)
        xstar_i = sol_i.x[gnep.i1[i]:gnep.i2[i]]
        fstar_i = sol_i.f
        fi = gnep.f[i](x) if p is None else gnep.f[i](x, p)
        df[i] = fi - fstar_i
        dx[gnep.i1[i]:gnep.i2[i]] = xi - xstar_i
        if verbose:
            dx_norm = np.linalg.norm(dx[gnep.i1[i]:gnep.i2[i]])
            print(f"Agent {i:>2d}'s BR: ‖x[{i:>2d}] - br(x[-{i:>2d}])‖ = {dx_norm:>12.4E}, ", end="")
            print(f"f[{i:>2d}](x) - f*[{i:>2d}] = {np.abs(df[i]):>12.4E}")
    return dx, df