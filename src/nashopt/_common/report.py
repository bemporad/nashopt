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

