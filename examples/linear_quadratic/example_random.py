"""
Generate a random monotone variational GNEP (vGNE) LQ problem with N players
and given monotonicity parameter mu >= 0. 

Player i solves:
    min_{x_i}  f_i(x) = (1/2) x_i^T Q[i,i] x_i + Q[i,-i] x_{-i} + c_i[i]^T x_i
    subject to  A x <= b,   A_eq x = b_eq,   lb <= x <= ub

with x_i having dim[i] decision variables.

(C) 2026 A. Bemporad
"""

import numpy as np
from nashopt.lq.generate_random import generate_random

np.random.seed(0)

dim = [3, 2, 4]  # dims[i] = number of decision variables of agent i
gnep_lq, data = generate_random(dim=dim, m=0, m_act=0, q=0, seed=None, mu=0.1, solver="dr_daqp")
sol = gnep_lq.solve(verbose=0)

print("x* (build)= ", data['x_star'])
print("x* (solve):", sol.x)

G=data["G"] 
print(np.linalg.eigvalsh(0.5 * (G + G.T)).min())
M=gnep_lq.pseudogradient_matrix()
print(np.linalg.eigvalsh(0.5 * (M + M.T)).min())
