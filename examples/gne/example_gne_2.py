"""
Solve the generalized Nash equilibrium problem described in [1, Fig. 6], originally proposed in [2, Section 5] for n=20 agents.

[1] F. Fabiani and A. Bemporad, “An active learning method for solving competitive multi-agent decision-making and control problems,” 2024, http://arxiv.org/abs/2212.12561. 

[2] F. Salehisadaghiani, W. Shi, and L. Pavel, “An ADMM approach to the problem of distributed Nash equilibrium seeking.” CoRR, 2017.

(C) 2025 A. Bemporad
"""
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from nashopt import GNEP

N = 10  # number of agents
sizes = [1]*N  # n agents of dimension 1
nvar = np.sum(sizes)


@jax.jit
def cost(x, i):
    # Cost function minimized by agent #i, i=0,...,N-1
    ci = N*(1.+i/2.)
    return ci*x[i]-x[i]*(60.*N-jnp.sum(x))


f = [partial(cost, i=i) for i in range(N)]

lb = 7. * np.ones(nvar)
ub = 100. * np.ones(nvar)

gnep = GNEP(sizes, f=f, lb=lb, ub=ub)

print("Solving GNEP with N =", N, "agents ... ", end="")
x0 = jnp.zeros(nvar)
sol = gnep.solve(x0, verbose=0)
x_star, lam_star, residual, stats = sol.x, sol.lam, sol.res, sol.stats
print("done.")

np.set_printoptions(precision=4, suppress=True)

print("=== GNE solution ===")
print(f"x = {x_star}")
for i in range(gnep.N):
    print(f"lambda[{i}] = {lam_star[i]}")

print(f"KKT residual norm = {float(jnp.linalg.norm(residual)): 10.7g}")
print(f"KKT evaluations   = {int(stats.kkt_evals): 3d}")
print(f"Elapsed time:       {stats.elapsed_time: .2f} seconds")
