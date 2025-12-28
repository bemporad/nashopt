"""
(C) 2025 A. Bemporad
"""
import numpy as np
import jax
import jax.numpy as jnp
from nashopt import GNEP

N = 2  # number of agents
sizes = [10, 10]  # 2 agents of dimension 10
nvar = np.sum(sizes)


@jax.jit
def f1(x):
    return jnp.sum((x[:sizes[0]]+x[sizes[0]:])**2)+jnp.sum(x**2)


@jax.jit
def f2(x):
    return jnp.sum((x[:sizes[0]]+x[sizes[0]:]-10.)**2)+jnp.sum(x**2)


f = [f1, f2]  # agents' objectives

# Linear inequality constraints A*x<=b
A = np.vstack((np.ones((1, nvar)), -np.ones((1, nvar))))
b = np.array([15., 15.])


@jax.jit
def g(x):
    gx = A @ x - b
    return gx


lb = -3. * np.ones(nvar)
ub = 3. * np.ones(nvar)

gnep = GNEP(sizes, f=f, g=g, ng=2, lb=lb, ub=ub)

x0 = jnp.zeros(nvar)
sol = gnep.solve(x0)
x_star, lam_star, residual, stats = sol.x, sol.lam, sol.res, sol.stats


np.set_printoptions(precision=4, suppress=True)

print("=== GNE solution ===")
print(f"x = {x_star}")
for i in range(gnep.N):
    print(f"lambda[{i}] = {lam_star[i]}")
