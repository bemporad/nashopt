import numpy as np
import jax
import jax.numpy as jnp
from nashopt import GNEP

# Solve a GNEP with 3 agents:
sizes = [2, 1, 1]      # [n1, n2, n3]

# Agent 1 objective:
@jax.jit
def f1(x):
    return jnp.sum((x[0:2] - jnp.array([1.0, -0.5]))**2)

# Agent 2 objective:
@jax.jit
def f2(x):
    return (x[2] + 0.3)**2

# Agent 3 objective:
@jax.jit
def f3(x):
    return (x[3] - 0.5*(x[0] + x[2]))**2

f = [f1, f2, f3]

# Shared constraint:
@jax.jit
def g(x):
    return jnp.array([x[3] + x[0] + x[2] - 2.0])


# Aeq = np.array([[1,1,1,1]])
# beq = np.array([2.0])
Aeq = None  # no equality constraints
beq = None

nvar = sum(sizes)
lb = np.zeros(nvar)  # lower bounds
ub = np.ones(nvar)  # upper bounds

gnep = GNEP(sizes, f=f, g=g, ng=1, lb=lb, ub=ub, Aeq=Aeq, beq=beq)

x0 = jnp.zeros(nvar)

sol = gnep.solve(x0, verbose=0)
x_star, lam_star, residual, stats = sol.x, sol.lam, sol.res, sol.stats

np.set_printoptions(precision=4, suppress=True)

print(
    f"CPU time = {stats.elapsed_time: .2f} s, KKT evaluations = {int(stats.kkt_evals): 3d}")

print("=== GNE solution ===")
print(f"x = {x_star}")
for i in range(gnep.N):
    print(f"lambda[{i}] = {lam_star[i]}")

print(f"KKT residual norm = {float(jnp.linalg.norm(residual)): 10.7g}")

# check best responses of all agents at the computed GNE
for i in range(gnep.N):
    x_br, fbr_opt, iters = gnep.best_response(i, x_star, rho=1e8)
    print(f"Agent {i}'s BR at the GNE: ", end="")
    print(f"|x_br-x_star| = {jnp.linalg.norm(x_br-x_star): 10.2g}", end="")
    # print(f"fbr_opt = {fbr_opt: 10.7g}")
    print(f" [{iters: 2d} L-BFGS-B iter(s)]")
