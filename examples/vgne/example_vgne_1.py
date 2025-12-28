import numpy as np
import jax
import jax.numpy as jnp
from nashopt import GNEP
from functools import partial

# Solve a Variational GNE problem with N agents:
np.random.seed(3)

N = 3  # number of agents
sizes = [2]*N  # sizes of each agent
nvar = sum(sizes)

# Agents' objective functions:
f = []
Q = []
c = []
for i in range(len(sizes)):
    Qi = np.random.randn(nvar, nvar)
    Qi = Qi@Qi.T + 1.e-3*np.eye(nvar)
    ci = 10.*np.random.randn(nvar)
    f.append(jax.jit(partial(lambda x, Qi, ci: 0.5*x@Qi@x + ci@x, Qi=Qi, ci=ci)))
    Q.append(Qi)
    c.append(ci)

# Shared constraints:
ncon = 2*3  # number of inequality constraints
A = .5*np.random.randn(int(ncon/2), sum(sizes))
A = np.vstack((A, -A))
b = np.random.rand(ncon)


@jax.jit
def g(x):
    return A@x-b


Aeq = np.ones(nvar).reshape(1, -1)
beq = np.array([1.0])

nvar = sum(sizes)
lb = -np.ones(nvar)  # lower bounds
ub = np.ones(nvar)  # upper bounds

# create GNEP object and solve for vGNE
gnep = GNEP(sizes, f=f, g=g, ng=ncon, lb=lb, ub=ub,
            Aeq=Aeq, beq=beq, variational=True)

x0 = jnp.zeros(nvar)
sol = gnep.solve(x0)
x_star_vgne, lam_star_vgne, residual_vgne, stats_vgne = sol.x, sol.lam, sol.res, sol.stats

np.set_printoptions(precision=4, suppress=True)

print("=== vGNE solution ===")
print(f"x = {x_star_vgne}")
for i in range(gnep.N):
    print(f"lambda[{i}] = {lam_star_vgne[i]}")


print(f"KKT residual norm = {float(jnp.linalg.norm(residual_vgne)): 10.7g}")
print(f"KKT evaluations   = {int(stats_vgne.kkt_evals): 3d}")
print(f"Elapsed time:       {stats_vgne.elapsed_time: .2f} seconds")

# check best responses of all agents at the computed GNE
for i in range(gnep.N):
    sol_vgne = gnep.best_response(i, x_star_vgne, rho=1.e8)
    x_br_vgne, fbr_opt_vgne, iters_vgne = sol_vgne.x, sol_vgne.f, sol_vgne.stats.iters
    print(f"Agent {i}'s BR at the GNE: ", end="")
    print(
        f"|x_br-x_star| = {jnp.linalg.norm(x_br_vgne-x_star_vgne): 10.2g}", end="")
    # print(f", fbr_opt = {fbr_opt_vgne: 10.7g}")
    print(f" [{iters_vgne: 2d} L-BFGS-B iter(s)]")
