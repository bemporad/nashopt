import numpy as np
import jax
import jax.numpy as jnp
from nashopt import ParametricGNEP
from functools import partial

# Solve a GNE problem with N agents and look for sparse solutions
np.random.seed(3)

N = 10  # number of agents
sizes = [2]*N  # sizes of each agent
nvar = sum(sizes)
npar = 2  # number of parameters

# Agents' objective functions:
f = []
for i in range(len(sizes)):
    Qi = np.random.randn(nvar, nvar)
    Qi = Qi@Qi.T  # + 1.e-3*np.eye(nvar)
    ci = 10.*np.random.randn(nvar)
    Fi = 0.*np.random.randn(nvar, npar)
    f.append(jax.jit(partial(lambda x, p, Qi, ci, Fi: 0.5 *
             x@Qi@x + (ci+Fi@p).T@x, Qi=Qi, ci=ci, Fi=Fi)))

# Shared constraints:
ncon = 2*1  # number of inequality constraints
A = np.random.randn(int(ncon/2), sum(sizes))
A = np.vstack((A, -A))
b = np.random.rand(ncon)
S = np.random.rand(ncon, npar)


@jax.jit
def g(x, p):
    return A@x - b - S@p
# g=None
# ncon = 0


nvar = sum(sizes)
# ub=10.*np.ones(nvar) # upper bounds
lb = None  # np.zeros(nvar)
ub = None
pmin = 0.*np.ones(npar)  # lower bounds on parameters
pmax = 10.*np.ones(npar)  # upper bounds on parameters

pgnep = ParametricGNEP(sizes, npar=npar, f=f, g=g, ng=ncon, lb=lb, ub=ub)


def J(x, p):
    return -jnp.sum(x)


sol = pgnep.solve(J=None, pmin=pmin, pmax=pmax, rho=1.e-3,
                  gne_warm_start=False, refine_gne=True)

x_star, p_star, lam_star, residual, stats = sol.x, sol.p, sol.lam, sol.res, sol.stats
np.set_printoptions(precision=4, suppress=True)

print("=== GNE solution ===")
print(f"x = {x_star}")
if ncon > 0 or lb is not None or ub is not None:
    for i in range(pgnep.N):
        print(f"lambda[{i}] = {lam_star[i]}")


print(f"KKT residual norm = {float(jnp.linalg.norm(residual)): 10.7g}")
print(f"KKT evaluations   = {int(stats.kkt_evals): 3d}")
print(f"Elapsed time:       {stats.elapsed_time: .2f} seconds")

# check best responses of all agents at the computed GNE
for i in range(pgnep.N):
    x_br, fbr_opt, iters = pgnep.best_response(i, x_star, p_star, rho=1.e8)
    print(f"Agent {i}'s BR at the GNE: ", end="")
    print(f"|x_br-x_star| = {jnp.linalg.norm(x_br-x_star): 10.2g}", end="")
    # print(f", fbr_opt = {fbr_opt: 10.7g}")
    print(f" [{iters: 2d} L-BFGS-B iter(s)]")
