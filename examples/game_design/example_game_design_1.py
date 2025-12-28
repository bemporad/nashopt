import numpy as np
import jax
import jax.numpy as jnp
from nashopt import ParametricGNEP

np.random.seed(0)

# Parametric GNEP with 3 agents:

sizes = [2, 1, 1]      # [n1, n2, n3]
npar = 5              # number of parameters

# Agent 1 objective:
@jax.jit
def f1(x, p):
    return jnp.sum((x[0:2] - p[0]*jnp.array([1.0, -0.5]))**2)

# Agent 2 objective:
@jax.jit
def f2(x, p):
    return (x[2] + p[1])**2

# Agent 3 objective:
@jax.jit
def f3(x, p):
    return (x[3] - p[2]*(x[0] + x[2]))**2

f = [f1, f2, f3]

# Shared constraint:
@jax.jit
def g(x, p):
    return jnp.array([x[3] + x[0] + x[2] - 5.*p[3]])

@jax.jit
def h(x, p):
    return jnp.array([x[0]**2 + x[1]**2 - p[4]])


nvar = sum(sizes)
lb = -10.*np.ones(nvar)  # lower bounds
ub = 10.*np.ones(nvar)  # upper bounds

pgnep = ParametricGNEP(sizes, npar=npar, f=f, g=g,
                       ng=1, lb=lb, ub=ub, h=h, nh=1)


def J(x, p):
    # Design objective:
    # J = jnp.sum(jnp.array([pgnep.f[i](x,p) for i in range(pgnep.N)])) # minimize social welfare
    # J = 0.
    J = (x[0]-.5)**2
    return J


# if pmin = pmax, just solve a standard GNEP for that p
pmin = -10.*np.ones(npar)
pmax = 10.*np.ones(npar)
pmax[0:2] = np.inf  # p1,p2 unbounded above
pmin[0:2] = -np.inf  # p1,p2 unbounded below

p0 = np.clip(np.random.randn(npar), pmin, pmax)
sol = pgnep.solve(J, pmin, pmax, p0, rho=1., maxiter=200,
                  tol=1e-10, refine_gne=True)
x_star = sol.x
p_star = sol.p
J_star = sol.J
stats = sol.stats


np.set_printoptions(precision=4, suppress=True)

print("=== GNE solution ===")
print(f"x = {x_star}")
print(f"p = {p_star}")
print(f"J_star = {J_star: 10.7g}")
print(f"[{stats.kkt_evals:2d} KKT evaluation(s), elapsed time = {stats.elapsed_time:10.4g} s]")

# check best responses of all agents at the computed GNE
for i in range(pgnep.N):
    sol = pgnep.best_response(i, x_star, p_star, rho=1e8)
    x_br, fbr_opt, iters = sol.x, sol.f, sol.stats.iters
    print(f"Agent {i}'s BR at the GNE: ", end="")
    print(f"|x_br-x_star| = {jnp.linalg.norm(x_br-x_star): 10.2g}", end="")
    # print(f"fbr_opt = {fbr_opt: 10.7g}")
    print(f" [{iters: 2d} L-BFGS-B iter(s)]")
