import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from nashopt import ParametricGNEP

np.random.seed(2)
np.set_printoptions(precision=4, suppress=True)

N = 8  # number of agents
C = 1.0          # shared capacity
D = 0.9          # desired total load
eta = 0.1        # weight on parameter deviation
rho = 0.3        # weight on leader's revenue

a = np.array([
    1.0,
    1.5,
    0.8,
    1.2,
    2.0,
    0.9,
    1.8,
    1.1
])

Gamma = np.array([
    [0.0, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
    [0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.1, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.15, 0.0, 0.1, 0.0, 0.0, 0.0],
    [0.1, 0.0, 0.0, 0.1, 0.0, 0.2, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.1, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.2],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0]
])

# p1_init = np.array([
#     -2.0,
#     -1.5,
#     -2.5,
#     -1.8,
#     -1.2,
#     -2.2,
#     -1.6,
#     -2.0
# ])
# p2_init = 0.05
# p0 = np.hstack((p1_init, p2_init))  # initial parameter guess

p1_bar = -2.*np.ones(N)  # reference

p1_min = -5.0
p1_max = 0.0

p2_min = 0.0
p2_max = 0.2

# initial parameter guess
p0 = np.hstack(((p1_min+p1_max)/2.*np.ones(N), (p2_min+p2_max)/2.))
x0 = C/N*np.ones(N)  # initial decision guess

lb = np.zeros(N)  # lower bounds on variables: x_i >=0
# C*np.ones(N)  # upper bounds on variables: x_i <= C, as sum(xi) <= C
ub = np.inf*np.ones(N)

npar = N+1  # number of parameters: p = [p1(N),p2]

pmin = np.hstack((-5.*np.ones(N), 0.))  # lower bounds on parameters
pmax = np.hstack((0.*np.ones(N), .2))  # upper bounds on parameters

sizes = [1]*N  # 1 dimension per each agent -> NP producers
nvar = N  # number of variables

f = []
for i in range(len(sizes)):
    f.append(jax.jit(partial(
        lambda x, p, i: a[i]*x[i]**2 + 2.*Gamma[i, :]@x + (p[i]+p[N]*jnp.sum(x)**2)*x[i], i=i)))


@jax.jit
def g(x, p):
    return jnp.array(jnp.sum(x)-C).reshape(1,)


ncon = 1


@jax.jit
def J(x, p):
    obj = (jnp.sum(x)-D)**2 + eta*jnp.sum((p[:N]-p1_bar)**2)
    for i in range(N):
        obj -= rho*(p[i]+p[N]*jnp.sum(x)**2)*x[i]
    return obj


gnep = ParametricGNEP(sizes, f, g, ncon, variational=False,
                      npar=npar, lb=lb, ub=ub)

M = 1  # number of random initializations
J_best = np.inf
for i in range(M):
    sol = gnep.solve(J=J, pmin=pmin, pmax=pmax, p0=p0, x0=x0, rho=1e8, alpha1=0., alpha2=0.,
                     maxiter=200, tol=1e-10, gne_warm_start=False, refine_gne=False, verbose=True)
    print(f"\n\033[1;33m k = {i+1}, J* = {sol.J}\033[0m", end="")
    if sol.J < J_best:
        best_sol = sol
        J_best = sol.J
        print("\033[1;33m   <-- best so far\033[0m")
    else:
        print("")
    
    if i<M-1:
        p0 = (pmax + pmin)/2. + .5*np.random.rand(npar)

sol = best_sol    
x_star, p_star, J_star, lam_star, residual, stats = sol.x, sol.p, sol.J, sol.lam, sol.res, sol.stats

print(f"\033[1;32mCPU time (LM): {stats.elapsed_time:.4f} s\033[0m")
print(f"\033[1;34mLM iters:\033[0m {stats.kkt_evals}")

print(f"\nOptimal J(x,p): {J_star:7.4f}")
print("\nOptimal parameters (p1, p2):")
print(p_star)
print(f"\nOptimal decisions (x_i): (sum(x_i) = {jnp.sum(x_star)})")
print(x_star)
print("\nOptimal multipliers (lam_i):")
print(lam_star)

# check best responses of all agents at the computed GNE
for i in range(gnep.N):
    x_br, fbr_opt, iters = gnep.best_response(i, x_star, p_star, rho=1e8)
    print(f"Agent {i}'s BR at the GNE: ", end="")
    print(f"|x_br-x_star| = {jnp.linalg.norm(x_br-x_star): 10.2g}", end="")
    # print(f"fbr_opt = {fbr_opt: 10.7g}")
    print(f" [{iters: 2d} L-BFGS-B iter(s)]")
