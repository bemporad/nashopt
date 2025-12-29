import numpy as np
import jax
import jax.numpy as jnp
from nashopt import ParametricGNEP
from functools import partial
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 32,
    "font.size": 32,
    "legend.fontsize": 32,
    "xtick.labelsize": 32,
    "ytick.labelsize": 32,
})

# Solve a GNE problem with N agents and look for sparse solutions
np.random.seed(3)

N = 2*20  # number of agents
sizes = [1]*N  # agent's sizes
npar = 1  # number of parameters

# Agents' objective functions on pairs of agents. Each value of x for which each pair (2k,2k+1) is equal is a Nash equilibrium, as each agent has no incentive to deviate from its companion.
f = []
for i in range(len(sizes)):
    def fi(x, p, i):
        k = int(i/2)
        return (x[2*k]-x[2*k+1])**2
    f.append(jax.jit(partial(lambda x, p, i: fi(x, p, i), i=i)))

g = None
ncon = 0

nvar = sum(sizes)
# ub=10.*np.ones(nvar) # upper bounds
lb = None  # np.zeros(nvar)
ub = None
pmin = -1.*np.ones(npar)  # lower bounds on parameters
pmax = 1.*np.ones(npar)  # upper bounds on parameters

pgnep = ParametricGNEP(sizes, npar=npar, f=f, g=g, ng=ncon, lb=lb, ub=ub)

xref = jnp.repeat(np.arange(1, int(N/2)+1)/10., 2)


def J(x, p):
    # + alpha1.*jnp.sum(jnp.abs(x)) <- doesn't work!
    return jnp.sum((x-xref)**2)


alpha1_set = np.arange(0., 5.1, .1)
x_star_set = []
p_star_set = []
nnz_set = []
J_set = []
time_set = []
for alpha1 in alpha1_set:
    sol = pgnep.solve(J=J, pmin=pmin, pmax=pmax, rho=1.e4, alpha1=alpha1,
                      alpha2=0.,  gne_warm_start=False, refine_gne=False)
    x_star, p_star, lam_star, residual, stats = sol.x, sol.p, sol.lam, sol.res, sol.stats
    x_star_set.append(x_star)
    p_star_set.append(p_star)
    time_set.append(stats.elapsed_time)
    nnz = jnp.sum(jnp.abs(x_star) > 1e-6)
    nnz_set.append(nnz)
    J_star = J(x_star, p_star)
    J_set.append(J_star)
    print(f"alpha1 = {alpha1:10.7g}, ||x*||_0 = {nnz:3d}, J(x*,p*) = {J_star:7.4f}, ||R(x*,p*)||_2 = {jnp.linalg.norm(residual):7.2g}")


fig, ax1 = plt.subplots(figsize=(12, 7))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
ax1.plot(alpha1_set, nnz_set, color=colors[0], linewidth=4)
ax1.set_xlabel(r'$\alpha_1$')
ax1.set_ylabel(r'number of nonzeros', color=colors[0])
ax1.tick_params(axis='y', colors=colors[0])   # y-tick labels + ticks

ax2 = ax1.twinx()
ax2.plot(alpha1_set, J_set, color=colors[2], linewidth=4)
ax2.set_ylabel(r'optimal cost', color=colors[2])
ax2.tick_params(axis='y', colors=colors[2])   # y-tick labels + ticks
plt.grid()
plt.show()
plt.savefig("example_sparse.pdf", bbox_inches='tight')

print(
    f"CPU time: init = {time_set[0]: 7.4f} s, subsequent (average) = {sum(time_set[1:])/len(time_set[1:]):7.4f} s")

cpu_time = []
alpha1 = 2
M = 200
f = []
for i in range(2*M):
    def fi(x, p, i):
        k = int(i/2)
        return (x[2*k]-x[2*k+1])**2
    f.append(jax.jit(partial(lambda x, p, i: fi(x, p, i), i=i)))

k_range = list(range(1, 10))+list(range(10, M+1, 10))
for k in k_range:
    N = 2*k  # number of agents
    sizes = [1]*N  # agent's sizes
    pgnep = ParametricGNEP(
        sizes, npar=npar, f=f[:N], g=g, ng=ncon, lb=lb, ub=ub)

    xref = jnp.repeat(np.arange(1, int(N/2)+1)/10., 2)

    def J(x, p):
        # + alpha1.*jnp.sum(jnp.abs(x)) <- doesn't work!
        return jnp.sum((x-xref)**2)

    sol = pgnep.solve(J=J, pmin=pmin, pmax=pmax, rho=1.e4, alpha1=alpha1,
                      alpha2=0.,  gne_warm_start=False, refine_gne=False)
    x_star, p_star, lam_star, residual, stats = sol.x, sol.p, sol.lam, sol.res, sol.stats
    cpu_time.append(stats.elapsed_time)
    nnz = jnp.sum(jnp.abs(x_star) > 1e-6)
    print(f"N = {N:3d}, CPU time = {stats.elapsed_time: 7.4f} s, ||x*||_0 = {nnz:3d}, J(x*,p*) = {J_star:7.4f}, ||R(x*,p*)||_2 = {jnp.linalg.norm(residual):7.2g}")

fig, ax1 = plt.subplots(figsize=(12, 7))
ax1.plot([2*k for k in k_range], cpu_time, color=colors[0], linewidth=4)
ax1.set_xlabel(r'$N$')
ax1.set_ylabel(r'CPU time (s)', color=colors[0])
plt.grid()
plt.show()
plt.savefig("example_sparse_time.pdf", bbox_inches='tight')
