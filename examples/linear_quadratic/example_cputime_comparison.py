import numpy as np
import jax
from nashopt import GNEP_LQ, GNEP
from functools import partial
import matplotlib.pyplot as plt

np.random.seed(2)
np.set_printoptions(precision=4, suppress=True)

Aeq = None
beq = None
Seq = None
lb = None  # no lower bounds on variables
ub = None  # no upper bounds on variables
pmin = None
pmax = None
max_solutions = None

max_size = 20
cpu_time_milp_highs = []
cpu_time_milp_gurobi = []
cpu_time_lm = []

for N in range(1, max_size+1):
    print(f"\n\n\033[1;35mNumber of agents: {N}\033[0m")
    sizes = [2]*N  # sizes of each agent
    ncon = 2*N  # number of inequality constraints
    npar = 0   # number of parameters
    A = np.round(np.random.randn(ncon, sum(sizes))*10.)/10.
    b = np.ones(ncon)

    N = len(sizes)  # number of agents
    nvar = sum(sizes)  # number of variables

    Q = []
    c = []
    F = []
    for i in range(N):
        Qi = np.eye(nvar)
        Q.append(Qi)
        ci = i*np.ones(nvar)
        c.append(ci)
        F.append(np.random.randn(nvar, npar))

    def solve_gnep_lq_milp(solver):
        gnep_lq = GNEP_LQ(sizes, Q, c, F=None, lb=lb, ub=ub, pmin=pmin,
                        pmax=pmax, A=A, b=b, S=None, M=1e4, variational=False, solver=solver)
        sol = gnep_lq.solve()

        if isinstance(sol, list):
            print("No GNE found")
            cpu_time = 0.
        else:
            cpu_time = sol.elapsed_time
        return cpu_time

    cpu_time_milp_highs.append(solve_gnep_lq_milp('highs'))
    cpu_time_milp_gurobi.append(solve_gnep_lq_milp('gurobi'))

    # Recompute variational GNE using Levenberg-Marquardt
    f = []
    for i in range(len(sizes)):
        f.append(
            jax.jit(partial(lambda x, Qi, ci: 0.5*x@Qi@x + ci@x, Qi=Q[i], ci=c[i])))

    @jax.jit
    def g(x):
        return A@x-b
    gnep = GNEP(sizes, f, g, ncon, variational=True)
    sol = gnep.solve(solver='lm')
    x_star_vgne, lam_star_vgne, residual_vgne, stats_vgne = sol.x, sol.lam, sol.res, sol.stats

    cpu_time_lm.append(stats_vgne.elapsed_time)

plt.rcParams.update({'font.size': 18})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.semilogy(range(1, max_size+1), cpu_time_milp_highs,
             color=colors[0], linewidth=4, label='MILP - HiGHS')
ax1.semilogy(range(1, max_size+1), cpu_time_milp_gurobi,
             color=colors[1], linewidth=4, label='MILP - Gurobi')
ax1.semilogy(range(1, max_size+1), cpu_time_lm,
             color=colors[2], linewidth=4, label='LM')
ax1.set_xlabel(r'number $N$ of agents')
ax1.set_ylabel(r'CPU time (s)')
ax1.legend()
plt.grid()
plt.show()
plt.savefig("example_milp_vs_lm.pdf", bbox_inches='tight')
