import numpy as np
import jax
from nashopt import GNEP_LQ, GNEP
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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

max_constraints = 300
cpu_time_milp_highs = []
cpu_time_milp_gurobi = []
cpu_time_admm = []
cpu_time_lm = []

admm_iters = []

N=5 # number of agents
m_range = range(20, max_constraints+20, 20)
for m in m_range:
    print(f"\n\n\033[1;35mNumber of constraints: {m}\033[0m")
    sizes = [2]*N  # sizes of each agent
    ncon = m  # number of inequality constraints
    npar = 0   # number of parameters
    A = np.round(np.random.randn(ncon, sum(sizes))*10.)/10.
    b = np.ones(ncon)

    nvar = sum(sizes)  # number of variables

    Q = []
    c = []
    for i in range(N):
        Qi = np.eye(nvar)
        Q.append(Qi)
        ci = i*np.ones(nvar)
        c.append(ci)

    def solve_gnep_lq(solver):
        gnep_lq = GNEP_LQ(sizes, Q, c, F=None, lb=lb, ub=ub, pmin=pmin,
                        pmax=pmax, A=A, b=b, S=None, M=1e4, 
                        variational=True if solver=='prox_admm' else False, 
                        solver=solver)
        if solver == 'prox_admm':
            solver_options={'maxiter': 50000}
        else: # solver=='highs' or solver=='gurobi':
            solver_options={'time_limit': 600}
        sol = gnep_lq.solve(solver_options=solver_options)

        if isinstance(sol, list) or sol is None:
            print("No GNE found")
            cpu_time = np.nan
        else:
            cpu_time = sol.elapsed_time
            if solver == 'prox_admm' and not isinstance(sol, list):
                admm_iters.append(sol.num_iters)
        
        return cpu_time

    cpu_time_milp_highs.append(solve_gnep_lq('highs'))
    cpu_time_milp_gurobi.append(solve_gnep_lq('gurobi'))
    cpu_time_admm.append(solve_gnep_lq('prox_admm'))

    # Recompute variational GNE using Levenberg-Marquardt
    f = []
    for i in range(len(sizes)):
        f.append(
            jax.jit(partial(lambda x, Qi, ci: 0.5*x@Qi@x + ci@x, Qi=Q[i], ci=c[i])))

    @jax.jit
    def g(x):
        return A@x-b
    gnep = GNEP(sizes, f, g, ncon, variational=True)
    sol = gnep.solve(solver='lm', max_nfev=2000)
    x_star_vgne, lam_star_vgne, residual_vgne, stats_vgne = sol.x, sol.lam, sol.res, sol.stats

    cpu_time_lm.append(stats_vgne.elapsed_time)

print("Iterations required by Proximal ADMM: min =", min(admm_iters), ", max =", max(admm_iters))

plt.rcParams.update({'font.size': 12})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax1 = plt.subplots(figsize=(7, 5))
ax1.semilogy(m_range, cpu_time_milp_highs,
             color=colors[0], linewidth=4, label='MILP - HiGHS')
ax1.semilogy(m_range, cpu_time_milp_gurobi,
             color=colors[1], linewidth=4, label='MILP - Gurobi')
ax1.semilogy(m_range, cpu_time_admm,
             color=colors[2], linewidth=4, label='ADMM')
ax1.semilogy(m_range, cpu_time_lm,
             color=colors[3], linewidth=4, label='LM')
ax1.set_xlabel(r'number of constraints')
ax1.set_ylabel(r'CPU time (s)')
ax1.legend(loc='lower right')
plt.grid()
vals = list(m_range)
ticks = vals[::2]
if vals[-1] not in ticks:
    ticks.append(vals[-1])
ax1.set_xticks(ticks)
plt.show()
plt.savefig("example_cputime_comparison_constraints.pdf", bbox_inches='tight')
