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

max_constraints = 300
cpu_time_milp_highs = []
cpu_time_milp_gurobi = []
cpu_time_qp = []
cpu_time_extragrad = []
cpu_time_op_extrap = []
cpu_time_admm = []
cpu_time_lm = []
cpu_time_lemke = []
cpu_time_log_ipm = []
cpu_time_dr_daqp = []

qp_iters = []
extragrad_iters = []
op_extrap_iters = []
admm_iters = []
lemke_iters = []
log_ipm_iters = []
dr_daqp_iters = []

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
        Qi = np.eye(nvar) # Note that Qi = I, then the game is potential
        Q.append(Qi)
        ci = i*np.ones(nvar)
        c.append(ci)

    def solve_gnep_lq(solver):
        gnep_lq = GNEP_LQ(sizes, Q, c, F=None, lb=lb, ub=ub, pmin=pmin,
                        pmax=pmax, A=A, b=b, S=None, M=1e4,
                        variational=True if solver in ['qp_gnep','extragradient','op_extrap','prox_admm','lemke','log_ipm','dr_daqp'] else False,
                        solver=solver)
        if solver in ('highs', 'gurobi'):
            solver_options = {'time_limit': 600}
        elif solver == 'prox_admm':
            solver_options = {'maxiter': 50000}
        elif solver == 'extragradient':
            solver_options = {'maxiter': 2000}
        else:
            solver_options = None
        sol = gnep_lq.solve(solver_options=solver_options)

        if isinstance(sol, list) or sol is None:
            print("No GNE found")
            cpu_time = np.nan
        else:
            cpu_time = sol.elapsed_time
            if solver == 'qp_gnep' and not isinstance(sol, list):
                qp_iters.append(sol.num_iters)
            elif solver == 'extragradient' and not isinstance(sol, list):
                extragrad_iters.append(sol.num_iters)
            elif solver == 'op_extrap' and not isinstance(sol, list):
                op_extrap_iters.append(sol.num_iters)
            elif solver == 'prox_admm' and not isinstance(sol, list):
                admm_iters.append(sol.num_iters)
            elif solver == 'lemke' and not isinstance(sol, list):
                lemke_iters.append(sol.num_iters)
            elif solver == 'log_ipm' and not isinstance(sol, list):
                log_ipm_iters.append(sol.num_iters)
            elif solver == 'dr_daqp' and not isinstance(sol, list):
                dr_daqp_iters.append(sol.num_iters)

        return cpu_time

    cpu_time_milp_highs.append(solve_gnep_lq('highs'))
    cpu_time_milp_gurobi.append(solve_gnep_lq('gurobi'))
    cpu_time_qp.append(solve_gnep_lq('qp_gnep'))
    cpu_time_extragrad.append(solve_gnep_lq('extragradient'))
    cpu_time_op_extrap.append(solve_gnep_lq('op_extrap'))
    cpu_time_admm.append(solve_gnep_lq('prox_admm'))
    cpu_time_lemke.append(solve_gnep_lq('lemke'))
    cpu_time_log_ipm.append(solve_gnep_lq('log_ipm'))
    cpu_time_dr_daqp.append(solve_gnep_lq('dr_daqp'))

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

print("Iterations required by QP-GNEP: min =", min(qp_iters), ", max =", max(qp_iters))
print("Iterations required by Extragradient: min =", min(extragrad_iters), ", max =", max(extragrad_iters))
print("Iterations required by Operator Extrapolation: min =", min(op_extrap_iters), ", max =", max(op_extrap_iters))
print("Iterations required by Proximal ADMM: min =", min(admm_iters), ", max =", max(admm_iters))
print("Iterations required by Lemke's algorithm: min =", min(lemke_iters), ", max =", max(lemke_iters))
print("Iterations required by Logarithmic IPM: min =", min(log_ipm_iters), ", max =", max(log_ipm_iters))
print("Iterations required by DR-DAQP: min =", min(dr_daqp_iters), ", max =", max(dr_daqp_iters))

plt.rcParams.update({'font.size': 12})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax1 = plt.subplots(figsize=(7, 5))
ax1.semilogy(m_range, cpu_time_milp_highs,
             color=colors[0], linewidth=4, label='MILP - HiGHS')
ax1.semilogy(m_range, cpu_time_milp_gurobi,
             color=colors[1], linewidth=4, label='MILP - Gurobi')
ax1.semilogy(m_range, cpu_time_qp,
             color=colors[2], linewidth=4, label='QP-GNEP')
ax1.semilogy(m_range, cpu_time_extragrad,
             color=colors[7], linewidth=4, label='Extragradient')
ax1.semilogy(m_range, cpu_time_op_extrap,
             color=colors[8], linewidth=4, label='Op-Extrapolation')
ax1.semilogy(m_range, cpu_time_admm,
             color=colors[3], linewidth=4, label='Prox-ADMM')
ax1.semilogy(m_range, cpu_time_lm,
             color=colors[4], linewidth=4, label='LM')
ax1.semilogy(m_range, cpu_time_lemke,
             color=colors[5], linewidth=4, label='Lemke')
ax1.semilogy(m_range, cpu_time_log_ipm,
             color=colors[6], linewidth=4, label='Log-IPM')
ax1.semilogy(m_range, cpu_time_dr_daqp,
             color=colors[9], linewidth=4, label='DR-DAQP')
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
#plt.savefig("example_cputime_comparison_constraints.pdf", bbox_inches='tight')
