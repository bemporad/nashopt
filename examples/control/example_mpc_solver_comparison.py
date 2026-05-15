"""
Game-theoretic linear MPC example comparing different solvers, as reported in:

A. Bemporad, "GoldNash: A Goldfarb-Idnani Variant for Strongly Monotone Linear-Quadratic Games,"
arXiv preprint, 2026.


(C) 2026 A. Bemporad
"""
import numpy as np
from scipy.linalg import block_diag
from nashopt import NashLinearMPC
import matplotlib.pyplot as plt
import matplotlib as mpl

np.random.seed(3)

np.set_printoptions(precision=4, suppress=True)

sizes = [2,2,2]
N = len(sizes)  # number of agents
nu = sum(sizes)  # number of inputs
nxi = 3
nx = nxi * N  # number of states
ny = sum(sizes)  # number of outputs

Tc_ratio = 1.0  # Tc = T (full constraint horizon)
umin = -3.*np.ones(nu)
umax = 3.*np.ones(nu)
dumin = -2.*np.ones(nu)
dumax = 2.*np.ones(nu)
ymin = 0.*np.ones(ny)
ymax = 2.*np.ones(ny)

pert = 0.02  # perturbation level for coupling between agents

# Random stable dynamics
A = block_diag(*[np.random.randn(nxi, nxi) for _ in range(N)])  # random A, possibly unstable
A += np.random.randn(nx, nx)*pert  # add some coupling between agents
# scale to have spectral radius = .99
A = A / max(abs(np.linalg.eigvals(A)))*0.95

B = block_diag(*[np.random.randn(nxi, sizes[i]) for i in range(N)])  # random B, block-diagonal
B += np.random.randn(nx, nu)*pert  # add some coupling between agents

C = block_diag(*[np.random.randn(sizes[i], nxi) for i in range(N)])  # random C, block-diagonal
C += np.random.randn(ny, nx)*pert  # add some coupling between agents
if nu == ny:
    DC = C@np.linalg.inv(np.eye(nx)-A)@B
    C = np.linalg.inv(DC)@C  # scale C to have DC gain = I

Qy = []
Qdu = []
Qeps = 1.e3
Qeps2 = 1.e-3
for i in range(N):
    Qyi = 1.*np.eye(ny)
    Qyi[np.ix_(range(i*2, (i+1)*2), range(i*2, (i+1)*2))] = 1.5*np.eye(2)  # higher weight on own outputs
    Qy.append(Qyi)
    Qdui = 0.1*np.eye(sizes[i])
    Qdu.append(Qdui)

Tsim = 50  # number of closed-loop simulation steps
x0 = np.zeros(nx)
ref = np.array([1,2,1,2,1,2]) #np.arange(1, ny+1)  # reference trajectory

variational = True
centralized = False

T_values = [10, 15, 20, 25, 30]
solvers = ['lemke', 'lemke_dual', 'dr_daqp', 'goldnash'] # 'prox_admm', 'log_ipm'

times_results = {solver: [] for solver in solvers}
Y_traj = None
U_traj = None

for T in T_values:
    Tc = T
    print(f"\n{'='*60}")
    print(f"  T = {T}")
    print(f"{'='*60}")

    nash_mpc = NashLinearMPC(sizes, A, B, C, Qy, Qdu, T, ymin=ymin, ymax=ymax,
                                umin=umin, umax=umax, dumin=dumin, dumax=dumax,
                                Qeps=Qeps, Qeps2=Qeps2, Tc=Tc, check_monotone=True)
    if not nash_mpc.is_monotone[0]:
        print(f"\033[1;31mLQ-GNE is not monotone for T = {T}!\033[0m")

    for solver in solvers:
        T_tot = []
        x = x0.copy()
        u1 = np.zeros(nu)
        collect = (T == T_values[-1] and solver == solvers[-1])
        if collect:
            Y_list, U_list = [], []

        for k in range(Tsim):
            if collect:
                Y_list.append(C @ x)
                
            # Warm-up solvers at first run, without collecting times
            if k == 0 and T == T_values[0]:
                print(f"  {solver: <12}: warm-up... ", end='', flush=True)
                sol = nash_mpc.solve(
                    x, u1, ref, variational=variational, centralized=centralized,
                    solver=solver, solver_options={}
                )
                G=sol.gnep.pseudogradient_matrix()
                print(f"  {solver: <12}: ||G-G^T||_F = {np.linalg.norm(G-G.T):.4e}")
                print("done.")
            
            sol = nash_mpc.solve(
                x, u1, ref, variational=variational, centralized=centralized,
                solver=solver, solver_options={}
            )
            uk = sol.u
            if collect:
                U_list.append(uk)
            u1 = uk
            x = A@x + B@uk
            T_tot.append(sol.elapsed_time)
            print('.', end='', flush=True)

        if collect:
            Y_traj = np.array(Y_list)
            U_traj = np.array(U_list)
        times_results[solver].append(np.array(T_tot))
        print(f"  {solver: <12}: min={np.min(T_tot)*1e3:5.2f} ms  "
                f"med={np.median(T_tot)*1e3:5.2f} ms  "
                f"max={np.max(T_tot)*1e3:5.2f} ms")

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
n_solvers = len(solvers)
group_width = 0.7
bar_width = group_width / n_solvers

fig, ax = plt.subplots(figsize=(11, 6))

for idx, solver in enumerate(solvers):
    color = colors[idx % len(colors)]
    medians = np.array([np.median(times_results[solver][i]) for i in range(len(T_values))]) * 1e3
    mins    = np.array([np.min(   times_results[solver][i]) for i in range(len(T_values))]) * 1e3
    maxs    = np.array([np.max(   times_results[solver][i]) for i in range(len(T_values))]) * 1e3

    x_pos = np.arange(len(T_values)) + (idx - (n_solvers - 1) / 2) * bar_width
    ax.bar(x_pos, maxs - mins, bottom=mins, width=bar_width * 0.9,
           color=color, alpha=0.5, label=solver)
    ax.hlines(medians, x_pos - bar_width * 0.4, x_pos + bar_width * 0.4,
              colors=color, linewidth=2.5)

ax.set_xticks(np.arange(len(T_values)))
ax.set_xticklabels(T_values)
ax.set_xlabel('prediction horizon $T$', fontsize=22)
ax.set_ylabel('CPU time (ms)', fontsize=22)
ax.set_yscale('log')
ax.legend(title=None, loc='upper left', fontsize=20, labelspacing=0.2)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')
fig.tight_layout()
plt.show()


cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
agent_colors = [cycle_colors[i % len(cycle_colors)] for i in range(N)]
# Map each output/input index to its agent index
signal_agent = []
for ag, sz in enumerate(sizes):
    signal_agent.extend([ag] * sz)

fig, ax = plt.subplots(2, 1, figsize=(8, 8))
time = range(Tsim)
for i in range(ny):
    c = agent_colors[signal_agent[i]]
    ax[0].plot(time, Y_traj[:, i], color=c,
                linewidth=3, label=f'$y_{i+1}$')
    ax[0].plot(time, ref[i]*np.ones(Tsim), '--',
                color=c, linewidth=2)
y_by_agent = [[] for _ in range(N)]
for i in range(ny):
    y_by_agent[signal_agent[i]].append(i + 1)
u_by_agent = [[] for _ in range(N)]
for i in range(nu):
    u_by_agent[signal_agent[i]].append(i + 1)

y_handles = [mpl.lines.Line2D([0], [0], color=agent_colors[i], linewidth=3,
                               label='$' + ','.join(f'y_{{{j}}}' for j in y_by_agent[i]) + '$')
             for i in range(N)]
u_handles = [mpl.lines.Line2D([0], [0], color=agent_colors[i], linewidth=3,
                               label='$' + ','.join(f'u_{{{j}}}' for j in u_by_agent[i]) + '$')
             for i in range(N)]

ax[0].legend(handles=y_handles, loc='lower right', fontsize=18, labelspacing=0.2)
ax[0].grid()

for i in range(nu):
    ax[1].step(time, U_traj[:, i], linewidth=3,
                color=agent_colors[signal_agent[i]])
ax[1].legend(handles=u_handles, loc='lower right', fontsize=18, labelspacing=0.2)
ax[1].grid()
the_title = 'closed-loop MPC trajectories'
the_title += f" ($T={T_values[-1]}$)"
ax[0].set_title(the_title, fontsize =22)
ax[1].set_xlabel('time step $t$', fontsize=22)
plt.show()
