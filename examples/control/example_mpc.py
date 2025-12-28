import numpy as np
from nashopt import NashLinearMPC
from functools import partial
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 20,
    "font.size": 20,
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

np.random.seed(1)

np.set_printoptions(precision=4, suppress=True)

nx = 6  # number of states
nu = 3  # number of inputs
ny = 3  # number of outputs
sizes = [1]*nu  # each agent controls one input
N = nu  # number of agents
T = 10  # prediction horizon
Tc = 3  # constraints horizon
umin, umax, dumin, dumax, ymin, ymax = None, None, None, None, None, None
umin = 0.*np.ones(nu)
umax = 4.*np.ones(nu)
# dumin = -.2*np.ones(nu)
# dumax = .2*np.ones(nu)
# ymin = -2.*np.ones(ny)
# ymax = np.array([1., np.inf, np.inf])

# Random stable dynamics
A = np.random.rand(nx, nx)  # random A, possibly unstable
# scale to have spectral radius = .99
A = A / max(abs(np.linalg.eigvals(A)))*0.95
B = np.random.randn(nx, nu)
C = np.random.randn(ny, nx)
if ny == ny:
    DC = C@np.linalg.inv(np.eye(nx)-A)@B
    C = np.linalg.inv(DC)@C  # scale C to have DC gain = I

Qy = []
Qdu = []
Qeps = 1.e3
for i in range(N):
    Qyi = np.zeros((ny, ny))
    Qyi[i, i] = 1.
    Qy.append(Qyi)
    Qdui = .5
    Qdu.append(Qdui)

nash_mpc = NashLinearMPC(sizes, A, B, C, Qy, Qdu, T, ymin=ymin, ymax=ymax,
                         umin=umin, umax=umax, dumin=dumin, dumax=dumax, Qeps=Qeps, Tc=Tc)

Tsim = 40  # number of closed-loop simulation steps
x0 = np.zeros(nx)
u1 = np.zeros(nu)
ref = np.array([1., 2., 3.])

for case in range(2):
    if case == 0:
        variational = False
        centralized = False
        print("Nash-MPC")
    elif case == 1:
        variational = False
        centralized = True
        print("Centralized MPC")
    cpu_build_time = []
    X = [x0]
    Y = []
    U = []
    T_tot = []
    T_solver = []
    x = x0.copy()
    for k in range(Tsim):
        X.append(x)
        Y.append(C@x)
        sol = nash_mpc.solve(
            x, u1, ref, variational=variational, centralized=centralized)
        uk = sol.u
        u1 = uk
        U.append(uk)
        x = A@x + B@uk
        T_tot.append(sol.elapsed_time)
        T_solver.append(sol.elapsed_time_solver)

    print(
        f"Total time per step ({'nash' if not centralized else 'centralized'}): {np.min(T_tot)*1.e3:.2f} <= t <= {np.max(T_tot)*1.e3:.2f} ms")
    # print(f"Solver time per step: {np.min(T_solver)*1.e3:.2f} <= t <= {np.max(T_solver)*1.e3:.2f} ms")
    X = np.array(X)
    Y = np.array(Y)
    U = np.array(U)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    time = range(Tsim)
    for i in range(ny):
        ax[0].plot(time, Y[:, i], color=colors[i],
                   linewidth=4, label=f'$y_{i+1}$')
        ax[0].plot(time, ref[i]*np.ones(Tsim), '--',
                   color=colors[i], linewidth=2)
    ax[0].legend(loc='lower right')
    ax[0].grid()

    for i in range(nu):
        ax[1].step(time, U[:, i], linewidth=2,
                   color=colors[i], label=f'$u_{i+1}$')
    ax[1].legend(loc='lower right')
    ax[1].grid()
    ax[0].set_title(
        f"{'game-theoretic MPC' if not centralized else 'centralized MPC'}")
    ax[1].set_xlabel('time step $t$')
    plt.show()
    plt.savefig(
        f"/Users/bemporad/Alberto/Lavori/Optimization/Nash_Game_Theory/NashOpt/latex/figures/example_linear_MPC_{'nash' if not centralized else 'centralized'}.pdf", bbox_inches='tight')
