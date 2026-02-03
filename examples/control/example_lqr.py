"""
(C) 2025 A. Bemporad
"""
import numpy as np
import jax.numpy as jnp
import time
from nashopt import NashLQR
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

t0 = time.time()

nx = 10 # number of states
nu = 1  # number of inputs per agent
N = 10  # number of agents
dare_iters = 50  # number of iterations for DARE solver

sizes = [nu]*N  # number of variables per agent
nvar = sum(sizes)

# Random unstable dynamics
A = np.random.rand(nx,nx) # random A, possibly unstable
A = A / max(abs(np.linalg.eigvals(A)))*1.1 # scale to have spectral radius = 1.1
B = np.random.randn(nx, nu*N)

Q=[]
R=[]
not_i = []
for i in range(N):
    # LQR weights of each agent
    #Qi = np.random.randn(nx, nx)
    Qi = np.zeros((nx,nx))
    Qi[i,i] = 1.0
    #Qi = Qi.T @ Qi
    #Ri = np.random.randn(nu, nu)
    Ri = np.zeros((nu, nu))
    Ri = Ri.T @ Ri + 10.*np.eye(nu)  # make it positive definite
    Q.append(Qi)
    R.append(Ri)   

nash_lqr = NashLQR(sizes, A, B, Q, R, dare_iters=dare_iters)
sol = nash_lqr.solve(method = 'residual', verbose=2)

K_Nash = sol.K_Nash
residual = sol.residual
stats = sol.stats
K_cen = sol.K_centralized

res = float(jnp.linalg.norm(residual))
np.set_printoptions(precision=4, suppress=True)

if res<1.e-6:
    print("=== GNE solution found ===")
    print(f"K_Nash = {K_Nash}")
else:
    print("=== GNE solution NOT found ===")
    
print(f"KKT residual norm:   {res: 10.7g}")
print(f"KKT evaluations:     {int(stats.kkt_evals): 3d}")
print(f"Elapsed time:        {stats.elapsed_time: .2f} seconds")

# Check stability of closed-loop system with Nash gains
rad_nash = np.abs(np.linalg.eigvals(A - B @ K_Nash)[0])
rad_cen = np.abs(np.linalg.eigvals(A - B @ K_cen)[0])
rad_ol = np.abs(np.linalg.eigvals(A)[0])
print("\n\033[1;34mSpectral radius:\033[0m")
print(f"\033[1;32mopen-loop\033[0m system:                         {rad_ol:.4f}")
print(f"closed-loop system with \033[1;31mNash gains\033[0m:       {rad_nash:.4f}")
print(f"closed-loop system with \033[1;35mcentralized LQR\033[0m:  {rad_cen:.4f}")


Tsim = 20 # number of closed-loop simulation steps
M = 10   # number of initial conditions
x0 = np.random.randn(nx,M)
C = np.ones(nx).reshape(1,-1)  # output matrix
ny = 1  # number of outputs

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
time = range(Tsim)
fig, ax = plt.subplots(2,1,figsize=(8,8))
fig.subplots_adjust(hspace=0.3)

for case in range(2):
    if case==0:
        centralized = False
        print("Nash-LQR")
    elif case==1:
        centralized = True
        print("Centralized LQR")
    for j in range(M):
        
        X=[x0[:,j]]
        Y=[]
        U=[]
        T_tot = []
        T_solver = []
        x = x0[:,j].copy()
        for k in range(Tsim):
            X.append(x)
            Y.append(C@x)
            uk = -K_Nash @ x if not centralized else -K_cen @ x
            U.append(uk)
            x = A@x + B@uk

        X = np.array(X)
        Y = np.array(Y)
        U = np.array(U)
            
        ax[case].plot(time, Y, color=colors[j], linewidth=4)
        ax[case].set_title(f"{'game-theoretic' if not centralized else 'centralized'} LQR")
ax[1].set_xlabel('time step $t$')
ax[0].grid()
ax[1].grid()
plt.show()
#plt.savefig("example_LQR.pdf", bbox_inches='tight')

K1 = nash_lqr.dare(agent=0)
print(f"First agent's gain from GNE-residual method: {K_Nash[0,:]}")
print(f"First agent's gain from JAX DARE solver:     {K1.reshape(-1)}")

# Now solve using the Riccati-based method
sol2 = nash_lqr.solve(method = 'riccati')
K_Nash2 = sol2.K_Nash
print(f"First agent's gain from Riccati-based method: {K_Nash2[0,:]}")
print(f"Elapsed time:        {sol2.stats.elapsed_time: .2f} seconds, {int(sol2.stats.riccati_iters)} Riccati iterations")
print(f"Difference between the two methods' gains: {np.linalg.norm(K_Nash - K_Nash2):.4g}")

