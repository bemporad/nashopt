import numpy as np
import jax
import jax.numpy as jnp
import time
from nashopt import GNEP_LQ, GNEP

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)

t0 = time.time()
sizes = [2, 1, 1]      # [n1, n2, n3]

variational = True

Q1 = np.diag(np.array([1.0, 1.0, 0.0, 0.0]))
c1 = np.array(np.array([-1.0, 0.5, 0.0, 0.0]))
F1 = np.zeros((4, 1))

Q2 = np.diag(np.array([0.0, 0.0, 1.0, 0.0]))
c2 = np.array(np.array([0.0, 0.0, -0.3, 0.0]))
F2 = np.zeros((4, 1))

Q3 = np.array([[0.25, 0.0, 0.5, -0.5],
               [0.0, 0.0, 0.0, 0.0],
               [0.5, 0.0, 0.25, -0.5],
               [-0.5, 0.0, -0.5, 1.0]])
c3 = np.array(np.array([0.0, 0.0, 0.0, 0.0]))
F3 = np.zeros((4, 1))

lb = np.zeros(4)
ub = np.ones(4)

Q = [Q1, Q2, Q3]
c = [c1, c2, c3]
F = [F1, F2, F3]

N = len(sizes)  # number of agents
nvar = sum(sizes)  # number of variables
npar = 1   # number of parameters
# pmin=-10.*np.ones(npar) # lower bounds on parameters
# pmax=10.*np.ones(npar) # upper bounds on parameters
pmin = np.zeros(npar)  # fix p=0
pmax = np.zeros(npar)   # fix p=0

def print_residuals(x,p,lam):
    print("Check residuals:")
    print(np.linalg.norm((Q[0] @ x + c[0] + F[0] @ p + A.T@lam[0][:ncon])[0:sizes[0]] +
        np.array([-lam[0][ncon]+lam[0][ncon+2], -lam[0][ncon+1]+lam[0][ncon+3]])))
    print(np.linalg.norm((Q[1] @ x + c[1] + F[1] @ p + A.T@lam[1][:ncon])[
        sizes[0]:sizes[0]+sizes[1]] + np.array([-lam[1][ncon]+lam[1][ncon+1]])))
    if N > 2:
        print(np.linalg.norm((Q[2] @ x + c[2] + F[2] @ p + A.T@lam[2][:ncon])[sizes[0]+sizes[1]:]
                            + np.array([-lam[2][ncon]+lam[2][ncon+1]])))
for test in range(0,2):
    if test == 0:
        ncon = 1
        A = np.array([[1.0, 0.0, 1.0, 1.0]])
        b = np.array([2.0])
        S = np.array([[0.0]])
    else:
        ncon = 3
        A = np.random.randn(ncon, 4)
        b = np.random.rand(ncon)
        S = np.random.randn(ncon, npar)

    for solver in ["highs","prox_admm"]:
        gnep = GNEP_LQ(sizes, Q, c, F, lb=lb, ub=ub, pmin=pmin,
                        pmax=pmax, A=A, b=b, S=S, M=1e4, variational=variational, solver=solver)
        sol = gnep.solve()
        print(f"{solver} status: {sol.status_str}, CPU time: {sol.elapsed_time:.4f} s")
        x = sol.x
        p = sol.p
        lam = sol.lam
        delta = sol.delta

        print("x:", x)
        print("p:", p)
        for i in range(N):
            print(f"lambda_{i}:", lam[i])
        print("delta:", delta)

        if sol.status_str in ["Optimal", "Converged"]:
            print_residuals(x,p,lam)

    f = []
    for i in range(N):
        f.append(jax.jit(lambda x, i=i: 0.5*x.T@Q[i]@x + c[i].T@x))

    @jax.jit
    def g(x):
        return A@x-b

    # Aeq = np.array([[1,1,1,1]])
    # beq = np.array([2.0])
    Aeq = None
    beq = None

    gnep_kkt = GNEP(sizes, f=f, g=g, ng=A.shape[0], lb=lb,
                ub=ub, Aeq=Aeq, beq=beq, variational=variational)

    x0 = jnp.zeros(nvar)
    sol_kkt = gnep_kkt.solve(x0)
    x_star, lam_star, residual, stats = sol_kkt.x, sol_kkt.lam, sol_kkt.res, sol_kkt.stats

    print("=== GNE solution ===")
    print(f"x = {x_star}")
    for i in range(gnep_kkt.N):
        print(f"lambda[{i}] = {lam_star[i]}")
    print_residuals(x_star,p,lam_star)

    print(f"KKT residual norm = {float(jnp.linalg.norm(residual)): 10.7g}")
    print(f"KKT evaluations   = {int(stats.kkt_evals): 3d}")
    print(f"Elapsed time:       {stats.elapsed_time: .4f} seconds")
