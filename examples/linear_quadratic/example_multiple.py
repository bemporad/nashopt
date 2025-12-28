import numpy as np
import jax
from nashopt import GNEP_LQ, GNEP
from functools import partial

np.random.seed(2)
np.set_printoptions(precision=4, suppress=True)

sizes = [2, 2, 2]  # sizes of each agent
ncon = 4  # number of inequality constraints
npar = 1   # number of parameters
A = np.round(np.random.randn(ncon, sum(sizes))*10.)/10.
b = np.ones(ncon)
S = 0.*np.random.randn(ncon, npar)
max_solutions = 2**ncon  # search for multiple solutions
Aeq = None
beq = None
Seq = None
lb = None  # no lower bounds on variables
ub = None  # no upper bounds on variables
pmin = None
pmax = None

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

if npar > 0:
    pmin = 0.*np.ones(npar)  # lower bounds on parameters
    pmax = 0.*np.ones(npar)  # upper bounds on parameters

for variational in [False, True]:
    print("\n\n\033[1;31mVariational GNEP:\033[0m", variational)
    gnep_lq = GNEP_LQ(sizes, Q, c, F, lb, ub, pmin, pmax, A,
                      b, S, Aeq, beq, Seq, M=1e4, variational=variational)
    sol = gnep_lq.solve(max_solutions=max_solutions, verbose=1)

    if not isinstance(sol, list):
        sol = [sol]
    for i in range(len(sol)):
        print("Solution ", i+1)
        x = sol[i].x
        delta = sol[i].delta
        p = sol[i].p

        print("HiGHS status:", sol[i].status_str)
        x = sol[i].x
        p = sol[i].p
        lam = sol[i].lam
        delta = sol[i].delta
        mu = sol[i].mu
        eps = sol[i].eps
        G = sol[i].G
        Geq = sol[i].Geq
        elapsed_time = sol[i].elapsed_time

        print("\033[1;34mx\033[0m:", x)
        print("\033[1;34mp\033[0m:", p)
        # print("y:", y)
        # print("lam:", lam)
        print("\033[1;34mdelta:\033[0m", delta)
        print("\033[1;34melapsed time:\033[0m", elapsed_time, "s")

        print("Check residuals:")

        ncon = A.shape[0] if A is not None else 0
        if npar == 0:
            p = np.array([])
        if A is not None and Aeq is None:
            print(
                f"max constraint violation: {np.maximum(np.max(A @ x - S @ p - b),0.)}")
        elif A is None and Aeq is not None:
            print(
                f"eq. constraint violation: {np.linalg.norm(Aeq @ x - Seq @ p - beq)}")
        elif A is not None and Aeq is not None:
            print(
                f"max constraint violation: {np.maximum(np.max(A @ x - S @ p - b),0.)}")
            print(
                f"eq. constraint violation: {np.linalg.norm(Aeq @ x - Seq @ p - beq)}")

    cpu_time = [sol[i].elapsed_time for i in range(len(sol))]
    print(
        f"\033[1;32mCPU time: {min(cpu_time):.4f} <= CPU time <= {max(cpu_time):.4f} s\033[0m")

# Recompute variational GNE using Levenberg-Marquardt
f = []
for i in range(len(sizes)):
    f.append(
        jax.jit(partial(lambda x, Qi, ci: 0.5*x@Qi@x + ci@x, Qi=Q[i], ci=c[i])))


@jax.jit
def g(x):
    return A@x-b


gnep = GNEP(sizes, f, g, ncon, variational=True)
sol = gnep.solve()
x_star_vgne, lam_star_vgne, residual_vgne, stats_vgne = sol.x, sol.lam, sol.res, sol.stats
print(f"\033[1;32mCPU time (LM): {stats_vgne.elapsed_time:.4f} s\033[0m")
print(f"\033[1;34mLM iters:\033[0m {stats_vgne.kkt_evals}")
