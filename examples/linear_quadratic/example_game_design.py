import numpy as np
from nashopt import GNEP_LQ

np.random.seed(2)
np.set_printoptions(precision=3, suppress=True)

sizes = [10]*10 # sizes of each agent
npar = 5   # number of parameters
ncon = 50  # number of inequality constraints
nconeq = 5  # number of equality constraints
lb = -10.*np.ones(sum(sizes))  # lower bounds on variables
ub = 10.*np.ones(sum(sizes))   # upper bounds on variables
max_solutions = 1  # only one solution
variational = False  # not variational GNEP

Aeq = np.random.randn(nconeq, sum(sizes))
beq = np.random.rand(nconeq)
Seq = np.random.randn(nconeq, npar)
A = np.random.randn(ncon, sum(sizes))
b = np.random.rand(ncon)
S = np.random.randn(ncon, npar)
pmin = -100.*np.ones(npar)  # lower bounds on parameters
pmax = 100.*np.ones(npar)  # upper bounds on parameters

eval_xdes = True  # compute a GNE and set it as desired GNE

N = len(sizes)  # number of agents
nvar = sum(sizes)  # number of variables

# Agents' cost functions
Q = []
c = []
F = []
for i in range(N):
    Qi = np.random.randn(nvar, nvar)
    Qi = Qi@Qi.T + 1.e-3*np.eye(nvar)
    Q.append(Qi)
    c.append(np.random.randn(nvar))
    F.append(np.random.randn(nvar, npar))

if eval_xdes:
    # Solve GNEP, no objective function for game design
    gnep_lq = GNEP_LQ(sizes, Q, c, F, lb, ub, pmin, pmax,
                      A, b, S, Aeq, beq, Seq, M=1e4)
    sol = gnep_lq.solve(max_solutions=1, verbose=0)
    xdes = sol.x  # set desired equilibrium point to the computed one

for norm in ['inf', '2']:
    if norm == 'inf':
        # |x - xdes|_inf -> eps >= +/- (x - xdes)
        D_pwa = np.vstack((np.eye(nvar), -np.eye(nvar)))
        E_pwa = np.zeros((2*nvar, npar))
        h_pwa = np.hstack([-xdes, xdes])
        Q_J = None
        c_J = None
    else:
        # min .5*|x - xdes|_2^2
        Q_J = np.block([[np.eye(nvar), np.zeros((nvar, npar))],
                        [np.zeros((npar, nvar)), np.zeros((npar, npar))]])
        c_J = np.hstack((-xdes, np.zeros(npar)))
        D_pwa = None
        E_pwa = None
        h_pwa = None

    gnep_lq = GNEP_LQ(sizes, Q, c, F, lb, ub, pmin, pmax, A, b, S,
                    Aeq, beq, Seq, D_pwa=D_pwa, E_pwa=E_pwa, h_pwa=h_pwa,
                    Q_J=Q_J, c_J=c_J, M=1e4, variational=variational, solver='gurobi')
    sol = gnep_lq.solve(max_solutions=max_solutions, verbose=1)

    x = sol.x
    p = sol.p
    lam = sol.lam
    delta = sol.delta
    mu = sol.mu
    eps = sol.eps

    if norm == 'inf':
        print(f"|x-x_des|_inf: {np.max(np.abs(x-xdes))}, CPU_time: {sol.elapsed_time: 5.4f} s")
    else:
        print(f"|x-x_des|_2: {np.linalg.norm(x - xdes)}, CPU_time: {sol.elapsed_time: 5.4f} s")

    print("x = ", x)
    print("p=", p)

