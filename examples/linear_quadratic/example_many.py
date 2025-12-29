import numpy as np
from nashopt import GNEP_LQ

np.random.seed(2)
np.set_printoptions(precision=3, suppress=True)

for ex in range(1, 11+1):
    print(f"\n\n\033[31m\033[43mExample {ex}:\033[0m")

    # default: no constraints, no parameters
    Aeq = None
    beq = None
    Seq = None
    A = None
    b = None
    S = None
    Aeq = None
    beq = None
    Seq = None
    lb = None  # no lower bounds on variables
    ub = None  # no upper bounds on variables
    pmin = None  # no parameters
    pmax = None  # no parameters
    xdes = None  # no desired equilibrium point
    eval_xdes = False
    D_pwa = None
    E_pwa = None
    h_pwa = None
    max_solutions = 1  # only one solution
    variational = False  # not variational GNEP

    match ex:
        case 1:
            # Unconstrained problem with 4 agents, each with 1 variable, no parameters
            sizes = [1]*4  # sizes of each agent * number of agents
            npar = 0   # number of parameters
        case 2:
            # smaller problem
            sizes = [3, 4, 2]  # sizes of each agent
            ncon = 5  # number of constraints
            npar = 2   # number of parameters
        case 3:
            # no constraints, no parameters
            sizes = [2, 3, 4]  # sizes of each agent
            npar = 0   # number of parameters
        case 4:
            # with inequality constraints and parameters
            sizes = [5]*20  # sizes of each agent
            ncon = 5  # number of inequality constraints
            npar = 5   # number of parameters
            A = np.random.randn(ncon, sum(sizes))
            b = np.random.rand(ncon)
            S = np.random.randn(ncon, npar)
        case 5:
            # with equality constraints and parameters
            sizes = [3, 3]  # sizes of each agent
            nconeq = 1  # number of constraints
            npar = 2   # number of parameters
            Aeq = np.random.randn(nconeq, sum(sizes))
            beq = np.random.rand(nconeq)
            Seq = np.random.randn(nconeq, npar)
        case 6:
            # with equality and inequality constraints and parameters
            sizes = [3, 3]  # sizes of each agent
            nconeq = 1  # number of constraints
            npar = 2   # number of parameters
            Aeq = np.random.randn(nconeq, sum(sizes))
            beq = np.random.rand(nconeq)
            Seq = np.random.randn(nconeq, npar)
            ncon = 3  # number of inequality constraints
            A = np.random.randn(ncon, sum(sizes))
            b = np.random.rand(ncon)
            S = np.random.randn(ncon, npar)
        case 7:
            sizes = [1, 1]  # sizes of each agent
            npar = 2   # number of parameters
            ncon = 1  # number of inequality constraints
            A = np.random.randn(ncon, sum(sizes))
            b = np.random.rand(ncon)
            S = np.random.randn(ncon, npar)
            nvar = sum(sizes)
            lb = -100.*np.ones(nvar)  # lower bounds on variables
            ub = 100.*np.ones(nvar)   # upper bounds on variables

        case 8:
            sizes = [2, 2]  # sizes of each agent
            ncon = 3  # number of inequality constraints
            npar = 2   # number of parameters
            A = np.random.randn(ncon, sum(sizes))
            b = np.random.rand(ncon)
            S = np.random.randn(ncon, npar)
            xdes = np.zeros(sum(sizes))  # desired equilibrium point

        case 9:
            sizes = [2, 2]  # sizes of each agent
            ncon = 3  # number of inequality constraints
            npar = 2   # number of parameters
            A = np.random.randn(ncon, sum(sizes))
            b = np.random.rand(ncon)
            S = np.random.randn(ncon, npar)
            xdes = np.zeros(sum(sizes))  # desired equilibrium point
            variational = True  # variational GNEP
        case 10:
            sizes = [2, 2]  # sizes of each agent
            ncon = 3  # number of inequality constraints
            npar = 2   # number of parameters
            A = np.random.randn(ncon, sum(sizes))
            b = np.random.rand(ncon)
            S = np.random.randn(ncon, npar)
            eval_xdes = True
        case 11:
            sizes = [2, 2]  # sizes of each agent
            ncon = 3  # number of inequality constraints
            npar = 2   # number of parameters
            A = np.random.randn(ncon, sum(sizes))
            b = np.random.rand(ncon)
            S = np.random.randn(ncon, npar)
            eval_xdes = True
            variational = True  # variational GNEP

    N = len(sizes)  # number of agents
    nvar = sum(sizes)  # number of variables

    Q = []
    c = []
    F = []
    for i in range(N):
        Qi = np.random.randn(nvar, nvar)
        Qi = Qi@Qi.T + 1.e-3*np.eye(nvar)
        Q.append(Qi)
        c.append(np.random.randn(nvar))
        F.append(np.random.randn(nvar, npar))

    if npar > 0:
        pmin = -100.*np.ones(npar)  # lower bounds on parameters
        pmax = 100.*np.ones(npar)  # upper bounds on parameters

    if eval_xdes:
        gnep_lq = GNEP_LQ(sizes, Q, c, F, lb, ub, pmin, pmax,
                          A, b, S, Aeq, beq, Seq, D_pwa=None, E_pwa=None, h_pwa=None, M=1e4)
        sol = gnep_lq.solve(max_solutions=1, verbose=0)
        xdes = sol.x  # set desired equilibrium point to the computed one
    if xdes is not None:
        # |x - xdes|_inf -> eps >= +/- (x - xdes)
        D_pwa = np.vstack((np.eye(nvar), -np.eye(nvar)))
        E_pwa = np.zeros((2*nvar, npar))
        h_pwa = np.hstack([-xdes, xdes])

    gnep_lq = GNEP_LQ(sizes, Q, c, F, lb, ub, pmin, pmax, A, b, S,
                      Aeq, beq, Seq, D_pwa=D_pwa, E_pwa=E_pwa, h_pwa=h_pwa, M=1e4, variational=variational)
    sol = gnep_lq.solve(max_solutions=max_solutions, verbose=1)

    if not isinstance(sol, list):
        print("HiGHS status:", sol.status_str)
        x = sol.x
        p = sol.p
        lam = sol.lam
        delta = sol.delta
        mu = sol.mu
        eps = sol.eps
        G = sol.G
        Geq = sol.Geq

        if xdes is not None:
            print(f"|x-x_des|_inf: {np.max(np.abs(x-xdes))}")
        else:
            print("x:", x)
        print("p:", p)
        # print("y:", y)
        # print("lam:", lam)
        print("delta:", delta)

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
    else:
        for i in range(len(sol)):
            print("Solution ", i+1)
            x = sol[i].x
            delta = sol[i].delta
            p = sol[i].p
            print("x:", x)
            print("p:", p)
            print("delta:", delta)
