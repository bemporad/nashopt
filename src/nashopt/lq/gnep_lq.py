# NashOpt: A Python package for computing Generalized Nash Equilibria (GNE) in noncooperative games.
#
# Linear-Quadratic GNEPs.
#
# (C) 2025-2026 Alberto Bemporad

import numpy as np
import time
from types import SimpleNamespace
from .._common.optional_deps import get_gurobi, get_highspy
from .prox_admm_gne import solve as solve_prox_admm

class GNEP_LQ():
    def __init__(self, dim, Q, c, F=None, lb=None, ub=None, pmin=None, pmax=None, A=None, b=None, S=None, Aeq=None, beq=None, Seq=None, D_pwa=None, E_pwa=None, h_pwa=None, Q_J=None, c_J=None, M=1e4, variational=False, solver="highs"):
        """Given a (multiparametric) generalized Nash equilibrium problem with N agents,
        convex quadratic objectives, and linear constraints, solve the following game-design problem

            min_{x*,p} f(x*,p)

            s.t.      x* is a generalized Nash equilibrium of the parametric GNEP:    

                    min_{x_i} 0.5 x^T Qi x + (c_i+ F_i p)^T x
                    s.t.      A x <= b + S p
                                Aeq x = beq + Seq p
                                lb <= x <= ub

        where f is either the sum of convex piecewise affine (PWA) functions

                    f(x,p) = sum_{k=1..nk} max_{i=1..nk} { D_pwa[k](i,:) x + E_pwa[k](i,:) p + h_pwa[k](i) }
        
        or the convex quadratic function
        
                    f(x,p) = 0.5 [x p]^T Q_J [x;p] + c_J^T [x;p]

        or the sum of both. Here, x = [x_1; x_2; ...; x_N] is the stacked vector of all agents' variables,
        p is a vector of parameters (possibly empty), and Qi, c_i, F_i are the cost function data for agent i.

        Special cases of the general problem are:
        1) If p is empty and f(x,p)=0, we simply look for a generalized Nash equilibrium of the linear
        quadratic game;
        2) If p is not empty and f(x,p)=||x-xdes||_inf or f(x,p)=||x-xdes||_1, we solve the game design problem of finding the parameter vector p such that the resulting general Nash equilibrium x* is as close as possible to the desired equilibrium point xdes.

        If max_solutions > 1, multiple solutions are searched for (if they exist, up to max_solutions), each corresponding to a different combination of active constraints at the equilibrium.

        To search for a variational GNE with no parameter, or parameter p=pmin=pmax, set the flag variational=True. In this case, the KKT conditions require equal Lagrange multipliers for all agents for each shared constraint.

            Parameters
        ----------
        dim : list of int
            List with number of variables for each agent.
        Q : list of (nx, nx) np.ndarray
            Q matrices for each agent.
        c : list of (nx,) np.ndarray
            c vectors for each agent.
        F : list of (nx, np) np.ndarray or None
            F matrices for each agent.
        lb : (nx,) np.ndarray or None
            Range lower bounds on x (unbounded if None or -inf).
        ub : (nx,) np.ndarray or None
            Range upper bounds on x (unbounded if None or +inf).
        pmin : (npar,) np.ndarray or None
            Range lower bounds on p.
        pmax : (npar,) np.ndarray or None
            Range upper bounds on p.
        A : (nA, nx) np.ndarray or None
            Shared inequality constraint matrix.
        b : (nA,) np.ndarray or None
            Shared inequality constraint RHS vector.
        S : (nA, npar) np.ndarray or None
            Shared inequality constraint parameter matrix.
        Aeq : (nAeq, nx) np.ndarray or None
            Shared equality constraint matrix.
        beq : (nAeq,) np.ndarray or None
            Shared equality constraint RHS vector.
        Seq : (nAeq, npar) np.ndarray or None
            Shared equality constraint parameter matrix.        
        D_pwa : (list of) (nf,nx) np.ndarray(s) or None
            Matrix defining the convex PWA objective function for designing the game. If None, no objective function is used, and only an equilibrium point is searched for.
        E_pwa : (list of) (nf, npar) np.ndarray(s) or None
            Parameter matrix defining the convex PWA objective function for designing the game. 
        h_pwa : (list of) (nf,) np.ndarray(s) or None
            Vector defining the convex PWA objective function for designing the game. 
        Q_J : (nx+npar, nx+npar) np.ndarray or None
            Hessian matrix defining the convex quadratic objective function for designing the game. If None, no quadratic objective function is used.
        c_J : (nx+npar,) np.ndarray or None
            Linear term of the convex quadratic objective function for designing the game.
        M   : float
            Big-M constant for complementary slackness condition. This must be an upper bound
            on the Lagrange multipliers lam(i,j) and on the slack variables y(j).
        variational : bool
            If True, search for a variational GNE.
        solver : str
            Solver used to solve the GNE:
            - "highs" (default) mixed-integer programming solver
            - "gurobi" mixed-integer programming solver
            - "prox_admm" proximal ADMM algorithm (Borgens and Kanzow, 2021), only for variational non-parametric GNEPs

        (C) 2025-2026 Alberto Bemporad
        """

        nx = sum(dim)  # total number of variables
        N = len(dim)  # number of agents
        if not len(Q) == N:
            raise ValueError("Length of Q must be equal to number of agents")
        if not len(c) == N:
            raise ValueError("Length of c must be equal to number of agents")
        if F is not None and not len(F) == N:
            raise ValueError("Length of F must be equal to number of agents")

        for i in range(N):
            if not Q[i].shape == (nx, nx):
                raise ValueError(f"Q[{i}] must be of shape ({nx},{nx})")
            if not c[i].shape == (nx,):
                raise ValueError(f"c[{i}] must be of shape ({nx},)")

        has_pwa_objective = (D_pwa is not None)
        if has_pwa_objective:
            if E_pwa is None or h_pwa is None:
                raise ValueError("E_pwa and h_pwa must be provided if D_pwa is provided")
            if not isinstance(D_pwa, list):
                D_pwa = [D_pwa]
            if not isinstance(E_pwa, list):
                E_pwa = [E_pwa]
            if not isinstance(h_pwa, list):
                h_pwa = [h_pwa]
            if not (len(D_pwa) == len(E_pwa) == len(h_pwa)):
                raise ValueError("D, E, and h must be lists of the same length")

            nJ = len(D_pwa)
            for k in range(nJ):
                nk = D_pwa[k].shape[0]
                if not D_pwa[k].shape == (nk, nx):
                    raise ValueError(f"D[{k}] must be of shape ({nk},{nx})")
                if not h_pwa[k].shape == (nk,):
                    raise ValueError(f"h[{k}] must be of shape ({nk},)")

        if pmin is not None:
            pmin = np.asarray(pmin).reshape(-1)
        if pmax is not None:
            pmax = np.asarray(pmax).reshape(-1)
        has_params = (pmin is not None) and (pmax is not None) and (
            pmin.size > 0) and (pmax.size > 0)
        if has_params:
            is_single_p = np.all(pmin == pmax)
            if is_single_p:
                single_p = pmin  # fixed parameter value
        else:
            is_single_p = False
            single_p = None
                
        if has_params:
            npar = F[0].shape[1]
            for i in range(N):
                if not F[i].shape == (nx, npar):
                    raise ValueError(f"F[{i}] must be of shape ({nx},{npar})")
            if has_pwa_objective:
                for k in range(nJ):
                    nk = D_pwa[k].shape[0]
                    if not E_pwa[k].shape == (nk, npar):
                        raise ValueError(f"E_pwa[{k}] must be of shape ({nk},{npar})")
            if not pmin.size == npar:
                raise ValueError(f"pmin must have {npar} elements")
            if not pmax.size == npar:
                raise ValueError(f"pmax must have {npar} elements")
            if is_single_p:
                for i in range(N):
                    c[i] = c[i] + F[i] @ single_p  # absorb fixed p into c
                if has_pwa_objective:
                    for k in range(nJ):
                        h_pwa[k] = h_pwa[k] + E_pwa[k] @ single_p  # absorb fixed p into h
        else:
            npar = 0

        has_quad_objective = (Q_J is not None) or (c_J is not None)
        if has_quad_objective:
            if Q_J is None:
                raise ValueError("No quadratic term specified for game objective, use J(x,p) = c_J @ [x;p] = D_pwa@x + E_pwa@p for linear objectives")
            if solver == "highs":
                raise ValueError("HiGHS solver does not support quadratic objective functions, use solver='gurobi'")
            if c_J is None:
                    c_J = np.zeros((nx + npar))
            if not Q_J.shape == (nx + npar, nx + npar):
                raise ValueError(f"Q_J must be of shape ({nx+npar},{nx+npar})")
            if not c_J.shape == (nx + npar,):
                raise ValueError(f"c_J must be of shape ({nx+npar},)")

        has_ineq_constraints = (A is not None) and (A.size > 0)
        if has_ineq_constraints:
            if b is None:
                raise ValueError("b must be provided if A is provided")
            if has_params:
                if S is None:
                    S = np.zeros((A.shape[0], npar))
            ncon = A.shape[0]
            if not A.shape[1] == nx:
                raise ValueError(f"A must have {nx} columns")
            if not b.size == ncon:
                raise ValueError(f"b must have {ncon} elements")
            if has_params:
                if not S.shape == (ncon, npar):
                    raise ValueError(f"S must be of shape ({ncon},{npar})")
                if is_single_p:
                    b = b + S @ single_p  # absorb fixed p into b
        else:
            ncon = 0
            nlam = 0  # no lam, delta, y variables

        has_eq_constraints = (Aeq is not None)

        if has_eq_constraints:
            if beq is None:
                raise ValueError("beq must be provided if Aeq is provided")

            if has_params:
                if Seq is None:
                    raise ValueError(
                        "Seq must be provided if Aeq is provided and pmin/pmax are provided")

            nconeq = Aeq.shape[0]
            if not Aeq.shape[1] == nx:
                raise ValueError(f"Aeq must have {nx} columns")
            if not beq.size == nconeq:
                raise ValueError(f"beq must have {nconeq} elements")

            if has_params:
                if not Seq.shape == (nconeq, npar):
                    raise ValueError(f"Seq must be of shape ({nconeq},{npar})")
                if is_single_p:
                    beq = beq + Seq @ single_p  # absorb fixed p into beq
        else:
            nconeq = 0
            nmu = 0  # no Lagrange multipliers mu

        if variational:
            if not (has_ineq_constraints or has_eq_constraints):
                print(
                    "\033[1;31mVariational GNE requested but no shared constraints are defined.\033[0m")
                variational = False

        if has_params and is_single_p:
            has_params = False  # no parameters anymore
            npar = 0

        solver = solver.lower()

        if solver not in ["highs", "gurobi", "prox_admm"]:
            raise ValueError("solver must be 'highs' or 'gurobi' or 'prox_admm'")
        
        if solver == 'gurobi':
            gp = get_gurobi()
            if gp is None:
                print(
                    "\033[1;33mWarning: Gurobi not installed, switching to HiGHS solver.\033[0m")
                solver = "highs"
            is_mip = True
        
        if solver == "prox_admm":
            if not variational:
                raise ValueError("Proximal ADMM algorithm can only solve variational GNEPs")
            if has_params:
                raise ValueError("Proximal ADMM algorithm only solves non-parametric GNEPs")
            if has_pwa_objective or has_quad_objective:
                raise ValueError("Proximal ADMM algorithm only solves GNEPs without game-design objective function")
            if len(dim)<2:
                raise ValueError("Proximal ADMM algorithm only solves GNEPs with at least 2 agents")
            is_mip = False

        if solver == "highs":
            highspy = get_highspy()
            if highspy is None:
                raise ValueError("HiGHS solver not installed.")
            inf = highspy.kHighsInf
            is_mip = True
        elif solver == "gurobi":
            inf = gp.GRB.INFINITY
        else:
            inf = np.inf
        
        self.solver = solver
        self.is_mip = is_mip

        # Deal with variable bounds
        if lb is None:
            lb = -inf * np.ones(nx)
        if ub is None:
            ub = inf * np.ones(nx)
        if not lb.size == nx:
            raise ValueError(f"lb must have {nx} elements")
        if not ub.size == nx:
            raise ValueError(f"ub must have {nx} elements")
        if not np.all(ub >= lb):
            raise ValueError("Inconsistent variable bounds: some ub < lb")
        if is_mip and (any(ub < inf) or any(lb > -inf)):
            # Embed variable bounds into inequality constraints
            AA = []
            bb = []
            SS = []
            for i in range(nx):
                ei = np.zeros(nx)
                ei[i] = 1.0
                if ub[i] < inf:
                    AA.append(ei.reshape(1, -1))
                    bb.append(ub[i])
                    if has_params:
                        SS.append(np.zeros((1, npar)))
                if lb[i] > -inf:
                    AA.append(-ei.reshape(1, -1))
                    bb.append(-lb[i])
                    if has_params:
                        SS.append(np.zeros((1, npar)))
            if has_ineq_constraints:
                A = np.vstack((A, np.vstack(AA)))
                b = np.hstack((b, np.hstack(bb)))
                if has_params:
                    S = np.vstack((S, np.vstack(SS)))
            else:
                A = np.vstack(AA)
                b = np.hstack(bb)
                has_ineq_constraints = True
                if has_params:
                    S = np.vstack(SS)
            ncon = A.shape[0]
            nbox = len(AA)  # the last nbox constraints are box constraints
        else:
            nbox = 0  # no box constraints added

        cum_dim_x = np.cumsum([0]+dim[:-1])  # cumulative sum of dim

        if has_ineq_constraints and is_mip:
            # Determine where each agent's vars appear in the inequality constraints
            # G[i,j] = 1 if constraint i depends on agent j's variables
            G = np.zeros((ncon, N), dtype=bool)
            for i in range(N):
                G[:, i] = np.any(
                    A[:, cum_dim_x[i]:cum_dim_x[i]+dim[i]] != 0, axis=1)
            # number of constraints involving each agent, for each agent
            dim_lam = np.sum(G, axis=0)
            # cumulative sum of dim_lam
            cum_dim_lam = np.cumsum([0]+list(dim_lam[:-1]))
            nlam = np.sum(dim_lam)  # total number of lam and delta variables
        else:
            nlam = 0
            G = None
            dim_lam = None

        if has_eq_constraints and is_mip:
            # Determine where each agent's vars appear in the equality constraints
            # G[i,j] = 1 if constraint i depends on agent j's variables
            Geq = np.zeros((nconeq, N), dtype=bool)
            for i in range(N):
                Geq[:, i] = np.any(
                    Aeq[:, cum_dim_x[i]:cum_dim_x[i]+dim[i]] != 0, axis=1)
            # number of constraints involving each agent, for each agent
            dim_mu = np.sum(Geq, axis=0)
            # cumulative sum of dim_mu
            cum_dim_mu = np.cumsum([0]+list(dim_mu[:-1]))
            # total number of y and lam and delta variables
            nmu = np.sum(dim_mu)
        else:
            nmu = 0
            dim_mu = None
            Geq = None

        # Variable index ranges in the *single* Highs column space (j=agent index)
        # [ x (0..nx-1) | p (nx..nx+npar-1) | y | lam | delta ]
        def idx_x(j, i): return cum_dim_x[j] + i
            
        if solver == 'highs':
            if has_params:
                def idx_p(t): return nx + t
            else:
                idx_p = None

            if has_ineq_constraints:
                def idx_lam(j, k): return nx + npar + cum_dim_lam[j] + k
                def idx_delta(k): return nx + npar + nlam + k
            else:
                idx_lam = None
                idx_delta = None

            if has_eq_constraints:
                def idx_mu(j, k): return nx + npar + (nlam + ncon) * \
                    has_ineq_constraints + cum_dim_mu[j] + k
            else:
                idx_mu = None

            if has_pwa_objective:
                def idx_eps(j): return nx + npar + (nlam + ncon) * \
                    has_ineq_constraints + nmu*has_eq_constraints + j
            else:
                idx_eps = None
                
            self.idx_lam = idx_lam
            self.idx_mu = idx_mu
            self.idx_delta = idx_delta
            self.idx_eps = idx_eps

            mip = highspy.Highs()
            self.kHighsInf = highspy.kHighsInf

            # ------------------------------------------------------------------
            # 1. Add variables with bounds
            # ------------------------------------------------------------------
            # All costs default to 0 => min 0 (feasibility problem).

            # x: free (or set bounds as needed)
            for i in range(nx):
                mip.addVar(lb[i], ub[i])

            if has_params:
                # p: free (or set bounds as needed)
                for t in range(npar):
                    mip.addVar(pmin[t], pmax[t])
        
            if has_ineq_constraints:
                # lam: lam >=0
                for j in range(N):
                    for k in range(dim_lam[j]):
                        mip.addVar(0.0, inf)

                # delta: binary => bounds [0,1] + integrality = integer
                for k in range(ncon):
                    mip.addVar(0.0, 1.0)

                # Mark delta columns as integer
                # (Binary is simply integer with bounds [0,1])
                for k in range(ncon):
                    col = idx_delta(k)
                    mip.changeColIntegrality(col, highspy.HighsVarType.kInteger)

            if has_eq_constraints:
                # mu: free
                for j in range(N):
                    for k in range(dim_mu[j]):
                        mip.addVar(-inf, inf)

            if has_pwa_objective:
                for k in range(nJ):
                    # eps variable for PWA objective, unconstrained
                    mip.addVar(-inf, inf)

            # ------------------------------------------------------------------
            # 2. Add constraints
            # ------------------------------------------------------------------
            # (a) Qi x + Fi p + Ai^T lam_i + Q(i,-i) x(-i) = - ci
            for j in range(N):

                if has_ineq_constraints:
                    Gj = G[:, j]  # constraints involving agent j
                    nGj = np.sum(Gj)
                if has_eq_constraints:
                    Geqj = Geq[:, j]  # equality constraints involving agent j
                    nGeqj = np.sum(Geqj)

                for i in range(dim[j]):
                    indices = []
                    values = []

                    # Qx part: Q[j,:]@x = Q[j,i]@x(i) + Q[j,(-i)]@x(-i)
                    row_Q = Q[j][idx_x(j, i), :]
                    for k in range(nx):
                        if row_Q[k] != 0.0:
                            indices.append(k)
                            values.append(row_Q[k])

                    if has_params:
                        # Fp part: sum_t F[j,t] * p_t
                        row_F = F[j][idx_x(j, i), :]
                        for t in range(npar):
                            if row_F[t] != 0.0:
                                indices.append(idx_p(t))
                                values.append(row_F[t])

                    if has_ineq_constraints:
                        # A^T lam part: sum_k A[k,j] * lam_k
                        # A is (nA, nx), so column j is A[:, j]
                        col_Aj = A[Gj, idx_x(j, i)]
                        for k in range(nGj):
                            if col_Aj[k] != 0.0:
                                indices.append(idx_lam(j, k))
                                values.append(col_Aj[k])

                    if has_eq_constraints:
                        # Aeq^T mu part: sum_k Aeq[k,j] * mu_k
                        # Aeq is (nAeq, nx), so column j is Aeq[:, j]
                        col_Aeqj = Aeq[Geqj, idx_x(j, i)]
                        for k in range(nGeqj):
                            if col_Aeqj[k] != 0.0:
                                indices.append(idx_mu(j, k))
                                values.append(col_Aeqj[k])

                    # Equality: lower = upper = -c_j
                    rhs = -float(c[j][idx_x(j, i)])
                    num_nz = len(indices)
                    if num_nz == 0:
                        # still add the row with empty pattern
                        mip.addRow(rhs, rhs, 0, [], [])
                    else:
                        mip.addRow(rhs, rhs, num_nz,
                                    np.array(indices, dtype=np.int64),
                                    np.array(values, dtype=np.double))

            if has_ineq_constraints:
                # (b) 0 <= lam(i,j) <= M * delta(j)
                for j in range(N):
                    ind_lam = 0
                    for k in range(ncon):
                        if G[k, j]:  # agent j involved in constraint k or vGNE
                            indices = np.array(
                                [idx_lam(j, ind_lam), idx_delta(k)], dtype=np.int64)
                            values = np.array([1.0, -M], dtype=np.double)
                            lower = -inf
                            upper = 0.
                            mip.addRow(lower, upper, len(
                                indices), indices, values)
                            ind_lam += 1

                # (c) b + S p - A x <= M (1-delta)
                for i in range(ncon):
                    indices = [idx_delta(i)]
                    values = [M]
                    upper = float(M - b[i])
                    for k in range(nx):
                        if A[i, k] != 0.0:
                            indices.append(k)
                            values.append(-A[i, k])
                    if has_params:
                        for t in range(npar):
                            if S[i, t] != 0.0:
                                indices.append(idx_p(t))
                                values.append(S[i, t])
                    indices = np.array(indices, dtype=np.int64)
                    values = np.array(values, dtype=np.double)
                    mip.addRow(-inf, upper, len(indices), indices, values)

                # (d) A x <= b + S p
                for i in range(ncon):
                    indices = []
                    values = []

                    # Ai x part
                    row_Ai = A[i, :]
                    for k in range(nx):
                        if row_Ai[k] != 0.0:
                            indices.append(k)
                            values.append(row_Ai[k])

                    if has_params:
                        # Si p part
                        row_Si = S[i, :]
                        for t in range(npar):
                            if row_Si[t] != 0.0:
                                indices.append(idx_p(t))
                                values.append(-row_Si[t])

                    rhs = float(b[i])
                    num_nz = len(indices)
                    mip.addRow(-inf, rhs, num_nz,
                                np.array(indices, dtype=np.int64),
                                np.array(values, dtype=np.double))

            if has_eq_constraints:
                # (d2) Aeq x - Seq p = beq
                for i in range(nconeq):
                    indices = []
                    values = []

                    # Aeqi x part
                    row_Aeqi = Aeq[i, :]
                    for k in range(nx):
                        if row_Aeqi[k] != 0.0:
                            indices.append(k)
                            values.append(row_Aeqi[k])

                    if has_params:
                        # Seqi p part
                        row_Seqi = Seq[i, :]
                        for t in range(npar):
                            if row_Seqi[t] != 0.0:
                                indices.append(idx_p(t))
                                values.append(-row_Seqi[t])

                    rhs = float(beq[i])
                    num_nz = len(indices)
                    mip.addRow(rhs, rhs, num_nz,
                                np.array(indices, dtype=np.int64),
                                np.array(values, dtype=np.double))
            if variational:
                if has_ineq_constraints:
                    # exclude box constraints, they have their own multipliers. Shared constraints are first, indexed 0, ..., ncon-nbox-1
                    for j in range(ncon-nbox):
                        # indices of agents involved in constraint j
                        ii = np.argwhere(G[j, :])
                        i1 = int(ii[0].item())  # first agent involved
                        for k in range(1, len(ii)):  # loop not executed if only one agent involved
                            i2 = int(ii[k].item())  # other agent involved
                            indices = [idx_lam(i1, j),
                                    idx_lam(i2, j)]
                            num_nz = 2
                            values = [1.0, -1.0]
                            mip.addRow(0.0, 0.0, num_nz, np.array(indices, dtype=np.int64),
                                        np.array(values, dtype=np.double))

                if has_eq_constraints:
                    for j in range(nconeq):
                        # indices of agents involved in constraint j
                        ii = np.argwhere(Geq[j, :])
                        i1 = int(ii[0].item())  # first agent involved
                        for k in range(1, len(ii)):  # loop not executed if only one agent involved
                            i2 = int(ii[k].item())  # other agent involved
                            indices = [idx_mu(i1, j),
                                    idx_mu(i2, j)]
                            num_nz = 2
                            values = [1.0, -1.0]
                            mip.addRow(0.0, 0.0, num_nz, np.array(indices, dtype=np.int64),
                                        np.array(values, dtype=np.double))

            if has_pwa_objective:
                # (e) eps[k] >= D_pwa[k](i,:) x + E_pwa[k](i,:) p + h_pwa[k](i), i=1..nk
                for k in range(nJ):
                    for i in range(D_pwa[k].shape[0]):
                        indices = []
                        values = []

                        # D x part
                        row_Di = D_pwa[k][i, :]
                        for t in range(nx):
                            if row_Di[t] != 0.0:
                                indices.append(t)
                                values.append(row_Di[t])

                        # E p part
                        if has_params:
                            row_Ei = E_pwa[k][i, :]
                            for t in range(npar):
                                if row_Ei[t] != 0.0:
                                    indices.append(idx_p(t))
                                    values.append(row_Ei[t])

                        # eps part
                        indices.append(idx_eps(k))
                        values.append(-1.0)

                        rhs = float(-h_pwa[k][i])
                        num_nz = len(indices)
                        mip.addRow(-inf, rhs, num_nz,
                                    np.array(indices, dtype=np.int64),
                                    np.array(values, dtype=np.double))

                    # Define objective function: min eps
                    mip.changeColCost(idx_eps(k), 1.0)
                    
        elif solver == "gurobi":

            m = gp.Model("GNEP_LQ_MIP")
            mip = SimpleNamespace()
            mip.model = m
            
            # x variables
            x = m.addVars(range(nx), lb=lb.tolist(), ub=ub.tolist(), vtype=gp.GRB.CONTINUOUS, name="x")
            mip.x = x
            p = m.addVars(range(npar), lb=pmin.tolist(), ub=pmax.tolist(), vtype=gp.GRB.CONTINUOUS, name="p") if has_params else None
            mip.p = p

            if has_ineq_constraints:
                # lam: lam >=0
                lam = []
                for j in range(N):
                    lam_j = m.addVars(range(dim_lam[j]), lb=0.0, ub=inf, vtype=gp.GRB.CONTINUOUS, name=f"lam_{j}")
                    lam.append(lam_j)
                # delta: binary
                delta = m.addVars(range(ncon), vtype=gp.GRB.BINARY, name="delta")
                mip.lam = lam
                mip.delta = delta
            else:
                lam = None
                delta = None
            
            if has_eq_constraints:
                # mu: free
                mu = []
                for j in range(N):
                    mu_j = m.addVars(range(dim_mu[j]), lb=-inf, ub=inf, vtype=gp.GRB.CONTINUOUS, name=f"mu_{j}")
                    mu.append(mu_j)
                mip.mu = mu
            else:
                mu = None
            
            if has_pwa_objective:
                eps = m.addVars(range(nJ), lb=-inf, ub=inf, vtype=gp.GRB.CONTINUOUS, name="eps") 
                mip.eps = eps

            # ------------------------------------------------------------------
            # 2. Add constraints
            # ------------------------------------------------------------------
            # (a) Qi x + Fi p + Ai^T lam_i + Q(i,-i) x(-i) = - ci
            for j in range(N):
                if has_ineq_constraints:
                    Gj = G[:, j]  # constraints involving agent j
                    nGj = np.sum(Gj)
                if has_eq_constraints:
                    Geqj = Geq[:, j]  # equality constraints involving agent j
                    nGeqj = np.sum(Geqj)

                KKT1 = []
                for i in range(dim[j]):
                    # Qx part: Q[j,:]@x = Q[j,i]@x(i) + Q[j,(-i)]@x(-i)
                    row_Q = Q[j][cum_dim_x[j] + i, :]
                    KKT1_i = gp.quicksum(row_Q[t]*x[t] for t in range(nx)) + c[j][cum_dim_x[j] + i]

                    if has_params:
                        # Fp part: sum_t F[j,t] * p_t
                        row_F = F[j][idx_x(j, i), :]
                        KKT1_i += gp.quicksum(row_F[t]*p[t] for t in range(npar))
                        
                    if has_ineq_constraints:
                        # A^T lam part: sum_k A[k,j] * lam_k
                        # A is (nA, nx), so column j is A[:, j]
                        col_Aj = A[Gj, idx_x(j, i)]
                        KKT1_i += gp.quicksum(col_Aj[k]*lam[j][k] for k in range(nGj))

                    if has_eq_constraints:
                        # Aeq^T mu part: sum_k Aeq[k,j] * mu_k
                        # Aeq is (nAeq, nx), so column j is Aeq[:, j]
                        col_Aeqj = Aeq[Geqj, idx_x(j, i)]
                        KKT1_i += gp.quicksum(col_Aeqj[k]*mu[j][k] for k in range(nGeqj))
                    
                    KKT1.append(KKT1_i)

                m.addConstrs((KKT1[i] == 0. for i in range(dim[j])), name=f"KKT1_agent_{j}")

            if has_ineq_constraints:
                # (b) 0 <= lam(i,j) <= M * delta(j)
                for j in range(N):
                    ind_lam = 0
                    for k in range(ncon):
                        if G[k, j]:  # agent j involved in constraint k or vGNE
                            m.addConstr(lam[j][ind_lam] <= M * delta[k], name=f"big-M-lam_{j}_constr_{k}")
                            ind_lam += 1

                # (c) b + S p - A x <= M (1-delta)
                for i in range(ncon):
                    m.addConstr(b[i] + gp.quicksum(S[i,t]*p[t] for t in range(npar) if has_params) - gp.quicksum(A[i,k]*x[k] for k in range(nx)) <= M * (1. - delta[i]), name=f"big-M-slack_constr_{i}")

                # (d) A x <= b + S p
                for i in range(ncon):
                    m.addConstr(gp.quicksum(A[i,k]*x[k] for k in range(nx)) <= b[i] + gp.quicksum(S[i,t]*p[t] for t in range(npar) if has_params), name=f"shared_ineq_constr_{i}")

            if has_eq_constraints:
                # (d2) Aeq x - Seq p = beq
                for i in range(nconeq):
                    m.addConstr(gp.quicksum(Aeq[i,k]*x[k] for k in range(nx)) - gp.quicksum(Seq[i,t]*p[t] for t in range(npar) if has_params) == beq[i], name=f"shared_eq_constr_{i}")
                    
            if variational:
                if has_ineq_constraints:
                    # exclude box constraints, they have their own multipliers. Shared constraints are first, indexed 0, ..., ncon-nbox-1
                    for j in range(ncon-nbox):
                        # indices of agents involved in constraint j
                        ii = np.argwhere(G[j, :])
                        i1 = int(ii[0].item())  # first agent involved
                        for k in range(1, len(ii)):  # loop not executed if only one agent involved
                            i2 = int(ii[k].item())  # other agent involved
                            m.addConstr(lam[i1][j] == lam[i2][j], name=f"variational_ineq_constr_{j}")
                if has_eq_constraints:
                    for j in range(nconeq):
                        # indices of agents involved in constraint j
                        ii = np.argwhere(Geq[j, :])
                        i1 = int(ii[0].item())  # first agent involved
                        for k in range(1, len(ii)):  # loop not executed if only one agent involved
                            i2 = int(ii[k].item())  # other agent involved
                            m.addConstr(mu[i1][j] == mu[i2][j], name=f"variational_eq_constr_{j}")

            if has_pwa_objective:
                # (e) eps[k] >= D[k](i,:) x + E[k](i,:) p + h[k](i), i=1..nk
                for k in range(nJ):
                    for i in range(D_pwa[k].shape[0]):
                        m.addConstr(eps[k] >= gp.quicksum(D_pwa[k][i,t]*x[t] for t in range(nx)) + gp.quicksum(E_pwa[k][i,t]*p[t] for t in range(npar) if has_params) + h_pwa[k][i], name=f"pwa_obj_constr_{k}_{i}")
                        row_Di = D_pwa[k][i, :]
                        if has_params:
                            row_Ei = E_pwa[k][i, :]

                J_PWA = gp.quicksum(eps[k] for k in range(nJ)) # Define objective function term: min sum(eps)
            else:
                J_PWA = 0.0
            
            if has_quad_objective:
                J_Q = 0.5 * gp.quicksum(Q_J[i, j] * (x[i] if i < nx else p[i - nx]) * (x[j] if j < nx else p[j - nx]) for i in range(nx + npar) for j in range(nx + npar)) + gp.quicksum(c_J[i] * (x[i] if i < nx else p[i - nx]) for i in range(nx + npar))
            else:
                J_Q = 0.0
                
            m.setObjective(J_PWA + J_Q, gp.GRB.MINIMIZE)
            
        elif solver == "prox_admm":
            mip = SimpleNamespace() # store problem data in the object for use in the ADMM algorithm. It's called mip for consistency with the other solvers, even if no MILP model is created.  
            mip.Q = Q
            mip.c = c
            mip.F = F
            mip.b = b
            mip.Aeq = Aeq
            mip.beq = beq
                            
        self.mip = mip
        self.dim = dim
        self.has_params = has_params
        self.is_single_p = is_single_p
        self.single_p = single_p
        self.has_ineq_constraints = has_ineq_constraints
        self.has_eq_constraints = has_eq_constraints
        self.has_pwa_objective = has_pwa_objective
        self.nx = nx
        self.npar = npar
        self.ncon = ncon
        self.nconeq = nconeq
        self.N = N
        self.G = G
        self.A = A
        self.Geq = Geq
        self.dim_lam = dim_lam
        self.dim_mu = dim_mu
        self.M = M
        self.lb = lb
        self.ub = ub
        self.nbox = nbox
        self.pmin = pmin
        self.pmax = pmax
        if has_pwa_objective:
            self.nJ = nJ
        else:
            self.nJ = 0

    def solve(self, max_solutions=1, verbose=0, solver_options=None):
        """Solve a linear quadratic generalized GNE problem and associated game-design problem via mixed-integer linear programming (MILP) or mixed-integer quadratic programming (MIQP):

            min_{x,p,y,lam,delta,eps} sum(eps[k])  (if D,E,h provided)
                                      + 0.5 *[x;p]^T Q_J [x;p] + c_J^T [x;p]  (if Q_J,c_J provided)
            s.t.
                eps[k] >= D_pwa[k](i,:) x + E_pwa[k](i,:) p + h_pwa[k](i), i=1,...,nk  
                Q_ii x_i + c_i + F_i p + Q_{i(-i)} x(-i) + A_i^T lam_i + Aeq_i^T mu_i = 0 
                                                (individual 1st KKT condition)
                A x <= b + S  p                  (shared inequality constraints)
                Aeq x = beq + Seq p              (shared equality constraints)
                lam(i,j) >= 0                    (individual Lagrange multipliers)
                b + S p - A x <= M (1-delta)     (delta(j) = 1 -> constraint j is active)
                0 <= lam(i,j) <= M * delta(j)    (delta(j) = 0 -> lam(i,j) = 0 for all agents i)
                delta(j) binary
                lb <= x <= ub                    (variable bounds, possibly infinite)
                pmin <= p <= pmax            

        If D_pwa, E_pwa, h_pwa, Q_J, and c_J are None, the objective function is omitted, and only an equilibrium point is searched for. If pmin = pmax (or pmin,pmax are None), the problem reduces to finding a solution to a standard (non-parametric) GNEP-QP (or, in case infinitely many exist, the one
        minimizing f(x,p)). The MILP solver specified during object construction is used to solve the problem.

        When multiple solutions are searched for (max_solutions > 1), the MIP is solved multiple times, adding a "no-good" cut after each solution found to exclude it from the feasible set.
        
        MILP is used when no quadratic objective function is specified, otherwise MIQP is used (only Gurobi supported). 
        
        Alternatively, if the solver specified is "prox_admm", a proximal ADMM algorithm is used to solve the variational GNEP without parameters (or with a fixed parameter) and without PWA or quadratic objective function. 

        Parameters
        ----------
        max_solutions : int
            Maximum number of solutions to look for (1 by default).
        verbose : int
            Verbosity level: 0 = None, 1 = minimal, 2 = detailed.
        solver_options : dict
            Dictionary of solver-specific options to set before solving (not required by MIP solvers).

        Returns
        -------
        sol : SimpleNamespace (or list of SimpleNamespace, if multiple solutions are searched for)
            Each entry has the following fields:
                x = generalized Nash equilibrium
                p = parameter vector (if any)
                lam = list of Lagrange multipliers for each agent (if any) in the order:
                    - shared inequality constraints
                    - finite lower bounds for agent i
                    - finite upper bounds for agent i
                delta = binary variables for shared inequalities (if any)
                mu = list of Lagrange multipliers for equalities for each agent (if any)
                eps = optimal value of the objective function (if xdes is provided)
                G = boolean matrix indicating which constraints involve which agents (if any inequalities)
                Geq = boolean matrix indicating which equalities involve which agents (if any equalities)
                status_str = HiGHS MIP model status (or other solver status) as string
                elapsed_time = time taken to solve the MILP (in seconds)

        (C) 2025-2026 Alberto Bemporad
        """

        nx = self.nx
        npar = self.npar
        ncon = self.ncon
        nconeq = self.nconeq
        nbox = self.nbox
        N = self.N
        A = self.A
        if self.is_mip:
            G = self.G
            Geq = self.Geq
            dim_lam = self.dim_lam
            dim_mu = self.dim_mu

        pmin = self.pmin

        if not self.has_ineq_constraints and max_solutions > 1:
            print(
                "\033[1;31mCannot search for multiple solutions if no inequality constraints are present.\033[0m")
            max_solutions = 1
        if max_solutions > 1 and self.solver == 'prox_admm':
            print(
                "\033[1;31mProximal ADMM solver does not support multiple solution search.\033[0m")
            max_solutions = 1

        if self.is_mip:
            if verbose >= 1:
                print("Solving MIP problem ...")

            if self.solver == 'highs':
                highspy = get_highspy()
                idx_lam = self.idx_lam
                idx_mu = self.idx_mu
                idx_delta = self.idx_delta
                idx_eps = self.idx_eps
                inf = highspy.kHighsInf
                if verbose < 2:
                    self.mip.setOptionValue("log_to_console", False)
            else:
                gp = get_gurobi()
                self.mip.model.setParam('OutputFlag', verbose >=2)

            x = None
            p = None
            lam = None
            delta = None
            mu = None
            eps = None

            go = True
            solutions = []  # store found solutions
            found = 0
            while go and (found < max_solutions):

                t0 = time.time()
                
                if self.solver == 'highs':
                    status = self.mip.run()
                    model_status = self.mip.getModelStatus()
                    status_str = self.mip.modelStatusToString(model_status)

                    if (status != highspy.HighsStatus.kOk) or (model_status != highspy.HighsModelStatus.kOptimal):
                        go = False
                else:
                    self.mip.model.optimize()
                    go = (self.mip.model.status == gp.GRB.OPTIMAL)
                    status_str = 'optimal solution found' if go else 'not solved'
                    
                t0 = time.time() - t0

                if go:
                    found += 1
                    if self.solver == 'highs':
                        sol = self.mip.getSolution()
                        x_full = np.array(sol.col_value, dtype=float)
                    else:
                        x_full = np.array(list(self.mip.model.getAttr('X', self.mip.x).values()))

                    if verbose == 1 and max_solutions > 1:
                        print(".", end="")
                        if found % 50 == 0 and found > 0:
                            print("")

                    # Extract slices
                    x = x_full[0:nx].reshape(-1)
                    if self.has_params:
                        if self.solver == 'highs':
                            p = x_full[nx:nx+npar].reshape(-1)
                        else:
                            p = np.array(list(self.mip.model.getAttr('X', self.mip.p).values()))
                    else:
                        p = pmin  # fixed p (or None)

                    if self.has_ineq_constraints:
                        if self.solver == 'highs':
                            delta = x_full[idx_delta(0):idx_delta(ncon)].reshape(-1)
                        else:
                            delta = np.array(list(self.mip.model.getAttr('X', self.mip.delta).values()))
                        # Round delta to {0,1} just in case
                        delta = 0 + (delta > 0.5)
                        
                        lam = []
                        for j in range(N):
                            lam_j = np.zeros(ncon)
                            if self.solver == 'highs':
                                lam_j[G[:, j]] = x_full[idx_lam(
                                    j, 0):idx_lam(j, dim_lam[j])]
                            else:
                                lam_j[G[:, j]] = np.array(list(self.mip.model.getAttr('X', self.mip.lam[j]).values()))
                            lam_g = lam_j[:ncon - nbox]  # exclude box constraints
                            # add only multipliers for box constraints involving agent j
                            # Start with finite lower bounds
                            for k in range(ncon - nbox, ncon):
                                if G[k, j] and sum(A[k, :]) < -0.5:
                                    lam_g = np.hstack((lam_g, lam_j[k]))
                            for k in range(ncon - nbox, ncon):
                                if G[k, j] and sum(A[k, :]) > 0.5:
                                    lam_g = np.hstack((lam_g, lam_j[k]))
                            lam.append(lam_g.reshape(-1))

                    if self.has_eq_constraints:
                        mu = []
                        for j in range(N):
                            mu_j = np.zeros(nconeq)
                            if self.solver == 'highs':
                                mu_j[Geq[:, j]] = x_full[idx_mu(
                                    j, 0):idx_mu(j, dim_mu[j])]
                            else:
                                mu_j[Geq[:, j]] = np.array(list(self.mip.model.getAttr('X', self.mip.mu[j]).values()))
                            mu.append(mu_j.reshape(-1))

                    if self.has_pwa_objective:
                        if self.solver == 'highs':
                            eps = np.array(x_full[idx_eps(0):idx_eps(self.nJ)]).reshape(-1)
                        else:
                            eps = np.array(list(self.mip.model.getAttr('X', self.mip.eps).values()))

                    solutions.append(SimpleNamespace(x=x, p=p, lam=lam, delta=delta, mu=mu,
                                    eps=eps, status_str=status_str, G=G, Geq=Geq, elapsed_time=t0))

                    if found < max_solutions:
                        # Append no-good constraint to exclude this delta in future iterations
                        # sum_{i: delta_k(i)=1} delta(i) - sum_{i: delta_k(i)=0} delta(i) <= -1 + sum(delta_k(i))
                        if self.solver == 'highs':
                            indices = np.array([idx_delta(k)
                                            for k in range(ncon)], dtype=np.int64)
                            values = np.ones(ncon, dtype=np.double)
                            values[delta < 0.5] = -1.0
                            lower = -inf
                            upper = np.sum(delta) - 1.
                            self.mip.addRow(lower, upper, len(
                                indices), indices, values)
                        else:
                            self.mip.model.addConstr(
                                gp.quicksum(self.mip.delta[k] if delta[k] > 0.5 else -self.mip.delta[k] for k in range(ncon)) <= - 1. + np.sum(delta),
                                name=f"no_good_cut_{found}")

            if verbose == 1:
                print(f" done. {found} combinations found")
        
        else:
            if verbose >= 1:
                print(f"Solving via '{self.solver}' method ...")
            
            if solver_options is None:
                solver_options = {}
            
            if self.solver == 'prox_admm':
                # Check if solver_options has the required options for the ADMM algorithm, and set defaults if not provided.
                if "x0" not in solver_options:
                    x0 = None
                else:
                    x0 = solver_options["x0"]
                if "maxiter" not in solver_options:
                    maxiter = 1000
                else:
                    maxiter = solver_options["maxiter"]
                if "tol" not in solver_options:
                    tol = 1e-6
                else:
                    tol = solver_options["tol"]
                if "rho" not in solver_options:
                    rho = 1.0
                else:
                    rho = solver_options["rho"]
                if "gamma" not in solver_options:
                    gamma = 1.0
                else:
                    gamma = solver_options["gamma"]
                if "cvx_solver" not in solver_options:
                    cvx_solver = "OSQP"
                else:                    
                    cvx_solver = solver_options["cvx_solver"]
            
            sol = solve_prox_admm(self.dim, self.mip.Q, self.mip.c, self.A, self.mip.b, C=self.mip.Aeq, d=self.mip.beq, lb=self.lb, ub=self.ub, x0=x0, rho=rho, gamma=gamma, maxiter=maxiter, tol=tol, verbose=verbose, cvx_solver=cvx_solver)
            sol.p = self.single_p            
            sol.eps = None
            sol.delta = None

            solutions = [sol]

        if len(solutions) == 1:
            return solutions[0]
        elif len(solutions) == 0:
            if verbose == 1:
                print(" done. No solution found")
            return None
        else:
            return solutions
