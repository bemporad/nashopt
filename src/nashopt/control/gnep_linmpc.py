# NashOpt: A Python package for computing Generalized Nash Equilibria (GNE) in noncooperative games.
#
# Game-theoretic Linear Model Predictive Control (MPC).
#
# (C) 2025-2026 Alberto Bemporad

import numpy as np
from scipy.linalg import block_diag
from types import SimpleNamespace
import osqp
from scipy.sparse import csc_matrix
from scipy.sparse import eye as speye
from scipy.sparse import vstack as spvstack
import time
from ..lq.gnep_lq import GNEP_LQ

class NashLinearMPC():
    def __init__(self, sizes, A, B, C, Qy, Qdu, T, ymin=None, ymax=None, umin=None, umax=None, dumin=None, dumax=None, Qeps=None, Tc=None, Acx=None, Acu=None, Acdu=None, bc=None):
        """Set up a game-theoretic linear MPC problem with N agents for set-point tracking. 

        The dynamics are given by

              x(t+1) = A x(t) + B u (t)
                y(t) = C x(t)
                u(t) = u(t-1) + du(t)

        where x(t) is the state vector, du(t) the vector of input increments, and y(t) the output vector at time t. The input vector u(t) is partitioned among N agents as u = [u_1; ...; u_N], where u_i(t) is the control input of agent i, with u_i of dimension sizes[i].

        Each agent i minimizes its finite-horizon cost

            J_i(du,w) = sum_{k=0}^{T-1} (y(k+1)-w)^T Qy[i] (y(k+1) - w) + du_i(k)^T Qdu[i] du_i(k) 

        subject to the above dynamics and the following (local) input constraints

            u_i_min <= u_i(k) <= u_i_max
            du_i_min <= du_i(k) <= du_i_max

        and (shared) output constraints

            -sum_i(eps[i]) + y_min <= y(k+1) <= y_max + sum_i(eps[i])

        where eps[i] >= 0 is a slack variable penalized in the cost function with the linear term Qeps[i]*eps[i] to soften the output constraints and prevent infeasibility issues. By default, the constraints are imposed at all time steps k=0 ... T-1, but if a constraint horizon Tc<T is provided, they are only imposed up to time Tc-1.

        The problem is solved via MILP to compute the first input increment du_{0,i} of each agent to apply to the system to close the loop.

        If variational=True at solution time, a variational equilibrium is computed by adding the necessary equality constraints on the multipliers of the shared output constraints.

        If centralized=True at solution time, a centralized MPC problem is solved instead of the game-theoretic one.
        
        An additional polyhedral shared hard constraint on the initial state and inputs can be specified using Acx, Acu, Acdu, and bc. The constraint is of the form Acx x(t) + Acu u(t) + Acdu (u(t)-u(t-1)) <= bc(t). The RHS bc(t) may vary over time. Infeasibility may occur if no input exists that satisfies the constraints for the current state and previous input.

        (C) 2025 Alberto Bemporad, December 26, 2025.

        Parameters
        ----------
        sizes : list of int
            List of dimensions of each agent's input vector.
        A : ndarray
            State matrix of the discrete-time system.
        B : ndarray
            Input matrix of the discrete-time system.
        C : ndarray
            Output matrix of the discrete-time system.
        Qy : list of ndarray
            List of output weighting matrices for each agent.
        Qdu : list of ndarray
            List of input increment weighting matrices for each agent.
        T : int
            Prediction horizon.
        ymin : ndarray, optional
            Minimum output constraints (shared among all agents). If None, no lower bound is applied.
        ymax : ndarray, optional
            Maximum output constraints (shared among all agents). If None, no upper bound is applied.
        umin : ndarray, optional
            Lower bound on input vector. If None, no lower bound is applied.
        umax : ndarray, optional
            Upper bound on input vector. If None, no upper bound is applied.
        dumin : ndarray, optional
            Lower bound on input increments. If None, no lower bound is applied.
        dumax : ndarray, optional
            Upper bound on input increments. If None, no upper bound is applied.
        Qeps : float, list, or None, optional
            List of slack variable penalties for each agent. If None, a default value of 1.e3 is used for all agents.
        Tc : int, optional
            Constraint horizon. If None, constraints are applied over the entire prediction horizon T.
        Acx : ndarray, optional
            Matrix for additional shared polyhedral constraints on the initial state. If None, no such constraints are applied.
        Acu : ndarray, optional
            Matrix for additional shared polyhedral constraints on the initial input. If None, no such constraints are applied.
        Acdu : ndarray, optional
            Matrix for additional shared polyhedral constraints on the initial input increment. If None, no such constraints are applied.
        bc : ndarray, optional
            RHS for additional shared polyhedral constraints. If None, no such constraints are applied. It can be modified at runtime by passing the value bc(t) via the solve() method.
        """

        self.sizes = sizes
        self.A = A
        N = len(sizes)
        self.N = N
        nx = A.shape[0]
        self.nx = nx
        nu = sum(sizes)
        self.nu = nu
        if not B.shape == (nx, nu):
            raise ValueError(
                f"B must be of shape ({nx},{nu}), you provided {B.shape}")
        self.B = B
        ny = C.shape[0]
        if not C.shape == (ny, nx):
            raise ValueError(
                f"C must be of shape ({ny},{nx}), you provided {C.shape}")
        self.C = C
        self.ny = ny

        if len(Qy) != len(sizes):
            raise ValueError(
                f"Qy must be a list of matrices with length equal to number {N} of agents")
        for i in range(N):
            if not isinstance(Qy[i], np.ndarray):
                # scalar output case
                Qy[i] = np.array(Qy[i]).reshape(1, 1)
            if not Qy[i].shape == (ny, ny):
                raise ValueError(f"Qy[{i}] must be of shape ({ny},{ny})")
        self.Qy = Qy
        if len(Qdu) != N:
            raise ValueError(
                f"Rdu must be a list of matrices with length equal to number {N} of agents")
        for i in range(N):
            if not isinstance(Qdu[i], np.ndarray):
                # scalar input case
                Qdu[i] = np.array(Qdu[i]).reshape(1, 1)
            if not Qdu[i].shape == (sizes[i], sizes[i]):
                raise ValueError(
                    f"Rdu[{i}] must be of shape ({sizes[i]},{sizes[i]})")
        self.Qdu = Qdu

        if Qeps is None:
            Qeps = [1.e3]*N
        elif not isinstance(Qeps, list):
            Qeps = [Qeps]*N
        else:
            if len(Qeps) != N:
                raise ValueError(
                    f"Qeps must be a list of length equal to number {N} of agents")
        self.Qeps = Qeps
        self.T = T  # prediction horizon
        self.Tc = min(Tc, T) if Tc is not None else T  # constraint horizon

        if ymin is not None:
            if not isinstance(ymin, np.ndarray):
                ymin = np.array(ymin).reshape(1,)
            if not ymin.shape == (ny,):
                raise ValueError(
                    f"ymin must be of shape ({ny},), you provided {ymin.shape}")
        else:
            ymin = -np.inf * np.ones(ny)
        self.ymin = ymin
        if ymax is not None:
            if not isinstance(ymax, np.ndarray):
                ymax = np.array(ymax).reshape(1,)
            if not ymax.shape == (ny,):
                raise ValueError(
                    f"ymax must be of shape ({ny},), you provided {ymax.shape}")
        else:
            ymax = np.inf * np.ones(ny)
        self.ymax = ymax
        if umin is not None:
            if not isinstance(umin, np.ndarray):
                umin = np.array(umin).reshape(1,)
            if not umin.shape == (nu,):
                raise ValueError(
                    f"umin must be of shape ({nu},), you provided {umin.shape}")
        else:
            umin = -np.inf * np.ones(nu)
        self.umin = umin
        if umax is not None:
            if not isinstance(umax, np.ndarray):
                umax = np.array(umax).reshape(1,)
            if not umax.shape == (nu,):
                raise ValueError(
                    f"umax must be of shape ({nu},), you provided {umax.shape}")
        else:
            umax = np.inf * np.ones(nu)
        self.umax = umax
        if dumin is not None:
            if not isinstance(dumin, np.ndarray):
                dumin = np.array(dumin).reshape(1,)
            if not dumin.shape == (nu,):
                raise ValueError(
                    f"dumin must be of shape ({nu},), you provided {dumin.shape}")
        else:
            dumin = -np.inf * np.ones(nu)
        self.dumin = dumin
        if dumax is not None:
            if not isinstance(dumax, np.ndarray):
                dumax = np.array(dumax).reshape(1,)
            if not dumax.shape == (nu,):
                raise ValueError(
                    f"dumax must be of shape ({nu},), you provided {dumax.shape}")
        else:
            dumax = np.inf * np.ones(nu)
        self.dumax = dumax
        
        if (Acx is not None) and (Acu is not None) and (Acdu is not None) and (bc is not None):
            if not Acx.shape[1] == nx:
                raise ValueError(f"Acx must have {nx} columns, you provided {Acx.shape[1]}")
            if not Acu.shape[1] == nu:
                raise ValueError(f"Acu must have {nu} columns, you provided {Acu.shape[1]}")
            if not Acdu.shape[1] == nu:
                raise ValueError(f"Acdu must have {nu} columns, you provided {Acdu.shape[1]}")
            if not Acx.shape[0] == Acu.shape[0] == Acdu.shape[0] == bc.shape[0]:
                raise ValueError(f"Acx, Acu, Acdu, and bc must have the same number of rows, you provided {Acx.shape[0]}, {Acu.shape[0]}, {Acdu.shape[0]}, and {bc.shape[0]} rows")
            self.has_poly_constraints = True
        else:
            self.has_poly_constraints = False       
        self.Acx = Acx
        self.Acu = Acu
        self.Acdu = Acdu
        self.bc = bc

        def build_qp(A, B, C, Qy, Qdu, Qeps, sizes, N, T, ymin, ymax, umin, umax, dumin, dumax, Tc, Acx, Acu, Acdu, bc):
            # Construct QP problem to solve linear MPC for a generic input sequence du
            nx, nu = B.shape
            ny = C.shape[0]

            # Build extended system matrices (input = du, state = (x,u), output = y)
            Ae = np.block([[A, B],
                           [np.zeros((nu, nx)), np.eye(nu)]])
            Be = np.vstack((B, np.eye(nu)))
            Ce = np.hstack((C, np.zeros((ny, nu))))

            Ak = [np.eye(nx+nu)]
            for k in range(1, T+1):
                Ak.append(Ak[-1] @ Ae)  # [A,B;0,I]^k

            # Determine x(k) = Sx * x0 + Su * du_sequence, k=1,...,T
            Sx = np.zeros((T * (nx+nu), nx+nu))
            Su = np.zeros((T * (nx+nu), T*nu))

            for k in range(1, T+1):
                # row block for x_k is from idx_start to idx_end
                i1 = (k-1) * (nx+nu)
                i2 = k * (nx+nu)

                # x_k = A^k x0 + sum_{j=0..k-1} A^{k-1-j} Bu u_j
                Sx[i1:i2, :] = Ak[k]

                for j in range(k):  # j = 0..k-1
                    Su[i1:i2, nu*j:nu*(j+1)] += Ak[k-1-j] @ Be

            Qblk = [np.kron(np.eye(T), Qy[i])
                    for i in range(N)]  # [(T*ny x T*ny)]
            # [du_1(0); ...; du_N(0); ...; du_1(T-1); ...; du_N(T-1)]
            Rblk = np.zeros((T*nu, T*nu))
            cumsizes = np.cumsum([0]+sizes)
            for k in range(T):
                off = k*nu
                for i in range(N):
                    Rblk[off+cumsizes[i]:off+cumsizes[i+1], off +
                         cumsizes[i]:off+cumsizes[i+1]] = Qdu[i]

            Cbar = np.kron(np.eye(T), Ce)  # (T*ny x T*(nx+nu))
            # Determine y(k) = Sx_y * x0 + Su_y * du_sequence
            Sx_y = Cbar @ Sx    # (T*ny x (nx+nu))
            Su_y = Cbar @ Su    # (T*ny x N)
            # (T*ny x ny), for reference tracking
            E = np.kron(np.ones((T, 1)), np.eye(ny))

            # Y -E@w = Cbar@X - E@w = Sx_y@x0 + Su_y@dU -E@w
            # .5*(Y -E@w)' Qblk (Y -E@w) = .5*dU' Su_y' Qblk Su_y dU + (Sx_y x0 - E w)' Qblk Su_y dU + const

            # The overall optimization vector is z = [du_0; ...; du_{T-1}, eps, lambda, w]
            # Cost function: .5*[[dU;eps]' H [dU;eps] + (c + F @ [x0;u(-1);w])' [U;eps] + const
            H = [block_diag(Su_y.T @ Qblk[i] @ Su_y + Rblk, np.zeros((N, N)))
                 for i in range(N)]  # [(T*nu+N x T*nu+N)]
            F = [np.vstack((np.hstack((Su_y.T @ Qblk[i] @ Sx_y, -Su_y.T @ Qblk[i] @ E)),
                           np.zeros((N, nx+nu+ny)))) for i in range(N)]  # [(N*nu+1 x (nx + nu + ny))]
            c = [np.hstack((np.zeros(T*nu), np.array(Qeps)))
                 for _ in range(N)]  # [(T*nu+N,)]

            # Output constraint for k=1,...,Tc:
            #          -> Ce*(Ae*[x(t);u(t-1)]+Be*delta_u(t) <= ymax
            #          -> -(Ce*(Ae*[x(t);u(t-1)]+Be*delta_u(t))) <= -ymin

            # Constraint matrices for all agents
            A_con = np.hstack(
                (np.vstack((Su_y[:Tc*ny], -Su_y[:Tc*ny])), -np.ones((Tc*ny*2, N))))
            # Constraint bounds for all agents
            b_con = np.hstack(
                (np.kron(np.ones(Tc), ymax), -np.kron(np.ones(Tc), ymin)))
            # Constraint matrix for [x(t);u(t-1)]
            B_con = np.vstack((-Sx_y[:Tc*ny], Sx_y[:Tc*ny]))

            # Input increment constraints
            # lower bound for all agents
            lb = np.hstack((np.kron(np.ones(T), dumin), np.zeros(N)))
            # upper bound for all agents
            ub = np.hstack((np.kron(np.ones(T), dumax), np.inf*np.ones(N)))

            # Bounds for input-increment constraints due to input constraints
            # u_k = u(t-1) + sum{j=0}^{k-1} du(j)  <= umax -> sum{j=0}^{k-1} du(j) <= umax - u(t-1)
            #                                      >= umin -> -sum{j=0}^{k-1} du(j) <= -umin + u(t-1)
            AI = np.kron(np.tril(np.ones((Tc, T))),
                         np.eye(nu))  # (Tc*nu x T*nu)
            A_con = np.vstack((A_con,
                               np.hstack((AI, np.zeros((Tc*nu, N)))),
                               np.hstack((-AI, np.zeros((Tc*nu, N))))
                               ))
            b_con = np.hstack((b_con,
                               np.kron(np.ones(Tc), umax),
                               np.kron(np.ones(Tc), -umin)
                               ))
            B_con = np.vstack((B_con,
                               np.hstack((
                                   np.zeros((2*Tc*nu, nx)),
                                   np.vstack((np.kron(np.ones((Tc, 1)), -np.eye(nu)),
                                              np.kron(
                                                  np.ones((Tc, 1)), np.eye(nu))
                                              ))
                               ))
                               ))
            if self.has_poly_constraints:
                ii_c = np.zeros(A_con.shape[0]+Acu.shape[0], dtype=bool)
                # Add constraint Acx x(t) + Acu (u(t-1) + du(0)) + Acdu (du(0)) <= bc 
                # --> (Acu + Acdu) du(0) <= bc - Acx x(t) - Acu u(t-1)
                A_con = np.vstack((A_con, np.hstack((Acu + Acdu, np.zeros((Acu.shape[0], A_con.shape[1]-nu))))))
                B_con = np.vstack((B_con, np.hstack((-Acx, -Acu))))
                b_con = np.hstack((b_con, bc))
                ii_c[-Acu.shape[0]:] = True  # indices of the additional constraints
            else:
                ii_c = None

            # Final QP problem: each agent i solves
            #
            # min_{du_sequence, eps1...epsN} .5*[du_sequence;eps1..epsN]' H[i] [du_sequence;eps1...epsN]
            #                   + (c + F[i] @ [x0;u(-1);ref])' [du_sequence;eps1...epsN]
            #
            # # s.t. A_con [du_sequence;eps1...epsN] <= b_con + B_con [x0;u(-1)]
            #        lb <= [du_sequence;eps1...epsN] <= ub

            return H, c, F, A_con, b_con, B_con, lb, ub, ii_c

        H, c, F, A_con, b_con, B_con, lb, ub, ii_c = build_qp(
            A, B, C, Qy, Qdu, Qeps, sizes, N, T, ymin, ymax, umin, umax, dumin, dumax, self.Tc, Acx, Acu, Acdu, bc)

        # Rearrange optimization variables to have all agents' variables together at each time step
        # Original z ordering:
        #   [du_1(0); ...; du_N(0); du_1(1); ...; du_N(1); ...; du_1(T-1); ...; du_N(T-1); eps1; ...; epsN]
        # New z_new ordering:
        #   [du_1(0); du_1(1); ...; du_1(T-1); eps1; du_2(0); ...; du_2(T-1); eps2; ...; du_N(0); ...; du_N(T-1); epsN]
        perm = []
        cum_sizes = np.cumsum([0] + list(sizes))
        for i in range(N):
            i_start = int(cum_sizes[i])
            i_end = int(cum_sizes[i + 1])
            # Collect all du_i(k) blocks across the horizon
            for k in range(T):
                koff = k * nu
                perm.extend(range(koff + i_start, koff + i_end))
            # Append eps_i (which is stored after all du's)
            perm.append(T * nu + i)

        # P = np.eye(T*nu+N)[perm,:]  # permutation matrix: z_new = P z -> z = P' z_new
        # .5 z' H z = .5 z_new' (P H P') z_new
        self.H = [Hi[perm, :][:, perm] for Hi in H]  # same as P@Hi@P.T
        # (c + F @ p)' z = (c + F @ p)' P' z_new = (P (c + F @ p))' z_new
        self.c = [ci[perm] for ci in c]  # same as P@ci
        self.F = [Fi[perm, :] for Fi in F]  # same as P@Fi

        # A_con @z = ... -> A_con P' @ z_new = ...
        iscon = np.isfinite(b_con)
        self.A_con = A_con[:, perm][iscon, :]  # same as A_con@P.T
        self.b_con = b_con[iscon]
        self.B_con = B_con[iscon, :]
        if self.has_poly_constraints:
            self.ii_c = ii_c[iscon]  # indices of the additional hard-constraints introduced for step 0

        # z >= lb -> P' z_new >= lb -> z_new >= P lb
        self.lb = lb[perm]
        self.ub = ub[perm]
        
        # remove constraints beyond constraint horizon Tc
        off = 0
        Tc = self.Tc
        self.lb_original = lb.copy() # save original bounds for use with Lemke's method, which requires lower-bounded variables
        for i in range(N):
            si = sizes[i]
            # constraint eps_i>=0 is not removed
            self.lb[off+Tc*si:off+T*si] = -np.inf
            self.ub[off+Tc*si:off+T*si] = np.inf
            off += T*si + 1  # Each agent optimizes du_i(0)..du_i(T-1), eps_i
        self.iperm = np.argsort(perm)  # inverse permutation

    def solve(self, x0, u1, ref, M=1.e4, variational=False, centralized=False, solver='highs', bc=None):
        """Solve game-theoretic linear MPC problem for a given reference via MILP.
        
        Parameters
        ----------
        x0 : ndarray
            Current state vector x(t).
        u1 : ndarray
            Previous input vector u(t-1).
        ref : ndarray
            Reference output vector r(t) to track.
        M : float, optional
            Big-M parameter for MILP formulation.
        variational : bool, optional
            If True, compute a variational equilibrium by adding the necessary equality constraints on the multipliers of the shared output constraints. 
        centralized : bool, optional
            If True, solve a centralized MPC problem via QP using osQP instead of the game-theoretic one.
        solver : str, optional
            LQ-GNEP solver to use: 'highs', 'gurobi' (MILP) or 'prox_admm', 'lemke', 'log_ipm' (only when 'variational=True').
        bc : ndarray, optional
            RHS for additional shared polyhedral constraints possibly imposed at current time step. If None, the value provided during initialization is used, or no such constraints were specified at construction.
            
        Returns
        -------
        sol : SimpleNamespace
            Solution object with the following fields:
            - u : ndarray
                First input of the optimal sequence to apply to the system as input u(t).
            - U : ndarray
                Full input sequence over the prediction horizon.
            - eps : ndarray
                Optimal slack variables for soft output constraints.
            - elapsed_time : float
                Total elapsed time (build + solve) in seconds.
            - elapsed_time_solver : float
                Elapsed time for solver only in seconds.
        """
        T = self.T
        # each agent's variable is [du_i(0); ...; du_i(T-1); eps_i]
        sizes = [si*T+1 for si in self.sizes]
        nu = self.nu
        if variational and centralized:
            print(
                "\033[1;31mWarning: variational equilibrium ignored in centralized MPC.\033[0m")

        if not self.has_poly_constraints and bc is not None:
            raise ValueError("No additional constraints were specified at construction, but bc value is provided at solve time.")
        b = self.b_con.copy() 
        if self.has_poly_constraints and bc is not None:
            # Update the RHS of the additional constraints with the provided bc value at current time step
            b[self.ii_c] = bc
        b += self.B_con @ np.hstack((x0, u1))
        c = [self.c[i] + self.F[i] @ np.hstack((x0, u1, ref)) for i in range(self.N)]

        t0 = time.perf_counter()
        if not centralized:
            # Set up and solve GNEP 
            lb = self.lb_original if variational and solver == 'lemke' else self.lb
            gnep = GNEP_LQ(sizes, self.H, c, F=None, lb=lb, ub=self.ub, pmin=None, pmax=None,
                           A=self.A_con, b=b, S=None, D_pwa=None, E_pwa=None, h_pwa=None, M=M, variational=variational, solver=solver)
        else:
            # Centralized MPC: total cost = sum of all agents' costs, solve via QP
            H_cen = csc_matrix(sum(self.H[i] for i in range(self.N)))
            c_cen = sum(c[i] for i in range(self.N))
            nvar = c_cen.size
            A_cen = spvstack([csc_matrix(self.A_con), speye(
                nvar, format="csc")], format="csc")
            lb_cen = np.hstack((-np.inf*np.ones(self.A_con.shape[0]), self.lb))
            ub_cen = np.hstack((b, self.ub))

            prob = osqp.OSQP()
            prob.setup(P=H_cen, q=c_cen, A=A_cen, l=lb_cen,
                       u=ub_cen, verbose=False, polish=True, max_iter=10000, eps_abs=1.e-6, eps_rel=1.e-6, polish_refine_iter=3)
        elapsed_time_build = time.perf_counter() - t0

        if not centralized:
            gnep_sol = gnep.solve()
            if gnep_sol is None:
                raise ValueError("No GNE solution found for game-theoretic MPC problem.")
            z = gnep_sol.x
            elapsed_time_solver = gnep_sol.elapsed_time
        else:
            # prob.update(q=c_cen, u=b) # We could speedup by storing prob and reusing previous factorizations
            res = prob.solve()  # Solve QP problem
            if res.info.status_val != 1:
                raise ValueError(
                    f"Centralized MPC QP solver failed with status {res.info.status}")
            z = res.x
            elapsed_time_solver = res.info.run_time

        # permutation matrix: z_new = P z -> z = P' z_new
        zeps_seq = z[self.iperm]  # rearranged optimization vector
        U = []
        uk = u1.copy()
        for k in range(T):
            uk = uk + zeps_seq[k*nu: (k+1)*nu]
            U.append(uk)

        sol = SimpleNamespace()
        sol.u = U[0]  # first input to apply
        sol.U = np.array(U)  # full input sequence
        # optimal slack variables for soft output constraints
        sol.eps = zeps_seq[-self.N:]
        sol.elapsed_time = elapsed_time_build + elapsed_time_solver
        sol.elapsed_time_solver = elapsed_time_solver
        return sol
