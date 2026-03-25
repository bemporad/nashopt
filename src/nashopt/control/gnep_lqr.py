# NashOpt: A Python package for computing Generalized Nash Equilibria (GNE) in noncooperative games.
#
# Game-theoretic Linear Quadratic Regulator (LQR).
#
# (C) 2025-2026 Alberto Bemporad

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
from scipy.linalg import block_diag, solve_discrete_are
from types import SimpleNamespace
from functools import partial
import time
from ..nonlinear.gnep_base import GNEP

class NashLQR():
    def __init__(self, sizes, A, B, Q, R, dare_iters=50):
        """Set up a discrete-time feedback linear quadratic dynamic game (Nash-LQR game) with N agents.

        The dynamics are given by

            x(k+1) = A x(k) + sum_{i=1..N} B_i u_i(k)

        where x(k) is the state vector at time k, and u_i(k) is the control input of agent i at time k and has dimension sizes[i].

        Each agent i minimizes its LQR cost K_i

            J_i = sum_{k=0}^\infty x(k)^T Q_i x(k) + u_i(k)^T R_i u_i(k) 

        subject to dynamics x(k+1) = (A -B_{-i}K_{-i})x(k) + B_i u_i(k).
        
        The LQR gain is computed in JAX approximately by evaluating the LQ cost over "dare_iters" time steps.

        (C) 2025 Alberto Bemporad, December 20, 2025
        """
        self.sizes = sizes
        self.A = A
        N = len(sizes)
        self.N = N
        nx = A.shape[0]
        self.nx = nx
        nu = sum(sizes)
        if not B.shape == (nx, nu):
            raise ValueError(
                f"B must be of shape ({nx},{nu}), you provided {B.shape}")
        self.B = B
        if len(Q) != len(sizes):
            raise ValueError(
                f"Q must be a list of matrices with length equal to number {N} of agents")
        for i in range(N):
            if not Q[i].shape == (nx, nx):
                raise ValueError(f"Q[{i}] must be of shape ({nx},{nx})")
            # We should also check that Q[i] is symmetric and positive semidefinite ...

        self.Q = Q
        if len(R) != N:
            raise ValueError(
                f"R must be a list of matrices with length equal to number {N} of agents")
        for i in range(N):
            if not R[i].shape == (sizes[i], sizes[i]):
                raise ValueError(
                    f"R[{i}] must be of shape ({sizes[i]},{sizes[i]})")
            # We should also check that R[i] is symmetric and positive definite ...

        self.R = R
        nu = sum(sizes)
        self.nu = nu
        sum_i = np.cumsum(sizes)
        not_i = [list(range(sizes[0], nu))]
        for i in range(1, N):
            not_i.append(list(range(sum_i[i-1])) + list(range(sum_i[i], nu)))
        self.not_i = not_i
        self.ii = [list(range(sum_i[i]-sizes[i], sum_i[i])) for i in range(N)]
        self.dare_iters = dare_iters
        self.K_Nash = None
        
    def solve(self, method='residual', riccati_iters=100, stop_tol=1e-5, **kwargs):
        """Solve the Nash-LQR game.

        K_Nash = self.solve() provides the Nash equilibrium feedback gain matrix K_Nash, where u = -K_Nash x.

        We provide two methods to compute the Nash equilibrium feedback gain K_Nash = [K_1; ...; K_N]:
        
        - 'residual': The Nash equilibrium is found by letting agent i minimize the difference between K_i and the LQR gain K_i for the dynamics (A -B_{-i}K_{-i}, B_i). 
        
        - 'riccati': The Nash equilibrium is found by solving the coupled discrete-time algebraic Riccati equations using "riccati_iters" Riccati-based iterations (best responses) as described in [1, Section III.B], until convergence within "stop_tol" (infinity norm of the difference between successive K_Nash iterates).
        
        Both methods are initialized from the centralized LQR solution with matrix Q=sum(Q_i) and R=block_diag(R_1, ..., R_N), obtained by "dare_iters" Riccati iterations.
        
        K_Nash = self.solve(**kwargs) allows passing additional keyword arguments to the underlying GNEP solver when the 'residual' method is used, see GNEP.solve() for details.
        
        [1] B. Nortman, A. Monti, M. Sassano, T. Mylvaganam, ``Nash Equilibria for Linear Quadratic
        Discrete-Time Dynamic Games via Iterative and Data-Driven Algorithms'', IEEE Trans. Autom. Contr.,
        vol. 69, no. 10, October 2024.
        """

        dare_iters = self.dare_iters
        sol = SimpleNamespace()

        @jax.jit
        def jax_dare(A, B, Q, R):
            """ Solve the discrete-time ARE

                    X = A^T X A - A^T X B (R + B^T X B)^(-1) B^T X A + Q

            in jax using the following simple fixed-point iterations

                K_{k} = (R + B^T X_k B)^(-1) B^T X_k A
                A_cl = A - B K_{k}
                X_{k+1} = Q + A_cl^T X_k A_cl + K_{k}^T R K_{k}

            """

            A = jnp.asarray(A)
            B = jnp.asarray(B)
            Q = jnp.asarray(Q)
            R = jnp.asarray(R)

            def get_K(X, A, B, R):
                S = R + B.T @ X @ B
                L, lower = cho_factor(S, lower=True)
                # Equivalent to K = (R + B^T X B)^-1 B^T X A
                K = cho_solve((L, lower), B.T @ X @ A)
                return K

            def update(X, _):
                K = get_K(X, A, B, R)
                A_cl = A - B @ K
                X_next = Q + A_cl.T @ X @ A_cl + K.T @ R @ K
                return X_next, _

            # initial state: X = Q (or zeros)
            X_final, _ = jax.lax.scan(update, Q, xs=None, length=dare_iters)

            K_final = get_K(X_final, A, B, R)
            return X_final, K_final

        self.jax_dare = jax_dare  # store for possible later use
        
        # Initial guess = centralized LQR
        nu = self.nu
        bigR = block_diag(*self.R)
        bigQ = sum(self.Q[i] for i in range(self.N))
        _, K_cen = jax_dare(self.A, self.B, bigQ, bigR)
        # # Check for comparison using python control library
        # from control import dare
        # P1, _, K1 = dare(A, B, bigQ, bigR)
        # print("Max difference between LQR gains: ", np.max(np.abs(K_cen - K1)))
        # print("Max difference between Riccati matrices: ", np.max(np.abs(P - P1)))

        sol.K_centralized = K_cen
        
        print(f"Solving Nash-LQR problem using method '{method}'... ", end='')

        if method == 'residual':
            @partial(jax.jit, static_argnums=(1,))  # i is static
            def lqr_fun(K_flat, i, A, B, Q, R):
                K = K_flat.reshape(self.nu, self.nx)
                Ai = A - B[:, self.not_i[i]]@K[self.not_i[i], :]
                Bi = B[:, self.ii[i]]
                _, Ki = jax_dare(Ai, Bi, Q[i], R[i])  # best response gain
                return jnp.sum((K[self.ii[i], :]-Ki)**2)  # Frobenius norm squared
            self.lqr_fun = lqr_fun  # store for possible later use outside solve()

            f = []
            for i in range(self.N):
                f.append(partial(lqr_fun, i=i, A=self.A,
                        B=self.B, Q=self.Q, R=self.R))

            # each agent's variable is K_i (size[i] x nx) flattened
            sizes = [self.sizes[i]*self.nx for i in range(self.N)]
            gnep = GNEP(sizes, f=f)

            K0 = K_cen.flatten()
            sol_residual = gnep.solve(x0=K0, **kwargs)
            K_Nash, residual, stats = sol_residual.x, sol_residual.res, sol_residual.stats
            print("done.")
            K_Nash = K_Nash.reshape(nu, self.nx)
            sol.residual = residual
            sol.stats = stats
        
        elif method == 'riccati':
    
            t0 = time.time()
            K_Nash = np.array(K_cen.copy())
            A = self.A
            B = self.B
            N = self.N
            ii = self.ii
            not_i = self.not_i

            P=[None]*self.N
            for it in range(riccati_iters):
                K_old = K_Nash.copy()

                for i in range(N):
                    A_cl_minus_i = A - B[:, not_i[i]] @ K_Nash[not_i[i],:]
                    #A_cl_minus_i = A - B[:, not_i[i]] @ K_old[not_i[i],:]
                    
                    #from scipy.linalg import solve_discrete_are
                    #P[i] = solve_discrete_are(A_cl_minus_i, B[:,ii[i]], Q[i], R[i])
                    #Ki = np.linalg.solve(R[i] + B[:,ii[i]].T @ P[i] @ B[:,ii[i]], B[:,ii[i]].T @ P[i] @ A_cl_minus_i)
                    P[i], Ki = jax_dare(A_cl_minus_i, B[:,ii[i]], self.Q[i], self.R[i])
                    
                    K_Nash[ii[i], :] = Ki

                # check convergence of gains
                max_diff = max(np.linalg.norm(K_Nash[ii[i], :] - K_old[ii[i], :]) for i in range(N))
                if max_diff <= stop_tol:
                    break
                
            t0 = time.time() - t0
            print("done.") 
            
            sol.stats = SimpleNamespace()
            sol.stats.elapsed_time = t0
            sol.stats.riccati_iters = it+1
            
        else:
            raise ValueError(f"Unknown method '{method}'. Available methods are 'residual' and 'riccati'.")
            
        sol.K_Nash = K_Nash
        self.K_Nash = K_Nash  # store in object for possible later use. e.g., in dare()
        return sol
    
    def dare(self, agent = 0):
        """Compute the discrete-time algebraic Riccati equation (DARE) solution for agent i, given the current Nash equilibrium feedback gain K_Nash.

        This can be used to check the optimality of the Nash equilibrium by verifying that K_Nash[i,:] is close to the LQR gain for the dynamics (A - B_{-i}K_{-i}, B_i) and cost matrices Q[i], R[i].
        """
        
        if self.K_Nash is None:
            raise ValueError("Nash equilibrium not computed yet. Call solve() first.")
        
        i = agent

        Ai = self.A - self.B[:, self.not_i[i]]@self.K_Nash[self.not_i[i], :]
        Bi = self.B[:, self.ii[i]]
        Qi = self.Q[i]
        Ri = self.R[i]
        Pi = solve_discrete_are(Ai, Bi, Qi, Ri)
        Ki = np.linalg.solve(Ri + Bi.T @ Pi @ Bi, Bi.T @ Pi @ Ai)
        return Ki
