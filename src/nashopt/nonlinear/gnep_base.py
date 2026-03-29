# NashOpt: A Python package for computing Generalized Nash Equilibria (GNE) in noncooperative games.
#
# Nonlinear GNEP problem. 
#
# (C) 2025-2026 Alberto Bemporad

import numpy as np
import jax
import jax.numpy as jnp
import time
from jaxopt import ScipyBoundedMinimize
from scipy.optimize import least_squares
from types import SimpleNamespace

from .._common.report import eval_residual, check_equilibrium_common

jax.config.update("jax_enable_x64", True)

class GNEP():
    def __init__(self, sizes, f, g=None, ng=None, lb=None, ub=None, Aeq=None, beq=None, h=None, nh=None, variational=False, parametric=False):
        """
        Generalized Nash Equilibrium Problem (GNEP) with N agents, where agent i solves:

            min_{x_i} f_i(x)
            s.t. g(x) <= 0          (shared inequality constraints)
                 Aeq x = beq        (shared linear equality constraints)
                 h(x) = 0           (shared nonlinear equality constraints)
                 lb <= x <= ub      (box constraints on x_i)
                 i= 1,...,N         (N = number of agents)

        Parameters:
        -----------
        sizes : list of int
            List containing the number of variables for each agent.
        f : list of callables
            List of objective functions for each agent. Each function f[i](x) takes the full variable vector x as input.    
        g : callable, optional
            Shared inequality constraint function g(x) <= 0, common to all agents.
        ng : int, optional
            Number of shared inequality constraints. Required if g is provided.
        lb : array-like, optional
            Lower bounds for the variables. If None, no lower bounds are applied.
        ub : array-like, optional
            Upper bounds for the variables. If None, no upper bounds are applied.
        Aeq : array-like, optional
            Equality constraint matrix. If None, no equality constraints are applied.
        beq : array-like, optional
            Equality constraint vector. If None, no equality constraints are applied.
        h : callable, optional
            Shared nonlinear equality constraint function h(x) = 0, common to all agents.
        nh : int, optional
            Number of shared nonlinear equality constraints. Required if h is provided.
        variational : bool, optional
            If True, solve for a variational GNE by imposing equal Lagrange multipliers.

        (C) 2025 Alberto Bemporad
        """

        self.sizes = sizes
        self.N = len(sizes)  # number of agents
        self.nvar = sum(sizes)  # number of variables
        self.i2 = np.cumsum(sizes)  # x_i = x(i1[i]:i2[i])
        self.i1 = np.hstack((0, self.i2[:-1]))
        not_i = [list(range(sizes[0], self.nvar))]
        for i in range(1, self.N):
            not_i.append(list(range(self.i2[i-1])) + list(range(self.i2[i], self.nvar)))
        self.not_i = not_i

        if len(f) != self.N:
            raise ValueError(
                f"List of functions f must contain {self.N} elements, you provided {len(f)}.")
        self.f = f
        self.g = g  # shared inequality constraints
        # number of shared inequality constraints, taken into account by all agents
        self.ng = int(ng) if ng is not None else 0
        if self.ng > 0 and g is None:
            raise ValueError("If ng>0, g must be provided.")

        if lb is None:
            lb = -np.inf * np.ones(self.nvar)
        if ub is None:
            ub = np.inf * np.ones(self.nvar)

        # Make bounds JAX arrays
        self.lb = jnp.asarray(lb)
        self.ub = jnp.asarray(ub)

        # Use *integer indices* of bounded variables per agent
        self.lb_idx = []
        self.ub_idx = []
        self.nlb = []
        self.nub = []
        self.is_lower_bounded = []
        self.is_upper_bounded = []
        self.is_bounded = []

        for i in range(self.N):
            sl = slice(self.i1[i], self.i2[i])
            lb_mask = np.isfinite(lb[sl])
            ub_mask = np.isfinite(ub[sl])
            lb_idx_i = np.nonzero(lb_mask)[0]
            ub_idx_i = np.nonzero(ub_mask)[0]
            self.lb_idx.append(lb_idx_i)
            self.ub_idx.append(ub_idx_i)
            self.nlb.append(len(lb_idx_i))
            self.nub.append(len(ub_idx_i))
            self.is_lower_bounded.append(self.nlb[i] > 0)
            self.is_upper_bounded.append(self.nub[i] > 0)
            self.is_bounded.append(
                self.is_lower_bounded[i] or self.is_upper_bounded[i])

        if Aeq is not None:
            if beq is None:
                raise ValueError(
                    "If Aeq is provided, beq must also be provided.")
            if Aeq.shape[1] != self.nvar:
                raise ValueError(f"Aeq must have {self.nvar} columns.")
            if Aeq.shape[0] != beq.shape[0]:
                raise ValueError(
                    "Aeq and beq must have compatible dimensions.")
            self.Aeq = jnp.asarray(Aeq)
            self.beq = jnp.asarray(beq)
            self.neq = Aeq.shape[0]
        else:
            self.Aeq = None
            self.beq = None
            self.neq = 0

        self.h = h  # shared nonlinear equality constraints
        # number of shared nonlinear equality constraints, taken into account by all agents
        self.nh = int(nh) if nh is not None else 0
        if self.nh > 0 and h is None:
            raise ValueError("If nh>0, h must be provided.")

        self.has_eq = self.neq > 0 or self.nh > 0
        self.has_constraints = any(self.is_bounded) or (
            self.ng > 0) or self.has_eq

        if variational:
            if self.ng == 0 and not self.has_eq:
                print(
                    "\033[1;31mVariational GNE requested but no shared constraints are defined.\033[0m")
                variational = False
        self.variational = variational

        n_shared = self.ng + self.neq + self.nh  # number of shared multipliers
        self.nlam = [int(self.nlb[i] + self.nub[i] + n_shared)
                     for i in range(self.N)]  # Number of multipliers per agent

        if not variational:
            self.nlam_sum = sum(self.nlam)  # total number of multipliers
            i2_lam = np.cumsum(self.nlam)
            i1_lam = np.hstack((0, i2_lam[:-1]))
            self.ii_lam = [np.arange(i1_lam[i], i2_lam[i], dtype=int) for i in range(
                self.N)]  # indices of multipliers for each agent
        else:
            # all agents have the same multipliers for shared constraints
            self.ii_lam = []
            j = n_shared
            for i in range(self.N):
                self.ii_lam.append(np.hstack((np.arange(self.ng, dtype=int),  # shared inequality-multipliers
                                              # agent-specific box multipliers
                                              np.arange(
                                                  j, j + self.nlb[i] + self.nub[i], dtype=int),
                                              np.arange(self.ng, self.ng + self.neq + self.nh, dtype=int))))  # shared equality-multipliers
                j += self.nlb[i] + self.nub[i]
            self.nlam_sum = n_shared + \
                sum([self.nlb[i] + self.nub[i] for i in range(self.N)])

        # Gradients of the agents' objectives
        if not parametric:
            self.df = [
                jax.jit(
                    jax.grad(
                        lambda xi, x, i=i: self.f[i](
                            x.at[self.i1[i]:self.i2[i]].set(xi)
                        ),
                        argnums=0,
                    )
                )
                for i in range(self.N)
            ]

            if self.ng > 0:
                self.g = jax.jit(self.g)
                self.dg = jax.jit(jax.jacobian(self.g))

            if self.nh > 0:
                self.h = jax.jit(self.h)
                self.dh = jax.jit(jax.jacobian(self.h))
        else:
            self.df = [
                jax.jit(
                    jax.grad(
                        lambda xi, x, p, i=i: self.f[i](
                            x.at[self.i1[i]:self.i2[i]].set(xi), p
                        ),
                        argnums=0,
                    )
                )
                for i in range(self.N)
            ]

            if self.ng > 0:
                self.g = jax.jit(self.g)
                self.dg = jax.jit(jax.jacobian(self.g, argnums=0))

            if self.nh > 0:
                self.h = jax.jit(self.h)
                self.dh = jax.jit(jax.jacobian(self.h, argnums=0))

        self.parametric = parametric
        self.npar = 0

    def kkt_residual_shared(self, z):
        # KKT residual function (shared constraints part)
        x = z[:self.nvar]
        isparam = self.parametric
        if isparam:
            p = z[-self.npar:]

        res = []

        ng = self.ng
        if ng > 0:
            if not isparam:
                gx = self.g(x)            # (ng,)
                dgx = self.dg(x)           # (ng, nvar)
            else:
                gx = self.g(x, p)         # (ng,)
                dgx = self.dg(x, p)        # (ng, nvar)
        else:
            gx = None
            dgx = None

        nh = self.nh  # number of nonlinear equalities
        if nh > 0:
            if not isparam:
                hx = self.h(x)            # (nh,)
                dhx = self.dh(x)           # (nh, nvar)
            else:
                hx = self.h(x, p)         # (nh,)
                dhx = self.dh(x, p)        # (nh, nvar)
        else:
            dhx = None

        # primal feasibility for shared constraints
        neq = self.neq  # number of linear equalities
        if ng > 0:
            # res.append(jnp.maximum(gx, 0.0))  # This is redundant, due to the Fischer–Burmeister function used below in kkt_residual_i
            pass
        if neq > 0:
            if not isparam:
                res.append(self.Aeq @ x - self.beq)
            else:
                res.append(self.Aeq @ x - (self.beq + self.Seq @ p))
        if nh > 0:
            res.append(hx)
        return res, gx, dgx, dhx

    def kkt_residual_i(self, z, i, gx, dgx, dhx):
        # KKT residual function for agent i
        x = z[:self.nvar]
        isparam = self.parametric
        if not isparam:
            if self.has_constraints:
                lam = z[self.nvar:]
        else:
            if self.has_constraints:
                lam = z[self.nvar:-self.npar]
            p = z[-self.npar:]

        ng = self.ng
        nh = self.nh
        neq = self.neq  # number of linear equalities
        nh = self.nh  # number of nonlinear equalities

        is_bounded = self.is_bounded
        is_lower_bounded = self.is_lower_bounded
        is_upper_bounded = self.is_upper_bounded

        res = []
        i1 = int(self.i1[i])
        i2 = int(self.i2[i])

        if is_bounded[i]:
            zero = jnp.zeros(self.sizes[i])
        if is_bounded[i] or ng > 0 or neq > 0:  # we have inequality constraints
            nlam_i = self.nlam[i]
            lam_i = lam[self.ii_lam[i]]

        # 1st KKT condition
        if not isparam:
            res_1st = self.df[i](x[i1:i2], x)
        else:
            res_1st = self.df[i](x[i1:i2], x, p)

        if ng > 0:
            res_1st += dgx[:, i1:i2].T @ lam_i[:ng]
        if is_lower_bounded[i]:
            lb_idx_i = self.lb_idx[i]
            # Add -sum(e_i * lam_lb_i), where e_i is a unit vector
            res_1st -= zero.at[lb_idx_i].set(lam_i[ng:ng + self.nlb[i]])
        if is_upper_bounded[i]:
            ub_idx_i = self.ub_idx[i]
            # Add sum(e_i * lam_ub_i)
            res_1st += zero.at[ub_idx_i].set(lam_i[ng +
                                             self.nlb[i]:ng + self.nlb[i] + self.nub[i]])
        if neq > 0:
            res_1st += self.Aeq[:, i1:i2].T @ lam_i[-neq-nh:][:neq]
        if nh > 0:
            res_1st += dhx[:, i1:i2].T @ lam_i[-nh:]
        res.append(res_1st)

        x_i = x[i1:i2]

        if is_bounded[i] or ng > 0:
            # inequality constraints
            if ng > 0:
                g_parts = [gx]
            else:
                g_parts = []
            if is_lower_bounded[i]:
                g_parts.append(-x_i[lb_idx_i] + self.lb[i1:i2][lb_idx_i])
            if is_upper_bounded[i]:
                g_parts.append(x_i[ub_idx_i] - self.ub[i1:i2][ub_idx_i])
            gix = jnp.concatenate(g_parts)

            # complementary slackness
            # Use Fischer–Burmeister NCP function: min phi(a,b) = sqrt(a^2 + b^2) - (a + b)
            # where here a = lam_i>=0 and b = -gix>=0
            res.append(jnp.sqrt(lam_i[:nlam_i-neq-nh] **
                       2 + gix**2) - lam_i[:nlam_i-neq-nh] + gix)
            # res.append(jnp.minimum(lam_i[:nlam_i-neq-nh], -gix))
            # res.append(lam_i[:nlam_i-neq-nh]*gix)

            # dual feasibility
            # res.append(jnp.minimum(lam_i[:nlam_i-neq-nh], 0.0)) # This is redundant, due to the Fischer–Burmeister function above
        return res

    def kkt_residual(self, z):
        # KKT residual function: append agent-specific parts to shared constraints part

        res, gx, dgx, dhx = self.kkt_residual_shared(z)

        for i in range(self.N):
            res += self.kkt_residual_i(z, i, gx, dgx, dhx)

        return jnp.concatenate(res)

    def solve(self, x0=None, max_nfev=200, tol=1e-12, solver="trf", verbose=1):
        """ Solve the GNEP starting from initial guess x0.

        The residuals of the KKT optimality conditions of all agents are minimized jointly as a 
        nonlinear least-squares problem, solved via a Trust Region Reflective algorithm or Levenberg-Marquardt method. Strict complementarity is enforced via the Fischer–Burmeister NCP function. Variational GNEs are also supported by simply imposing equal Lagrange multipliers.

        Parameters:
        -----------
        x0 : array-like or None
            Initial guess for the Nash equilibrium x.
        max_nfev : int, optional
            Maximum number of function evaluations.
        tol : float, optional
            Tolerance used for solver convergence.
        solver : str, optional
            Solver method used by scipy.optimize.least_squares: "lm" (Levenberg-Marquardt) or "trf" (Trust Region Reflective algorithm). Method "dogbox" is another option.
        verbose : int, optional
            Verbosity level (0: silent, 1: termination report, 2: progress (not supported by "lm")).

        Returns:
        --------
        sol : SimpleNamespace
            Solution object with fields:
            x : ndarray
                Computed GNE solution (if one is found).
            res : ndarray
                KKT residual at the solution x*
            lam : list of ndarrays
                List of Lagrange multipliers for each agent at the GNE solution (if constrains are present).
                For each agent i, lam_star[i] contains the multipliers in the order:
                    - shared inequality constraints
                    - finite lower bounds for agent i
                    - finite upper bounds for agent i
                    - shared linear equality constraints
                    - shared nonlinear equality constraints
            stats : Statistics about the optimization result.
        """
        t0 = time.time()

        solver = solver.lower()

        if x0 is None:
            x0 = jnp.zeros(self.nvar)
        else:
            x0 = jnp.asarray(x0)

        if self.has_constraints:
            lam0 = 0.1 * jnp.ones(self.nlam_sum)
            z0 = jnp.hstack((x0, lam0))
        else:
            z0 = x0

        # Solve the KKT residual minimization problem via SciPy least_squares
        f = jax.jit(self.kkt_residual)
        df = jax.jit(jax.jacobian(self.kkt_residual))
        try:
            solution = least_squares(f, z0, jac=df, method=solver, verbose=verbose,
                                     ftol=tol, xtol=tol, gtol=tol, max_nfev=max_nfev)
        except Exception as e:
            raise RuntimeError(
                f"Error in least_squares solver: {str(e)} If you are using 'lm', try using 'trf' instead.") from e
        z_star = solution.x
        res = solution.fun
        kkt_evals = solution.nfev  # number of function evaluations
        if verbose>0 and kkt_evals == max_nfev:
            print(
                "\033[1;33mWarning: maximum number of function evaluations reached.\033[0m")

        x = z_star[:self.nvar]
        lam = []
        if self.has_constraints:
            lam_star = z_star[self.nvar:]
            for i in range(self.N):
                lam.append(np.asarray(lam_star[self.ii_lam[i]]))

        t0 = time.time() - t0

        norm_res = eval_residual(res, verbose, kkt_evals, t0)
                                        
        stats = SimpleNamespace()
        stats.solver = solver
        stats.kkt_evals = kkt_evals
        stats.elapsed_time = t0
        
        sol = SimpleNamespace()
        sol.x = np.asarray(x)
        sol.res = np.asarray(res)
        sol.lam = lam
        sol.stats = stats
        sol.norm_residual = norm_res
        return sol

    def best_response(self, i, x, p=None, rho=1e5, maxiter=200, tol=1e-8):
        """
        Compute best response for agent i via SciPy's L-BFGS-B:

            min_{x_i} f_i(x_i, x_{-i}) + rho * (sum_j max(g_i(x), 0)^2 + ||Aeq x - beq||^2 + ||h(x)||^2)
            s.t. lb_i <= x_i <= ub_i
            
        For parametric problems, the agent's objective is f_i(x_i, x_{-i}, p) and the shared constraints are g(x, p), h(x, p), Aeq x = beq + Seq p. The best response is computed at the current joint strategy x and parameters p. 

        Parameters:
        -----------
        i : int
            Index of the agent for which to compute the best response.
        x : array-like
            Current joint strategy of all agents.
        p : array-like, optional
            Current game parameters. Only used for parametric GNEPs.
        rho : float, optional
            Penalty parameter for constraint violations.
        maxiter : int, optional
            Maximum number of L-BFGS-B iterations.
        tol : float, optional
            Tolerance used in L-BFGS-B optimization.

        Returns:
        -----------
        sol : SimpleNamespace
            Solution object with fields:
            x : ndarray
                best response of agent i, within the full vector x.
            f : ndarray
                optimal objective value for agent i at best response, fi(x).
            stats : Statistics about the optimization result.
        """

        i1 = self.i1[i]
        i2 = self.i2[i]
        x = jnp.asarray(x)

        t0 = time.time()

        if p is None:
            @jax.jit
            def fun(xi):
                # reconstruct full x with x_i replaced
                x_i = x.at[i1:i2].set(xi)
                f = jnp.array(self.f[i](x_i)).reshape(-1)
                if self.ng > 0:
                    f += rho*jnp.sum(jnp.maximum(self.g(x_i), 0.0)**2)
                if self.neq > 0:
                    f += rho*jnp.sum((self.Aeq @ x_i - self.beq)**2)
                if self.nh > 0:
                    f += rho*jnp.sum(self.h(x_i)**2)
                return f[0]
        else:
            @jax.jit
            def fun(xi):
                # reconstruct full x with x_i replaced
                x_i = x.at[i1:i2].set(xi)
                f = jnp.array(self.f[i](x_i, p)).reshape(-1)
                if self.ng > 0:
                    f += rho*jnp.sum(jnp.maximum(self.g(x_i, p), 0.0)**2)
                if self.neq > 0:
                    f += rho*jnp.sum((self.Aeq @ x_i - self.beq - self.Seq @ p)**2)
                if self.nh > 0:
                    f += rho*jnp.sum(self.h(x_i, p)**2)
                return f[0]

        li = self.lb[i1:i2]
        ui = self.ub[i1:i2]

        options = {'iprint': -1, 'maxls': 20, 'gtol': tol, 'eps': tol,
                   'ftol': tol, 'maxfun': maxiter, 'maxcor': 10}

        solver = ScipyBoundedMinimize(
            fun=fun, tol=tol, method="L-BFGS-B", maxiter=maxiter, options=options)
        xi, state = solver.run(x[i1:i2], bounds=(li, ui))
        x_new = np.asarray(x.at[i1:i2].set(xi))
        
        t0 = time.time() - t0

        stats = SimpleNamespace()
        stats.elapsed_time = t0
        stats.solver = state
        stats.iters = state.iter_num

        sol = SimpleNamespace()
        sol.x = x_new
        sol.f = self.f[i](x_new) if p is None else self.f[i](x_new, p)
        sol.stats = stats
        return sol


    def check_equilibrium(self, x, p=None, verbose=True, **kwargs):
        """ Check if x is a GNE by evaluating the best response of each agent at x and comparing it with x, as well as comparing the associated objective values.
        
        Parameters:
        -----------
        x : array-like
            Joint strategy to check for equilibrium.
        p : array-like, optional
            Game parameters to check for equilibrium. Only used for parametric GNEPs.
        verbose : bool, optional
            If True, print the distance between x and the best response for each agent, as well as the difference in objective values.
        kwargs : dict, optional
            Additional keyword arguments to pass to the best response computation, such as rho, maxiter, tol.
            
        Returns:
        -----------
        dx : ndarray
            Difference between the current strategy and the collection of best responses for each agent.
        df : ndarray
            Difference between the current objective values and the optimal objective values for each agent.
        """

        return check_equilibrium_common(self, x, p, verbose, **kwargs)
