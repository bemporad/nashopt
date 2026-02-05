"""
NashOpt - A Python Library for Solving Generalized Nash Equilibrium (GNE) and Game-Design Problems.

For general nonlinear problems, the KKT conditions of all agents are enforced jointly by solving a nonlinear least-squares problem, using JAX for automatic differentiation. Game-design problems can also be solved on multi-parametric GNE problems to determine optimal game parameters.

For linear-quadratic problems, the KKT conditions are enforced instead via mixed-integer linear programming (MILP). MILP enables finding GNE solutions, if they exist, and to possibly enumerate all GNEs in case of multiple equilibria. Game-design problems based on convex piecewise affine objectives can also be solved via mip.

(C) 2025 Alberto Bemporad
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
from jaxopt import ScipyBoundedMinimize
from scipy.optimize import least_squares
from scipy.linalg import block_diag, solve_discrete_are
from types import SimpleNamespace
import highspy
from functools import partial
import osqp
from scipy.sparse import csc_matrix
from scipy.sparse import eye as speye
from scipy.sparse import vstack as spvstack
import time

try:
    import gurobipy as gp
    GUROBI_INSTALLED = True
except ImportError:
    GUROBI_INSTALLED = False

jax.config.update("jax_enable_x64", True)

def eval_residual(res, verbose, f_evals, elapsed_time):
    warn_tol = 1.e-4
    norm_res = np.sqrt(np.sum(res**2))
    if verbose > 0:
        print(
            f"GNEP solved: ||KKT residual||_2 = {norm_res:.3e} found in {f_evals} function evaluations, time = {elapsed_time:.3f} seconds.")  
        if norm_res > warn_tol:
            print(
                f"\033[1;33mWarning: the KKT residual norm > {warn_tol}, an equilibrium may not have been found.\033[0m")
    return norm_res

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

    def best_response(self, i, x, rho=1e5, maxiter=200, tol=1e-8):
        """
        Compute best response for agent i via SciPy's L-BFGS-B:

            min_{x_i} f_i(x_i, x_{-i}) + rho * (sum_j max(g_i(x), 0)^2 + ||Aeq x - beq||^2 + ||h(x)||^2)
            s.t. lb_i <= x_i <= ub_i

        Parameters:
        -----------
        i : int
            Index of the agent for which to compute the best response.
        x : array-like
            Current joint strategy of all agents.
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
        sol.f = self.f[i](x_new)
        sol.stats = stats
        return sol


class ParametricGNEP(GNEP):
    def __init__(self, *args, **kwargs):
        """
        Multiparametric Generalized Nash Equilibrium Problem (mpGNEP).

        We consider a multiparametric GNEP with N agents, where agent i solves:

            min_{x_i} f_i(x,p)
            s.t. g(x,p) <= 0          (shared inequality constraints)
                 Aeq x = beq + Seq p  (shared linear equality constraints)
                 h(x,p) = 0           (shared nonlinear equality constraints)
                 lb <= x <= ub        (box constraints on x_i)
                 i= 1,...,N           (N = number of agents)

        where p are the game parameters.

        Parameters:
        -----------
        sizes : list of int
            List containing the number of variables for each agent.
        f : list of callables
            List of objective functions for each agent. Each function f[i](x) takes the full variable vector x as input.    
        g : callable, optional
            Shared inequality constraint function g(x,p) <= 0, common to all agents.
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
            Shared inequality constraint function h(x,p) <= 0, common to all agents.
        nh : int, optional
            Number of shared inequality constraints. Required if h is provided.
        variational : bool, optional
            If True, solve for a variational GNE by imposing equal Lagrange multipliers.
        npar: int, optional
            Number of game parameters p.
        Seq : array-like, optional
            Parameter dependence matrix for equality constraints. If None, no parameter dependence is applied on equality constraints.

        (C) 2025 Alberto Bemporad
        """

        Seq = kwargs.pop("Seq", None)
        npar = kwargs.pop("npar", None)
        if npar is None:
            raise ValueError(
                "npar (number of parameters) must be provided for ParametricGNEP.")

        super().__init__(*args, **kwargs, parametric=True)

        self.npar = int(npar)

        if Seq is not None:
            if self.Aeq is None:
                raise ValueError(
                    "If Seq is provided, Aeq must also be provided.")
            if Seq.shape[0] != self.Aeq.shape[0]:
                raise ValueError(
                    "Seq and Aeq must have the same number of rows.")
            if Seq.shape[1] != self.npar:
                raise ValueError(f"Seq must have {self.npar} columns.")
            self.Seq = jnp.asarray(Seq)
        else:
            self.Seq = jnp.zeros(
                (self.Aeq.shape[0], self.npar)) if self.Aeq is not None else None

    def solve(self, J=None, pmin=None, pmax=None, p0=None, x0=None, rho=1e5, alpha1=0., alpha2=0., maxiter=200, tol=1e-10, gne_warm_start=False, refine_gne=False, verbose=True):
        """
        Design game-parameter vector p for the GNEP by solving:

            min_{p} J(x*(p), p)
            s.t. pmin <= p <= pmax

        where x*(p) is the GNE solution for parameters p.

        Parameters:
        -----------
        J : callable or None
            Design objective function J(x, p) to be minimized. If None, the default objective J(x,p) = 0 is used.
        pmin : array-like or None
            Lower bounds for the parameters p.
        pmax : array-like or None
            Upper bounds for the parameters p.
        p0 : array-like or None
            Initial guess for the parameters p.
        x0 : array-like or None
            Initial guess for the GNE solution x.
        rho : float, optional
            Penalty parameter for KKT violation in best-response.
        alpha1 : float or None, optional
            If provided, add the regularization term alpha1*||x||_1
        alpha2 : float or None, optional
            If provided, add the regularization term alpha2*||x||_2^2
            When alpha2>0 and J is None, a GNE solution is computed nonlinear least squares.
        maxiter : int, optional
            Maximum number of solver iterations.
        tol : float, optional
            Optimization tolerance.
        gne_warm_start : bool, optional
            If True, warm-start the optimization by computing a GNE.
        refine_gne : bool, optional
            If True, try refining the solution to get a GNE after solving the problem for the optimal parameter p found. Mainly useful when J is provided or regularization is used. 
        verbose : bool, optional
            If True, print optimization statistics.
            
        Returns:
        --------
        sol : SimpleNamespace
            Solution object with fields:
            x : ndarray
                Computed GNE solution at optimal parameters p*.
            p : ndarray
                Computed optimal parameters p*.
            res : ndarray
                KKT residual at the solution (x*(p*), p*).
            lam : list of ndarrays
                List of Lagrange multipliers for each agent at the GNE solution (if constraints are present).
            J : float
                Optimal value of the design objective J at (x*(p*), p*).
            stats : Statistics about the optimization result.
        """
        t0 = time.time()

        is_J = J is not None
        if not is_J:
            def J(x, p): return 0.0

        L1_regularized = alpha1 > 0.
        L2_regularized = alpha2 > 0.

        if p0 is None:
            p0 = jnp.zeros(self.npar)
        if x0 is None:
            x0 = jnp.zeros(self.nvar)
        if self.has_constraints:
            lam0 = 0.1 * jnp.ones(self.nlam_sum)
        else:
            lam0 = jnp.array([])

        z0 = jnp.hstack((x0, lam0, p0)) if not L1_regularized else jnp.hstack(
            (jnp.maximum(x0, 0.), jnp.maximum(-x0, 0.), lam0, p0))

        nvars = self.nvar*(1+L1_regularized) + self.npar + self.nlam_sum
        lb = -np.inf*np.ones(nvars)
        ub = np.inf*np.ones(nvars)
        if pmin is not None:
            lb[-self.npar:] = pmin
        if pmax is not None:
            ub[-self.npar:] = pmax
        if not L1_regularized:
            lb[:self.nvar] = self.lb
            ub[:self.nvar] = self.ub
        else:
            lb[:self.nvar] = jnp.maximum(self.lb, 0.0)
            ub[:self.nvar] = jnp.maximum(self.ub, 0.0)
            lb[self.nvar:2*self.nvar] = jnp.maximum(-self.ub, 0.0)
            ub[self.nvar:2*self.nvar] = jnp.maximum(-self.lb, 0.0)

        stats = SimpleNamespace()
        stats.kkt_evals = 0

        if gne_warm_start:
            # Compute a GNE for initial guess
            dR_fun = jax.jit(jax.jacobian(self.kkt_residual))
            solution = least_squares(self.kkt_residual, z0, jac=dR_fun, method="trf",
                                     verbose=0, ftol=tol, xtol=tol, gtol=tol, max_nfev=maxiter, bounds=(lb, ub))
            z0 = solution.x
            stats.kkt_evals += solution.nfev

        # also include the case of no J and pure L1-regularization, since alpha2 = 0 cannot be handled by least_squares
        if is_J or (L1_regularized and alpha2 == 0.0):
            stats.solver = "L-BFGS"
            options = {'iprint': -1, 'maxls': 20, 'gtol': tol, 'eps': tol,
                       'ftol': tol, 'maxfun': maxiter, 'maxcor': 10}

            if not L1_regularized:
                @jax.jit
                def obj(z):
                    x = z[:self.nvar]
                    p = z[-self.npar:]
                    return J(x, p) + 0.5*rho * jnp.sum(self.kkt_residual(z)**2) + alpha2*jnp.sum(x**2)

                solver = ScipyBoundedMinimize(
                    fun=obj, tol=tol, method="L-BFGS-B", maxiter=maxiter, options=options)
                z, state = solver.run(z0, bounds=(lb, ub))
                x = z[:self.nvar]
                R = self.kkt_residual(z)

            else:  # L1-regularized
                @jax.jit
                def obj(z):
                    xp = z[:self.nvar]
                    xm = z[self.nvar:2*self.nvar]
                    p = z[-self.npar:]
                    return J(xp-xm, p) + alpha1 * jnp.sum(xp+xm) + alpha2 * (jnp.sum(xp**2+xm**2))

                solver = ScipyBoundedMinimize(
                    fun=obj, tol=tol, method="L-BFGS-B", maxiter=maxiter, options=options)
                z, state = solver.run(z0, bounds=(lb, ub))
                x = z[:self.nvar]-z[self.nvar:2*self.nvar]
                R = self.kkt_residual(jnp.concatenate((x, z[2*self.nvar:])))

            stats.kkt_evals += state.num_fun_eval

        else:
            # No design objective, just solve for a GNE with possible regularization
            stats.solver = "TRF"
            srho = jnp.sqrt(rho)
            alpha3 = jnp.sqrt(2.*alpha2)

            if not L1_regularized:
                if not L2_regularized:
                    R_obj = jax.jit(self.kkt_residual)
                else:
                    @jax.jit
                    def R_obj(z):
                        return jnp.concatenate((srho*self.kkt_residual(z), alpha3*x))
            else:
                # The case (L1_regularized and alpha2==0.0) was already handled above, so here alpha2>0 --> alpha3>0
                alpha4 = alpha1/alpha3

                @jax.jit
                def R_obj(z):
                    zx = jnp.concatenate(
                        (z[:self.nvar]-z[self.nvar:2*self.nvar], z[2*self.nvar:]))
                    res, gx, dgx, dhx = self.kkt_residual_shared(zx)
                    for i in range(self.N):
                        res += self.kkt_residual_i(zx, i, gx, dgx, dhx)
                    for i in range(len(res)):
                        res[i] = srho*res[i]
                    res += [alpha3*z[:self.nvar] + alpha4]
                    res += [alpha3*z[self.nvar:2*self.nvar] + alpha4]
                    return jnp.concatenate(res)

            # Solve the KKT residual minimization via SciPy least_squares
            dR_obj = jax.jit(jax.jacobian(R_obj))
            solution = least_squares(R_obj, z0, jac=dR_obj, method="trf", verbose=0,
                                     ftol=tol, xtol=tol, gtol=tol, max_nfev=maxiter, bounds=(lb, ub))
            z = solution.x
            if not L1_regularized:
                x = z[:self.nvar]
                zx = z
            else:
                x = z[:self.nvar]-z[self.nvar:2*self.nvar]
                zx = jnp.concatenate((x, z[2*self.nvar:]))

            R = self.kkt_residual(zx)

            # This actually returns the number of function evaluations, not solver iterations
            stats.kkt_evals += solution.nfev

        x = np.asarray(x)
        R = np.asarray(R)
        p = np.asarray(z[-self.npar:])

        if refine_gne:
            if verbose and (not L1_regularized and not is_J):
                # No need to refine
                print(
                    "\033[1;33mWarning: refine_gne=True has no effect when no design objective and regularization are used. Skipping refinement.\033[0m")
            else:
                def kkt_residual_refine(z, p):
                    zx = jnp.concatenate((z, p))
                    return self.kkt_residual(zx)
                Rx = partial(jax.jit(kkt_residual_refine), p=p)
                dRx = jax.jit(jax.jacobian(Rx))
                lam0 = z[self.nvar+self.nvar*L1_regularized: -self.npar]
                z0 = jnp.hstack((x, lam0))
                lbz = jnp.concatenate((self.lb, -np.inf*np.ones(len(lam0))))
                ubz = jnp.concatenate((self.ub,  np.inf*np.ones(len(lam0))))
                solution = least_squares(Rx, z0, jac=dRx, method="trf", verbose=0,
                                         ftol=tol, xtol=tol, gtol=tol, max_nfev=maxiter, bounds=(lbz, ubz))
                z = solution.x
                x = z[:self.nvar]
                z = jnp.hstack((z, p))
                R = np.asarray(self.kkt_residual(z))
                stats.kkt_evals += solution.nfev

        t0 = time.time() - t0
        J_opt = J(x, p) if is_J else 0.0
        lam = []
        if self.has_constraints:
            lam_star = z[self.nvar+self.nvar*L1_regularized:]
            for i in range(self.N):
                lam.append(np.asarray(lam_star[self.ii_lam[i]]))

        stats.elapsed_time = t0

        norm_res = eval_residual(R, verbose, stats.kkt_evals, t0)
        
        sol = SimpleNamespace()
        sol.x = x
        sol.p = p
        sol.lam = lam
        sol.J = J_opt
        sol.res = R
        sol.stats = stats
        sol.norm_residual = norm_res
        return sol

    def best_response(self, i, x, p, rho=1e5, maxiter=200, tol=1e-8):
        """
        Compute best response for agent i via SciPy's L-BFGS-B:

            min_{x_i} f_i(x_i, x_{-i}, p) + rho * (sum_j max(g_i(x,p), 0)^2 + ||Aeq x - beq -Seq p||^2 + ||h(x,p)||^2)
            s.t. lb_i <= x_i <= ub_i

        Parameters:
        -----------
        i : int
            Index of the agent for which to compute the best response.
        x : array-like
            Current joint strategy of all agents.
        p : array-like
            Current game parameters.
        rho : float, optional
            Penalty parameter for constraint violations.
        maxiter : int, optional
            Maximum number of L-BFGS-B iterations.
        tol : float, optional
            Tolerance used in L-BFGS-B optimization.

        Returns:
        x_i     : best response of agent i
        res     : SciPy optimize result
        """
        t0 = time.time()
        
        i1 = self.i1[i]
        i2 = self.i2[i]
        x = jnp.asarray(x)

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
        sol.f = self.f[i](x_new, p)
        sol.stats = stats
        return sol


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

        To search for a variational GNE, set the flag variational=True. In this case, the KKT conditions require equal Lagrange multipliers for all agents for each shared constraint.

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
            Solver used to solve the resulting mixed-integer program. "highs" (default) or "gurobi".

        (C) 2025 Alberto Bemporad, December 20, 2025
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
            if np.all(pmin == pmax):
                for i in range(N):
                    c[i] = c[i] + F[i] @ pmin  # absorb fixed p into c
                if has_pwa_objective:
                    for k in range(nJ):
                        h_pwa[k] = h_pwa[k] + E_pwa[k] @ pmin  # absorb fixed p into h
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
                if np.all(pmin == pmax):
                    b = b + S @ pmin  # absorb fixed p into b
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
                if np.all(pmin == pmax):
                    beq = beq + Seq @ pmin  # absorb fixed p into beq
        else:
            nconeq = 0
            nmu = 0  # no Lagrange multipliers mu

        if variational:
            if not (has_ineq_constraints or has_eq_constraints):
                print(
                    "\033[1;31mVariational GNE requested but no shared constraints are defined.\033[0m")
                variational = False

        if has_params and np.all(pmin == pmax):
            has_params = False  # no parameters anymore
            npar = 0

        solver = solver.lower()

        if solver == 'gurobi' and not GUROBI_INSTALLED:
            print(
                "\033[1;33mWarning: Gurobi not installed, switching to HiGHS solver.\033[0m")
            solver = "highs"
        if solver == "highs":
            inf = highspy.kHighsInf
        elif solver == "gurobi":
            inf = gp.GRB.INFINITY
        else:
            raise ValueError("solver must be 'highs' or 'gurobi'")
        
        self.solver = solver

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
        if any(ub < inf) or any(lb > -inf):
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

        if has_ineq_constraints:
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

        if has_eq_constraints:
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
                        i1 = int(ii[0])  # first agent involved
                        for k in range(1, len(ii)):  # loop not executed if only one agent involved
                            i2 = int(ii[k])  # other agent involved
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
                        i1 = int(ii[0])  # first agent involved
                        for k in range(1, len(ii)):  # loop not executed if only one agent involved
                            i2 = int(ii[k])  # other agent involved
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
                    
        else:  # gurobi

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
                        i1 = int(ii[0])  # first agent involved
                        for k in range(1, len(ii)):  # loop not executed if only one agent involved
                            i2 = int(ii[k])  # other agent involved
                            m.addConstr(lam[i1][j] == lam[i2][j], name=f"variational_ineq_constr_{j}")
                if has_eq_constraints:
                    for j in range(nconeq):
                        # indices of agents involved in constraint j
                        ii = np.argwhere(Geq[j, :])
                        i1 = int(ii[0])  # first agent involved
                        for k in range(1, len(ii)):  # loop not executed if only one agent involved
                            i2 = int(ii[k])  # other agent involved
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
                
        self.mip = mip
        self.has_params = has_params
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

    def solve(self, max_solutions=1, verbose=0):
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
        minimizing f(x,p). The MILP solver specified during object construction is used to solve the problem.

        When multiple solutions are searched for (max_solutions > 1), the MIP is solved multiple times, adding a "no-good" cut after each solution found to exclude it from the feasible set.
        
        MILP is used when no quadratic objective function is specified, otherwise MIQP is used (only Gurobi supported). 

        Parameters
        ----------
        max_solutions : int
            Maximum number of solutions to look for (1 by default).
        verbose : int
            Verbosity level: 0 = None, 1 = minimal, 2 = detailed.

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
                status_str = HiGHS MIP model status as string
                elapsed_time = time taken to solve the MILP (in seconds)

        (C) 2025 Alberto Bemporad, December 18, 2025    
        """

        nx = self.nx
        npar = self.npar
        ncon = self.ncon
        nconeq = self.nconeq
        nbox = self.nbox
        N = self.N
        G = self.G
        A = self.A
        Geq = self.Geq
        dim_lam = self.dim_lam
        dim_mu = self.dim_mu

        pmin = self.pmin

        if not self.has_ineq_constraints and max_solutions > 1:
            print(
                "\033[1;31mCannot search for multiple solutions if no inequality constraints are present.\033[0m")
            max_solutions = 1

        if verbose >= 1:
            print("Solving MIP problem ...")

        if self.solver == 'highs':
            idx_lam = self.idx_lam
            idx_mu = self.idx_mu
            idx_delta = self.idx_delta
            idx_eps = self.idx_eps
            inf = highspy.kHighsInf
            if verbose < 2:
                self.mip.setOptionValue("log_to_console", False)
        else:
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

        if len(solutions) == 1:
            return solutions[0]
        else:
            return solutions


class NashLQR():
    def __init__(self, sizes, A, B, Q, R, dare_iters=50):
        """Set up a discrete-time linear quadratic dynamic game (Nash-LQR game) with N agents.

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

                K = (R + B^T X_k B)^(-1) B^T X_k A
                A_cl = A - B K
                X_{k+1} = Q + A_cl^T X_k A_cl + K^T R K

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


class NashLinearMPC():
    def __init__(self, sizes, A, B, C, Qy, Qdu, T, ymin=None, ymax=None, umin=None, umax=None, dumin=None, dumax=None, Qeps=None, Tc=None):
        """Set up a game-theoretic linear MPC problem with N agents for set-point tracking. 

        The dynamics are given by

            x(t+1) = A x(t) + B u (t)
                y(t) = C x(t)
                u(t) = u(t-1) + du(t)

        where x(t) is the state vector, du(t) the vector of input increments, and y(t) the output vector at time t. The input vector u(t) is partitioned among N agents as u = [u_1; ...; u_N], where u_i(t) is the control input of agent i, with u_i of dimension sizes[i].

        Each agent i minimizes its finite-horizon cost

            J_i(du,w) = sum_{k=0}^{T-1} (y(k+1)-w)^T Q[i] (y(k+1) - w) + du_i(k)^T Qdu[i] du_i(k) 

        subject to the above dynamics and the following (local) input constraints

            u_i_min <= u_i(k) <= u_i_max
            du_i_min <= du_i(k) <= du_i_max

        and (shared) output constraints

            -sum_i(eps[i]) + y_min <= y(k+1) <= y_max + sum_i(eps[i])

        where eps[i] >= 0 is a slack variable penalized in the cost function with the linear term Qeps[i]*eps[i] to soften the output constraints and prevent infeasibility issues. By default, the constraints are imposed at all time steps k=0 ... T-1, but if a constraint horizon Tc<T is provided, they are only imposed up to time Tc-1.

        The problem is solved via MILP to compute the first input increment du_{0,i} of each agent to apply to the system to close the loop.

        If variational=True at solution time, a variational equilibrium is computed by adding the necessary equality constraints on the multipliers of the shared output constraints.

        If centralized=True at solution time, a centralized MPC problem is solved instead of the game-theoretic one.

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

        def build_qp(A, B, C, Qy, Qdu, Qeps, sizes, N, T, ymin, ymax, umin, umax, dumin, dumax, Tc):
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

            # Final QP problem: each agent i solves
            #
            # min_{du_sequence, eps1...epsN} .5*[du_sequence;eps1..epsN]' H[i] [du_sequence;eps1...epsN]
            #                   + (c + F[i] @ [x0;u(-1);ref])' [du_sequence;eps1...epsN]
            #
            # # s.t. A_con [du_sequence;eps1...epsN] <= b_con + B_con [x0;u(-1)]
            #        lb <= [du_sequence;eps1...epsN] <= ub

            return H, c, F, A_con, b_con, B_con, lb, ub

        H, c, F, A_con, b_con, B_con, lb, ub = build_qp(
            A, B, C, Qy, Qdu, Qeps, sizes, N, T, ymin, ymax, umin, umax, dumin, dumax, self.Tc)

        # Rearrange optimization variables to have all agents' variables together at each time step
        # Original z ordering:
        #   [du_1(0); ...; du_N(0); du_1(1); ...; du_N(1); ...; du_1(T-1); ...; du_N(T-1); eps1; ...; epsN]
        # Desired z_new ordering:
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

        # z >= lb -> P' z_new >= lb -> z_new >= P lb
        self.lb = lb[perm]
        self.ub = ub[perm]
        # remove constraints beyond constraint horizon Tc
        off = 0
        Tc = self.Tc
        for i in range(N):
            si = sizes[i]
            # constraint eps_i>=0 is not removed
            self.lb[off+Tc*si:off+T*si] = -np.inf
            self.ub[off+Tc*si:off+T*si] = np.inf
            off += T*si + 1  # Each agent optimizes du_i(0)..du_i(T-1), eps_i
        self.iperm = np.argsort(perm)  # inverse permutation

    def solve(self, x0, u1, ref, M=1.e4, variational=False, centralized=False, solver='highs'):
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
            If True, solve a centralized MPC problem via QP instead of the game-theoretic one via MILP.
        solver : str, optional
            MILP solver to use ('highs' or 'gurobi').
            
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

        b = self.b_con + self.B_con @ np.hstack((x0, u1))
        c = [self.c[i] + self.F[i] @
             np.hstack((x0, u1, ref)) for i in range(self.N)]

        t0 = time.time()
        if not centralized:
            # Set up and solve GNEP via MILP
            gnep = GNEP_LQ(sizes, self.H, c, F=None, lb=self.lb, ub=self.ub, pmin=None, pmax=None,
                           A=self.A_con, b=b, S=None, D=None, E=None, h=None, M=M, variational=variational, solver=solver)
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
        elapsed_time_build = time.time() - t0

        if not centralized:
            gnep_sol = gnep.solve()
            z = gnep_sol.x
            elapsed_time_solver = gnep_sol.elapsed_time
        else:
            # prob.update(q=c_cen, u=b) # We could speedup by storing prob and reusing previous factorizations
            res = prob.solve()  # Solve QP problem
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
