# NashOpt: A Python package for computing Generalized Nash Equilibria (GNE) in noncooperative games.
#
# Nonlinear Parametric GNEP problem. 
#
# (C) 2025-2026 Alberto Bemporad

import numpy as np
import jax
import jax.numpy as jnp
from jaxopt import ScipyBoundedMinimize
from scipy.optimize import least_squares
from types import SimpleNamespace
from functools import partial
import time
from .._common.report import eval_residual
from .gnep_base import GNEP

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
