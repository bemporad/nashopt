"""
Solve the generalized Nash equilibrium problem described in [1, Fig. 6], originally proposed in [2, Section 5] for n=20 agents.

[1] F. Fabiani and A. Bemporad, “An active learning method for solving competitive multi-agent decision-making and control problems,” 2024, http://arxiv.org/abs/2212.12561.

[2] F. Salehisadaghiani, W. Shi, and L. Pavel, “An ADMM approach to the problem of distributed Nash equilibrium seeking.” CoRR, 2017.

(C) 2025-2026 A. Bemporad
"""
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from nashopt import GNEP

N = 10  # number of agents
sizes = [1]*N  # n agents of dimension 1
nvar = np.sum(sizes)


@jax.jit
def cost(x, i):
    # Cost function minimized by agent #i, i=0,...,N-1
    ci = N*(1.+i/2.)
    return ci*x[i]-x[i]*(60.*N-jnp.sum(x))


f = [partial(cost, i=i) for i in range(N)]

lb = 7. * np.ones(nvar)
ub = 100. * np.ones(nvar)

gnep = GNEP(sizes, f=f, lb=lb, ub=ub)

np.set_printoptions(precision=4, suppress=True)

x0 = (lb+ub)/2.

# --- 1) Solve via the default KKT-based solver ---
print("Solving GNEP with N =", N, "agents (default KKT solver) ... ", end="")
sol = gnep.solve(x0, verbose=0)
x_star, lam_star, residual, stats = sol.x, sol.lam, sol.res, sol.stats
print("done.")

print("=== GNE solution (KKT solver) ===")
print(f"x = {x_star}")
for i in range(gnep.N):
    print(f"lambda[{i}] = {lam_star[i]}")

print(f"KKT residual norm = {float(jnp.linalg.norm(residual)): 10.7g}")
print(f"KKT evaluations   = {int(stats.kkt_evals): 3d}")
print(f"Elapsed time:       {stats.elapsed_time: .2f} seconds")

# check best responses of all agents at the computed GNE
dx, df = gnep.check_equilibrium(x_star)

# --- 2) Solve again via Korpelevich's extragradient method ---
# Estimate the step size alpha < 1/L, where L is the Lipschitz constant of the
# pseudogradient F(x) = (nabla_{x_i} f_i(x))_i, from the spectral norm of its
# Jacobian at a reference point (exact here, since F is affine in x).
def pseudogradient(x):
    xj = jnp.asarray(x)
    return jnp.concatenate([gnep.df[i](xj[gnep.i1[i]:gnep.i2[i]], xj) for i in range(gnep.N)])


xref = 0.5 * (lb + ub)
L = np.linalg.norm(np.asarray(jax.jacobian(pseudogradient)(jnp.asarray(xref))), 2)
alpha = 0.99 / L

print("\nSolving GNEP with N =", N, "agents (extragradient method) ... ", end="")

projection_solver = "trf" # usually faster than "ipopt"
# projection_solver = "ipopt" 
sol_eg = gnep.solve(x0, solver="extragrad", verbose=0,
                     extragrad_opts={"tol": 1e-9, "maxiter": 5000, "alpha": alpha, "projection_solver": projection_solver})
x_star_eg, stats_eg = sol_eg.x, sol_eg.stats
print("done.")

print("=== GNE solution (extragradient method) ===")
print(f"x = {x_star_eg}")

print(f"Final step norm   = {float(sol_eg.norm_residual): 10.7g}")
print(f"Iterations        = {int(stats_eg.kkt_evals): 3d}")
print(f"Elapsed time:       {stats_eg.elapsed_time: .2f} seconds")

# check best responses of all agents at the computed GNE
dx_eg, df_eg = gnep.check_equilibrium(x_star_eg)

print(f"\n||x_KKT - x_extragrad|| = {float(jnp.linalg.norm(x_star - x_star_eg)): 10.7g}")
