"""
Generate a random monotone variational nonlinear GNEP with N players, extending a
linear-quadratic vGNE (see lq/generate_random.py) with nonlinear convex shared
constraints of "multi-energy system" style (Example 1):

    log(sum_{k=1}^{n_exp_terms[s]} exp(a_exp_s[k,:] . x + b_exp_s[k])) <= c_exp_s,   s = 1,...,n_exp
    x^T Q_quad_s x + a_quad_s^T x <= c_quad_s,                                       s = 1,...,n_quad

Player i solves:
    min_{x_i}  f_i(x) = (1/2) x^T Q[i] x + c[i]^T x
    subject to  g(x) <= 0,   lb <= x <= ub

with all players sharing the same dimension d (no shared linear inequality/equality
constraints are used in this example, only box constraints). The problem is solved
with four solvers supported by nashopt.GNEP.solve() (toggle each with the RUN_*
flags below), and the results are compared against the (x_star, lambda_nl_star)
used to construct it:
    1) solver="trf": least-squares KKT root-finding (nashopt.GNEP built-in
       solver).
    2) solver="extragrad": Korpelevich's extragradient method, targeting the
       variational GNE via projections (see nl_extragrad.py).
    3) solver="golden_ratio": adaptive Golden Ratio Algorithm on the lifted
       primal-dual VI (see golden_ratio.py).
    4) solver="op_extrapolation": Operator Extrapolation method applied
       directly to the original (non-lifted) variational GNE problem, with
       parameters set from the quadratic costs (see op_extrapolation.py).

(C) 2026 A. Bemporad
"""

import numpy as np
from nashopt.nonlinear.generate_nl_game import generate_nl_game
from nashopt.nonlinear.op_extrapolation import compute_L_mu

RUN_TRF = True
RUN_EXTRAGRAD = True
RUN_GOLDEN_RATIO = True
RUN_OP_EX = True

N = 5
d = 1
dim = [d] * N
gnep, data = generate_nl_game(
    dim=dim,
    m=0,
    m_act=0,
    n_exp=5,
    n_quad=5,
    m_nl_act=2,
    n_box=N*d,
    n_box_act=0,
    seed=0,
    mu=0.1,
    box_slack_min=0.5,
    box_slack_max=1.5,
)

x0 = np.zeros(data["nvar"])

np.set_printoptions(precision=4, suppress=True)

sol = sol_eg = sol_gr = sol_oe = None

# ------------------------------------------------------------------
# 1) Baseline: KKT root-finding (nashopt.GNEP built-in solver)
# ------------------------------------------------------------------
if RUN_TRF:
    sol = gnep.solve(x0=x0, solver="trf", verbose=1)

# ------------------------------------------------------------------
# 2) Korpelevich's extragradient method
# ------------------------------------------------------------------
if RUN_EXTRAGRAD:
    # Costs are exactly quadratic, so the pseudogradient is affine: use the exact
    # Lipschitz constant L (see compute_L_mu) instead of extragrad's default
    # finite-difference estimate, for a step size alpha < 1/L.
    L, _, _ = compute_L_mu(gnep, x0)
    sol_eg = gnep.solve(x0=x0, solver="extragrad", verbose=1,
                         solver_opts={"tol": 1e-9, "maxiter": 5000, "alpha": 0.99 / L})

# ------------------------------------------------------------------
# 3) Adaptive Golden Ratio Algorithm on the lifted primal-dual VI
# ------------------------------------------------------------------
if RUN_GOLDEN_RATIO:
    sol_gr = gnep.solve(x0=x0, solver="golden_ratio", verbose=1,
                         solver_opts={"theta0": 0.3, "max_iter": 50000, "tol": 1e-10})

# ------------------------------------------------------------------
# 4) Operator Extrapolation on the original (non-lifted) VI, with
#    parameters set from the quadratic costs.
# ------------------------------------------------------------------
if RUN_OP_EX:
    sol_oe = gnep.solve(x0=x0, solver="op_extrapolation", verbose=1,
                         solver_opts={"max_iter": 5000, "tol": 1e-10})

print()
print("=== x* comparison ===")
print("x* (build)                :", data["x_star"])
if RUN_TRF:
    print("x* (trf)                  :", sol.x)
if RUN_EXTRAGRAD:
    print("x* (extragrad)            :", sol_eg.x)
if RUN_GOLDEN_RATIO:
    print("x* (golden ratio)         :", sol_gr.x)
if RUN_OP_EX:
    print("x* (op_extrapolation)     :", sol_oe.x)
if RUN_TRF:
    print("||x*_trf - x_star||               =", np.linalg.norm(sol.x - data["x_star"]))
if RUN_EXTRAGRAD:
    print("||x*_extragrad - x_star||         =", np.linalg.norm(sol_eg.x - data["x_star"]))
if RUN_GOLDEN_RATIO:
    print("||x*_golden_ratio - x_star||      =", np.linalg.norm(sol_gr.x - data["x_star"]))
if RUN_OP_EX:
    print("||x*_op_extrapolation - x_star||  =", np.linalg.norm(sol_oe.x - data["x_star"]))

print()
print("=== lambda* comparison (shared nonlinear inequality multipliers) ===")
# m=0 in this example (no shared linear inequality constraints), so all of gnep's shared
# multipliers (gnep.ng = m_nl) correspond to the nonlinear constraints built on top of
# generate_random()'s vGNE.
print("lambda_nl* (build)        :", data["lambda_nl_star"])
if RUN_TRF:
    lam_trf = sol.lam[0][:gnep.ng]
    print("lambda_nl* (trf, agent 0) :", lam_trf)
    print("||lam*_trf - lambda_nl_star||          =", np.linalg.norm(lam_trf - data["lambda_nl_star"]))
if RUN_GOLDEN_RATIO:
    print("lambda_nl* (golden ratio) :", sol_gr.lam)
    print("||lam*_golden_ratio - lambda_nl_star|| =", np.linalg.norm(sol_gr.lam - data["lambda_nl_star"]))
print("(extragrad and op_extrapolation are multiplier-free: no lambda* estimate)")

# check best responses of all agents at the constructed equilibrium
dx, df = gnep.check_equilibrium(data["x_star"])

# ------------------------------------------------------------------
# Summary table (only for the solvers that were actually run):
# CPU time and ||x-x_star|| found by each one.
# ------------------------------------------------------------------
summary = []
if RUN_TRF:
    summary.append(("trf", sol))
if RUN_EXTRAGRAD:
    summary.append(("extragrad", sol_eg))
if RUN_GOLDEN_RATIO:
    summary.append(("golden_ratio", sol_gr))
if RUN_OP_EX:
    summary.append(("op_extrapolation", sol_oe))

print()
print("=== Summary ===")
if summary:
    name_w = max(len(name) for name, _ in summary)
    header = f"{'Solver':<{name_w}}  CPU time (s)  Iterations  ||x-x_star||"
    print(header)
    print("-" * len(header))
    for name, s in summary:
        err = np.linalg.norm(s.x - data["x_star"])
        print(f"{name:<{name_w}}  {s.stats.elapsed_time:12.4f}  {s.stats.kkt_evals:10d}  {err:12.4e}")
else:
    print("(no solver was run: all RUN_* flags are False)")
