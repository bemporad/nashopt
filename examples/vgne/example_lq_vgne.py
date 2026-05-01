# Solve variational GNEP with linear-quadratic structure, using different solvers (Lemke, IPM, ADMM, MILP).
#
# (C) 2026 A. Bemporad

import time
import numpy as np
from nashopt import GNEP_LQ

np.random.seed(1)

N, n_p = 5, 3       # players, variables per player
ncon   = 50         # coupling inequality constraints
nvar   = N * n_p    # total variables

# Local PSD Hessians  Q_i = A_i^T A_i + 0.5 I > 0
Qi = []
for _ in range(N):
    A_tmp = np.random.standard_normal((n_p, n_p))
    Qi.append(A_tmp.T @ A_tmp + 0.5 * np.eye(n_p))

# Asymmetric cross-player coupling (non-potential game when S[i][j] != S[j][i]^T)
#k = 0.05 # monotone
k = 0.5 # non monotone
S = [[None] * N for _ in range(N)]
for i in range(N):
    for j in range(N):
        if i != j:
            S[i][j] = k * np.random.standard_normal((n_p, n_p))

p  = [np.random.standard_normal(n_p) for _ in range(N)]

# Assemble into "nashopt" format
Q_nash = [np.zeros((nvar, nvar)) for _ in range(N)]
c_nash = [np.zeros(nvar)       for _ in range(N)]
for i in range(N):
    si, ei = i * n_p, (i + 1) * n_p
    Q_nash[i][si:ei, si:ei]  = Qi[i]
    c_nash[i][si:ei]         = p[i]
    for j in range(N):
        if i != j:
            Q_nash[i][si:ei, j * n_p:(j + 1) * n_p] = S[i][j]

lb = -5. * np.ones(nvar)
ub  =  5. * np.ones(nvar)

AA = np.random.randn(ncon, nvar)    # coupling constraint directions
bb = np.random.rand(ncon)         # coupling offsets (bb > 0 -> weak coupling)

gnep_lemke = GNEP_LQ(
    dim=[n_p] * N, Q=Q_nash, c=c_nash,
    A=AA, b=bb, lb=lb, ub=ub,
    variational=True, solver="lemke",
)

flag, sigma = gnep_lemke.is_monotone(verbose=False, return_min_eig=True)
print("=" * 62)
print(f"  N={N} players,  n={n_p} vars/player,  {ncon} coupling constraints")
print(f"  Monotonicity sigma = min eig((W+W^T)/2) = {sigma:.4f}"
      f"  ({'monotone (ok)' if sigma >= 0 else 'non-monotone!'})")
print("=" * 62)

res_lemke = gnep_lemke.solve()
x_lemke = res_lemke.x
t_lemke = res_lemke.elapsed_time


gnep_ipm = GNEP_LQ(
    dim=[n_p] * N, Q=Q_nash, c=c_nash,
    A=AA, b=bb, lb=lb, ub=ub,
    variational=True, solver="log_ipm",
)
res_ipm = gnep_ipm.solve()
x_ipm = res_ipm.x
t_ipm = res_ipm.elapsed_time
info_ipm = res_ipm.info

gnep_admm = GNEP_LQ(
    dim=[n_p] * N, Q=Q_nash, c=c_nash,
    A=AA, b=bb, lb=lb, ub=ub,
    variational=True, solver="prox_admm",
)
sol_admm  = gnep_admm.solve()
t_admm    = sol_admm.elapsed_time
x_admm    = sol_admm.x

if N* n_p <= 100:
    gnep_milp = GNEP_LQ(
        dim=[n_p] * N, Q=Q_nash, c=c_nash,
        A=AA, b=bb, lb=lb, ub=ub,
        variational=True, solver="gurobi", M=1e3,
    )
    t2 = time.perf_counter()
    sol_milp  = gnep_milp.solve()
    t_milp    = time.perf_counter() - t2
    x_milp    = sol_milp.x
else:
    t_milp = float("nan")
    x_milp = np.full(nvar, np.nan)

# -------------------------------------------------------------------------------
# 5.  Best-response distances  (zero at a true VE)
# -------------------------------------------------------------------------------
def br_distance(x_cand, gnep_obj):
    """||x - x_br||_2: each component replaced by its best response."""
    x_br = np.zeros_like(x_cand)
    for i in range(N):
        si, ei = i * n_p, (i + 1) * n_p
        try:
            x_br[si:ei] = gnep_obj.best_response(i, x_cand).x[si:ei]
        except Exception as e:
            print(f"  Warning: best response for player {i+1} failed")
            x_br[si:ei] = np.full(n_p, np.nan)  # mark as NaN if BR fails
    return np.linalg.norm(x_cand - x_br)

d_lemke = br_distance(x_lemke, gnep_admm) if res_lemke.success else float("nan")
d_ipm   = br_distance(x_ipm,       gnep_admm)
d_admm  = br_distance(x_admm,      gnep_admm)
d_milp  = br_distance(x_milp,      gnep_admm)

STATUS_LEMKE = {0: "solution", 1: "iter-limit", 2: "ray-term"}

print(f"\n{'-'*62}")
print(f"  {'Method':<14}  {'CPU (s)':>8}  {'||x-x_br||':>10}  {'Status / info'}")
print(f"{'-'*62}")
print(f"  {'Lemke':<14}  {t_lemke:8.4f}  {d_lemke:10.2e}"
      f"  {STATUS_LEMKE[res_lemke.status]},"
      f" {res_lemke.num_iter} pivots")
print(f"  {'Log-IPM':<14}  {t_ipm:8.4f}  {d_ipm:10.2e}"
      f"  {'converged (ok)' if info_ipm['converged'] else 'NOT converged'},"
      f" {info_ipm['outer_iters']} outer iters,  mu={info_ipm['mu']:.1e}")
print(f"  {'Prox-ADMM':<14}  {t_admm:8.4f}  {d_admm:10.2e}")
print(f"  {'MILP':<14}  {t_milp:8.4f}  {d_milp:10.2e}")
print(f"{'-'*62}")

print("\n  Solution differences:")
print(f"  ||x_Lemke  - x_IPM || = {np.linalg.norm(x_lemke - x_ipm):.3e}")
print(f"  ||x_Lemke  - x_ADMM|| = {np.linalg.norm(x_lemke - x_admm):.3e}")
print(f"  ||x_Lemke  - x_MILP|| = {np.linalg.norm(x_lemke - x_milp):.3e}")

if n_p <= 3:
    print(f"\n{'-'*62}")
    print(f"  Player strategies x*")
    print(f"{'-'*62}")
    print(f"  {'Player':>8}  {'Lemke':^24}  {'Log-IPM':^24}  {'Prox-ADMM':^24}  {'MILP':^24}")
    print(f"  {'-'*8}  {'-'*24}  {'-'*24}  {'-'*24}  {'-'*24}")
    for i in range(N):
        si, ei = i * n_p, (i + 1) * n_p
        xl = "  ".join(f"{v:6.3f}" for v in x_lemke[si:ei])
        xi = "  ".join(f"{v:6.3f}" for v in x_ipm[si:ei])
        xa = "  ".join(f"{v:6.3f}" for v in x_admm[si:ei])
        xm = "  ".join(f"{v:6.3f}" for v in x_milp[si:ei])
        print(f"  Player {i+1:2d}  [{xl}]  [{xi}]  [{xa}]  [{xm}]")

print("\n  Objective values f_i(x*):")
print(f"  {'Player':>8}  {'Lemke':>12}  {'Log-IPM':>12}  {'Prox-ADMM':>12}  {'MILP':>12}")
print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")
for i in range(N):
    fl = gnep_admm.cost_eval(i, x_lemke)
    fi = gnep_admm.cost_eval(i, x_ipm)
    fa = gnep_admm.cost_eval(i, x_admm)
    fm = gnep_admm.cost_eval(i, x_milp)
    print(f"  Player {i+1:2d}  {fl:+12.6f}  {fi:+12.6f}  {fa:+12.6f}  {fm:+12.6f}")
