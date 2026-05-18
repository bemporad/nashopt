"""
Benchmark strongly monotone variational GNEP (vGNE) LQ problems with N players, each with n decision variables. 

Player i solves:
    min_{x_i}  f_i(x) = (1/2) x^T Q_i x + c_i^T x
    subject to  A x <= b,   A_eq x = b_eq,   lb <= x <= ub

Strong monotonicity is enforced by computing

  sigma = lambda_min((G + G^T)/2)

and, if sigma < eps, adding (eps - sigma)*I_n to every diagonal block Q_i^{(ii)}.
This shifts all eigenvalues of (G + G^T)/2 by (eps - sigma) exactly, giving
lambda_min((G_new + G_new^T)/2) = eps > 0.

This example has been reported in:

[1] A. Bemporad, "GoldNash: A Goldfarb-Idnani Variant for Strongly Monotone Linear-Quadratic Games," 
arXiv preprint 2605.16002, 2026.   https://arxiv.org/abs/2605.16002

(C) 2026 A. Bemporad
"""

import numpy as np
import warnings
from nashopt import GNEP_LQ

N_INSTANCES = 100

SOLVERS = ["lemke", "lemke_dual", "goldnash", "dr_daqp", "prox_admm", "log_ipm"]

# N values; n=5, m=2*N*n; q varies per run
CONFIGS = [30] #[2, 3, 5, 10, 20, 30, 50, 100]

RUNS = [
    ("no equalities ($q=0$)",          lambda N: (N, 5, 2*N*5,    0)),
    ("equalities ($q=\\lfloor N/2 \\rfloor$)", lambda N: (N, 5, 2*N*5, N//2)),
]

MONOTONE_EPS = 1e-4   # target lambda_min((G+G^T)/2) after diagonal shift

def make_instance(N, n, m, q, rng):
    """Generate one strongly monotone vGNE LQ instance."""
    nvar = N * n

    # Q_i = B_i^T B_i (symmetric PSD, full coupling across all agents)
    Q_sym = []
    for _ in range(N):
        B = rng.standard_normal((nvar, nvar))
        Q_sym.append(B.T @ B)

    c_nash = [5.0 * rng.standard_normal(nvar) for _ in range(N)] # linear terms for each player

    # Build pseudogradient G: block row i is Q_sym[i][si:ei, :]
    G = np.zeros((nvar, nvar))
    for i in range(N):
        si, ei = i * n, (i + 1) * n
        G[si:ei, :] = Q_sym[i][si:ei, :]

    # Enforce strong monotonicity: shift diagonal blocks so lambda_min = MONOTONE_EPS
    lam_min = np.linalg.eigvalsh(0.5 * (G + G.T)).min()
    shift = MONOTONE_EPS + min(-lam_min,0.) 
    if shift > 0:
        for i in range(N):
            si, ei = i * n, (i + 1) * n
            Q_sym[i][si:ei, si:ei] += shift * np.eye(n)

    lb = - rng.uniform(0.1, 5., nvar)
    ub =  rng.uniform(0.1, 5., nvar)

    x_feas = rng.uniform(0., .1, nvar)*ub+lb # Feasible point
    AA = rng.standard_normal((m, nvar))    
    bb = np.max(AA@x_feas) + rng.uniform(0.1, 0.5, m)

    # Coupling equalities: set beq = Aeq @ x_feas for a feasible x_feas
    if q > 0:
        Aeq    = rng.standard_normal((q, nvar))
        beq    = Aeq @ x_feas
    else:
        Aeq = None
        beq = None

    return Q_sym, c_nash, AA, bb, lb, ub, Aeq, beq


def solve_one(N, n, Q_sym, c_nash, AA, bb, lb, ub, Aeq, beq, solver):
    """
    Returns (elapsed_time, success).
    elapsed_time is NaN when the solver is skipped, raises, or returns None.
    """
    if solver == "lemke" and ((Aeq is not None) or N>50):
        return float("nan"), False  # lemke does not support equality constraints
    if solver == "prox_admm" and N > 20:
        return float("nan"), False
    if solver == "log_ipm" and N > 20:
        return float("nan"), False

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gnep = GNEP_LQ(
                dim=[n] * N, Q=Q_sym, c=c_nash,
                A=AA, b=bb, lb=lb, ub=ub, Aeq=Aeq, beq=beq,
                variational=True, solver=solver,
            )
            res = gnep.solve()

        if res is None:
            return float("nan"), False

        t = float(res.elapsed_time)

        # Determine success from solver-specific fields
        if hasattr(res, "success"):
            ok = bool(res.success)
        elif hasattr(res, "info") and isinstance(res.info, dict):
            ok = bool(res.info.get("converged", True))
        else:
            ok = True

        return t, ok

    except Exception:
        return float("nan"), False

def warmup():
    """Run each solver once on a tiny problem to absorb import/JIT overhead."""
    print("Warming up solvers...")
    rng = np.random.default_rng(0)
    Q_sym, c_nash, AA, bb, lb, ub, Aeq, beq = make_instance(2, 2, 4, 0, rng)
    for solver in SOLVERS:
        solve_one(2, 2, Q_sym, c_nash, AA, bb, lb, ub, Aeq, beq, solver)
    print("  done.\n")


def run_benchmark(expand_fn, rng):
    """Run all configs/solvers for one expand function; return (times, success)."""
    times   = {s: [[] for _ in CONFIGS] for s in SOLVERS}
    success = {s: [[] for _ in CONFIGS] for s in SOLVERS}

    for cfg_idx, N_raw in enumerate(CONFIGS):
        N, n, m, q = expand_fn(N_raw)
        nvar = N * n
        print(f"\n{'='*66}")
        print(f"  Config {cfg_idx + 1}/{len(CONFIGS)}: "
              f"N={N}, n={n}, m={m}, q={q}  (nvar={nvar})")
        print(f"{'='*66}")

        for inst in range(N_INSTANCES):
            if (inst + 1) % 5 == 0:
                print(f"  instance {inst + 1}/{N_INSTANCES}")

            Q_sym, c_nash, AA, bb, lb, ub, Aeq, beq = make_instance(
                N, n, m, q, rng
            )

            for solver in SOLVERS:
                t, ok = solve_one(
                    N, n, Q_sym, c_nash, AA, bb, lb, ub, Aeq, beq, solver
                )
                times[solver][cfg_idx].append(t)
                success[solver][cfg_idx].append(ok)

        # Print per-config timing summary to console
        print(f"\n  {'Solver':<14}  {'mean(s)':>10}  {'worst(s)':>10}  {'succ':>8}")
        print(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*8}")
        for solver in SOLVERS:
            ts    = times[solver][cfg_idx]
            scs   = success[solver][cfg_idx]
            valid = [t for t in ts if not np.isnan(t)]
            n_ok  = sum(scs)
            if valid:
                print(f"  {solver:<14}  {np.mean(valid):10.4f}  {np.max(valid):10.4f}"
                      f"  {n_ok:3d}/{N_INSTANCES}")
            else:
                print(f"  {solver:<14}  {'--':>10}  {'--':>10}  {n_ok:3d}/{N_INSTANCES}")

    return times, success


def main():
    warmup()
    rng = np.random.default_rng(42)

    all_results = []
    for run_label, expand_fn in RUNS:
        print(f"\n{'#'*66}")
        print(f"  RUN: {run_label}")
        print(f"{'#'*66}")
        times, _ = run_benchmark(expand_fn, rng)
        all_results.append((run_label, times))
        print()

if __name__ == "__main__":
    main()
