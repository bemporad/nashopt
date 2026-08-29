""" Generate a random linear-quadratic variational generalized Nash equilibrium problem (LQ-GNEP)
with N players of dimensions (n1,n2,...,nN) and given monotonicity constant mu, in accordance with
Lemma 4.2 in [1].

[1] A. Bemporad, T. Tatarenko, "Learning Parametric Monotone Games," arXiv preprint, 2026.

(C) 2026 A. Bemporad
"""

import numpy as np
from .gnep_lq import GNEP_LQ

def generate_random(
    dim: list[int],
    m: int,
    m_act: int,
    q: int = 0,
    seed: int | None = None,
    mu: float = 0., # desired monotonicity constant, set mu = 0 for merely monotone GNEs
    inactive_slack_min: float = 0.5,
    inactive_slack_max: float = 1.5,
    lambda_min: float = 0.5,
    lambda_max: float = 1.5,
    mu_scale: float = 1.0,
    solver: str = "dr_daqp"
    ):
    """
    Generate a linear-quadratic generalized Nash equilibrium problem.

    There are N = len(dim) agents, agent i having dim[i] variables, so the aggregate
    variable is

        x = col(x_1, ..., x_N) in R^nvar, x_i in R^(dim[i]), nvar = sum(dim).

    Each agent i minimizes

        J_i(x) = 0.5 x^T Q_i x + c_i^T x_i

    w.r.t. x_i given x_-i, where Q_i is symmetric positive semidefinite and c_i is a linear term.

    The shared constraints are

        A x <= b,
        E x  = h.

    The function constructs a known variational GNE x_star with exactly m_act active inequality constraints.

    Parameters
    ----------
    dim : list of int
        Number of decision variables per agent, dim[i] for agent i. N = len(dim).
    m : int
        Number of shared inequality constraints.
    m_act : int
        Number of active inequalities at the constructed variational GNE.
    q : int, default 0
        Number of shared equality constraints.
    seed : int or None
        Random seed.
    mu : float
        Desired lower bound on lambda_min(0.5*(G+G.T)), i.e., the monotonicity constant.
    inactive_slack_min, inactive_slack_max : float
        Range for strictly positive slacks of inactive inequalities.
    lambda_min, lambda_max : float
        Range for strictly positive active inequality multipliers.
    mu_scale : float
        Scale for equality multipliers.

    Returns
    -------
    data : dict
        Dictionary containing the generated LQ-GNEP data.
    """

    N = len(dim)
    if N <= 0:
        raise ValueError("dim must be nonempty.")
    if any(ni <= 0 for ni in dim):
        raise ValueError("all entries of dim must be positive.")
    if m < 0:
        raise ValueError("m must be nonnegative.")
    if q < 0:
        raise ValueError("q must be nonnegative.")
    if not (0 <= m_act <= m):
        raise ValueError("m_act must satisfy 0 <= m_act <= m.")

    nvar = sum(dim)
    offsets = np.concatenate(([0], np.cumsum(dim)))
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # 1. Choose a target variational equilibrium.
    # ------------------------------------------------------------
    x_star = rng.standard_normal(nvar)

    # ------------------------------------------------------------
    # 2. Generate quadratic cost matrices Q_i.
    #
    # Q = C'C + D - D' + (mu-lambda_min(C'C))*I
    # ------------------------------------------------------------
    
    C = rng.standard_normal((nvar, nvar))
    mask_C = np.triu(np.ones((nvar, nvar), dtype=bool))
    C = np.where(mask_C, C, 0.0) # Make block upper triangular
    CtC = C.T @ C
    
    block_id = np.repeat(np.arange(N), dim)
    block_mask_D = block_id[:, None] < block_id[None, :]
    D = rng.standard_normal((nvar, nvar))
    D = np.where(block_mask_D, D, 0.0) # Make block strictly lower triangular
        
    lambda_min = np.linalg.eigvalsh(CtC).min()
    G = CtC + D - D.T + (mu - lambda_min) * np.eye(nvar) # Pseudogradient matrix 
    
    Q_agents = []
    for i in range(N):
        Qi = np.zeros((nvar, nvar))
        si, ei = offsets[i], offsets[i + 1]
        Qi[si:ei, si:ei] = G[si:ei, si:ei]
        Qi[si:ei, :si]   = 2.0 * G[si:ei, :si]
        Qi[si:ei, ei:]   = 2.0 * G[si:ei, ei:]
        Q_agents.append(Qi)

    lam_min_shifted = np.linalg.eigvalsh(0.5 * (G + G.T)).min()
    print(f"Monotonicity check: lambda_min(0.5*(G+G.T)) = {lam_min_shifted:.4e}")
    lam_min_Q = [np.linalg.eigvalsh(Q_agents[i][offsets[i]:offsets[i + 1], offsets[i]:offsets[i + 1]]).min() for i in range(N)]
    print(f"Monotonicity check: lambda_min(Q_i) = {[f'{l:.4e}' for l in lam_min_Q]}")

    # ------------------------------------------------------------
    # 5. Generate inequality constraints A x <= b.
    # ------------------------------------------------------------
    A = rng.standard_normal((m, nvar))

    active = np.arange(m_act)
    inactive = np.arange(m_act, m)

    b = A @ x_star

    if m_act < m:
        slacks = rng.uniform(
            inactive_slack_min,
            inactive_slack_max,
            size=m - m_act,
        )
        b[inactive] += slacks
    else:
        slacks = np.zeros(0)

    # ------------------------------------------------------------
    # 6. Generate equality constraints E x = h.
    # ------------------------------------------------------------
    if q > 0:
        E = rng.standard_normal((q, nvar))
        h = E @ x_star
        mu_star = mu_scale * rng.standard_normal(q)
    else:
        E = np.zeros((0, nvar))
        h = np.zeros(0)
        mu_star = np.zeros(0)

    # ------------------------------------------------------------
    # 7. Choose inequality multipliers.
    #
    # Active constraints get strictly positive multipliers.
    # Inactive constraints get zero multipliers.
    # ------------------------------------------------------------
    lambda_star = np.zeros(m)

    if m_act > 0:
        lambda_star[active] = rng.uniform(lambda_min, lambda_max, size=m_act)

    # ------------------------------------------------------------
    # 8. Choose linear cost terms c so that stationarity holds:
    #
    #     G x_star + c + A.T lambda_star + E.T mu_star = 0.
    #
    # Therefore:
    #
    #     c = -G x_star - A.T lambda_star - E.T mu_star.
    # ------------------------------------------------------------
    c = -G @ x_star - A.T @ lambda_star - E.T @ mu_star

    # Split c into agent-wise linear terms c_i.
    c_agents = []
    for i in range(N):
        si, ei = offsets[i], offsets[i + 1]
        ci = np.random.standard_normal(nvar)
        ci[si:ei] = c[si:ei]
        c_agents.append(ci)

    gnep_lq_prob = GNEP_LQ(list(dim), Q_agents, c_agents, A=A, b=b, Aeq=E, beq=h, variational=True, solver=solver)
    
    # ------------------------------------------------------------
    # 9. Diagnostics.
    # ------------------------------------------------------------
    ineq_residual = A @ x_star - b
    eq_residual = E @ x_star - h
    stationarity = G @ x_star + c + A.T @ lambda_star + E.T @ mu_star

    detected_active = np.where(np.abs(ineq_residual) <= 1e-8)[0]

    data = {
        # Dimensions
        "N": N,
        "dims": dim,
        "nvar": nvar,
        "m": m,
        "m_act": m_act,
        "q": q,

        # Pseudogradient matrix
        "G": G,

        # Constructed vGNE
        "x_star": x_star,
        "lambda_star": lambda_star,
        "mu_star": mu_star,

        # Active-set information
        "active": active,
        "inactive": inactive,
        "inactive_slacks": slacks,

        # Diagnostics
        "ineq_residual": ineq_residual,
        "eq_residual": eq_residual,
        "stationarity_residual": stationarity,
        "stationarity_residual_norm": np.linalg.norm(stationarity),
        "eq_residual_norm": np.linalg.norm(eq_residual),
        "detected_active": detected_active,
    }

    return gnep_lq_prob, data