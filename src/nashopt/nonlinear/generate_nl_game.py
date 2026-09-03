""" Generate a random nonlinear generalized Nash equilibrium problem (NL-GNEP)
with N players of common dimension d and given monotonicity constant mu.

The cost functions are generated exactly as in the linear-quadratic case (see
lq/generate_random.py), i.e., convex quadratic and monotone with pseudogradient
constant mu, in accordance with Lemma 4.2 in [1].

The shared constraints are nonlinear convex, of the form described in the
"multi-energy system" example (Example 1): for a joint strategy
x = col(x_1,...,x_N), x_i in R^d,

    sum_{i=1}^N exp(a_l^T x_i) <= C_l,          l = 1,...,n_exp   (transmission lines)
    sum_{i=1}^N ||B_r x_i||^2  <= D_r,           r = 1,...,n_norm (transformers)
    sum_{i=1}^N x_i^T H_s x_i  <= E_s,           s = 1,...,n_quad (voltage stability)

with H_s symmetric positive semidefinite, so that each constraint is convex in x
(it is a sum, over the agents, of a convex function of each agent's own block).
Box constraints lb <= x <= ub are also generated, with a subset of variables
active at the constructed equilibrium.

[1] A. Bemporad, T. Tatarenko, "Learning Parametric Monotone Games," arXiv preprint 2609.02494, 2026. https://arxiv.org/abs/2609.02494. 

(C) 2026 A. Bemporad
"""

import numpy as np
import jax
import jax.numpy as jnp
from .gnep_base import GNEP

jax.config.update("jax_enable_x64", True)


def generate_nl_game(
    N: int = 2,
    d: int = 1,
    n_exp: int = 0,
    n_norm: int = 0,
    n_quad: int = 0,
    m_act: int = 0,
    n_box: int = 0,
    n_box_act: int = 0,
    k_norm: int | None = None,
    seed: int | None = None,
    mu: float = 0., # desired monotonicity constant, set mu = 0 for merely monotone GNEs
    inactive_slack_min: float = 0.5,
    inactive_slack_max: float = 1.5,
    lambda_min: float = 0.5,
    lambda_max: float = 1.5,
    box_slack_min: float = 0.5,
    box_slack_max: float = 1.5,
    a_scale: float = 1.0,
    B_scale: float = 1.0,
    H_scale: float = 1.0,
):
    """
    Generate a nonlinear generalized Nash equilibrium problem with convex quadratic,
    monotone costs and nonlinear convex shared constraints.

    There are N agents, all of the same dimension d (required),
    so the aggregate variable is

        x = col(x_1, ..., x_N) in R^nvar, x_i in R^d, nvar = N*d.

    Each agent i minimizes

        J_i(x) = 0.5 x^T Q_i x + c_i^T x_i

    w.r.t. x_i given x_-i, where Q_i is symmetric positive semidefinite and c_i is a
    linear term (built exactly as in lq/generate_random.py).

    The shared nonlinear inequality constraints are g(x) <= 0, where g stacks:

        n_exp   constraints  sum_i exp(a_l^T x_i) - C_l           (transmission lines)
        n_norm  constraints  sum_i ||B_r x_i||^2 - D_r            (transformers)
        n_quad  constraints  sum_i x_i^T H_s x_i - E_s            (voltage stability)

    for a total of m = n_exp + n_norm + n_quad shared inequality constraints, of which
    m_act are constructed to be active (tight) at the equilibrium x_star.

    Box constraints lb <= x <= ub are generated for n_box randomly chosen entries of x
    (the remaining entries are unbounded), of which n_box_act are active at x_star
    (each one active either at its lower or upper bound, chosen at random).

    Parameters
    ----------
    N : int
        Number of agents (players).
    d : int
        Number of decision variables per agent. All agents share the same dimension d.
    n_exp : int
        Number of exponential ("transmission line") shared inequality constraints.
    n_norm : int
        Number of quadratic-norm ("transformer") shared inequality constraints.
    n_quad : int
        Number of quadratic-form ("voltage stability") shared inequality constraints.
    m_act : int
        Number of active shared inequality constraints at the constructed GNE, out of
        m = n_exp + n_norm + n_quad. Which of the m constraints are active is chosen
        at random.
    n_box : int
        Number of decision variables (out of nvar) that get finite box constraints.
        The remaining nvar - n_box variables are left unbounded.
    n_box_act : int
        Number of box constraints active at x_star, out of n_box (each active either
        at its lower or upper bound, chosen at random).
    k_norm : int or None
        Output dimension of the matrices B_r used in the "transformer" constraints.
        Defaults to d if None.
    seed : int or None
        Random seed.
    mu : float
        Desired lower bound on lambda_min(0.5*(G+G.T)), i.e., the monotonicity constant
        of the pseudogradient.
    inactive_slack_min, inactive_slack_max : float
        Range for strictly positive slacks of inactive shared inequality constraints.
    lambda_min, lambda_max : float
        Range for strictly positive multipliers of active shared inequality constraints
        and active box constraints.
    box_slack_min, box_slack_max : float
        Range for strictly positive slacks of the inactive side of box constraints
        (both sides, for variables with inactive box constraints).
    a_scale : float
        Scale of the random vectors a_l defining the exponential constraints.
    B_scale : float
        Scale of the random matrices B_r defining the quadratic-norm constraints.
    H_scale : float
        Scale of the random matrices used to build the PSD matrices H_s defining the
        quadratic-form constraints.

    Returns
    -------
    gnep : GNEP
        The generated nonlinear GNEP object.
    data : dict
        Dictionary containing the generated NL-GNEP data.
    """

    if N < 2:
        raise ValueError("N must be greater or equal than 2.")
    if d <= 0:
        raise ValueError("dim must be positive.")
    dim = [d] * N  # all agents have the same dimension d

    m = n_exp + n_norm + n_quad
    if not (0 <= m_act <= m):
        raise ValueError("m_act must satisfy 0 <= m_act <= m.")
    if k_norm is None:
        k_norm = d

    nvar = N * d
    if not (0 <= n_box <= nvar):
        raise ValueError("n_box must satisfy 0 <= n_box <= nvar.")
    if not (0 <= n_box_act <= n_box):
        raise ValueError("n_box_act must satisfy 0 <= n_box_act <= n_box.")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # 1. Choose a target variational equilibrium.
    # ------------------------------------------------------------
    x_star = rng.standard_normal(nvar)

    # ------------------------------------------------------------
    # 2. Generate quadratic cost matrices Q_i, exactly as in the LQ case.
    #
    # Q = C'C + D - D' + (mu-lambda_min(C'C))*I
    # ------------------------------------------------------------
    C = rng.standard_normal((nvar, nvar))
    mask_C = np.triu(np.ones((nvar, nvar), dtype=bool))
    C = np.where(mask_C, C, 0.0)  # Make block upper triangular
    CtC = C.T @ C

    block_id = np.repeat(np.arange(N), dim)
    block_mask_D = block_id[:, None] < block_id[None, :]
    D = rng.standard_normal((nvar, nvar))
    D = np.where(block_mask_D, D, 0.0)  # Make block strictly lower triangular

    lambda_min_CtC = np.linalg.eigvalsh(CtC).min()
    G = CtC + D - D.T + (mu - lambda_min_CtC) * np.eye(nvar)  # Pseudogradient matrix

    offsets = np.concatenate(([0], np.cumsum(dim)))
    Q_agents = []
    for i in range(N):
        Qi = np.zeros((nvar, nvar))
        si, ei = offsets[i], offsets[i + 1]
        Qi[si:ei, si:ei] = G[si:ei, si:ei]
        Qi[si:ei, :si] = 2.0 * G[si:ei, :si]
        Qi[si:ei, ei:] = 2.0 * G[si:ei, ei:]
        Q_agents.append(0.5 * (Qi + Qi.T))  # symmetrize: grad_i(0.5 x'Qi x) = G[si:ei,:] @ x

    lam_min_shifted = np.linalg.eigvalsh(0.5 * (G + G.T)).min()
    print(f"Monotonicity check: lambda_min(0.5*(G+G.T)) = {lam_min_shifted:.4e}")

    # ------------------------------------------------------------
    # 3. Generate nonlinear convex shared inequality constraints.
    # ------------------------------------------------------------
    a_mat = rng.standard_normal((n_exp, d)) * a_scale / np.sqrt(d)
    B_arr = rng.standard_normal((n_norm, k_norm, d)) * B_scale / np.sqrt(d)
    M_arr = rng.standard_normal((n_quad, d, d)) * H_scale / np.sqrt(d)
    H_arr = np.einsum("sjk,sjl->skl", M_arr, M_arr)  # H_s = M_s^T M_s, PSD

    a_mat_j = jnp.asarray(a_mat)
    B_arr_j = jnp.asarray(B_arr)
    H_arr_j = jnp.asarray(H_arr)

    def phi(x):
        """Raw values of the m shared constraints (before subtracting the RHS)."""
        X = x.reshape(N, d)
        parts = []
        if n_exp > 0:
            parts.append(jnp.sum(jnp.exp(X @ a_mat_j.T), axis=0))          # (n_exp,)
        if n_norm > 0:
            BX = jnp.einsum("rkd,nd->rnk", B_arr_j, X)
            parts.append(jnp.sum(BX ** 2, axis=(1, 2)))                    # (n_norm,)
        if n_quad > 0:
            quad = jnp.einsum("nj,sjk,nk->sn", X, H_arr_j, X)
            parts.append(jnp.sum(quad, axis=1))                           # (n_quad,)
        if parts:
            return jnp.concatenate(parts)
        return jnp.zeros(0)

    x_star_j = jnp.asarray(x_star)
    phi_star = np.asarray(phi(x_star_j))

    # ------------------------------------------------------------
    # 4. Fix RHS = phi(x_star): active constraints tight, inactive ones with a
    #    random positive slack.
    # ------------------------------------------------------------
    perm = rng.permutation(m)
    active = perm[:m_act]
    inactive = perm[m_act:]

    RHS = phi_star.copy()
    if m_act < m:
        slacks = rng.uniform(inactive_slack_min, inactive_slack_max, size=m - m_act)
        RHS[inactive] += slacks
    else:
        slacks = np.zeros(0)

    RHS_j = jnp.asarray(RHS)

    def g(x):
        return phi(x) - RHS_j

    # ------------------------------------------------------------
    # 5. Choose multipliers for the active shared inequality constraints.
    # ------------------------------------------------------------
    lambda_star = np.zeros(m)
    if m_act > 0:
        lambda_star[active] = rng.uniform(lambda_min, lambda_max, size=m_act)

    dphi_star = np.asarray(jax.jacobian(phi)(x_star_j))  # (m, nvar), same as dg(x_star)

    # ------------------------------------------------------------
    # 6. Generate box constraints lb <= x <= ub, with n_box_act active at x_star.
    # ------------------------------------------------------------
    lb = -np.inf * np.ones(nvar)
    ub = np.inf * np.ones(nvar)
    lam_lb = np.zeros(nvar)
    lam_ub = np.zeros(nvar)

    box_idx = rng.choice(nvar, size=n_box, replace=False)
    box_act_idx = rng.choice(box_idx, size=n_box_act, replace=False) if n_box > 0 else np.zeros(0, dtype=int)
    box_inact_idx = np.setdiff1d(box_idx, box_act_idx)

    for j in box_act_idx:
        j = int(j)
        if rng.uniform() < 0.5:
            # active lower bound
            lb[j] = x_star[j]
            ub[j] = x_star[j] + rng.uniform(box_slack_min, box_slack_max)
            lam_lb[j] = rng.uniform(lambda_min, lambda_max)
        else:
            # active upper bound
            ub[j] = x_star[j]
            lb[j] = x_star[j] - rng.uniform(box_slack_min, box_slack_max)
            lam_ub[j] = rng.uniform(lambda_min, lambda_max)

    for j in box_inact_idx:
        j = int(j)
        lb[j] = x_star[j] - rng.uniform(box_slack_min, box_slack_max)
        ub[j] = x_star[j] + rng.uniform(box_slack_min, box_slack_max)

    # ------------------------------------------------------------
    # 7. Choose linear cost terms c so that KKT stationarity holds:
    #
    #     G x_star + c + dg(x_star)^T lambda_star - lam_lb + lam_ub = 0.
    #
    # Therefore:
    #
    #     c = -G x_star - dg(x_star)^T lambda_star + lam_lb - lam_ub.
    # ------------------------------------------------------------
    c = -G @ x_star - dphi_star.T @ lambda_star + lam_lb - lam_ub

    # Split c into agent-wise linear terms c_i (only the agent's own block matters).
    c_agents = []
    for i in range(N):
        si, ei = offsets[i], offsets[i + 1]
        ci = rng.standard_normal(nvar)
        ci[si:ei] = c[si:ei]
        c_agents.append(ci)

    # ------------------------------------------------------------
    # 8. Build the jax cost functions f_i(x) = 0.5 x^T Q_i x + c_i^T x.
    # ------------------------------------------------------------
    Q_agents_j = [jnp.asarray(Qi) for Qi in Q_agents]
    c_agents_j = [jnp.asarray(ci) for ci in c_agents]

    def make_f(Qi, ci):
        @jax.jit
        def f(x):
            return 0.5 * x @ Qi @ x + ci @ x
        return f

    f_agents = [make_f(Q_agents_j[i], c_agents_j[i]) for i in range(N)]

    lb_j = np.where(np.isfinite(lb), lb, -np.inf)
    ub_j = np.where(np.isfinite(ub), ub, np.inf)

    gnep = GNEP(list(dim), f_agents, g=g, ng=m, lb=lb_j, ub=ub_j, variational=True)

    # ------------------------------------------------------------
    # 9. Diagnostics.
    # ------------------------------------------------------------
    ineq_residual = phi_star - RHS
    box_lb_residual = lb[np.isfinite(lb)] - x_star[np.isfinite(lb)]
    box_ub_residual = x_star[np.isfinite(ub)] - ub[np.isfinite(ub)]
    stationarity = G @ x_star + c + dphi_star.T @ lambda_star - lam_lb + lam_ub

    detected_active = np.where(np.abs(ineq_residual) <= 1e-8)[0]

    data = {
        # Dimensions
        "N": N,
        "d": d,
        "dims": dim,
        "nvar": nvar,
        "n_exp": n_exp,
        "n_norm": n_norm,
        "n_quad": n_quad,
        "m": m,
        "m_act": m_act,
        "n_box": n_box,
        "n_box_act": n_box_act,

        # Pseudogradient matrix
        "G": G,

        # Nonlinear constraint data
        "a_mat": a_mat,
        "B_arr": B_arr,
        "H_arr": H_arr,
        "RHS": RHS,
        "phi": phi,
        "g": g,

        # Constructed vGNE
        "x_star": x_star,
        "lambda_star": lambda_star,
        "lb": lb,
        "ub": ub,
        "lam_lb": lam_lb,
        "lam_ub": lam_ub,

        # Active-set information
        "active": active,
        "inactive": inactive,
        "inactive_slacks": slacks,
        "box_active_idx": box_act_idx,
        "box_inactive_idx": box_inact_idx,

        # Diagnostics
        "ineq_residual": ineq_residual,
        "box_lb_residual": box_lb_residual,
        "box_ub_residual": box_ub_residual,
        "stationarity_residual": stationarity,
        "stationarity_residual_norm": np.linalg.norm(stationarity),
        "detected_active": detected_active,
    }

    return gnep, data
