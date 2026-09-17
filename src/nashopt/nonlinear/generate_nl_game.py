""" Generate a random nonlinear generalized Nash equilibrium problem (NL-GNEP) that extends
the linear-quadratic variational GNE built by lq/generate_random.py with additional nonlinear
convex shared inequality constraints.

generate_nl_game() first calls generate_random() to build the quadratic costs as in [1], the shared
linear inequality/equality constraints, and the box constraints of a variational GNE with a
known equilibrium x_star (exactly as in examples/linear_quadratic/example_random.py), and then
adds on top of that game the nonlinear convex shared inequality constraints

    log(sum_{k=1}^{n_exp_terms[s]} exp(a_exp_s[k,:] . x + b_exp_s[k])) <= c_exp_s,   s = 1,...,n_exp
    x^T Q_quad_s x + a_quad_s^T x <= c_quad_s,                                      s = 1,...,n_quad

with Q_quad_s symmetric positive semidefinite and n_exp_terms a list of positive integers (one
per exponential constraint, default 2) giving the number of terms summed in each log-sum-exp
constraint, for a total of m_nl = n_exp + n_quad extra shared inequality constraints, of which
m_nl_act are constructed to be active (tight) at the same x_star. The agents' linear cost terms
are corrected so that the KKT stationarity conditions of the enlarged game still hold at x_star.

Optionally (nonlinear_cost=True), a shared, non-quadratic convex potential term

    P(x) = log(sum_{k=1}^{n_pot_terms} exp(a_pot_exp[k,:] . x + b_pot_exp[k]))

(a log-sum-exp of n_pot_terms affine terms in the FULL x, built the same way as one of the
n_exp log-sum-exp constraints above) is added identically to every agent's cost on top of the
quadratic cost above. P is convex, so it only adds a symmetric PSD term to the game's
pseudogradient Jacobian and cannot break monotonicity; the agents' linear cost terms are
further corrected (folded into the same mechanism used for the nonlinear constraints) so that
x_star remains an exact KKT point of this further-enlarged game, with the same multipliers.

[1] A. Bemporad, T. Tatarenko, "Learning Parametric Monotone Games," arXiv preprint 2609.02494, 2026.
https://arxiv.org/abs/2609.02494

(C) 2026 A. Bemporad
"""

import numpy as np
import jax
import jax.numpy as jnp
from .gnep_base import GNEP
from ..lq.generate_random import generate_random

jax.config.update("jax_enable_x64", True)


def generate_nl_game(
    dim: list[int],
    m: int,
    m_act: int,
    q: int = 0,
    n_box: int = 0,
    n_box_act: int = 0,
    n_exp: int = 0,
    n_exp_terms: list[int] | None = None,
    n_quad: int = 0,
    m_nl_act: int = 0,
    seed: int | None = None,
    mu: float = 0., # desired monotonicity constant, set mu = 0 for merely monotone GNEs
    inactive_slack_min: float = 0.5,
    inactive_slack_max: float = 1.5,
    lambda_min: float = 0.5,
    lambda_max: float = 1.5,
    box_slack_min: float = 0.5,
    box_slack_max: float = 1.5,
    mu_scale: float = 1.0,
    nl_inactive_slack_min: float = 0.5,
    nl_inactive_slack_max: float = 1.5,
    nl_lambda_min: float = 0.5,
    nl_lambda_max: float = 1.5,
    a_exp_scale: float = 1.0,
    b_exp_scale: float = 1.0,
    Q_quad_scale: float = 1.0,
    a_quad_scale: float = 1.0,
    nonlinear_cost: bool = False,
    n_pot_terms: int = 2,
    a_pot_scale: float = 1.0,
    b_pot_scale: float = 1.0,
    solver: str = "dr_daqp",
    verbose: bool = False,
):
    """
    Generate a nonlinear generalized Nash equilibrium problem obtained by adding nonlinear
    convex shared inequality constraints on top of a linear-quadratic vGNE.

    There are N = len(dim) agents, agent i having dim[i] variables, so the aggregate
    variable is

        x = col(x_1, ..., x_N) in R^nvar, x_i in R^(dim[i]), nvar = sum(dim).

    Each agent i minimizes

        J_i(x) = 0.5 x^T Q_i x + c_i^T x_i + P(x)

    w.r.t. x_i given x_-i, where Q_i is symmetric positive semidefinite and c_i is a linear
    term. Q_i and the shared linear constraints/box constraints are built exactly as in
    lq/generate_random.py, by calling that function first. On top of that game, this function
    optionally adds a convex nonlinear potential term P(x) to each agent's cost, and the
    nonlinear convex shared inequality constraints

        log(sum_{k=1}^N exp(a_exp_s[k,:] . x + b_exp_s[k])) <= c_exp_s,   s = 1,...,n_exp
        x^T Q_quad_s x + a_quad_s^T x <= c_quad_s,                        s = 1,...,n_quad

    with Q_quad_s symmetric positive semidefinite, for a total of m_nl = n_exp + n_quad
    nonlinear shared inequality constraints, of which m_nl_act are constructed to be active
    (tight) at the same equilibrium x_star built by generate_random().

    The full set of shared inequality constraints (linear A x <= b from generate_random(),
    plus the nonlinear ones above) is passed to the nonlinear GNEP solver as a single
    g(x) <= 0 constraint; the shared linear equality constraints E x = h and the box
    constraints lb <= x <= ub are passed through unchanged.

    Parameters
    ----------
    dim : list of int
        Number of decision variables per agent, dim[i] for agent i. N = len(dim).
    m : int
        Number of shared linear inequality constraints (see generate_random()).
    m_act : int
        Number of active shared linear inequality constraints at x_star (see generate_random()).
    q : int, default 0
        Number of shared linear equality constraints (see generate_random()).
    n_box : int
        Number of decision variables that get finite box constraints (see generate_random()).
    n_box_act : int
        Number of box constraints active at x_star, out of n_box (see generate_random()).
    n_exp : int
        Number of exponential (log-sum-exp) shared inequality constraints.
    n_exp_terms : list of int or None
        Number of terms n_exp_terms[s] summed in the s-th log-sum-exp constraint, for
        s = 1,...,n_exp. If None, defaults to 2 terms per constraint. Must have length n_exp,
        with all entries strictly positive.
    n_quad : int
        Number of quadratic-form shared inequality constraints.
    m_nl_act : int
        Number of active nonlinear inequality constraints at x_star, out of
        m_nl = n_exp + n_quad. Which of the m_nl constraints are active is chosen at random.
    seed : int or None
        Random seed.
    mu : float
        Desired lower bound on min eigenvalue of 0.5*(G+G.T), i.e., the monotonicity constant of the
        pseudogradient (see generate_random()).
    inactive_slack_min, inactive_slack_max : float
        Range for strictly positive slacks of inactive shared linear inequality constraints
        (see generate_random()).
    lambda_min, lambda_max : float
        Range for strictly positive multipliers of active shared linear inequality
        constraints and active box constraints (see generate_random()).
    box_slack_min, box_slack_max : float
        Range for strictly positive slacks of the inactive side of box constraints
        (see generate_random()).
    mu_scale : float
        Scale for shared linear equality multipliers (see generate_random()).
    nl_inactive_slack_min, nl_inactive_slack_max : float
        Range for strictly positive slacks of inactive nonlinear shared inequality
        constraints.
    nl_lambda_min, nl_lambda_max : float
        Range for strictly positive multipliers of active nonlinear shared inequality
        constraints.
    a_exp_scale : float
        Scale of the random matrices a_exp_s defining the exponential constraints.
    b_exp_scale : float
        Scale of the random bias vectors b_exp_s defining the exponential constraints.
    Q_quad_scale : float
        Scale of the random matrices used to build the PSD matrices Q_quad_s defining the
        quadratic-form constraints.
    a_quad_scale : float
        Scale of the random vectors a_quad_s defining the quadratic-form constraints.
    nonlinear_cost : bool
        If True, add the shared, non-quadratic convex potential term
        P(x) = log(sum_{k=1}^{n_pot_terms} exp(a_pot_exp[k,:] . x + b_pot_exp[k])) identically
        to every agent's cost, on top of the quadratic cost 0.5 x^T Q_i x + c_i^T x_i (see
        module docstring). Off by default.
    n_pot_terms : int
        Number of terms summed in the shared potential's log-sum-exp (only used when
        nonlinear_cost=True); determines the shape of a_pot_exp (n_pot_terms, nvar) and
        b_pot_exp (n_pot_terms,). Must be positive.
    a_pot_scale : float
        Scale of the random matrix a_pot_exp defining the shared potential term (only used
        when nonlinear_cost=True).
    b_pot_scale : float
        Scale of the random bias vector b_pot_exp defining the shared potential term (only
        used when nonlinear_cost=True).
    solver : str
        Solver passed to generate_random() to build the underlying linear-quadratic vGNE data
        (see generate_random() and GNEP_LQ). Unrelated to the solver later used to solve the
        returned nonlinear GNEP via gnep.solve(solver=...).
    verbose : bool
        Passed through to generate_random(): if True, print its monotonicity check. Off by
        default.

    Returns
    -------
    gnep : GNEP
        The generated nonlinear GNEP object.
    data : dict
        Dictionary containing the generated NL-GNEP data.
    """

    N = len(dim)
    if n_exp < 0:
        raise ValueError("n_exp must be nonnegative.")
    if n_quad < 0:
        raise ValueError("n_quad must be nonnegative.")
    if n_exp_terms is None:
        n_exp_terms = [2] * n_exp
    if len(n_exp_terms) != n_exp:
        raise ValueError("n_exp_terms must have length n_exp.")
    if any(k <= 0 for k in n_exp_terms):
        raise ValueError("all entries of n_exp_terms must be positive.")
    m_nl = n_exp + n_quad
    if not (0 <= m_nl_act <= m_nl):
        raise ValueError("m_nl_act must satisfy 0 <= m_nl_act <= m_nl.")
    if nonlinear_cost and n_pot_terms <= 0:
        raise ValueError("n_pot_terms must be positive when nonlinear_cost=True.")

    # ------------------------------------------------------------
    # 1. Build the quadratic costs, shared linear constraints, and box constraints, exactly
    #    as in generate_random(). A separate child seed is spawned for the nonlinear part
    #    below so that its randomness does not repeat the stream used here.
    # ------------------------------------------------------------
    seed_lq, seed_nl = np.random.SeedSequence(seed).spawn(2)

    gnep_lq_prob, data = generate_random(
        dim=dim, m=m, m_act=m_act, q=q, n_box=n_box, n_box_act=n_box_act, seed=seed_lq, mu=mu,
        inactive_slack_min=inactive_slack_min, inactive_slack_max=inactive_slack_max,
        lambda_min=lambda_min, lambda_max=lambda_max,
        box_slack_min=box_slack_min, box_slack_max=box_slack_max,
        mu_scale=mu_scale, solver=solver, verbose=verbose,
    )

    nvar = data["nvar"]
    x_star = data["x_star"]
    G = data["G"]
    lambda_star = data["lambda_star"]
    mu_star = data["mu_star"]
    lb = data["lb"]
    ub = data["ub"]
    lam_lb = data["lam_lb"]
    lam_ub = data["lam_ub"]

    A = gnep_lq_prob.A
    b = gnep_lq_prob.b
    E = gnep_lq_prob.Aeq
    h = gnep_lq_prob.beq
    Q_agents = gnep_lq_prob.Q  # already symmetrized by GNEP_LQ
    offsets = np.concatenate(([0], np.cumsum(dim)))

    rng = np.random.default_rng(seed_nl)

    # ------------------------------------------------------------
    # 2. Generate the exponential (log-sum-exp) shared inequality constraints. Each
    #    constraint s sums n_exp_terms[s] terms; since these counts may differ across
    #    constraints, a_exp/b_exp are padded to the largest term count K_max, with padded
    #    entries given bias -inf so they contribute exp(-inf) = 0 to the sum regardless of
    #    the (zero) padding in a_exp.
    # ------------------------------------------------------------
    K_max = max(n_exp_terms) if n_exp > 0 else 0
    a_exp = np.zeros((n_exp, K_max, nvar))
    b_exp = -np.inf * np.ones((n_exp, K_max))
    for s in range(n_exp):
        K_s = n_exp_terms[s]
        a_exp[s, :K_s, :] = rng.standard_normal((K_s, nvar)) * a_exp_scale / np.sqrt(nvar)
        b_exp[s, :K_s] = rng.standard_normal(K_s) * b_exp_scale

    a_exp_j = jnp.asarray(a_exp)
    b_exp_j = jnp.asarray(b_exp)

    # ------------------------------------------------------------
    # 3. Generate the quadratic-form shared inequality constraints.
    # ------------------------------------------------------------
    M_arr = rng.standard_normal((n_quad, nvar, nvar)) * Q_quad_scale / np.sqrt(nvar)
    Q_quad = np.einsum("sjk,sjl->skl", M_arr, M_arr)  # Q_quad_s = M_s^T M_s, PSD
    a_quad = rng.standard_normal((n_quad, nvar)) * a_quad_scale / np.sqrt(nvar)

    Q_quad_j = jnp.asarray(Q_quad)
    a_quad_j = jnp.asarray(a_quad)

    def phi_nl(x):
        """Raw values of the m_nl = n_exp + n_quad nonlinear constraints (before subtracting
        the RHS)."""
        parts = []
        if n_exp > 0:
            lin = jnp.einsum("skj,j->sk", a_exp_j, x) + b_exp_j  # (n_exp, K_max)
            parts.append(jax.scipy.special.logsumexp(lin, axis=1))  # (n_exp,)
        if n_quad > 0:
            quad = jnp.einsum("sjk,j,k->s", Q_quad_j, x, x) + a_quad_j @ x  # (n_quad,)
            parts.append(quad)
        if parts:
            return jnp.concatenate(parts)
        return jnp.zeros(0)

    x_star_j = jnp.asarray(x_star)
    phi_nl_star = np.asarray(phi_nl(x_star_j))

    # ------------------------------------------------------------
    # 4. Fix RHS = phi_nl(x_star): active constraints tight, inactive ones with a random
    #    positive slack.
    # ------------------------------------------------------------
    perm_nl = rng.permutation(m_nl)
    active_nl = perm_nl[:m_nl_act]
    inactive_nl = perm_nl[m_nl_act:]

    RHS_nl = phi_nl_star.copy()
    if m_nl_act < m_nl:
        slacks_nl = rng.uniform(nl_inactive_slack_min, nl_inactive_slack_max, size=m_nl - m_nl_act)
        RHS_nl[inactive_nl] += slacks_nl
    else:
        slacks_nl = np.zeros(0)

    RHS_nl_j = jnp.asarray(RHS_nl)

    # ------------------------------------------------------------
    # 5. Choose multipliers for the active nonlinear shared inequality constraints.
    # ------------------------------------------------------------
    lambda_nl_star = np.zeros(m_nl)
    if m_nl_act > 0:
        lambda_nl_star[active_nl] = rng.uniform(nl_lambda_min, nl_lambda_max, size=m_nl_act)

    dphi_nl_star = np.asarray(jax.jacobian(phi_nl)(x_star_j))  # (m_nl, nvar)

    # ------------------------------------------------------------
    # 5b. Optionally build the shared, non-quadratic convex potential term
    #     P(x) = log(sum_k exp(a_pot_exp[k,:] . x + b_pot_exp[k])), a log-sum-exp of
    #     n_pot_terms affine terms in the FULL x (built the same way as one of the n_exp
    #     log-sum-exp constraints above), added identically to every agent's cost below
    #     (step 8). P is convex, so it only adds a symmetric PSD term to the game's
    #     pseudogradient Jacobian and cannot break monotonicity. Its gradient at x_star is
    #     folded into `correction` (step 6) below, exactly like the nonlinear constraints'
    #     own dphi_nl_star.T @ lambda_nl_star term, so that x_star remains stationary.
    # ------------------------------------------------------------
    if nonlinear_cost:
        a_pot_exp = rng.standard_normal((n_pot_terms, nvar)) * a_pot_scale / np.sqrt(nvar)
        b_pot_exp = rng.standard_normal(n_pot_terms) * b_pot_scale

        a_pot_exp_j = jnp.asarray(a_pot_exp)
        b_pot_exp_j = jnp.asarray(b_pot_exp)

        def potential(x):
            return jax.scipy.special.logsumexp(a_pot_exp_j @ x + b_pot_exp_j)

        grad_potential_star = np.asarray(jax.grad(potential)(x_star_j))
    else:
        potential = None
        a_pot_exp = None
        b_pot_exp = None
        grad_potential_star = np.zeros(nvar)

    # ------------------------------------------------------------
    # 6. Correct the agents' linear cost terms so that KKT stationarity still holds:
    #
    #     G x_star + c + A.T lambda_star + E.T mu_star - lam_lb + lam_ub
    #         + dphi_nl(x_star).T lambda_nl_star + grad P(x_star) = 0.
    #
    # generate_random() already ensured that the first four terms sum to zero (with c the
    # linear cost term it built), so only the extra nonlinear-constraint and (if
    # nonlinear_cost) potential terms need to be subtracted from each agent's own block of c.
    # ------------------------------------------------------------
    correction = dphi_nl_star.T @ lambda_nl_star + grad_potential_star  # (nvar,)

    c_agents = [ci.copy() for ci in gnep_lq_prob.c]
    for i in range(N):
        si, ei = offsets[i], offsets[i + 1]
        c_agents[i][si:ei] -= correction[si:ei]

    # ------------------------------------------------------------
    # 7. Combine the shared linear inequality constraints A x <= b (from generate_random())
    #    with the new nonlinear ones into a single g(x) <= 0 constraint for the GNEP.
    # ------------------------------------------------------------
    A_j = jnp.asarray(A)
    b_j = jnp.asarray(b)

    def g(x):
        parts = []
        if m > 0:
            parts.append(A_j @ x - b_j)
        if m_nl > 0:
            parts.append(phi_nl(x) - RHS_nl_j)
        if parts:
            return jnp.concatenate(parts)
        return jnp.zeros(0)

    # ------------------------------------------------------------
    # 8. Build the jax cost functions f_i(x) = 0.5 x^T Q_i x + c_i^T x (+ P(x), if
    #    nonlinear_cost).
    # ------------------------------------------------------------
    Q_agents_j = [jnp.asarray(Qi) for Qi in Q_agents]
    c_agents_j = [jnp.asarray(ci) for ci in c_agents]

    def make_f(Qi, ci):
        if potential is None:
            @jax.jit
            def f(x):
                return 0.5 * x @ Qi @ x + ci @ x
        else:
            @jax.jit
            def f(x):
                return 0.5 * x @ Qi @ x + ci @ x + potential(x)
        return f

    f_agents = [make_f(Q_agents_j[i], c_agents_j[i]) for i in range(N)]

    gnep = GNEP(list(dim), f_agents, g=g, ng=m + m_nl, lb=lb, ub=ub, Aeq=E, beq=h, variational=True)

    # ------------------------------------------------------------
    # 9. Diagnostics.
    # ------------------------------------------------------------
    nl_ineq_residual = phi_nl_star - RHS_nl
    detected_active_nl = np.where(np.abs(nl_ineq_residual) <= 1e-8)[0]

    # Reconstruct the true (per-agent-block) linear cost term used above.
    c_true = np.zeros(nvar)
    for i in range(N):
        si, ei = offsets[i], offsets[i + 1]
        c_true[si:ei] = c_agents[i][si:ei]

    stationarity = (G @ x_star + c_true + A.T @ lambda_star + E.T @ mu_star
                    - lam_lb + lam_ub + correction)

    data.update({
        # Dimensions
        "n_exp": n_exp,
        "n_exp_terms": n_exp_terms,
        "n_quad": n_quad,
        "m_nl": m_nl,
        "m_nl_act": m_nl_act,

        # Nonlinear constraint data
        "a_exp": a_exp,
        "b_exp": b_exp,
        "Q_quad": Q_quad,
        "a_quad": a_quad,
        "RHS_nl": RHS_nl,
        "phi_nl": phi_nl,
        "g": g,

        # Shared potential cost term (nonlinear_cost)
        "nonlinear_cost": nonlinear_cost,
        "n_pot_terms": n_pot_terms,
        "a_pot_exp": a_pot_exp,
        "b_pot_exp": b_pot_exp,
        "potential": potential,
        "grad_potential_star": grad_potential_star,

        # Nonlinear active-set information
        "lambda_nl_star": lambda_nl_star,
        "active_nl": active_nl,
        "inactive_nl": inactive_nl,
        "inactive_slacks_nl": slacks_nl,

        # Diagnostics (override the linear-only ones from generate_random with the full-game
        # stationarity residual; the linear-only diagnostics remain available unchanged)
        "nl_ineq_residual": nl_ineq_residual,
        "detected_active_nl": detected_active_nl,
        "stationarity_residual": stationarity,
        "stationarity_residual_norm": np.linalg.norm(stationarity),
    })

    return gnep, data
