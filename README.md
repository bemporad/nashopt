<img src="http://cse.lab.imtlucca.it/~bemporad/nashopt/images/nashopt-logo.png" alt="nashopt" width=40%/>

# NashOpt  
### A Python library for computing generalized Nash equilibria and solving game-design and game-theoretic control problems

This repository includes a library for solving different classes of nonlinear **Generalized Nash Equilibrium Problems** (GNEPs). The decision variables and Lagrange multipliers that jointly satisfy the KKT conditions for all agents are determined by solving a nonlinear least-squares problem. If a zero residual is obtained, this corresponds to a potential generalized Nash equilibrium, a property that can be verified by evaluating the individual **best responses**. For the special case of **Linear-Quadratic Games**, one or more equilibria are obtained by solving mixed-integer linear programming problems. The package can also solve **game-design** problems by optimizing the parameters of a **multiparametric GNEP** by box-constrained nonlinear optimization, as well as **game-theoretic control** problems, such as **Linear Quadratic Regulation** and **Model Predictive Control** problems.

---
## Installation

~~~python
pip install nashopt
~~~


## Overview

Consider a game with $N$ agents. Each agent $i$ solves the following problem

$$
x_i^\star \in \arg\min_{x_i \in \mathbb{R}^{n_i}} f_i(x)
$$

subject to the following shared and local constraints

$$
g(x) \leq 0, \qquad A_{\textrm eq}x = b_{\textrm eq}, \qquad h(x)=0, \qquad \ell_i \leq x_i \leq u_i
$$

where:

- $f_i$ is the objective of agent $i$, specified as a <a href="https://github.com/jax-ml/jax">JAX</a> function;
- $x = (x_1^\top \dots x_N^\top)^\top \in \mathbb{R}^n$ are the decision variables, $x_i\in\mathbb{R}^{n_i}$;
- $g : \mathbb{R}^n \to \mathbb{R}^{n_g}$ encodes shared inequality constraints (JAX function);
- $A_{\textrm eq}, b_{\textrm eq}$ define linear shared equality constraints;
- $h : \mathbb{R}^n \to \mathbb{R}^{n_h}$ encodes shared nonlinear equality constraints (JAX function);
- $\ell, u$ are local box constraints.

A **generalized Nash equilibrium** $x^\star$ is a vector such that no agent can reduce their cost given the others' strategies and feasibility constraints, i.e.,

$$f_i(x^\star_{i}, x^\star_{-i})\leq f_i(x_i, x^\star_{-i})$$ 

for all feasible $x=(x_i,x_{-i}^\star)$, or equivalently, in terms of **best responses**: 

$$
\begin{aligned}
x_i^\star \in \arg\min_{\ell_{i}\leq x_{i}\leq u_{i}} &f_i(x)\\
\textrm{s.t.} \quad &g(x) \leq 0 \\
&A_{\textrm eq}x = b_{\textrm eq}\\
&h(x) = 0\\
&x_{-i}=x_{-i}^\star.
\end{aligned}
$$


---

## KKT Conditions

For each agent $i$, the necessary KKT conditions are:

**1. Stationarity**

$$ \nabla_{x_i} f_i(x) + \nabla_{x_i} g(x)^\top \lambda_i + [A_i^\top\ \nabla_{x_i} h(x)^\top] \mu_i - v_i + y_i = 0 $$

**2. Primal Feasibility**

$$
g(x) \leq 0, \qquad Ax = b, \qquad h(x) = 0, \qquad \ell \le x \le u
$$

**3. Dual Feasibility**

$$
\lambda_i \ge 0, \qquad v_i\geq 0, \qquad y_i\geq 0
$$

**4. Complementary Slackness**

$$
\lambda_{i,j} \, g_j(x) = 0
$$

$$
v_{i,k} \, (x_{i,k} - \ell_{i,k}) = 0
$$

$$
y_{i,k} \, (u_{i,k} - x_{i,k}) = 0
$$

For general nonlinear problems, in `nashopt` primal feasibility (with respect to inequalities), dual feasibility, and complementary slackness conditions, which can be summarized as complementarity pairs $0\leq a\perp b\geq 0$, are enforced by using the nonlinear complementarity problem (NCP) Fischer–Burmeister function [1]

$$
\phi(a, b) = \sqrt{a^2 + b^2} - a - b
$$

which has the property

$$
\phi(a,b) = 0 \;\Longleftrightarrow\; a \ge 0,\; b \ge 0,\; ab = 0.
$$

Therefore, the above KKT conditions can be rewritten as the nonlinear system of equalities

$$R(z)=0$$

where $z = (x, \{\lambda_i\}, \{\mu_i\}, \{v_i\}, \{y_i\})$.  To find a solution, we solve the nonlinear least-squares problem

$$
   \min_z \frac{1}{2}\|R(z)\|_2^2
$$

using `scipy`'s nonlinear least squares methods in <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares">`least_squares`</a>, exploiting JAX's automatic differentiation capabilities.

After solving the nonlinear least-squares problem, if the residual $R(z^\star)=0$, we can check if indeed $x^\star$ is a GNE by computing the best responses of each agent

$$ \min_{\ell_i\leq x_i\leq u_i} f_i(x_i, x^\star_{-i}) $$

$$ \textrm{s.t.} \qquad g_i(x), \qquad Ax=b, \qquad h(x)=0$$

In `nashopt`, the best response of agent $i$ is computed by solving the following box-constrained nonlinear
programming problem with `scipy`'s <a href="https://jaxopt.github.io/stable/_autosummary/jaxopt.ScipyBoundedMinimize.html">`L-BFGS-B`</a> method via the `jaxopt` interface:

$$ \min_{x_i} f_i(x_i, x_{-i}) + \rho \left(\left(\sum_j \max(g_i(x), 0)^2\right) + \|A x - b\|_2^2 + \|h(x)\|_2^2\right) $$
            
$$ \textrm{s.t.} \qquad \ell_i \leq x_i \leq u_i$$

with $x_{-i}=x^\star_{-i}$, where $\rho\gg 1$ is a large penalty on the violation of shared constraints.

*Variational GNEs* can be obtained by making the Lagrange multipliers associated with the
shared constraints the same for all players, i.e., by replacing $\{\lambda_i\}$ with a single vector $\lambda$ and $\{\mu_i\}$ with a single vector $\mu$, which further reduces the dimension of the zero-finding problem.

### Example

We want to solve a simple GNEP with 3 agents, $x_1\in\mathbb{R}^2$, $x_2\in\mathbb{R}$, $x_3\in\mathbb{R}$, defined as follows:

```python
import numpy as np
import jax
import jax.numpy as jnp
from nashopt import GNEP

sizes = [2, 1, 1]      # [n1, n2, n3]

# Agent 1 objective:
@jax.jit
def f1(x):
    return jnp.sum((x[0:2] - jnp.array([1.0, -0.5]))**2)

# Agent 2 objective:
@jax.jit
def f2(x):
    return (x[2] + 0.3)**2

# Agent 3 objective:
@jax.jit
def f3(x):
    return (x[3] - 0.5*(x[0] + x[2]))**2

# Shared constraint:
def g(x): 
    return jnp.array([x[3] + x[0] + x[2] - 2.0])

lb=np.zeros(4) # lower bounds
ub=np.ones(4) # upper bounds

gnep = GNEP(sizes, f=[f1,f2,f3], g=g, ng=1, lb=lb, ub=ub)
```

We call `solve()` to solve the problem defined above:

```python
sol = gnep.solve()
x_star = sol.x
```

which gives the following solution:

```python
x* = [ 1.  0.  0.   0.5]
```

We can check if the KKT conditions are satisfied by looking at the residual norm $||R(x)||_2$:

```python
residual = sol.res
print(np.linalg.norm(residual))

1.223145e-16
```

We can also inspect the vector of Lagrange multipliers and other statistics about the solution process:

```python
lam_star = sol.lam
stats = sol.stats
```

After solving the problem, we can check if indeed $x^\star$ is an equilibrium by evaluating the agents' individual best responses:

```python
for i in range(gnep.N):
    sol = gnep.best_response(i, x_star)
    print(sol.x)
```

```
[ 1.   0.  -0.   0.5]
[ 1.  -0.   0.   0.5]
[ 1.  -0.  -0.   0.5]
```

To add linear equality constraints, use the following:

```python
Aeq = np.array([[1,1,1,1]])
beq = np.array([2.0])

gnep = GNEP(sizes, f=[f1,f2,f3], g=g, ng=1, lb=lb, ub=ub, Aeq=Aeq, beq=beq)
```

while for general nonlinear equality constraints:

```python
gnep = GNEP(sizes, f=[f1,f2,f3], g=g, ng=1, lb=lb, ub=ub, h=h, nh=nh)
```

where `h` is a vector function returning a `jax` array of length `nh`.


You can also specify an initial guess $x_0$ to the GNEP solver as follows:
```python
sol = gnep.solve(x0)
```

To compute a **variational GNE** solution, set flag `variational` = `True`:

```python
gnep = GNEP( ... , variational=True)
```

To decide the nonlinear least-squares solver used to compute the GNEP, use the following call:

```python
sol = gnep.solve(x0, solver = "trf")
```

or

```python
sol = gnep.solve(x0, solver = "lm")
```

where `trf` calls a trust-region reflective algorithm, while `lm` a Levenberg-Marquardt method.

## Game Design

By leveraging the above characterization of GNEs, we consider the **multiparametric Generalized Nash Equilibrium Problem** (mpGNEP) with $N$ agents, in which each agent $i$ solves:
        
$$
\begin{aligned}
\min_{x_i} \quad & f_i(x,p)\\
\textrm{s.t.} \quad & g(x,p) \leq  0\\
& A_{\textrm{eq}} x = b_{\textrm{eq}} + S_{\textrm{eq}} p\\
& h(x,p) = 0\\
& \ell \leq  x \leq  u
\end{aligned}
$$
                 
where $p\in\mathbb{R}^{n_p}$ is a vector of parameters defining the game. Our goal is to design the game-parameter vector $p$ to achieve a desired GNE, according to the following nested optimization problem:

$$
\begin{aligned}
\min_{x^\star,p}\quad & J(x^\star,p) \\
\text{s.t.} \quad & x_i^\star\in\arg\min_{x_i \in \mathbb{R}^{n_i}}\quad && f_i(x,p)\\
&\text{s.t. } \quad && g(x,p) \leq 0\\
&&&Ax = b+Sp\\
&&&h(x,p) = 0\\
&&&\ell \leq x\leq u\\
&&&x_{-i} = x_{-i}^\star,\qquad i = 1, \ldots, N
\end{aligned}
$$    

where $J$ is the objective function of the designer used to shape the resulting GNE. For example,
given an observed agents' equilibrium $x_{\textrm des}$, we can solve the inverse-game theoretical problem 
of finding a vector $p$ (if one exists) such that $x^\star\approx x_{\textrm des}$, by setting

$$J(x^\star,p)=\|x^\star-x_{\rm des}\|_2^2.$$

We solve the game-design problem as
$$
\begin{aligned}
    \min_{z,p}\quad & J(x,p) + \frac{\rho}{2}\|R(z,p)\|_2^2\\
    \text{s.t. }\quad & \ell_p\leq p\leq u_p
\end{aligned}
$$

via `L-BFGS-B`, where $R(z,p)$ is the parametric version of the KKT residual defined above and $\ell_p$, $u_p$ define the range of admissible $p$, $\ell_{pj}\in\mathbb{R}\cup \{-\infty\}$, $u_{pj}\in\mathbb{R}\cup \{+\infty\}$, $j=1,\ldots,n_p$.

Smooth and nonsmooth regularization terms $\alpha_1\|x\|_1 + \alpha_2\|x\|_2^2$ can be explicitly added to $J(x,p)$.

### Example
To solve a **game-design** problem with objective $J$, use the following structure:

```python
from nashopt import ParametricGNEP

pgnep = ParametricGNEP(sizes, npar=2, f=f, g=g, ng=1, lb=lb, ub=ub, Aeq=Aeq, beq=beq, h=n, nh=nh, Seq=Seq)

sol = pgnep.solve(J, pmin, pmax)
```

where now the functions listed in `f`, `g`, `h`, and `J` take $x$ and $p$ as input arguments,
and `pmin`, `pmax` define the admissible range of the parameter-vector $p$ (infinite bounds are allowed).

Regularization terms

$$
 \alpha_1\|x\|_1 + \alpha_2\|x\|_2^2
$$

where $\alpha_1,\alpha_2\geq 0$ can be added on the cost function $J$ as follows:

```python
sol = pgnep.solve(J, pmin, pmax, alpha1=alpha1, alpha2=alpha2)
```

You can specify two further flags: `gne_warm_start`, to warm-start the optimization by computing first a GNE, and `refine_gne`, to try getting a GNE after solving the problem by refining the solution $x$ for the optimal parameter $p$ found.

## Linear-Quadratic Games
When the agents' cost functions are quadratic and convex with respect to $x_i$ and all the constraints are linear, i.e.,

$$
\begin{array}{rrl}
    \min_{p, x^\star} \quad & J(x^\star,p)\\
    \text{s.t. } & x^\star_i\in \arg\min_{x_i} &f_i(x,p)=\frac{1}{2} x^\top Q^i x + (c^i + F^i p)^\top x \\
    & \text{s.t. } & A x \leq b + S p\\ 
    &&A_{\mathrm{eq}} x = b_{\mathrm{eq}} + S_{\mathrm{eq}} p \\
    &&\ell_i \leq x \leq u_i\\
    &&x_{-i} = x^\star_{-i}\\
    && i=1,\dots,N
\end{array}
$$

the equilibrium conditions can be expressed as a mixed-integer linear program (MILP) using a "big-M" approach. `nashopt` support both the open-source solver `HiGHS` and `Gurobi` to solve the MILP. 

Example:

```python
from nashopt import GNEP_LQ

gnep = GNEP_LQ(sizes, Q, c, F, lb=lb, ub=ub, pmin=pmin,
               pmax=pmax, A=A, b=b, S=S, M=1e4, variational=variational, solver='highs')
sol = gnep.solve()
x = sol.x
```

We can also extract multiple solutions, if any exist, that correspond to different combinations of active constraints at optimality. For example, to get a list of the first 10 solutions:

```python
sol = gnep.solve(max_solutions=10)
```

In addition, a game objective $J$ can be given as the (sum of) convex piecewise affine function(s)

$$
J(x,p) = \sum_{j=1}^{n_J}\max_{k=1,\dots,n_j} D^{PWA}_{jk} x + E^{PWA}_{jk} p + h^{PWA}_{jk}
$$

```python
    gnep_lq = GNEP_LQ(sizes, ... D_pwa=D_pwa, E_pwa=E_pwa, h_pwa=h_pwa, ...)
```

and the optimal parameters $p$ are also determined by MILP, or as the convex quadratic function
        
$$
                    f(x,p) = \frac{1}{2} [x^T\ p^T] Q_J \left[\begin{array}{c}x\\p\end{array}\right] + c_J^T \left[\begin{array}{c}x\\p\end{array}\right]
$$

```python
    gnep_lq = GNEP_LQ(sizes, ... Q_J=Q_J, c_J=c_J, ...)
```

or the sum of both, where in this case the optimal parameters $p$ are determined by MIQP (only Gurobi supported).

## Game-Theoretic Control
We consider non-cooperative multi-agent control problems where each agent only controls a subset of the input vector $u$ of a discrete-time linear dynamical system 

$$
\begin{aligned}
x(t+1) &= A x(t) + B u(t)\\
y(t) &= C x(t)
\end{aligned}
$$

where $u(t)$ stacks the agents' decision vectors $u_1(t),\ldots,u_N(t)$. 

### Game-Theoretic LQR
For solving non-cooperative linear quadratic regulation (LQR) games, you can use the `NashLQR` class:

```python
from nashopt import NashLQR

nash_lqr = NashLQR(sizes, A, B, Q, R, dare_iters=dare_iters)
sol = nash_lqr.solve(verbose=2)
sol.K_Nash=K_Nash
```

where `sizes` contains the input sizes $[n_1,\ldots,n_N]$, $Q=[Q_1,\ldots,Q_N]$ are the full-state weight matrices, and $R=[R_1,\ldots,R_N]$ the input weight matrices used by agent $i$ to weight $u_i$. The number `dare_iters` is the number of fixed-point iterations used to find an approximate solution of the discrete algebraic Riccati equation for each agent.

You can retrieve extra information after solving the Nash equilibrium problem, such as the KKT residual `sol.residual`, useful to verify whether an equilibrium was found, the centralized LQR gain `sol.K_centralized` (for comparison), and other statistics `sol.stats=stats`.


### Game-Theoretic Model Predictive Control
We now want to make the output vector $y(t)$ of the system track a given setpoint $r(t)$.
Each agent optimizes a sequence of input increments $\{\Delta u_{i,k}\}_{k=0}^{T-1}$ over a prediction horizon of $T$ steps, where $\Delta u_k=u_k-u_{k-1}$, by solving:

$$
\Delta u_i,\epsilon_i \in\arg\min \sum_{k=0}^{T-1}
\left( (y_{k+1}-r(t))^\top Q_i (y_{k+1}-r(t))
      + \Delta u_{i,k}^\top Q_{\Delta u,i}\Delta u_{i,k}\right)
+ q_{\epsilon,i}^\top \epsilon_i
$$

$$
\begin{array}{rll}
\text{s.t. } & x_{k+1} = A x_k + B u_k& y_{k+1} = C x_{k+1}\\
& u_{k,i} = u_{k-1,i} + \Delta u_{k,i}& u_{-1} = u(t-1)\\
&\Delta u_{\rm min} \leq \Delta u_k \leq \Delta u_{\rm max} 
& u_{\rm min} \leq  u_k \leq u_{\rm max}\\
& y_{\min} - \sum_{i=1}^N \epsilon_i \leq y_{k+1} \leq y_{\max} + \sum_{i=1}^N \epsilon_i&
\epsilon_i \geq 0\\
& i=1,\ldots,N,\ k=0,\ldots,T-1. 
\end{array}
$$
where $Q_i\succeq 0$, $Q_{\Delta u,i}\succeq 0$ and $\epsilon_i\geq 0$ is a slack variable
used to soften shared output constraints (with linear penalty $q_{\epsilon,i}\geq 0$). Each agent's MPC problem can be simplified by imposing the constraints only on a shorter constraint horizon of $T_c<T$ steps.

You can use the `NashLinearMPC` class to define the game-theoretic MPC problem:

```python
from nashopt import NashLinearMPC

nash_mpc = NashLinearMPC(sizes, A, B, C, Qy, Qdu, T, ymin=ymin, ymax=ymax, umin=umin, umax=umax, dumin=dumin, dumax=dumax, Qeps=Qeps, Tc=Tc)
```

and then evaluate the GNE control move `u` = $u(t)$ at each step $t$:

```python
sol = nash_mpc.solve(x, u1, r)
u = sol.u
```

where `r` = $r(t)$ is the current output reference signal, `x` = $x(t)$ the current state, and `u1` = $u(t-1)$ the previous input. 

To compute a *variational* GNE solution, use

```python
sol = nash_mpc.solve(x, u1, r, variational=True)
u = sol.u
```

For comparison, you can compute instead the *centralized* MPC move, where the cost function is the sum of all agents' costs, via standard quadratic programming:

```python
sol = nash_mpc.solve(x, u1, r, centralized=True)
u = sol.u
```

To specify the MILP solver to use to compute the game-theoretic MPC law, use the following:

```python
sol = nash_mpc.solve(x0, u1, ref, ..., solver='highs')
```

or

```python
sol = nash_mpc.solve(x0, u1, ref, ..., solver='gurobi')
```


## References

> [1] Alexander Fischer. *A special Newton-type optimization method.* **Optimization**, 24(3–4):269–284, 1992.

## Citation

```
@misc{nashopt,
    author={A. Bemporad},
    title={{NashOpt}: A {Python} Library for Computing Generalized {Nash} Equilibria and Game Design},
    howpublished = {\url{https://github.com/bemporad/nashopt}},
    year=2025
}
```

---
## Related packages

<a href="https://github.com/bemporad/nash-mpqp">**nash-mpqp**</a> a solver for solving linear-quadratic multi-parametric generalized Nash equilibrium (GNE) problems in *explicit* form.

<a href="https://github.com/bemporad/gnep-learn">**gnep-learn**</a> a Python package for solving generalized Nash equilibrium problems by *active learning* of best-response models.

---
## License

Apache 2.0

(C) 2025 A. Bemporad

## Acknowledgement
This work was funded by the European Union (ERC Advanced Research Grant COMPACT, No. 101141351). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.

<p align="center">
<img src="erc-logo.png" alt="ERC" width="400"/>
</p>
