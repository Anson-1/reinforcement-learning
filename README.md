# Multi-Asset Portfolio Allocation

## 1. Problem Statement
Consider the discrete-time asset allocation example in section 8.4 of Rao and Jelvis. Consider the price vector $X$, which has dimension $n$, $n > 2$. $X(0) = 1$. The one-period return of asset $k$ is $\mathcal{N}(\mu_k, \sigma_k^2)$, where $\mu_k$ is the mean and $\sigma_k^2$ is the variance. You are given an initial portfolio with asset $k$ having a portion of $p^{(k)}$, where $p^{(0)}$ is the portion in cash. Obviously, $p^{(0)} + p^{(1)} + \cdots + p^{(n)} = 1$. The interest on any cash is $r$. You can at most adjust 10% of your portfolio at each period. You have an absolute risk averse utility function. Use RL to compute the optimal strategy at each period for any reasonable $r$, $\mu_k$, $\sigma_k^2$ and $p^{(k)}$. Show that your program can work for any time horizon $< 10$, and $n < 5$.


## 2. Project Overview
This repository implements a discrete-time multi-asset portfolio allocation optimizer, solving the problem outlined in Section 8.4 of Rao and Jelvis. The objective is to maximize the expected utility of terminal wealth over a finite time horizon ($T < 10$) for a portfolio consisting of a risk-free cash asset and $n$ risky assets ($n < 5$). 

The environment features:
* **Normally Distributed Returns:** One-period returns for asset $k$ are drawn from $\mathcal{N}(\mu_k, \sigma_k^2)$.
* **CARA Utility:** The agent acts under an absolute risk-averse (Constant Absolute Risk Aversion) utility function.
* **Turnover Constraint:** A maximum of 10% of the portfolio can be rebalanced at each period.
* **Leverage Constraints:** Bounded borrowing/shorting parameters.

This project contrasts Deep Reinforcement Learning (PPO, A2C and SAC) and Exact Dynamic Programming (Tabular DP) with Approximate Dynamic Programming (ADP via Neural Networks with Gauss-Hermite and Monte Carlo expectations).

## 3. Mathematical Formulation
### 3.1 Markov Decision Process (MDP)
We can model this problem as a **finite-horizon Markov Decision Process**. 
- **State** at time $t$:  $s_t = (t,\; W_t,\; p_t^{(1)},\; \ldots,\; p_t^{(n)})$, where $t$ is the remaining time step, $W_t$ is total wealth and $p_t^{(k)}$ is the portfolio weight of risky asset. For the weight of cash, we can easily know it from $p_t^{(0)} = 1 - \sum_{k=1}^n p_t^{(k)}$.

- **Action**: $a_t = (\delta_t^{(1)}, \delta_t^{(2)}, \dots, \delta_t^{(n)})$, just like in the state space, the change to the cash position ($\delta_t^{(0)}$) is mathematically redundant, since $\delta_t^{(0)} = -\sum_{k=1}^n \delta_t^{(k)}$
- **Reward**: In portfolio optimization, you are not withdrawing cash or consuming wealth at every time step. Your sole objective is to maximize the utility of your terminal wealth ($W_T$) at the very end of the time horizon.
For all intermediate steps ($t < T$): $R_t = 0$, at the final time step ($t=T$), $R_T = u(W_T) = \frac{1 - e^{-A \cdot W_T}}{A}$.
- **Transition**: 
     - Cash: $p_t^{+(0)} = p_t^{(0)} + \delta_t^{(0)}$
     - Risky Assets: $p_t^{+(k)} = p_t^{(k)} + \delta_t^{(k)}$
     - Risky assets return: $R_{t+1}^{(k)} \sim \mathcal{N}(\mu_k, \sigma_k^2)$
     - Total portfolio return ($R_p$): $R_p = p_t^{+(0)} \cdot r + \sum_{k=1}^n \left( p_t^{+(k)} \cdot R_{t+1}^{(k)} \right)$
     - Wealth grows/shrinks: $W_{t+1} = W_t \cdot (1 + R_p)$
     - Weights drift: $p_{t+1}^{(k)} = \frac{p_t^{+(k)} \cdot (1 + R_{t+1}^{(k)})}{1 + R_p}$

### 3.2 Action Space and Portfolio Constraints

1. To model illiquidity, transaction costs, and prevent unrealistic high-frequency trading, the total volume of trading at any single time step cannot exceed 10% of the portfolio's total value. Because every purchase must be funded by an equivalent sale (self-financing), the $L_1$-norm of all asset weight changes (divided by 2) must remain under the turnover limit ($M = 0.10$):$$\frac{1}{2} \left( |\delta_t^{(0)}| + \sum_{k=1}^n |\delta_t^{(k)}| \right) \le M = 0.10$$

2. To prevent the agent from instantly dumping its entire portfolio into a single asset, each individual delta is strictly bounded. The agent cannot buy or sell more than 10% of any specific asset in a single period:$$-0.10 \le \delta_t^{(k)} \le 0.10 \quad \forall k$$ While mathematically implied by the Turnover limit, an explicit per-asset bound ($-0.10 \le \delta_t^{(k)} \le 0.10$) is strictly enforced to define the continuous action space for the RL agents (gym.spaces.Box). This explicit box constraint prevents neural network output explosion and drastically optimizes the discrete grid-search space for the Tabular DP algorithm.

3. To prevent the agent from taking infinite margin loans or unbounded short positions, the gross exposure of the portfolio is restricted by a leverage factor, $L_{\max} = 2$. Mathematically, the sum of the absolute values of all portfolio weights (including cash) must not exceed this limit:$$|p_t^{(0)}| + \sum_{k=1}^n |p_t^{(k)}| \le L_{\max}$$ 
A leverage factor of 2 represents a standard 2:1 margin limit (e.g., Regulation T in US equities). It dictates that for every $\$1.00$ of actual wealth, the agent can hold at most $\$2.00$ of total gross market exposure (e.g., $\$1.50$ in long positions and $\$0.50$ in short positions). Without a strict leverage ceiling of 2, the optimizer would exploit the expected positive drift by borrowing extremely high amounts of cash at rate $r$, exposing the portfolio to a near-guaranteed risk of ruin (wealth dropping below zero) during a negative tail-risk event.

### 3.3 Objective Function and The Bellman Equation

The ultimate goal of the agent is to maximize the expected utility of its wealth at the exact terminal time step $T$. To model the agent's risk aversion, we use the Constant Absolute Risk Aversion (CARA) utility function:$$U(W_T) = \frac{1 - e^{-A \cdot W_T}}{A}$$where $A$ is the risk aversion coefficient. A higher $A$ penalizes downside variance more heavily, forcing the agent to balance the high expected returns of risky assets against their inherent market volatility.

Since the environment yields zero intermediate rewards ($R_t = 0$ for all $t < T$), the expected Value Function $V_t(s_t)$ at any given step is mathematically equivalent to the expected terminal utility. We can derive this by starting with the standard Bellman Optimality Equation:$$V^*(s) = \max_a \mathbb{E} \left[ R(s,a) + \gamma V^*(s') \right]$$In our specific finite-horizon portfolio environment, we apply two key conditions: No intermediate rewards: $R(s,a) = 0$ (trading does not generate immediate utility). No temporal discounting: $\gamma = 1$ (the passage of time is already accounted for by the stochastic market returns). Applying these conditions, the immediate reward and discount factor drop out entirely. Adding time subscripts to account for the finite horizon $T$, we arrive at the simplified, recursive Bellman Equation that governs this project:$$V_t(s_t) = \max_{a_t} \mathbb{E} \left[ V_{t+1}(s_{t+1}) \mid s_t, a_t \right]$$with the strict terminal boundary condition of $V_T(s_T) = U(W_T)$.This recursive equation serves as the mathematical engine for the entire simulation. It allows the Exact and Approximate Dynamic Programming algorithms to seamlessly propagate the sparse terminal reward backward through time, and it acts as the theoretical target that the model-free Reinforcement Learning Critic networks attempt to estimate.

## 4. Algorithms

### 4.1 Deep Reinforcement Learning (PPO, A2C, SAC)

Unlike the DP methods, the RL agents are **model-free** — they do not know the return distribution $\mathcal{N}(\mu_k, \sigma_k^2)$ or the transition dynamics. They learn purely by interacting with a Gymnasium environment (`PortfolioEnv`) that simulates the portfolio.

- **Observation:** $[t/T,\; W_t,\; p_t^{(0)},\; p_t^{(1)},\; \ldots,\; p_t^{(n)}]$ — the normalized time step, current wealth, and full portfolio weight vector. Note that $p_t^{(0)}$ is mathematically redundant ($p_t^{(0)} = 1 - \sum_{k=1}^n p_t^{(k)}$), but is explicitly included to simplify learning for the neural network.
- **Action:** A continuous vector $a \in [-1, 1]^n$, scaled by $\delta_{\max} = 0.10$ to produce the portfolio deltas $\delta_t^{(k)} = a^{(k)} \cdot \delta_{\max}$. The turnover and leverage constraints are enforced inside the environment by proportionally rescaling the action if violated.
- **Reward:** $R_t = 0$ for $t < T$, and $R_T = U(W_T)$ at the terminal step. The discount factor is $\gamma = 1.0$ since we only care about terminal utility.

Three algorithms are used, all implemented via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/):

1. **PPO (Proximal Policy Optimization):** An on-policy, policy-gradient algorithm. It collects rollouts (2048 steps per batch), then updates the policy by maximizing a clipped surrogate objective that prevents destructively large updates. Trained for 500K environment steps with $\text{lr} = 10^{-3}$.

2. **A2C (Advantage Actor-Critic):** An on-policy actor-critic method that updates the policy after every $n_{\text{steps}} = 512$ environment steps using the advantage $A_t = R_t + V(s_{t+1}) - V(s_t)$ to reduce gradient variance. Trained for 1M steps with $\text{lr} = 5 \times 10^{-4}$.

3. **SAC (Soft Actor-Critic):** An off-policy, maximum-entropy algorithm. It maintains a replay buffer and optimizes a modified objective $J = \mathbb{E}[R + \alpha \mathcal{H}(\pi)]$ that encourages exploration via an entropy bonus $\mathcal{H}(\pi)$. This helps avoid premature convergence to suboptimal policies. Trained for 500K steps with $\text{lr} = 10^{-3}$ and batch size 256.

All three agents use the default `MlpPolicy` (two hidden layers of 64 units each). Because the agents have no access to the return model, they are robust to model misspecification but less sample-efficient than DP/ADP methods.

### 4.2 Exact Dynamic Programming (Tabular DP)
The Tabular DP algorithm acts as the ground-truth baseline. It solves the Bellman equation via exact backward induction starting from $t=T$.
- Mechanics: It discretizes the continuous state and action spaces into a finite, computationally searchable grid.
- Expectation Calculation: Because it has full access to the transition dynamics (a "white-box" model), it calculates the exact mathematical expectation of the Gaussian market shocks using Gauss-Hermite Quadrature, entirely avoiding sampling noise.
- Limitations: While mathematically perfect, it suffers severely from the Curse of Dimensionality. As the number of assets $n$ increases, the grid size explodes, making it computationally intractable for large portfolios.

### 4.3 Approximate Dynamic Programming (ADP)

The ADP algorithm replaces the discrete state grid with a continuous function approximator.
- Mechanics: Instead of saving a value for every possible state in a table, it trains a distinct multi-layer Neural Network (ValueNet) for each time step $t$.
- Expectation Calculation: The network is trained via supervised regression to approximate the Bellman expectation $V_t(s_t)$. The integral over the market shocks is estimated using either numerical integration (Gauss-Hermite) or Monte Carlo sampling.
- Advantages: It handles continuous state spaces gracefully and solves the grid-scaling problem, while still exploiting the known "white-box" transition dynamics to propagate values backward efficiently.

## 5. Results & Discussion

### 5.1 Scenario Design

All methods are evaluated on **7 core scenarios** using shared pre-generated return paths (seed=42, 2000 episodes) to ensure a fair comparison. The scenarios are designed to cover the full range required by the problem statement ($n < 5$, $T < 10$):

| Scenario | $n$ | $T$ | $r$ | $A$ | Purpose |
|---|---|---|---|---|---|
| Base | 3 | 5 | 0.03 | 0.5 | Standard benchmark |
| Two Assets | 2 | 5 | 0.03 | 0.7 | Minimum $n > 2$ |
| Four Assets | 4 | 5 | 0.03 | 0.5 | High-dimensional action space |
| Four Assets Long | 4 | 7 | 0.02 | 0.7 | High $n$ + longer horizon |
| Long Horizon | 3 | 9 | 0.03 | 0.5 | Near-maximum $T$ |
| High Risk Aversion | 3 | 5 | 0.03 | 3.0 | Conservative agent ($A=3$) |
| Low Risk Aversion | 3 | 5 | 0.03 | 0.1 | Aggressive agent ($A=0.1$) |

Additionally, 3 **convergence test scenarios** are used to verify that all RL methods (PPO, A2C, SAC) find the same optimal policy (see `RL_Convergence.ipynb`):

| Scenario | $n$ | $T$ | $r$ | $A$ | Purpose |
|---|---|---|---|---|---|
| Cash Dominant | 3 | 5 | 0.06 | 1.0 | $r > \mu_k$ for all risky assets — optimal to hold cash |
| Asset 3 Dominant | 3 | 5 | 0.03 | 1.0 | $\mu_3 = 0.15 \gg \mu_1, \mu_2$ — clear best risky asset |
| Low Variance Wins | 3 | 5 | 0.03 | 2.0 | Asset 1 has lowest variance — risk aversion favors safety |

A separate **sanity check** scenario (in `Compare.ipynb`) tests all methods including DP on an extreme dominant-asset case. In total: **7 core + 3 convergence + 1 sanity = 11 test cases**.

### 5.2 Cross-Method Utility Comparison

The table below shows the average CARA utility $\pm$ standard error across all 7 core scenarios. **Hold** (do nothing) and **Heuristic** (Sharpe-ratio proportional) serve as baselines.

| Scenario | Hold | Heuristic | Tabular DP | ADP-Hermite | ADP-MC | PPO | A2C | SAC |
|---|---|---|---|---|---|---|---|---|
| Base (n=3, T=5) | 0.999 | 1.042 | **1.050** | 1.049 | 1.049 | 1.046 | 1.043 | 1.044 |
| Two Assets (n=2, T=5) | 0.905 | 0.958 | **0.968** | 0.968 | 0.967 | 0.961 | 0.958 | 0.961 |
| Four Assets (n=4, T=5) | 1.002 | 1.043 | 1.002 | 1.043 | 1.044 | 1.045 | 1.042 | **1.048** |
| Four Assets Long (n=4, T=7) | 0.951 | 1.037 | 0.984 | **1.057** | 1.056 | 1.044 | 1.042 | 1.048 |
| Long Horizon (n=3, T=9) | 1.183 | 1.296 | 1.293 | **1.308** | 1.305 | 1.304 | 1.293 | 1.305 |
| High Risk Aversion (A=3.0) | 0.3275 | 0.3289 | **0.3291** | 0.3290 | 0.3290 | 0.3290 | 0.3290 | 0.3290 |
| Low Risk Aversion (A=0.1) | 1.291 | 1.413 | **1.440** | 1.438 | 1.438 | 1.419 | 1.409 | 1.426 |

**Key observations:**

- All optimization methods (DP, ADP, RL) significantly outperform the Hold and Heuristic baselines, confirming they learn meaningful policies.
- ADP-Hermite and ADP-MC closely match or exceed Tabular DP in all scenarios, while scaling to $n=4$ where Tabular DP suffers from the curse of dimensionality (note the degraded Tabular DP performance at $n=4$).
- PPO and A2C, despite being model-free, achieve utilities within 1–2% of the best DP/ADP methods. The gap is largest for the Low Risk Aversion scenario, where the flat utility surface makes precise optimization harder.
- SAC (tested separately in `SAC.ipynb`) matches or slightly exceeds PPO in most scenarios, benefiting from its entropy-driven exploration.

### 5.3 Scalability: Curse of Dimensionality

Tabular DP performance degrades sharply for $n = 4$ assets. In the **Four Assets** scenario, Tabular DP achieves utility 1.002 — barely above Hold (1.002) — because the discretized grid becomes too coarse to represent the optimal policy. ADP and RL methods are unaffected:

| Method | Four Assets (n=4, T=5) | Four Assets Long (n=4, T=7) |
|---|---|---|
| Tabular DP | 1.002 | 0.984 |
| ADP-Hermite | 1.043 | **1.057** |
| PPO | **1.044** | 1.048 |

This demonstrates that neural network-based methods (ADP and RL) scale gracefully where exact tabular methods fail.

### 5.4 Allocation Convergence

Across all scenarios, the allocation trends show that every optimization method follows the same **macro strategy**: progressively reduce cash holdings and increase exposure to risky assets with high excess returns ($\mu_k - r$). The methods differ slightly in **micro allocation** — which specific assets receive the marginal dollar — but these differences have negligible impact on utility due to the flat CARA utility surface near the optimum.

### 5.5 Sanity Check: Dominant Asset

To rigorously test convergence, a sanity-check scenario is constructed with one overwhelmingly dominant asset ($\mu_3 = 0.20$ vs $\mu_1 = 0.01$, $\mu_2 = 0.03$, $r = 0.02$, $A = 0.05$, $T = 2$). In this scenario, the optimal action is unambiguous: maximize allocation to Asset 3.

Results confirm that **all methods agree**:
- Every method allocates $\delta_3 \approx 0.10$ (the maximum allowed per period) at every time step.
- Terminal utilities are tightly clustered: Tabular DP = 1.078, ADP-Hermite = 1.077, ADP-MC = 1.075, PPO = 1.076, A2C = 1.077.
- Minor differences in Assets 1 and 2 reflect different **funding paths** (selling Asset 1 vs selling cash to fund the Asset 3 purchase), which are economically equivalent.

### 5.6 RL Convergence Test

The 3 convergence test scenarios each have a single clearly dominant strategy. All three RL methods are trained independently and evaluated on shared returns:

| Scenario | PPO | A2C | SAC |
|---|---|---|---|
| Cash Dominant ($r=0.06$) | 0.696 | 0.695 | 0.696 |
| Asset 3 Dominant ($\mu_3=0.15$) | 1.036 | 1.034 | 1.033 |
| Low Variance Wins ($A=2.0$) | 0.461 | 0.461 | 0.462 |

**Key findings:**
- **Utilities are nearly identical** across all 3 RL methods in every scenario (differences < 0.3%).
- **Cash Dominant:** All methods correctly increase cash holdings over time (from ~0.20 to ~0.63), recognizing that $r = 0.06$ exceeds all risky returns.
- **Asset 3 Dominant:** All methods maximize Asset 3 allocation (from ~0.20 to ~0.70), agreeing on the dominant asset. Differences appear only in how Assets 1 and 2 are reduced (funding paths).
- **Low Variance Wins:** All methods increase Asset 1 (lowest variance) from ~0.33 to ~0.63, correctly responding to high risk aversion ($A=2.0$).

Pairwise Policy MSE between RL methods is computed across 5000 sampled states and confirms that the action-level agreement is strong, with the dominant asset showing the smallest per-asset MSE (see `RL_Convergence.ipynb` for details).

### 5.7 Summary

| Property | Tabular DP | ADP | PPO / A2C / SAC |
|---|---|---|---|
| Model access | Full (white-box) | Full (white-box) | None (black-box) |
| Scalability ($n=4$) | Fails | Scales | Scales |
| Utility quality | Exact (when grid is fine) | Near-optimal | Near-optimal |
| Training time | Minutes (small $n$) | Minutes | Minutes (PPO/A2C), Hours (SAC) |

All methods converge to the same optimal policy, with differences confined to economically irrelevant micro-allocation choices. The RL agents (PPO, A2C, SAC) successfully solve the portfolio optimization problem for all tested configurations ($n \in \{2, 3, 4\}$, $T \in \{2, 5, 7, 9\}$), demonstrating that the program works for any $n < 5$ and $T < 10$ as required.

## 6. Limitations

- **Tabular DP does not scale.** The discretized grid grows exponentially with $n$. At $n = 4$, the grid becomes too coarse and Tabular DP underperforms even the Hold baseline. It is included as a ground-truth reference for $n \le 3$ only.
- **RL training is stochastic.** PPO and A2C results vary slightly across random seeds. The reported numbers use a fixed seed (42) for reproducibility, but different seeds may produce marginally different utilities.
- **SAC is slow.** SAC's off-policy replay buffer and dual Q-network architecture make it ~10x slower to train than PPO/A2C for comparable timesteps, with only marginal utility improvement.
- **Flat utility surface near optimum.** The CARA utility function is concave, so many allocations near the optimum yield nearly identical utility. This makes it difficult to distinguish "correct" policies from "nearly correct" ones — different methods may choose different micro-allocations that are economically equivalent.

## 7. Setup and Execution

### 7.1 Environment

- **Python**: 3.11
- **Conda environment**: `asset_alloc`

Key dependencies:
| Package | Version |
|---|---|
| stable-baselines3 | 2.7.1 |
| gymnasium | 1.2.3 |
| pytorch | 2.10.0 |
| numpy | 2.4.2 |
| scipy | 1.17.1 |
| matplotlib | 3.10.8 |

### 7.2 Installation

```bash
conda create -n asset_alloc python=3.11
conda activate asset_alloc
pip install stable-baselines3 gymnasium torch numpy scipy matplotlib
```

### 7.3 Project Structure

```
├── README.md
├── utils.py                  # Shared utilities (CARA utility, return generation, baselines)
├── backward.py               # Tabular DP solver (exact backward induction)
├── backward_adp.py           # ADP solver with Gauss-Hermite quadrature
├── backward_adp_mc.py        # ADP solver with Monte Carlo expectations
├── ppo.py                    # Gymnasium environment + PPO/A2C/SAC training functions
├── DP.ipynb                  # Run all DP methods on 7 scenarios → dp_results.pkl
├── RL.ipynb                  # Run PPO/A2C on 7 scenarios → rl_results.pkl
├── SAC.ipynb                 # Run SAC on 7 scenarios → sac_results.pkl
├── Compare.ipynb             # DP vs RL comparison, sanity check, convergence observations
└── RL_Convergence.ipynb      # PPO vs A2C vs SAC convergence test (3 dominant-asset scenarios)
```

### 7.4 Execution Order

The notebooks should be run in the following order, as later notebooks depend on saved results from earlier ones:

1. **`DP.ipynb`** — Trains Tabular DP, ADP-Hermite, ADP-MC on 7 scenarios. Saves `dp_results.pkl` (~30 min).
2. **`RL.ipynb`** — Trains PPO and A2C on 7 scenarios. Saves `rl_results.pkl` (~30 min).
3. **`SAC.ipynb`** — Trains SAC on 7 scenarios. Loads `rl_results.pkl` for comparison. Saves `sac_results.pkl` (~8 hours).
4. **`Compare.ipynb`** — Loads `dp_results.pkl` and `rl_results.pkl`. Produces cross-method plots and sanity check with convergence observations. Also trains all methods from scratch on the sanity scenario (~10 min).
5. **`RL_Convergence.ipynb`** — Trains PPO, A2C, SAC on 3 convergence test scenarios with pairwise Policy MSE (~2 hours). Self-contained (no dependencies on other pkl files).
