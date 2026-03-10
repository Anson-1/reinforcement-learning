"""
Analytical solution for the unconstrained single-asset CARA allocation problem.

Based on Rao & Jelvis, Section 8.4, Equation 8.25:
    x*_t = (mu - r) / (s * a * (1+r)^(T-t-1))

where (using assignment notation):
    mu    = a(k), expected return of the risky asset
    s     = s(k), variance of the risky asset return
    r     = risk-free rate
    a     = CARA risk-aversion coefficient
    T     = investment horizon (number of periods)
    t     = current time step
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class MarketParams:
    """Parameters for the single-asset CARA allocation problem.

    Uses assignment notation: s = variance (not std dev).
    """
    mu: float = 0.13       # a(k): expected return of risky asset
    s: float = 0.04        # s(k): variance of risky asset return (sigma^2)
    r: float = 0.07        # risk-free rate
    a: float = 1.0         # CARA risk-aversion coefficient
    T: int = 4             # investment horizon (number of periods)
    W0: float = 1.0        # initial wealth


def optimal_allocation(params: MarketParams, t: int) -> float:
    """Optimal dollar amount in the risky asset at time t (Eq. 8.25).

    x*_t = (mu - r) / (s * a * (1+r)^(T-t-1))
    where s is the variance.
    """
    return (params.mu - params.r) / (
        params.s * params.a * (1 + params.r) ** (params.T - t - 1)
    )


def cara_utility(W, a: float):
    """CARA utility: U(W) = -exp(-a*W) / a (textbook form, Eq. 8.26 compatible)."""
    return -np.exp(-a * W) / a


def simulate_optimal_policy(params: MarketParams, n_sims: int = 10000, seed: int = 42):
    """
    Simulate the unconstrained optimal policy over T periods.

    Returns:
        wealth_paths: array of shape (n_sims, T+1) with wealth at each time step
        allocations:  array of shape (T,) with optimal allocation at each step
    """
    rng = np.random.default_rng(seed)
    std = np.sqrt(params.s)  # convert variance to std dev for sampling
    wealth = np.full(n_sims, params.W0, dtype=np.float64)
    wealth_paths = np.zeros((n_sims, params.T + 1))
    wealth_paths[:, 0] = params.W0
    allocations = np.zeros(params.T)

    for t in range(params.T):
        x_star = optimal_allocation(params, t)
        allocations[t] = x_star
        # Risky asset return ~ N(mu, s) where s is variance
        Y = rng.normal(params.mu, std, size=n_sims)
        # W_{t+1} = x_t*(1 + Y_t) + (W_t - x_t)*(1 + r)
        wealth = x_star * (1 + Y) + (wealth - x_star) * (1 + params.r)
        wealth_paths[:, t + 1] = wealth

    return wealth_paths, allocations


def optimal_value(params: MarketParams, t: int, W: float) -> float:
    """
    Optimal value function V*_t(W) for the CARA problem (Eq. 8.26).

    V*_t(W) = -exp(-a * W * (1+r)^(T-t) - (T-t)*(mu-r)^2 / (2*s)) / a
    where s is the variance.
    """
    periods_left = params.T - t
    certainty_equiv = -params.a * W * (1 + params.r) ** periods_left
    risk_premium = -periods_left * (params.mu - params.r) ** 2 / (2 * params.s)
    return -np.exp(certainty_equiv + risk_premium) / params.a


def plot_analytical_solution(params: MarketParams = None):
    """Plot the analytical solution: allocations, wealth distribution, value function."""
    if params is None:
        params = MarketParams()

    # Optimal allocations over time
    times = np.arange(params.T)
    allocs = [optimal_allocation(params, t) for t in times]

    # Simulate wealth paths
    wealth_paths, _ = simulate_optimal_policy(params)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Optimal allocation over time
    axes[0].bar(times, allocs, color="steelblue", alpha=0.8)
    axes[0].set_xlabel("Time step t")
    axes[0].set_ylabel("Optimal risky allocation x*_t")
    axes[0].set_title("Optimal Allocation (Eq. 8.25)")
    axes[0].set_xticks(times)

    # Plot 2: Wealth distribution at each time step
    for t in range(params.T + 1):
        parts = axes[1].violinplot(wealth_paths[:, t], positions=[t], showmeans=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.5)
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Wealth")
    axes[1].set_title("Wealth Distribution Under Optimal Policy")

    # Plot 3: Terminal wealth histogram
    terminal_wealth = wealth_paths[:, -1]
    axes[2].hist(terminal_wealth, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
    axes[2].axvline(np.mean(terminal_wealth), color="red", linestyle="--",
                    label=f"Mean = {np.mean(terminal_wealth):.3f}")
    axes[2].set_xlabel("Terminal Wealth W_T")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Terminal Wealth Distribution")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("results/analytical_solution.png", dpi=150)
    plt.close()
    print("Saved results/analytical_solution.png")

    # Print summary
    print(f"\nParameters: {params}")
    print(f"\nOptimal allocations by time step:")
    for t in range(params.T):
        print(f"  t={t}: x*_t = {allocs[t]:.4f}")
    print(f"\nTerminal wealth stats:")
    print(f"  Mean:   {np.mean(terminal_wealth):.4f}")
    print(f"  Std:    {np.std(terminal_wealth):.4f}")
    print(f"  Median: {np.median(terminal_wealth):.4f}")
    mean_utility = np.mean(cara_utility(terminal_wealth, params.a))
    print(f"  Mean CARA utility: {mean_utility:.6f}")
    print(f"  Optimal value V*_0(W0): {optimal_value(params, 0, params.W0):.6f}")


if __name__ == "__main__":
    plot_analytical_solution()
