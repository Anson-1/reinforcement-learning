"""
Evaluation, validation, and baseline comparison for the asset allocation RL agent.

Includes:
- Validation against analytical solution (single-asset, unconstrained)
- Baseline policies: Hold, Equal-weight
- Plots: weight trajectories, wealth distributions, RL vs analytical
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from environment import AssetAllocationEnv, EnvParams
from analytical import MarketParams, optimal_allocation, optimal_value, cara_utility
from train import train, make_env


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_policy(model, env, n_episodes=1000, deterministic=True):
    """Evaluate a trained model. Returns terminal utilities and wealth arrays."""
    utilities = []
    wealths = []
    weight_trajectories = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=2000 + ep)
        weights_ep = [env.unwrapped.weights.copy()]
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            weights_ep.append(info["weights"].copy())
            done = terminated or truncated
        wealths.append(info["wealth"])
        utilities.append(reward)
        weight_trajectories.append(weights_ep)

    return np.array(utilities), np.array(wealths), weight_trajectories


def evaluate_hold_policy(env, n_episodes=1000):
    """Hold policy: never rebalance (action = 0)."""
    utilities = []
    wealths = []
    n_assets = env.unwrapped.params.n_assets

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=2000 + ep)
        done = False
        while not done:
            action = np.zeros(n_assets)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        wealths.append(info["wealth"])
        utilities.append(reward)

    return np.array(utilities), np.array(wealths)


def evaluate_equal_weight_policy(env, n_episodes=1000):
    """Equal-weight policy: rebalance toward 1/(n+1) each period (subject to constraint)."""
    utilities = []
    wealths = []
    n_assets = env.unwrapped.params.n_assets
    target = 1.0 / (n_assets + 1)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=2000 + ep)
        done = False
        while not done:
            # Push toward equal weight: action = direction toward target
            # obs layout: [cash_w, asset1_w, ..., assetN_w, wealth, time_remaining]
            current_risky = obs[1:n_assets + 1]
            direction = target - current_risky
            # Scale to [-1, 1] range for the environment
            max_dir = np.max(np.abs(direction)) if np.max(np.abs(direction)) > 0 else 1.0
            action = direction / max_dir
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        wealths.append(info["wealth"])
        utilities.append(reward)

    return np.array(utilities), np.array(wealths)


# ---------------------------------------------------------------------------
# Validation: RL vs analytical (single-asset, unconstrained)
# ---------------------------------------------------------------------------

def validate_against_analytical(
    mu=0.10, s=0.04, r=0.07, a=1.0, T=4, W0=1.0,
    total_timesteps=500_000,
):
    """
    Train RL on single-asset unconstrained case and compare to analytical solution.
    Uses mu=0.10, s=0.04 (variance), a=1.0 so that optimal allocation x*_t < 1
    (no leverage needed) and the utility signal is strong enough for PPO to learn.
    """
    print("=" * 60)
    print("VALIDATION: RL vs Analytical (single-asset, unconstrained)")
    print("=" * 60)

    # Analytical solution
    mkt = MarketParams(mu=mu, s=s, r=r, a=a, T=T, W0=W0)
    analytical_allocs = [optimal_allocation(mkt, t) for t in range(T)]
    analytical_V0 = optimal_value(mkt, 0, W0)
    print(f"\nParameters: mu={mu}, s(variance)={s}, r={r}, a={a}, T={T}")
    print(f"Analytical allocations: {[f'{x:.4f}' for x in analytical_allocs]}")
    print(f"Analytical V*_0(W0):    {analytical_V0:.6f}")

    # Train RL on matching environment (no rebalancing constraint)
    env_params = EnvParams(
        n_assets=1,
        means=[mu],
        variances=[s],
        r=r,
        a=a,
        T=T,
        max_rebalance=2.0,  # effectively unconstrained
        initial_weights=[1 - analytical_allocs[0], analytical_allocs[0]],
        initial_wealth=W0,
    )

    print(f"\nTraining PPO ({total_timesteps} timesteps)...")
    model, callback = train(
        env_params,
        total_timesteps=total_timesteps,
        save_path="results/ppo_validation",
    )

    # Evaluate RL: record allocations at each step
    env = make_env(env_params)
    n_eval = 1000
    rl_allocs_by_step = [[] for _ in range(T)]  # dollar amounts (weight * wealth)
    rl_weights_by_step = [[] for _ in range(T)]  # raw weights
    rl_utilities = []

    for ep in range(n_eval):
        obs, _ = env.reset(seed=3000 + ep)
        for t in range(T):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            # chosen_weights = agent's allocation BEFORE returns realize
            risky_weight = info["chosen_weights"][1]
            wealth_before = info["wealth_before"]
            # Dollar amount = weight * wealth (comparable to analytical x*_t)
            rl_allocs_by_step[t].append(risky_weight * wealth_before)
            rl_weights_by_step[t].append(risky_weight)
            if terminated:
                rl_utilities.append(reward)

    rl_mean_dollars = [np.mean(a) for a in rl_allocs_by_step]
    rl_mean_weights = [np.mean(w) for w in rl_weights_by_step]
    rl_mean_utility = np.mean(rl_utilities)

    print(f"\n{'Step':<6} {'Analytical x*_t':<18} {'RL mean $alloc':<18} {'RL mean weight':<18}")
    print("-" * 60)
    for t in range(T):
        print(f"t={t:<4} {analytical_allocs[t]:<18.4f} {rl_mean_dollars[t]:<18.4f} {rl_mean_weights[t]:<18.4f}")

    print(f"\nRL mean utility:        {rl_mean_utility:.6f}")
    print(f"Analytical V*_0(W0):    {analytical_V0:.6f}")
    print(f"Gap:                    {abs(rl_mean_utility - analytical_V0):.6f}")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Allocation comparison (dollar amounts — apples-to-apples)
    times = np.arange(T)
    width = 0.35
    axes[0].bar(times - width / 2, analytical_allocs, width, label="Analytical", color="steelblue", alpha=0.8)
    axes[0].bar(times + width / 2, rl_mean_dollars, width, label="RL (PPO)", color="coral", alpha=0.8)
    axes[0].set_xlabel("Time step t")
    axes[0].set_ylabel("Dollar amount in risky asset")
    axes[0].set_title("Allocation: Analytical vs RL (dollar amounts)")
    axes[0].set_xticks(times)
    axes[0].legend()

    # Utility comparison
    axes[1].hist(rl_utilities, bins=50, color="coral", alpha=0.7, edgecolor="black", label="RL episodes")
    axes[1].axvline(analytical_V0, color="steelblue", linewidth=2, linestyle="--",
                    label=f"Analytical V*_0 = {analytical_V0:.4f}")
    axes[1].axvline(rl_mean_utility, color="coral", linewidth=2, linestyle="--",
                    label=f"RL mean = {rl_mean_utility:.4f}")
    axes[1].set_xlabel("Terminal CARA Utility")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Terminal Utility Distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("results/validation_rl_vs_analytical.png", dpi=150)
    plt.close()
    print("\nSaved results/validation_rl_vs_analytical.png")

    return model


# ---------------------------------------------------------------------------
# Constrained vs unconstrained comparison
# ---------------------------------------------------------------------------

def compare_constrained_vs_unconstrained(
    mu=0.10, s=0.04, r=0.07, a=1.0, T=4,
    total_timesteps=500_000,
):
    """Show that the 10% constraint changes RL behavior vs unconstrained."""
    print("\n" + "=" * 60)
    print("COMPARISON: Constrained (10%) vs Unconstrained")
    print("=" * 60)

    mkt = MarketParams(mu=mu, s=s, r=r, a=a, T=T)
    analytical_allocs = [optimal_allocation(mkt, t) for t in range(T)]
    analytical_fracs = [x / mkt.W0 for x in analytical_allocs]

    results = {}
    for label, max_reb in [("Unconstrained", 2.0), ("Constrained (10%)", 0.10)]:
        print(f"\nTraining {label}...")
        env_params = EnvParams(
            n_assets=1, means=[mu], variances=[s], r=r, a=a, T=T,
            max_rebalance=max_reb,
            initial_weights=[0.5, 0.5],
        )
        model, _ = train(env_params, total_timesteps=total_timesteps,
                         save_path=f"results/ppo_{label.lower().replace(' ', '_')}")

        env = make_env(env_params)
        utils, wealths, weight_trajs = evaluate_policy(model, env, n_episodes=500)
        # Average weight trajectory
        avg_weights = np.mean([np.array(wt) for wt in weight_trajs], axis=0)
        results[label] = {
            "utilities": utils, "wealths": wealths,
            "avg_weights": avg_weights, "model": model,
        }
        print(f"  Mean utility: {np.mean(utils):.6f}, Mean wealth: {np.mean(wealths):.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for label, data in results.items():
        axes[0].plot(data["avg_weights"][:, 1], marker="o", label=label)
    axes[0].axhline(analytical_fracs[0], color="gray", linestyle=":", label="Analytical optimal")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Risky asset weight")
    axes[0].set_title("Weight Trajectory: Constrained vs Unconstrained")
    axes[0].legend()

    for label, data in results.items():
        axes[1].hist(data["utilities"], bins=40, alpha=0.5, label=f"{label} (mean={np.mean(data['utilities']):.4f})")
    axes[1].set_xlabel("Terminal CARA Utility")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Utility Distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("results/constrained_vs_unconstrained.png", dpi=150)
    plt.close()
    print("\nSaved results/constrained_vs_unconstrained.png")


# ---------------------------------------------------------------------------
# RL vs baselines (multi-asset, constrained)
# ---------------------------------------------------------------------------

def compare_rl_vs_baselines(
    env_params: EnvParams = None,
    total_timesteps=500_000,
):
    """Compare RL against Hold and Equal-weight baselines."""
    if env_params is None:
        env_params = EnvParams()

    print("\n" + "=" * 60)
    print("COMPARISON: RL vs Baselines (multi-asset, constrained)")
    print("=" * 60)
    print(f"Params: {env_params}")

    # Train RL
    print("\nTraining PPO...")
    model, _ = train(env_params, total_timesteps=total_timesteps,
                     save_path="results/ppo_baselines")

    env = make_env(env_params)
    rl_utils, rl_wealths, rl_trajs = evaluate_policy(model, env, n_episodes=500)

    # Baselines
    env = make_env(env_params)
    hold_utils, hold_wealths = evaluate_hold_policy(env, n_episodes=500)

    env = make_env(env_params)
    eq_utils, eq_wealths = evaluate_equal_weight_policy(env, n_episodes=500)

    # Summary
    print(f"\n{'Policy':<20} {'Mean Utility':<15} {'Std Utility':<15} {'Mean Wealth':<15}")
    print("-" * 65)
    for name, utils, wealths in [
        ("RL (PPO)", rl_utils, rl_wealths),
        ("Hold", hold_utils, hold_wealths),
        ("Equal-weight", eq_utils, eq_wealths),
    ]:
        print(f"{name:<20} {np.mean(utils):<15.6f} {np.std(utils):<15.6f} {np.mean(wealths):<15.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, utils, color in [
        ("RL (PPO)", rl_utils, "coral"),
        ("Hold", hold_utils, "steelblue"),
        ("Equal-weight", eq_utils, "seagreen"),
    ]:
        axes[0].hist(utils, bins=40, alpha=0.5, color=color,
                     label=f"{name} ({np.mean(utils):.4f})")
    axes[0].set_xlabel("Terminal CARA Utility")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Utility: RL vs Baselines")
    axes[0].legend()

    # Average weight trajectory for RL
    avg_rl_weights = np.mean([np.array(wt) for wt in rl_trajs], axis=0)
    n_assets = env_params.n_assets
    labels = ["Cash"] + [f"Asset {k+1}" for k in range(n_assets)]
    for k in range(n_assets + 1):
        axes[1].plot(avg_rl_weights[:, k], marker="o", label=labels[k])
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Portfolio weight")
    axes[1].set_title("RL Average Weight Trajectory")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("results/rl_vs_baselines.png", dpi=150)
    plt.close()
    print("\nSaved results/rl_vs_baselines.png")


if __name__ == "__main__":
    # 1. Validate RL against analytical
    validate_against_analytical(total_timesteps=500_000)

    # 2. Constrained vs unconstrained
    compare_constrained_vs_unconstrained(total_timesteps=500_000)

    # 3. RL vs baselines (3-asset constrained)
    compare_rl_vs_baselines(total_timesteps=500_000)
