"""
Parameter sweep experiments for the asset allocation RL agent.

Demonstrates the program works for any reasonable parameters:
- Vary n_assets: 2, 3, 4
- Vary T: 3, 5, 7
- Vary r: 0.02, 0.05, 0.10
- Vary risk aversion a: 0.5, 1.0, 2.0
- Vary initial weights: equal, concentrated, random
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from environment import AssetAllocationEnv, EnvParams
from train import train, make_env
from evaluate import evaluate_policy, evaluate_hold_policy, evaluate_equal_weight_policy


# s(k) = variance for each asset profile
ASSET_PROFILES = {
    2: {"means": [0.10, 0.13], "variances": [0.0225, 0.04]},
    3: {"means": [0.10, 0.13, 0.08], "variances": [0.0225, 0.04, 0.0144]},
    4: {"means": [0.10, 0.13, 0.08, 0.15], "variances": [0.0225, 0.04, 0.0144, 0.0625]},
}


def run_single_experiment(env_params, label, total_timesteps=200_000, n_eval=300):
    """Train and evaluate a single configuration. Returns result dict."""
    print(f"\n--- {label} ---")
    safe_label = label.replace(" ", "_").replace("=", "").replace(",", "").replace("(", "").replace(")", "")
    save_path = f"results/ppo_{safe_label}"

    model, callback = train(env_params, total_timesteps=total_timesteps, save_path=save_path)

    env = make_env(env_params)
    rl_utils, rl_wealths, _ = evaluate_policy(model, env, n_episodes=n_eval)

    env = make_env(env_params)
    hold_utils, hold_wealths = evaluate_hold_policy(env, n_episodes=n_eval)

    env = make_env(env_params)
    eq_utils, eq_wealths = evaluate_equal_weight_policy(env, n_episodes=n_eval)

    return {
        "label": label,
        "rl_utility": np.mean(rl_utils),
        "hold_utility": np.mean(hold_utils),
        "equal_utility": np.mean(eq_utils),
        "rl_wealth": np.mean(rl_wealths),
        "hold_wealth": np.mean(hold_wealths),
        "equal_wealth": np.mean(eq_wealths),
    }


def print_results_table(results):
    """Print a formatted summary table."""
    header = f"{'Experiment':<35} {'RL Utility':<14} {'Hold Utility':<14} {'EqWt Utility':<14} {'RL Wealth':<12}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        print(f"{r['label']:<35} {r['rl_utility']:<14.6f} {r['hold_utility']:<14.6f} "
              f"{r['equal_utility']:<14.6f} {r['rl_wealth']:<12.4f}")
    print("=" * len(header))


def plot_results_bar(results, filename, title, group_key):
    """Bar chart comparing RL vs baselines across experiments."""
    labels = [r["label"] for r in results]
    rl = [r["rl_utility"] for r in results]
    hold = [r["hold_utility"] for r in results]
    eq = [r["equal_utility"] for r in results]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2), 5))
    ax.bar(x - width, rl, width, label="RL (PPO)", color="coral", alpha=0.8)
    ax.bar(x, hold, width, label="Hold", color="steelblue", alpha=0.8)
    ax.bar(x + width, eq, width, label="Equal-weight", color="seagreen", alpha=0.8)

    ax.set_xlabel(group_key)
    ax.set_ylabel("Mean CARA Utility")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=150)
    plt.close()
    print(f"Saved results/{filename}")


def sweep_n_assets(timesteps=200_000):
    """Vary number of risky assets: 2, 3, 4."""
    print("\n" + "=" * 60)
    print("SWEEP: Number of assets (n = 2, 3, 4)")
    print("=" * 60)
    results = []
    for n in [2, 3, 4]:
        params = EnvParams(
            n_assets=n,
            means=ASSET_PROFILES[n]["means"],
            variances=ASSET_PROFILES[n]["variances"],
        )
        r = run_single_experiment(params, f"n={n}", total_timesteps=timesteps)
        results.append(r)
    print_results_table(results)
    plot_results_bar(results, "sweep_n_assets.png", "Utility vs Number of Assets", "n_assets")
    return results


def sweep_horizon(timesteps=200_000):
    """Vary horizon: T = 3, 5, 7."""
    print("\n" + "=" * 60)
    print("SWEEP: Horizon (T = 3, 5, 7)")
    print("=" * 60)
    results = []
    for T in [3, 5, 7]:
        params = EnvParams(n_assets=3, T=T, **ASSET_PROFILES[3])
        r = run_single_experiment(params, f"T={T}", total_timesteps=timesteps)
        results.append(r)
    print_results_table(results)
    plot_results_bar(results, "sweep_horizon.png", "Utility vs Horizon", "T")
    return results


def sweep_risk_free_rate(timesteps=200_000):
    """Vary risk-free rate: r = 0.02, 0.05, 0.10."""
    print("\n" + "=" * 60)
    print("SWEEP: Risk-free rate (r = 0.02, 0.05, 0.10)")
    print("=" * 60)
    results = []
    for r_val in [0.02, 0.05, 0.10]:
        params = EnvParams(n_assets=3, r=r_val, **ASSET_PROFILES[3])
        r = run_single_experiment(params, f"r={r_val}", total_timesteps=timesteps)
        results.append(r)
    print_results_table(results)
    plot_results_bar(results, "sweep_risk_free.png", "Utility vs Risk-Free Rate", "r")
    return results


def sweep_risk_aversion(timesteps=200_000):
    """Vary risk aversion: a = 0.5, 1.0, 2.0."""
    print("\n" + "=" * 60)
    print("SWEEP: Risk aversion (a = 0.5, 1.0, 2.0)")
    print("=" * 60)
    results = []
    for a_val in [0.5, 1.0, 2.0]:
        params = EnvParams(n_assets=3, a=a_val, **ASSET_PROFILES[3])
        r = run_single_experiment(params, f"a={a_val}", total_timesteps=timesteps)
        results.append(r)
    print_results_table(results)
    plot_results_bar(results, "sweep_risk_aversion.png", "Utility vs Risk Aversion", "a")
    return results


def sweep_initial_weights(timesteps=200_000):
    """Vary initial weights: equal, concentrated in asset 1, random."""
    print("\n" + "=" * 60)
    print("SWEEP: Initial weights")
    print("=" * 60)
    n = 3
    weight_configs = {
        "Equal": [1 / (n + 1)] * (n + 1),
        "Concentrated": [0.1, 0.7, 0.1, 0.1],
        "Cash-heavy": [0.7, 0.1, 0.1, 0.1],
    }
    results = []
    for name, weights in weight_configs.items():
        params = EnvParams(n_assets=n, initial_weights=weights, **ASSET_PROFILES[n])
        r = run_single_experiment(params, f"w0={name}", total_timesteps=timesteps)
        results.append(r)
    print_results_table(results)
    plot_results_bar(results, "sweep_initial_weights.png", "Utility vs Initial Weights", "Initial weights")
    return results


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    start = time.time()

    all_results = []
    all_results.extend(sweep_n_assets())
    all_results.extend(sweep_horizon())
    all_results.extend(sweep_risk_free_rate())
    all_results.extend(sweep_risk_aversion())
    all_results.extend(sweep_initial_weights())

    print("\n\n" + "=" * 70)
    print("FULL SUMMARY")
    print("=" * 70)
    print_results_table(all_results)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")
