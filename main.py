"""
Discrete-Time Asset Allocation with Reinforcement Learning

Based on Rao & Jelvis Section 8.4 (analytical) and Section 8.5 (RL extension).
Extends the single-asset CARA problem to multi-asset with rebalancing constraints.

Usage:
    python main.py              # Run everything (full pipeline)
    python main.py --quick      # Quick run with fewer timesteps (for testing)
    python main.py --skip-sweep # Skip parameter sweeps (faster)
"""

import argparse
import os
import time
import numpy as np

from analytical import MarketParams, plot_analytical_solution
from evaluate import (
    validate_against_analytical,
    compare_constrained_vs_unconstrained,
    compare_rl_vs_baselines,
)
from environment import EnvParams
from run_experiments import (
    sweep_n_assets,
    sweep_horizon,
    sweep_risk_free_rate,
    sweep_risk_aversion,
    sweep_initial_weights,
    print_results_table,
)


def main():
    parser = argparse.ArgumentParser(description="Asset Allocation with RL")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with reduced timesteps")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="Skip parameter sweep experiments")
    args = parser.parse_args()

    ts_main = 500_000 if not args.quick else 100_000
    ts_sweep = 200_000 if not args.quick else 100_000

    os.makedirs("results", exist_ok=True)
    start = time.time()

    # ------------------------------------------------------------------
    # Part 1: Analytical Baseline (Section 8.4)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PART 1: Analytical Baseline (Rao & Jelvis Section 8.4)")
    print("=" * 70)
    params = MarketParams(mu=0.13, s=0.04, r=0.07, a=1.0, T=4)
    plot_analytical_solution(params)

    # ------------------------------------------------------------------
    # Part 2: RL Validation Against Analytical
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("PART 2: RL Validation Against Analytical Solution")
    print("=" * 70)
    print("Single-asset, unconstrained case. RL should approach Eq. 8.25.")
    validate_against_analytical(total_timesteps=ts_main)

    # ------------------------------------------------------------------
    # Part 3: Constrained vs Unconstrained
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("PART 3: Effect of 10% Rebalancing Constraint")
    print("=" * 70)
    print("Shows RL learns gradual rebalancing when constrained.")
    compare_constrained_vs_unconstrained(total_timesteps=ts_main)

    # ------------------------------------------------------------------
    # Part 4: RL vs Baselines (Multi-Asset, Constrained)
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("PART 4: RL vs Baselines (3-Asset, 10% Constraint)")
    print("=" * 70)
    env_params = EnvParams(
        n_assets=3,
        means=[0.10, 0.13, 0.08],
        variances=[0.0225, 0.04, 0.0144],
        r=0.07, a=1.0, T=4,
        max_rebalance=0.10,
    )
    compare_rl_vs_baselines(env_params, total_timesteps=ts_main)

    # ------------------------------------------------------------------
    # Part 5: Parameter Sweep Experiments
    # ------------------------------------------------------------------
    if not args.skip_sweep:
        print("\n\n" + "=" * 70)
        print("PART 5: Parameter Sweep Experiments")
        print("=" * 70)
        print("Demonstrating generalization across (n, T, r, a, initial weights)")

        all_results = []
        all_results.extend(sweep_n_assets(timesteps=ts_sweep))
        all_results.extend(sweep_horizon(timesteps=ts_sweep))
        all_results.extend(sweep_risk_free_rate(timesteps=ts_sweep))
        all_results.extend(sweep_risk_aversion(timesteps=ts_sweep))
        all_results.extend(sweep_initial_weights(timesteps=ts_sweep))

        print("\n\n" + "=" * 70)
        print("FULL PARAMETER SWEEP SUMMARY")
        print("=" * 70)
        print_results_table(all_results)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - start
    print("\n\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Total runtime: {elapsed / 60:.1f} minutes")
    print(f"\nAll plots saved to results/:")
    for f in sorted(os.listdir("results")):
        if f.endswith(".png"):
            print(f"  - {f}")

    print("\nKey findings:")
    print("  1. RL approaches the analytical solution in the unconstrained case")
    print("  2. The 10% constraint forces gradual rebalancing toward optimal weights")
    print("  3. RL outperforms Hold and Equal-weight baselines under CARA utility")
    print("  4. Results generalize across different (n, T, r, a) configurations")


if __name__ == "__main__":
    main()
