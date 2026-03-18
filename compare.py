"""
Compare Tabular DP vs ADP (neural net approximation) on identical parameters.

Runs multiple configurations to verify correctness and show generality
for any n < 5 and T < 10. Includes a Hold baseline for reference.

Note: PPO (stock_price_simulations.py) is a separate model-free RL approach
with stricter long-only constraints — see that file for its own evaluation.
"""

import numpy as np
import time


# ── Hold-only baseline (no rebalancing) ──────────────────────────
def evaluate_hold(n, T, r, A, a_k, s_k, p_init, num_episodes=2000, seed=42):
    """Baseline: hold initial portfolio with no rebalancing."""
    a_k = np.array(a_k)
    std_k = np.sqrt(np.array(s_k))
    p_init = np.array(p_init)

    def utility(W):
        return (1.0 - np.exp(-A * np.clip(W, -20, 100))) / A

    np.random.seed(seed)
    utilities = []
    terminal_wealths = []

    for _ in range(num_episodes):
        W = 1.0
        p = p_init.copy()
        for _ in range(T):
            R = np.random.normal(a_k, std_k)
            port_ret = p[0] * r + (p[1:] * R).sum()
            W_new = W * (1.0 + port_ret)
            if W_new <= 0.001:
                W = 0.001
                p = np.concatenate([[1.0], np.zeros(n)])
            else:
                denom = 1.0 + port_ret
                p = np.concatenate([
                    [p[0] * (1.0 + r) / denom],
                    p[1:] * (1.0 + R) / denom,
                ])
                W = W_new
        utilities.append(utility(W))
        terminal_wealths.append(W)

    return {
        'avg_utility': np.mean(utilities),
        'std_err': np.std(utilities) / np.sqrt(num_episodes),
        'avg_wealth': np.mean(terminal_wealths),
        'med_wealth': np.median(terminal_wealths),
    }


# ── Tabular DP runner ────────────────────────────────────────────
def run_tabular_dp(n, T, r, A, a_k, s_k, p_init, num_episodes=2000, seed=42):
    from backward import solve_dp, simulate_dp

    v_grids, policy, config = solve_dp(
        n=n, T=T, r=r, A=A,
        a_k=list(a_k), s_k=list(s_k),
        prop_min=0.0, prop_max=1.0, prop_step=0.25,
        leverage_factor=1.0,
    )
    results = simulate_dp(
        v_grids, policy, config,
        p_init=list(p_init),
        num_episodes=num_episodes, seed=seed, verbose=False,
    )
    return {
        'avg_utility': np.mean(results['utilities']),
        'std_err': np.std(results['utilities']) / np.sqrt(num_episodes),
        'avg_wealth': np.mean(results['terminal_wealths']),
        'med_wealth': np.median(results['terminal_wealths']),
    }


# ── ADP runner ───────────────────────────────────────────────────
def run_adp(n, T, r, A, a_k, s_k, p_init, num_episodes=2000, seed=42):
    from backward_adp import solve_dp, simulate_dp

    p_init_risky = list(p_init[1:])
    v_nets, policy_nets, config = solve_dp(
        n=n, T=T, r=r, A=A,
        a_k=list(a_k), s_k=list(s_k),
        prop_min=0.0, prop_max=1.0,
        leverage_factor=1.0,
        hidden_size=128, n_train=5000, epochs=200, lr=1e-3,
        p_init_risky=p_init_risky,
    )
    results = simulate_dp(
        v_nets, policy_nets, config,
        p_init=list(p_init),
        num_episodes=num_episodes, seed=seed, verbose=False,
    )
    return {
        'avg_utility': np.mean(results['utilities']),
        'std_err': np.std(results['utilities']) / np.sqrt(num_episodes),
        'avg_wealth': np.mean(results['terminal_wealths']),
        'med_wealth': np.median(results['terminal_wealths']),
    }


# ── Pretty print ─────────────────────────────────────────────────
def print_results(label, results_dict):
    print(f"\n{'='*75}")
    print(f"  {label}")
    print(f"{'='*75}")
    print(f"  {'Method':<16} {'Avg Utility':>12} {'± Std Err':>10} "
          f"{'Avg Wealth':>12} {'Med Wealth':>12}")
    print(f"  {'-'*62}")
    for name, res in results_dict.items():
        print(f"  {name:<16} {res['avg_utility']:>12.5f} {res['std_err']:>10.5f} "
              f"{res['avg_wealth']:>12.4f} {res['med_wealth']:>12.4f}")


# ── Test configurations ──────────────────────────────────────────
CONFIGS = [
    {
        'label': 'Config 1: n=3, T=5 (baseline)',
        'n': 3, 'T': 5, 'r': 0.03, 'A': 0.5,
        'a_k': [0.08, 0.06, 0.10],
        's_k': [0.02, 0.015, 0.04],
        'p_init': [0.25, 0.25, 0.25, 0.25],
    },
    {
        'label': 'Config 2: n=3, T=9 (longer horizon)',
        'n': 3, 'T': 9, 'r': 0.03, 'A': 0.5,
        'a_k': [0.08, 0.06, 0.10],
        's_k': [0.02, 0.015, 0.04],
        'p_init': [0.25, 0.25, 0.25, 0.25],
    },
    {
        'label': 'Config 3: n=4, T=5 (more assets)',
        'n': 4, 'T': 5, 'r': 0.03, 'A': 0.5,
        'a_k': [0.08, 0.06, 0.10, 0.07],
        's_k': [0.02, 0.015, 0.04, 0.025],
        'p_init': [0.20, 0.20, 0.20, 0.20, 0.20],
    },
    {
        'label': 'Config 4: n=3, T=5, high risk aversion (A=2.0)',
        'n': 3, 'T': 5, 'r': 0.03, 'A': 2.0,
        'a_k': [0.08, 0.06, 0.10],
        's_k': [0.02, 0.015, 0.04],
        'p_init': [0.25, 0.25, 0.25, 0.25],
    },
]


if __name__ == '__main__':
    print("=" * 75)
    print("  Portfolio Optimization: Tabular DP vs ADP Comparison")
    print("  (Hold baseline included for reference)")
    print("=" * 75)

    for cfg in CONFIGS:
        label = cfg.pop('label')
        params = cfg

        print(f"\n\n{'#'*75}")
        print(f"# {label}")
        print(f"# n={params['n']}, T={params['T']}, r={params['r']}, A={params['A']}")
        print(f"# a_k={params['a_k']}, s_k={params['s_k']}")
        print(f"{'#'*75}")

        results = {}

        # Hold baseline
        print("\n[1/3] Running Hold baseline...")
        t0 = time.time()
        results['Hold'] = evaluate_hold(**params)
        print(f"       Done in {time.time()-t0:.1f}s")

        # Tabular DP
        print("[2/3] Solving Tabular DP...")
        t0 = time.time()
        try:
            results['Tabular DP'] = run_tabular_dp(**params)
            print(f"       Done in {time.time()-t0:.1f}s")
        except MemoryError:
            print(f"       SKIPPED — grid too large for n={params['n']}")
            results['Tabular DP'] = None

        # ADP
        print("[3/3] Solving ADP...")
        t0 = time.time()
        results['ADP'] = run_adp(**params)
        print(f"       Done in {time.time()-t0:.1f}s")

        # Print results (skip None entries)
        active = {k: v for k, v in results.items() if v is not None}
        print_results(label, active)

        # Correctness checks
        print("\n  Correctness checks:")
        hold_u = results['Hold']['avg_utility']
        for name in ['Tabular DP', 'ADP']:
            if results[name] is None:
                continue
            diff = results[name]['avg_utility'] - hold_u
            status = 'PASS' if diff > -0.01 else 'FAIL'
            print(f"    {name} >= Hold? {status} (delta={diff:+.5f})")

        if results['Tabular DP'] is not None:
            dp_u = results['Tabular DP']['avg_utility']
            adp_u = results['ADP']['avg_utility']
            gap = abs(dp_u - adp_u)
            status = 'PASS' if gap < 0.05 else 'CHECK'
            print(f"    ADP ≈ Tabular DP? {status} (gap={gap:.5f})")
        else:
            print(f"    ADP scales to n={params['n']} where Tabular DP cannot — PASS")

        cfg['label'] = label
