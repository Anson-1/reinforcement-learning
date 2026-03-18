import numpy as np
import itertools
import time
from scipy.special import roots_hermite
from scipy.interpolate import RegularGridInterpolator


def solve_dp(
    n=3, T=9, r=0.07, A=0.5,
    a_k=None, s_k=None,
    max_turnover=0.10, leverage_factor=2.0,
    wealth_points=15, wealth_min=0.05, wealth_max=3.5,
    prop_min=-0.5, prop_max=1.5, prop_step=0.25,
    action_step=0.05, action_max=0.10,
    n_quad=5,
):
    """Solve the portfolio optimization via backward induction.

    Returns:
        v_grids: dict mapping time step -> value function array
        policy:  dict mapping time step -> optimal action array
        config:  dict with all grid/parameter info needed for simulation
    """
    a_k = np.array(a_k if a_k is not None else [0.08, 0.06, 0.10][:n])
    s_k = np.array(s_k if s_k is not None else [0.02, 0.015, 0.04][:n])
    std_k = np.sqrt(s_k)

    def utility_function(W):
        return (1.0 - np.exp(-A * np.clip(W, -20, 100))) / A

    # Quadrature setup
    nodes_1d, weights_1d = roots_hermite(n_quad)
    weights_1d_norm = weights_1d / np.sqrt(np.pi)

    asset_returns = []
    for k in range(n):
        real_set = np.sqrt(2 * s_k[k]) * nodes_1d + a_k[k]
        asset_returns.append(real_set)

    grids = np.meshgrid(*asset_returns, indexing='ij') 
    joint_returns = np.stack([g.ravel() for g in grids], axis=1)

    wts_list = [weights_1d_norm] * n
    wgrids = np.meshgrid(*wts_list, indexing='ij')
    joint_weights = np.prod(
        np.stack([g.ravel() for g in wgrids], axis=1), axis=1
    )

    # Grids
    w_grid = np.linspace(wealth_min, wealth_max, wealth_points)
    p_single_grid = np.arange(prop_min, prop_max + prop_step * 0.5, prop_step)
    p_single_grid = np.round(p_single_grid, 6)
    n_pg = len(p_single_grid)

    all_p_indices = list(itertools.product(range(n_pg), repeat=n))
    all_p_values = np.array([
        [p_single_grid[i] for i in idx] for idx in all_p_indices
    ])

    # Actions
    action_vals = np.arange(-action_max, action_max + action_step * 0.5, action_step)
    action_vals = np.round(action_vals, 6)
    all_actions = np.array(list(itertools.product(action_vals, repeat=n)))
    d_cash = -all_actions.sum(axis=1)
    turnover = 0.5 * (np.abs(d_cash) + np.abs(all_actions).sum(axis=1))
    valid_actions = all_actions[turnover <= max_turnover + 1e-10]

    Q = len(joint_weights)
    print(f"Setup: n={n}, T={T}, r={r}, A={A}")
    print(f"  Mean returns:   {list(a_k)}")
    print(f"  Excess returns: {list(np.round(a_k - r, 4))}")
    print(f"  Wealth grid: {len(w_grid)} pts in [{wealth_min}, {wealth_max}]")
    print(f"  Prop grid: {n_pg} pts in [{p_single_grid[0]}, {p_single_grid[-1]}], step={prop_step}")
    print(f"  Prop combos: {len(all_p_indices)}")
    print(f"  Actions: {len(valid_actions)} feasible")
    print(f"  Quadrature: {n_quad}^{n} = {Q} points")

    # Storage
    grid_shape = (len(w_grid),) + (n_pg,) * n
    v_grids = {}
    policy = {}

    # Terminal Step
    v_grids[T] = np.zeros(grid_shape)
    for wi, W in enumerate(w_grid):
        v_grids[T][wi] = utility_function(W)

    # Backward induction
    print(f"\n{'='*60}")
    print("Backward Induction")
    print(f"{'='*60}")
    total_start = time.time()

    for t in reversed(range(T)):
        step_start = time.time()

        interp = RegularGridInterpolator(
            (w_grid,) + (p_single_grid,) * n,
            v_grids[t + 1],
            method='linear',
            bounds_error=False,
            fill_value=None,
        )

        V_t = np.full(grid_shape, -1e20)
        pol_t = np.zeros(grid_shape + (n,))
        n_computed = 0

        for wi, w in enumerate(w_grid):
            max_lev = 1.0 + leverage_factor * max(w, 0.0)

            for pi, pidx in enumerate(all_p_indices):
                p_risky = all_p_values[pi]

                p_cash = 1.0 - p_risky.sum()
                if np.abs(p_cash) + np.abs(p_risky).sum() > max_lev + 1.0:
                    V_t[(wi,) + pidx] = utility_function(0.01)
                    continue

                best_val = -np.inf
                best_act = None

                for a in valid_actions:
                    p_new = p_risky + a
                    new_cash = 1.0 - p_new.sum()

                    gross = np.abs(new_cash) + np.abs(p_new).sum()
                    if gross > max_lev + 1e-10:
                        continue

                    port_ret = new_cash * r + joint_returns @ p_new
                    w_next = w * (1.0 + port_ret)

                    denom = 1.0 + port_ret
                    p_next = (p_new * (1.0 + joint_returns)) / denom[:, np.newaxis]

                    w_clip = np.clip(w_next, w_grid[0], w_grid[-1])
                    p_clip = np.clip(p_next, p_single_grid[0], p_single_grid[-1])

                    pts = np.column_stack([w_clip, p_clip])
                    vals = interp(pts)

                    bankrupt = w_next <= 0
                    if np.any(bankrupt):
                        vals[bankrupt] = utility_function(0.001)

                    ev = np.dot(vals, joint_weights)

                    if ev > best_val:
                        best_val = ev
                        best_act = a.copy()

                if best_val == -np.inf:
                    best_val = utility_function(0.01)
                V_t[(wi,) + pidx] = best_val
                if best_act is not None:
                    pol_t[(wi,) + pidx] = best_act
                n_computed += 1

        v_grids[t] = V_t
        policy[t] = pol_t
        elapsed = time.time() - step_start
        print(f"  t={t}: {n_computed:,} states, {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\nTotal solve time: {total_elapsed:.1f}s")

    # Pack config for simulation
    config = dict(
        n=n, T=T, r=r, A=A, a_k=a_k, s_k=s_k, std_k=std_k,
        leverage_factor=leverage_factor,
        w_grid=w_grid, p_single_grid=p_single_grid,
        utility_function=utility_function,
    )

    return v_grids, policy, config


def simulate_dp(v_grids, policy, config, p_init=None,
                num_episodes=2000, seed=42, verbose=True):
    """Forward-simulate the optimal policy.

    Returns:
        results: dict with utilities, terminal_wealths, wealth_paths, final_props, short_steps
    """
    n, T, r, A = config['n'], config['T'], config['r'], config['A']
    a_k, std_k = config['a_k'], config['std_k']
    w_grid = config['w_grid']
    p_single_grid = config['p_single_grid']
    leverage_factor = config['leverage_factor']
    utility_function = config['utility_function']

    if p_init is None:
        w = 0.60 / n
        p_init = np.array([1.0 - n * w] + [w] * n)
    else:
        p_init = np.array(p_init, dtype=float)

    np.random.seed(seed)
    utilities = []
    terminal_wealths = []
    wealth_paths = np.zeros((num_episodes, T + 1))
    final_props = []
    short_steps = np.zeros(n, dtype=int)
    traj_first = []

    for ep in range(num_episodes):
        W = 1.0
        p = p_init.copy()
        wealth_paths[ep, 0] = W

        for t_step in range(T):
            p_risky = p[1:].copy()

            wi = int(np.argmin(np.abs(w_grid - W)))
            pidx = tuple(
                int(np.argmin(np.abs(p_single_grid - p_risky[k])))
                for k in range(n)
            )
            action = policy[t_step][(wi,) + pidx].copy()

            new_pr = p_risky + action
            new_cash = 1.0 - new_pr.sum()

            max_lev = 1.0 + leverage_factor * max(W, 0.0)
            gross = np.abs(new_cash) + np.abs(new_pr).sum()
            if gross > max_lev + 1e-10:
                scale = max_lev / gross
                new_pr = p_risky + action * scale
                new_cash = 1.0 - new_pr.sum()

            for k in range(n):
                if new_pr[k] < -1e-10:
                    short_steps[k] += 1

            if t_step == T - 1:
                final_props.append([new_cash] + list(new_pr))

            if ep == 0:
                traj_first.append((t_step, W, p.copy(), action.copy(),
                                   np.concatenate([[new_cash], new_pr])))

            R = np.random.normal(a_k, std_k)
            port_ret = new_cash * r + (new_pr * R).sum()
            W_new = W * (1.0 + port_ret)

            if W_new <= 0.001:
                W = 0.001
                p = np.concatenate([[1.0], np.zeros(n)])
            else:
                denom = 1.0 + port_ret
                p = np.concatenate([
                    [new_cash * (1.0 + r) / denom],
                    new_pr * (1.0 + R) / denom,
                ])
                W = W_new

            wealth_paths[ep, t_step + 1] = W

        utilities.append(utility_function(W))
        terminal_wealths.append(W)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Simulating {num_episodes} episodes")
        print(f"{'='*60}")

        if traj_first:
            print("\n--- Sample Episode ---")
            for ts, Ws, pb, act, pa in traj_first:
                names = [f"a{k+1}" for k in range(n)]
                before = (f"cash={pb[0]:.3f}, "
                          + ", ".join(f"{names[k]}={pb[k+1]:.3f}" for k in range(n)))
                deltas = ", ".join(f"d{names[k]}={act[k]:+.3f}" for k in range(n))
                after = (f"cash={pa[0]:.3f}, "
                         + ", ".join(f"{names[k]}={pa[k+1]:.3f}" for k in range(n)))
                flag = " [SHORT]" if any(pa[1:] < -1e-10) else ""
                print(f"  t={ts}: W={Ws:.4f}")
                print(f"    before: [{before}]")
                print(f"    action: [{deltas}]")
                print(f"    after:  [{after}]{flag}")
            print(f"  Terminal: W={terminal_wealths[0]:.4f}, U={utilities[0]:.4f}")

        avg_u = np.mean(utilities)
        std_err = np.std(utilities) / np.sqrt(num_episodes)
        total_steps = num_episodes * T

        print(f"\nResults ({num_episodes} episodes):")
        print(f"  Average utility:         {avg_u:.6f} +/- {std_err:.6f}")
        print(f"  Average terminal wealth: {np.mean(terminal_wealths):.4f}")
        print(f"  Median terminal wealth:  {np.median(terminal_wealths):.4f}")
        print(f"\nShort-selling frequency:")
        for k in range(n):
            pct = 100.0 * short_steps[k] / total_steps
            excess = a_k[k] - r
            print(f"  Asset {k+1} (mu={a_k[k]:.2f}, excess={excess:+.4f}): "
                  f"{short_steps[k]:,}/{total_steps:,} steps ({pct:.1f}%)")

        fp = np.array(final_props)
        print(f"\nAverage final allocation (t={T-1}):")
        print(f"  Cash:  {fp[:, 0].mean():.3f}")
        for k in range(n):
            print(f"  Asset {k+1}: {fp[:, k+1].mean():.3f}")

    return dict(
        utilities=np.array(utilities),
        terminal_wealths=np.array(terminal_wealths),
        wealth_paths=wealth_paths,
        final_props=np.array(final_props),
        short_steps=short_steps,
    )


# ============================================================
# Default run when executed directly
# ============================================================
if __name__ == '__main__':
    # Match dp_multi_asset.py parameters for comparison
    v_grids, policy, config = solve_dp(
        n=3, T=5, r=0.03, A=0.5,
        a_k=[0.08, 0.06, 0.10],
        s_k=[0.02, 0.015, 0.04],
        # Match dp_multi_asset.py constraints: no shorting, no leverage
        prop_min=0.0, prop_max=1.0, prop_step=0.25,
        leverage_factor=1.0,
    )

    results = simulate_dp(
        v_grids, policy, config,
        p_init=[0.25, 0.25, 0.25, 0.25],
    )
