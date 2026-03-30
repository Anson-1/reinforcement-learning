import numpy as np
import itertools
import time
from scipy.special import roots_hermite
from scipy.interpolate import RegularGridInterpolator
from utils import cara_utility, build_feasible_actions


def solve_dp(
    n=3, T=9, r=0.07, A=0.5,
    a_k=None, s_k=None, p_init=None,
    max_turnover=0.10, leverage_factor=1.0,
    wealth_points=15, wealth_min=0.05, wealth_max=3.5,
    prop_min=None, prop_max=None, prop_step=0.25,
    action_step=0.025, action_max=0.10,
    n_quad=5,
):
    """Solve the portfolio optimization via backward induction.

    Returns:
        v_grids: dict mapping time step -> value function array
        policy:  dict mapping time step -> optimal action array
        config:  dict with all grid/parameter info needed for simulation
    """
    # Auto-derive prop range from leverage_factor if not explicitly set
    if prop_min is None:
        prop_min = -(leverage_factor - 1.0) / 2.0
    if prop_max is None:
        prop_max = (leverage_factor + 1.0) / 2.0

    a_k = np.array(a_k if a_k is not None else [0.08, 0.06, 0.10][:n])
    s_k = np.array(s_k if s_k is not None else [0.02, 0.015, 0.04][:n])
    std_k = np.sqrt(s_k)

    # Default p_init: 40% cash, rest equally split among risky assets
    if p_init is None:
        w = 0.60 / n
        p_init = np.array([1.0 - n * w] + [w] * n)
    else:
        p_init = np.array(p_init, dtype=float)

    def utility_function(W):
        return cara_utility(W, A)

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
    valid_actions = build_feasible_actions(n, action_max, action_step, max_turnover)

    # Find zero action index for tie-breaking
    zero_act_idx = None
    for ai, a in enumerate(valid_actions):
        if np.allclose(a, 0.0):
            zero_act_idx = ai
            break

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

        Q = len(joint_weights)

        for wi, w in enumerate(w_grid):
            max_lev = leverage_factor

            for pi, pidx in enumerate(all_p_indices):
                p_risky = all_p_values[pi]

                p_cash = 1.0 - p_risky.sum()
                if np.abs(p_cash) + np.abs(p_risky).sum() > max_lev + 1e-10:
                    V_t[(wi,) + pidx] = utility_function(0.01)
                    continue

                # All candidate allocations at once: (N_act, n)
                new_p = p_risky + valid_actions
                new_cash = 1.0 - new_p.sum(axis=1)

                # Filter by leverage constraint
                gross = np.abs(new_cash) + np.abs(new_p).sum(axis=1)
                fidx = np.where(gross <= max_lev + 1e-10)[0]

                if len(fidx) == 0:
                    V_t[(wi,) + pidx] = utility_function(0.01)
                    continue

                new_p_f = new_p[fidx]          # (F, n)
                new_cash_f = new_cash[fidx]    # (F,)
                F = len(fidx)

                # Portfolio returns for all actions × all scenarios: (F, Q)
                port_ret = new_cash_f[:, None] * r + new_p_f @ joint_returns.T

                # Next-period wealth: (F, Q)
                w_next = w * (1.0 + port_ret)

                # Next-period proportions: (F, Q, n)
                denom = 1.0 + port_ret
                p_next = (new_p_f[:, None, :] * (1.0 + joint_returns[None, :, :])) / denom[:, :, None]

                # Clip for interpolation
                w_clip = np.clip(w_next, w_grid[0], w_grid[-1])
                p_clip = np.clip(p_next, p_single_grid[0], p_single_grid[-1])

                # Assemble interpolation points: (F*Q, 1+n)
                pts = np.column_stack([w_clip.ravel(), p_clip.reshape(F * Q, n)])
                vals = interp(pts)

                # Handle bankruptcy
                bankrupt = w_next.ravel() <= 0.001
                if np.any(bankrupt):
                    vals[bankrupt] = utility_function(0.001)

                # Expected values per action: (F,)
                evs = vals.reshape(F, Q) @ joint_weights

                best = np.argmax(evs)

                # Prefer zero action when improvement is negligible
                # (avoids floating-point noise picking random actions
                #  when all values are nearly identical, e.g. A=50)
                if zero_act_idx is not None and zero_act_idx in fidx:
                    zero_pos = np.where(fidx == zero_act_idx)[0][0]
                    zero_ev = evs[zero_pos]
                    best_ev = evs[best]
                    rel_improv = abs(best_ev - zero_ev) / (abs(zero_ev) + 1e-10)
                    if rel_improv < 1e-4:
                        best = zero_pos

                V_t[(wi,) + pidx] = evs[best]
                pol_t[(wi,) + pidx] = valid_actions[fidx[best]]
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
        leverage_factor=leverage_factor, p_init=p_init,
        w_grid=w_grid, p_single_grid=p_single_grid,
        utility_function=utility_function,
    )

    return v_grids, policy, config


def simulate_dp(v_grids, policy, config, p_init=None,
                num_episodes=2000, seed=42, verbose=True, returns=None):
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
        p_init = config.get('p_init')
    if p_init is None:
        w = 0.60 / n
        p_init = np.array([1.0 - n * w] + [w] * n)
    else:
        p_init = np.array(p_init, dtype=float)

    if returns is None:
        np.random.seed(seed)

    # Build policy interpolators for smooth action lookup
    policy_interps = {}
    for t_step in range(T):
        pol_t = policy[t_step]  # shape: (wealth_points, n_pg, ..., n_pg, n)
        interps = []
        for k in range(n):
            interp = RegularGridInterpolator(
                (w_grid,) + (p_single_grid,) * n,
                pol_t[..., k],
                method='linear',
                bounds_error=False,
                fill_value=None,
            )
            interps.append(interp)
        policy_interps[t_step] = interps

    utilities = []
    terminal_wealths = []
    wealth_paths = np.zeros((num_episodes, T + 1))
    alloc_history = np.zeros((num_episodes, T, n + 1))
    final_props = []
    short_steps = np.zeros(n, dtype=int)
    traj_first = []

    for ep in range(num_episodes):
        W = 1.0
        p = p_init.copy()
        wealth_paths[ep, 0] = W

        for t_step in range(T):
            p_risky = p[1:].copy()

            # Interpolate policy for smooth action (instead of snap-to-grid)
            W_clip = np.clip(W, w_grid[0], w_grid[-1])
            p_clip = np.clip(p_risky, p_single_grid[0], p_single_grid[-1])
            state_pt = np.concatenate([[W_clip], p_clip]).reshape(1, -1)
            action = np.array([
                policy_interps[t_step][k](state_pt)[0] for k in range(n)
            ])

            new_pr = p_risky + action
            new_cash = 1.0 - new_pr.sum()

            max_lev = leverage_factor
            gross = np.abs(new_cash) + np.abs(new_pr).sum()
            if gross > max_lev + 1e-10:
                scale = max_lev / gross
                new_pr = p_risky + action * scale
                new_cash = 1.0 - new_pr.sum()

            for k in range(n):
                if new_pr[k] < -1e-10:
                    short_steps[k] += 1

            alloc_history[ep, t_step] = np.concatenate([[new_cash], new_pr])

            if t_step == T - 1:
                final_props.append([new_cash] + list(new_pr))

            if ep == 0:
                traj_first.append((t_step, W, p.copy(), action.copy(),
                                   np.concatenate([[new_cash], new_pr])))

            R = returns[ep, t_step] if returns is not None else np.random.normal(a_k, std_k)
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
        alloc_history=alloc_history,
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
