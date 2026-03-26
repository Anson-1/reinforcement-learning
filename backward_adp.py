import numpy as np
import itertools
import time
import torch
import torch.nn as nn
from scipy.special import roots_hermite


class ValueNet(nn.Module):
    """2-layer MLP that maps state [W, p_1, ..., p_n] -> scalar value."""

    def __init__(self, input_dim, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PolicyNet(nn.Module):
    """2-layer MLP mapping state -> action in [-1, 1]^n (scaled by action_max at inference)."""

    def __init__(self, input_dim, action_dim, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def _sample_reachable_states(n, T, t, n_train, a_k, s_k, r, p_init_risky,
                             valid_actions, leverage_factor, rng):
    """Sample reachable states at time t by rolling forward from p_init with random actions.

    Returns:
        W_samples: (n_train,) wealth values
        p_samples: (n_train, n) risky proportions
    """
    std_k = np.sqrt(s_k)
    W = np.ones(n_train)
    p_risky = np.tile(p_init_risky, (n_train, 1))  # (n_train, n)

    n_valid = len(valid_actions)

    for step in range(t):
        # Pick random actions from valid set
        idx = rng.integers(0, n_valid, size=n_train)
        actions = valid_actions[idx]  # (n_train, n)

        p_new = p_risky + actions
        new_cash = 1.0 - p_new.sum(axis=1)  # (n_train,)

        # Enforce leverage constraint per sample
        gross = np.abs(new_cash) + np.abs(p_new).sum(axis=1)
        max_lev = leverage_factor
        over = gross > max_lev + 1e-10
        if np.any(over):
            scale = np.where(over, max_lev / np.maximum(gross, 1e-12), 1.0)
            p_new[over] = p_risky[over] + actions[over] * scale[over, np.newaxis]
            new_cash = 1.0 - p_new.sum(axis=1)

        # Sample returns and advance
        R = rng.normal(a_k, std_k, size=(n_train, n))  # (n_train, n)
        port_ret = new_cash * r + (p_new * R).sum(axis=1)
        W_new = W * (1.0 + port_ret)

        # Drift proportions
        denom = 1.0 + port_ret
        bankrupt = W_new <= 0.001
        p_risky_new = p_new * (1.0 + R) / denom[:, np.newaxis]

        # Handle bankruptcy
        p_risky_new[bankrupt] = p_init_risky
        W_new[bankrupt] = 0.001

        W = W_new
        p_risky = p_risky_new

    return W, p_risky


def solve_dp(
    n=3, T=9, r=0.07, A=0.5,
    a_k=None, s_k=None,
    max_turnover=0.10, leverage_factor=2.0,
    wealth_min=0.05, wealth_max=3.5,
    prop_min=None, prop_max=None,
    action_step=0.05, action_max=0.10,
    n_quad=5,
    hidden_size=128, n_train=5000, epochs=200, lr=1e-3,
    p_init=None, p_init_risky=None,
):
    """Solve portfolio optimization via backward induction with NN value approximation.

    Returns:
        v_nets:      dict mapping time step -> (ValueNet, v_mean, v_std)
                     or None for terminal
        policy_nets: dict mapping time step -> PolicyNet
        config:      dict with all parameters needed for simulation
    """
    # Auto-derive prop range from leverage_factor if not explicitly set
    if prop_min is None:
        prop_min = -(leverage_factor - 1.0) / 2.0
    if prop_max is None:
        prop_max = (leverage_factor + 1.0) / 2.0

    device = torch.device('cpu')
    a_k = np.array(a_k if a_k is not None else [0.08, 0.06, 0.10][:n])
    s_k = np.array(s_k if s_k is not None else [0.02, 0.015, 0.04][:n])
    std_k = np.sqrt(s_k)

    # Derive p_init_risky from p_init if provided
    if p_init is not None and p_init_risky is None:
        p_init = np.array(p_init, dtype=float)
        p_init_risky = p_init[1:]  # strip cash
    elif p_init_risky is None:
        w = 0.60 / n
        p_init_risky = np.array([w] * n)
        p_init = np.array([1.0 - n * w] + [w] * n)
    else:
        p_init_risky = np.array(p_init_risky, dtype=float)
        if p_init is None:
            p_init = np.array([1.0 - p_init_risky.sum()] + list(p_init_risky))

    def utility_function(W):
        return (1.0 - np.exp(-A * np.clip(W, -20, 100))) / A

    # Quadrature setup (same as tabular)
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

    # Actions (same filtering as tabular)
    action_vals = np.arange(-action_max, action_max + action_step * 0.5, action_step)
    action_vals = np.round(action_vals, 6)
    all_actions = np.array(list(itertools.product(action_vals, repeat=n)))
    d_cash = -all_actions.sum(axis=1)
    turnover = 0.5 * (np.abs(d_cash) + np.abs(all_actions).sum(axis=1))
    valid_actions = all_actions[turnover <= max_turnover + 1e-10]

    Q = len(joint_weights)
    print(f"ADP Setup: n={n}, T={T}, r={r}, A={A}")
    print(f"  Mean returns:   {list(a_k)}")
    print(f"  Excess returns: {list(np.round(a_k - r, 4))}")
    print(f"  Training samples per step: {n_train}")
    print(f"  Actions: {len(valid_actions)} feasible")
    print(f"  Quadrature: {n_quad}^{n} = {Q} points")
    print(f"  NN: hidden_size={hidden_size}, epochs={epochs}, lr={lr}")

    # Storage
    v_nets = {}  # t -> (net, v_mean, v_std) or None for terminal
    policy_nets = {}  # t -> PolicyNet

    # Terminal: no network needed, V_T = U(W)
    v_nets[T] = None

    rng = np.random.default_rng(42)

    print(f"\n{'='*60}")
    print("Backward Induction (ADP)")
    print(f"{'='*60}")
    total_start = time.time()

    for t in reversed(range(T)):
        step_start = time.time()

        # 1. Sample reachable states by rolling forward from p_init
        W_samples, p_samples = _sample_reachable_states(
            n, T, t, n_train, a_k, s_k, r, p_init_risky,
            valid_actions, leverage_factor, rng,
        )

        target_values = np.full(n_train, -1e20)
        best_actions = np.zeros((n_train, n))

        # 2-4. For each sampled state, find best action
        for i in range(n_train):
            w = W_samples[i]
            p_risky = p_samples[i]
            max_lev = leverage_factor

            # Check if current state is feasible
            p_cash = 1.0 - p_risky.sum()
            if np.abs(p_cash) + np.abs(p_risky).sum() > max_lev + 1.0:
                target_values[i] = utility_function(0.01)
                continue

            best_val = -np.inf
            zero_val = -np.inf
            best_act = np.zeros(n)

            for a in valid_actions:
                p_new = p_risky + a
                new_cash = 1.0 - p_new.sum()

                gross = np.abs(new_cash) + np.abs(p_new).sum()
                if gross > max_lev + 1e-10:
                    continue

                # Compute next-period wealth and proportions for all quadrature points
                port_ret = new_cash * r + joint_returns @ p_new
                w_next = w * (1.0 + port_ret)

                denom = 1.0 + port_ret
                p_next = (p_new * (1.0 + joint_returns)) / denom[:, np.newaxis]

                # Evaluate V_{t+1}
                if t + 1 == T:
                    # Terminal: use utility directly
                    vals = utility_function(w_next)
                else:
                    # Use V_{t+1} network (with denormalization)
                    w_clip = np.clip(w_next, wealth_min, wealth_max)
                    p_clip = np.clip(p_next, prop_min, prop_max)
                    state_next = np.column_stack([w_clip, p_clip])
                    state_t = torch.tensor(state_next, dtype=torch.float32, device=device)
                    net_next, v_mean, v_std = v_nets[t + 1]
                    with torch.no_grad():
                        vals_norm = net_next(state_t).numpy()
                    vals = vals_norm * v_std + v_mean

                # Handle bankruptcy
                bankrupt = w_next <= 0
                if np.any(bankrupt):
                    vals[bankrupt] = utility_function(0.001)

                ev = np.dot(vals, joint_weights)

                # Track zero-action value for comparison
                if np.allclose(a, 0.0):
                    zero_val = ev

                if ev > best_val:
                    best_val = ev
                    best_act = a.copy()

            if best_val == -np.inf:
                best_val = utility_function(0.01)

            # If best action is not meaningfully better than doing nothing,
            # default to zero action to avoid fitting NN approximation noise
            if zero_val > -np.inf:
                rel_improv = abs(best_val - zero_val) / (abs(zero_val) + 1e-10)
                if rel_improv < 1e-4:
                    best_act = np.zeros(n)

            target_values[i] = best_val
            best_actions[i] = best_act

        # 5. Train V_t network with target normalization
        states_np = np.column_stack([W_samples, p_samples])
        states_t = torch.tensor(states_np, dtype=torch.float32, device=device)
        targets_t = torch.tensor(target_values, dtype=torch.float32, device=device)

        # Normalize targets for stable training (skip if targets are near-constant)
        v_mean = float(targets_t.mean())
        v_std = float(targets_t.std())
        if v_std < 1e-6:
            # All targets nearly identical — normalization would amplify noise
            targets_norm = targets_t - v_mean
            v_std = 1.0  # denormalization becomes a no-op shift
        else:
            targets_norm = (targets_t - v_mean) / v_std

        net = ValueNet(1 + n, hidden_size).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _epoch in range(epochs):
            pred = net(states_t)
            loss = loss_fn(pred, targets_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        v_nets[t] = (net, v_mean, v_std)

        # 6. Train policy net: state -> action (normalized to [-1, 1])
        actions_norm = torch.tensor(
            best_actions / action_max, dtype=torch.float32, device=device
        )
        pol_net = PolicyNet(1 + n, n, hidden_size).to(device)
        pol_optimizer = torch.optim.Adam(pol_net.parameters(), lr=lr)
        for _epoch in range(epochs):
            pred_a = pol_net(states_t)
            pol_loss = nn.functional.mse_loss(pred_a, actions_norm)
            pol_optimizer.zero_grad()
            pol_loss.backward()
            pol_optimizer.step()
        pol_net.eval()
        policy_nets[t] = pol_net

        elapsed = time.time() - step_start
        print(f"  t={t}: {n_train} samples, v_loss={loss.item():.6f}, "
              f"pi_loss={pol_loss.item():.6f}, {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\nTotal solve time: {total_elapsed:.1f}s")

    config = dict(
        n=n, T=T, r=r, A=A, a_k=a_k, s_k=s_k, std_k=std_k,
        leverage_factor=leverage_factor, p_init=p_init,
        wealth_min=wealth_min, wealth_max=wealth_max,
        prop_min=prop_min, prop_max=prop_max,
        action_step=action_step, action_max=action_max,
        max_turnover=max_turnover,
        n_quad=n_quad,
        utility_function=utility_function,
        valid_actions=valid_actions,
        joint_returns=joint_returns,
        joint_weights=joint_weights,
    )

    return v_nets, policy_nets, config


def simulate_dp(v_nets, policy_nets, config, p_init=None,
                num_episodes=2000, seed=42, verbose=True, returns=None):
    """Forward-simulate the optimal policy using policy networks.

    At each step, uses the PolicyNet for fast action selection (single forward pass).

    Returns:
        results: dict with utilities, terminal_wealths, wealth_paths, final_props, short_steps
    """
    device = torch.device('cpu')
    n, T, r = config['n'], config['T'], config['r']
    a_k, std_k = config['a_k'], config['std_k']
    leverage_factor = config['leverage_factor']
    utility_function = config['utility_function']
    action_max = config['action_max']
    max_turnover = config['max_turnover']

    if p_init is None:
        p_init = config.get('p_init')
    if p_init is None:
        w = 0.60 / n
        p_init = np.array([1.0 - n * w] + [w] * n)
    else:
        p_init = np.array(p_init, dtype=float)

    if returns is None:
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
            max_lev = leverage_factor

            # Policy net: state -> action (single forward pass)
            state = np.concatenate([[W], p_risky])
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = policy_nets[t_step](state_t).squeeze(0).numpy()
            action = action * action_max  # denormalize from [-1,1] to [-action_max, action_max]

            # Project: enforce turnover constraint
            d_cash = -action.sum()
            turnover = 0.5 * (np.abs(d_cash) + np.abs(action).sum())
            if turnover > max_turnover + 1e-10:
                action = action * (max_turnover / turnover)

            new_pr = p_risky + action
            new_cash = 1.0 - new_pr.sum()

            # Project: enforce leverage constraint
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
        print(f"Simulating {num_episodes} episodes (ADP)")
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
    # Match backward.py __main__ parameters for comparison
    v_nets, policy_nets, config = solve_dp(
        n=3, T=5, r=0.03, A=0.5,
        a_k=[0.08, 0.06, 0.10],
        s_k=[0.02, 0.015, 0.04],
        prop_min=0.0, prop_max=1.0,
        leverage_factor=1.0,
        hidden_size=128, n_train=5000, epochs=200, lr=1e-3,
        p_init_risky=[0.25, 0.25, 0.25],
    )

    results = simulate_dp(
        v_nets, policy_nets, config,
        p_init=[0.25, 0.25, 0.25, 0.25],
    )
