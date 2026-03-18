"""
Approximate Dynamic Programming with fully continuous state AND action spaces.

Same backward induction as backward_adp.py, but replaces discrete action
enumeration with gradient-based action optimization through the V_{t+1}
network. All n_train states are optimized in parallel.
"""

import numpy as np
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
    """2-layer MLP mapping state -> action in [-1, 1]^n."""

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


def _sample_reachable_states(n, t, n_train, a_k, s_k, r, p_init_risky,
                             action_max, max_turnover, leverage_factor, rng):
    """Sample reachable states at time t by rolling forward with random continuous actions."""
    std_k = np.sqrt(s_k)
    W = np.ones(n_train)
    p_risky = np.tile(p_init_risky, (n_train, 1))

    for _step in range(t):
        # Random continuous actions in [-action_max, action_max]
        actions = rng.uniform(-action_max, action_max, size=(n_train, n))

        # Enforce turnover constraint
        d_cash = -actions.sum(axis=1)
        turnover = 0.5 * (np.abs(d_cash) + np.abs(actions).sum(axis=1))
        over = turnover > max_turnover + 1e-10
        if np.any(over):
            scale = np.where(over, max_turnover / np.maximum(turnover, 1e-12), 1.0)
            actions = actions * scale[:, np.newaxis]

        p_new = p_risky + actions
        new_cash = 1.0 - p_new.sum(axis=1)

        # Enforce leverage constraint
        gross = np.abs(new_cash) + np.abs(p_new).sum(axis=1)
        max_lev = 1.0 + leverage_factor * np.maximum(W, 0.0)
        over = gross > max_lev + 1e-10
        if np.any(over):
            scale = np.where(over, max_lev / np.maximum(gross, 1e-12), 1.0)
            p_new[over] = p_risky[over] + actions[over] * scale[over, np.newaxis]
            new_cash = 1.0 - p_new.sum(axis=1)

        # Sample returns and advance
        R = rng.normal(a_k, std_k, size=(n_train, n))
        port_ret = new_cash * r + (p_new * R).sum(axis=1)
        W_new = W * (1.0 + port_ret)

        denom = 1.0 + port_ret
        bankrupt = W_new <= 0.001
        p_risky_new = p_new * (1.0 + R) / denom[:, np.newaxis]

        p_risky_new[bankrupt] = p_init_risky
        W_new[bankrupt] = 0.001

        W = W_new
        p_risky = p_risky_new

    return W, p_risky


def _optimize_actions_batch(
    W_samples, p_samples, t, T, n, A, r, v_nets,
    joint_returns_t, joint_weights_t,
    action_max, max_turnover, leverage_factor,
    wealth_min, wealth_max, prop_min, prop_max,
    n_opt_steps=100, action_lr=0.01,
):
    """Optimize actions for all states in parallel via gradient ascent on E[V_{t+1}].

    Returns:
        best_actions: (n_train, n) numpy array of optimized actions
        target_values: (n_train,) numpy array of E[V*] at optimal actions
    """
    n_train = len(W_samples)
    Q = joint_returns_t.shape[0]

    w_t = torch.tensor(W_samples, dtype=torch.float32)          # (S,)
    p_t = torch.tensor(p_samples, dtype=torch.float32)          # (S, n)
    jr = joint_returns_t                                         # (Q, n)
    jw = joint_weights_t                                         # (Q,)

    # Bankruptcy utility value (constant)
    bankrupt_val = (1.0 - torch.exp(torch.tensor(-A * 0.001))) / A

    # Learnable actions, initialized to zero
    actions = torch.zeros(n_train, n, requires_grad=True)
    opt = torch.optim.Adam([actions], lr=action_lr)

    # Freeze V_{t+1} parameters to save memory
    if t + 1 < T and v_nets[t + 1] is not None:
        net_next, v_mean_next, v_std_next = v_nets[t + 1]
        for param in net_next.parameters():
            param.requires_grad_(False)

    # Pre-compute max leverage per sample
    max_lev = 1.0 + leverage_factor * torch.clamp(w_t, min=0.0)  # (S,)

    for _step in range(n_opt_steps):
        # Clamp actions to [-action_max, action_max]
        with torch.no_grad():
            actions.data.clamp_(-action_max, action_max)

            # Project: turnover constraint
            d_cash = -actions.data.sum(dim=1)
            turnover = 0.5 * (d_cash.abs() + actions.data.abs().sum(dim=1))
            over = turnover > max_turnover + 1e-10
            if over.any():
                scale = torch.where(
                    over, max_turnover / turnover.clamp(min=1e-12),
                    torch.ones_like(turnover),
                )
                actions.data.mul_(scale.unsqueeze(1))

            # Project: leverage constraint
            p_new_check = p_t + actions.data
            new_cash_check = 1.0 - p_new_check.sum(dim=1)
            gross = new_cash_check.abs() + p_new_check.abs().sum(dim=1)
            over_lev = gross > max_lev + 1e-10
            if over_lev.any():
                lev_scale = torch.where(
                    over_lev, max_lev / gross.clamp(min=1e-12),
                    torch.ones_like(gross),
                )
                actions.data.mul_(lev_scale.unsqueeze(1))

        # Forward: compute E[V_{t+1}]
        p_new = p_t + actions                          # (S, n)
        new_cash = 1.0 - p_new.sum(dim=1)             # (S,)

        # Portfolio return: (S, Q)
        port_ret = new_cash.unsqueeze(1) * r + (p_new @ jr.T)
        w_next = w_t.unsqueeze(1) * (1.0 + port_ret)  # (S, Q)

        if t + 1 == T:
            # Terminal: CARA utility (differentiable)
            w_safe = torch.clamp(w_next, min=0.001)
            vals = (1.0 - torch.exp(-A * w_safe)) / A  # (S, Q)
        else:
            # V_{t+1} network
            denom = 1.0 + port_ret                     # (S, Q)
            # p_next: (S, Q, n)
            p_next = p_new.unsqueeze(1) * (1.0 + jr.unsqueeze(0)) / denom.unsqueeze(2)

            w_clip = torch.clamp(w_next, wealth_min, wealth_max)
            p_clip = torch.clamp(p_next, prop_min, prop_max)

            # Flatten for batch V-net evaluation: (S*Q, 1+n)
            state_flat = torch.cat([
                w_clip.reshape(-1, 1), p_clip.reshape(-1, n)
            ], dim=1)
            vals_flat = net_next(state_flat)
            vals = vals_flat.reshape(n_train, Q) * v_std_next + v_mean_next

        # Mask bankruptcy
        vals = torch.where(w_next <= 0, bankrupt_val, vals)

        # Expected value: (S,)
        ev = (vals * jw.unsqueeze(0)).sum(dim=1)

        # Maximize expected value
        loss = -ev.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Final projection
    with torch.no_grad():
        actions.data.clamp_(-action_max, action_max)
        d_cash = -actions.data.sum(dim=1)
        turnover = 0.5 * (d_cash.abs() + actions.data.abs().sum(dim=1))
        over = turnover > max_turnover + 1e-10
        if over.any():
            scale = torch.where(
                over, max_turnover / turnover.clamp(min=1e-12),
                torch.ones_like(turnover),
            )
            actions.data.mul_(scale.unsqueeze(1))

        p_new_check = p_t + actions.data
        new_cash_check = 1.0 - p_new_check.sum(dim=1)
        gross = new_cash_check.abs() + p_new_check.abs().sum(dim=1)
        over_lev = gross > max_lev + 1e-10
        if over_lev.any():
            lev_scale = torch.where(
                over_lev, max_lev / gross.clamp(min=1e-12),
                torch.ones_like(gross),
            )
            actions.data.mul_(lev_scale.unsqueeze(1))

    # Compute final target values at optimized actions
    with torch.no_grad():
        p_new = p_t + actions.data
        new_cash = 1.0 - p_new.sum(dim=1)
        port_ret = new_cash.unsqueeze(1) * r + (p_new @ jr.T)
        w_next = w_t.unsqueeze(1) * (1.0 + port_ret)

        if t + 1 == T:
            w_safe = torch.clamp(w_next, min=0.001)
            vals = (1.0 - torch.exp(-A * w_safe)) / A
        else:
            denom = 1.0 + port_ret
            p_next = p_new.unsqueeze(1) * (1.0 + jr.unsqueeze(0)) / denom.unsqueeze(2)
            w_clip = torch.clamp(w_next, wealth_min, wealth_max)
            p_clip = torch.clamp(p_next, prop_min, prop_max)
            state_flat = torch.cat([
                w_clip.reshape(-1, 1), p_clip.reshape(-1, n)
            ], dim=1)
            vals_flat = net_next(state_flat)
            vals = vals_flat.reshape(n_train, Q) * v_std_next + v_mean_next

        vals = torch.where(w_next <= 0, bankrupt_val, vals)
        target_values = (vals * jw.unsqueeze(0)).sum(dim=1)

    # Re-enable gradients on V_{t+1} parameters for future use
    if t + 1 < T and v_nets[t + 1] is not None:
        for param in net_next.parameters():
            param.requires_grad_(True)

    return actions.data.numpy(), target_values.numpy()


def solve_dp(
    n=3, T=9, r=0.07, A=0.5,
    a_k=None, s_k=None,
    max_turnover=0.10, leverage_factor=2.0,
    wealth_min=0.05, wealth_max=3.5,
    prop_min=-0.5, prop_max=1.5,
    action_max=0.10,
    n_quad=5,
    hidden_size=128, n_train=5000, epochs=200, lr=1e-3,
    n_opt_steps=100, action_lr=0.01,
    p_init_risky=None,
):
    """Solve portfolio optimization via backward induction with continuous actions.

    Actions are optimized via gradient ascent through the V_{t+1} network,
    all states in parallel. No discrete action grid needed.

    Returns:
        v_nets:      dict mapping time step -> (ValueNet, v_mean, v_std) or None
        policy_nets: dict mapping time step -> PolicyNet
        config:      dict with parameters needed for simulation
    """
    device = torch.device('cpu')
    a_k = np.array(a_k if a_k is not None else [0.08, 0.06, 0.10][:n])
    s_k = np.array(s_k if s_k is not None else [0.02, 0.015, 0.04][:n])
    std_k = np.sqrt(s_k)

    if p_init_risky is None:
        w = 0.60 / n
        p_init_risky = np.array([w] * n)
    else:
        p_init_risky = np.array(p_init_risky, dtype=float)

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

    # Pre-convert to torch tensors for action optimization
    joint_returns_t = torch.tensor(joint_returns, dtype=torch.float32)
    joint_weights_t = torch.tensor(joint_weights, dtype=torch.float32)

    Q = len(joint_weights)
    print(f"ADP-Continuous Setup: n={n}, T={T}, r={r}, A={A}")
    print(f"  Mean returns:   {list(a_k)}")
    print(f"  Excess returns: {list(np.round(a_k - r, 4))}")
    print(f"  Training samples per step: {n_train}")
    print(f"  Action space: continuous, optimized via gradient ascent")
    print(f"  Action optimization: {n_opt_steps} steps, lr={action_lr}")
    print(f"  Quadrature: {n_quad}^{n} = {Q} points")
    print(f"  NN: hidden_size={hidden_size}, epochs={epochs}, lr={lr}")

    v_nets = {}
    policy_nets = {}
    v_nets[T] = None

    rng = np.random.default_rng(42)

    print(f"\n{'='*60}")
    print("Backward Induction (ADP-Continuous)")
    print(f"{'='*60}")
    total_start = time.time()

    for t in reversed(range(T)):
        step_start = time.time()

        # 1. Sample reachable states
        W_samples, p_samples = _sample_reachable_states(
            n, t, n_train, a_k, s_k, r, p_init_risky,
            action_max, max_turnover, leverage_factor, rng,
        )

        # 2-4. Optimize actions for all states in parallel
        best_actions, target_values = _optimize_actions_batch(
            W_samples, p_samples, t, T, n, A, r, v_nets,
            joint_returns_t, joint_weights_t,
            action_max, max_turnover, leverage_factor,
            wealth_min, wealth_max, prop_min, prop_max,
            n_opt_steps=n_opt_steps, action_lr=action_lr,
        )

        # 5. Train V_t network with target normalization
        states_np = np.column_stack([W_samples, p_samples])
        states_t = torch.tensor(states_np, dtype=torch.float32, device=device)
        targets_t = torch.tensor(target_values, dtype=torch.float32, device=device)

        v_mean = float(targets_t.mean())
        v_std = float(targets_t.std()) + 1e-8
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
        leverage_factor=leverage_factor,
        wealth_min=wealth_min, wealth_max=wealth_max,
        prop_min=prop_min, prop_max=prop_max,
        action_max=action_max,
        max_turnover=max_turnover,
        utility_function=utility_function,
    )

    return v_nets, policy_nets, config


def simulate_dp(v_nets, policy_nets, config, p_init=None,
                num_episodes=2000, seed=42, verbose=True):
    """Forward-simulate using policy networks (single forward pass per step)."""
    device = torch.device('cpu')
    n, T, r = config['n'], config['T'], config['r']
    a_k, std_k = config['a_k'], config['std_k']
    leverage_factor = config['leverage_factor']
    utility_function = config['utility_function']
    action_max = config['action_max']
    max_turnover = config['max_turnover']

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
            max_lev = 1.0 + leverage_factor * max(W, 0.0)

            # Policy net: single forward pass
            state = np.concatenate([[W], p_risky])
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = policy_nets[t_step](state_t).squeeze(0).numpy()
            action = action * action_max

            # Project: turnover
            d_cash = -action.sum()
            turnover = 0.5 * (np.abs(d_cash) + np.abs(action).sum())
            if turnover > max_turnover + 1e-10:
                action = action * (max_turnover / turnover)

            new_pr = p_risky + action
            new_cash = 1.0 - new_pr.sum()

            # Project: leverage
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
        print(f"Simulating {num_episodes} episodes (ADP-Continuous)")
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
if __name__ == '__main__':
    v_nets, policy_nets, config = solve_dp(
        n=3, T=5, r=0.03, A=0.5,
        a_k=[0.08, 0.06, 0.10],
        s_k=[0.02, 0.015, 0.04],
        prop_min=0.0, prop_max=1.0,
        leverage_factor=1.0,
        hidden_size=128, n_train=5000, epochs=200, lr=1e-3,
        n_opt_steps=100, action_lr=0.01,
        p_init_risky=[0.25, 0.25, 0.25],
    )

    results = simulate_dp(
        v_nets, policy_nets, config,
        p_init=[0.25, 0.25, 0.25, 0.25],
    )
