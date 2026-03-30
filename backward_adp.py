import numpy as np
import time
import torch
import torch.nn as nn
from scipy.special import roots_hermite
from utils import (
    cara_utility, build_feasible_actions, check_feasibility,
    sample_reachable_states,
)


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
    """2-layer MLP mapping state -> action (raw deltas, no activation on output)."""

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


def solve_dp(
    n=3, T=9, r=0.07, A=0.5,
    a_k=None, s_k=None,
    max_turnover=0.10, leverage_factor=2.0,
    wealth_min=0.05, wealth_max=3.5,
    prop_min=None, prop_max=None,
    action_step=0.025, action_max=0.10,
    n_quad=5,
    hidden_size=128, n_train=5000, epochs=200, lr=1e-3,
    p_init=None, p_init_risky=None,
):
    """Solve portfolio optimization via backward induction with NN value approximation.

    Vectorized: uses bulk numpy operations instead of per-sample Python loops.

    Returns:
        v_nets:      dict mapping time step -> (ValueNet, v_mean, v_std)
        policy_nets: dict mapping time step -> PolicyNet
        config:      dict with all parameters needed for simulation
    """
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
        p_init_risky = p_init[1:]
    elif p_init_risky is None:
        w = 0.60 / n
        p_init_risky = np.array([w] * n)
        p_init = np.array([1.0 - n * w] + [w] * n)
    else:
        p_init_risky = np.array(p_init_risky, dtype=float)
        if p_init is None:
            p_init = np.array([1.0 - p_init_risky.sum()] + list(p_init_risky))

    # Quadrature setup
    nodes_1d, weights_1d = roots_hermite(n_quad)
    weights_1d_norm = weights_1d / np.sqrt(np.pi)

    asset_returns = []
    for k in range(n):
        real_set = np.sqrt(2 * s_k[k]) * nodes_1d + a_k[k]
        asset_returns.append(real_set)

    grids = np.meshgrid(*asset_returns, indexing='ij')
    joint_returns = np.stack([g.ravel() for g in grids], axis=1)  # (Q, n)

    wts_list = [weights_1d_norm] * n
    wgrids = np.meshgrid(*wts_list, indexing='ij')
    joint_weights = np.prod(
        np.stack([g.ravel() for g in wgrids], axis=1), axis=1
    )  # (Q,)

    # Actions (feasible deltas pre-filtered by turnover)
    valid_actions = build_feasible_actions(n, action_max, action_step, max_turnover)
    n_act = len(valid_actions)

    Q = len(joint_weights)
    print(f"ADP Setup: n={n}, T={T}, r={r}, A={A}")
    print(f"  Mean returns:   {list(a_k)}")
    print(f"  Excess returns: {list(np.round(a_k - r, 4))}")
    print(f"  Training samples per step: {n_train}")
    print(f"  Actions: {n_act} feasible")
    print(f"  Quadrature: {n_quad}^{n} = {Q} points")
    print(f"  NN: hidden_size={hidden_size}, epochs={epochs}, lr={lr}")

    # Storage
    v_nets = {}
    policy_nets = {}
    v_nets[T] = None

    rng = np.random.default_rng(42)

    print(f"\n{'='*60}")
    print("Backward Induction (ADP - Vectorized)")
    print(f"{'='*60}")
    total_start = time.time()

    for t in reversed(range(T)):
        step_start = time.time()

        # 1. Sample reachable states at time t
        W_samples, p_samples = sample_reachable_states(
            n, t, a_k, s_k, r, p_init_risky,
            valid_actions, leverage_factor, rng, n_train,
        )
        S = len(W_samples)

        # 2. Vectorized feasibility check: (S, A, n), (S, A), (S, A)
        p_new, new_cash, feasible = check_feasibility(
            p_samples, valid_actions, leverage_factor
        )

        # 3. Compute Bellman targets in chunks to manage memory
        # Memory per chunk: cs * n_act * Q * 8 bytes * ~3 arrays
        mem_budget = 200_000_000  # 200 MB
        per_state = n_act * Q * 8 * 3
        chunk_size = max(1, mem_budget // per_state)

        targets = np.full(S, -1e20)
        best_actions = np.zeros((S, n))

        for s_start in range(0, S, chunk_size):
            s_end = min(s_start + chunk_size, S)
            cs = s_end - s_start

            p_new_c = p_new[s_start:s_end]       # (cs, A, n)
            nc_c = new_cash[s_start:s_end]         # (cs, A)
            feas_c = feasible[s_start:s_end]       # (cs, A)
            W_c = W_samples[s_start:s_end]         # (cs,)

            # Portfolio return: (cs, A, Q)
            port_ret = (nc_c[:, :, None] * r +
                        np.einsum('san,qn->saq', p_new_c, joint_returns))
            w_next = W_c[:, None, None] * (1.0 + port_ret)  # (cs, A, Q)

            if t + 1 == T:
                # Terminal: use utility directly
                vals = cara_utility(w_next, A)  # (cs, A, Q)
            else:
                # Use V_{t+1} network
                denom = 1.0 + port_ret  # (cs, A, Q)
                p_next = (p_new_c[:, :, None, :] *
                          (1.0 + joint_returns[None, None, :, :])) / denom[:, :, :, None]
                # p_next: (cs, A, Q, n)

                w_clip = np.clip(w_next, wealth_min, wealth_max)
                p_clip = np.clip(p_next, prop_min, prop_max)

                flat_w = w_clip.reshape(-1, 1)
                flat_p = p_clip.reshape(-1, n)
                nn_input = np.concatenate([flat_w, flat_p], axis=1)

                net_next, v_mean, v_std = v_nets[t + 1]
                with torch.no_grad():
                    inp_t = torch.tensor(nn_input, dtype=torch.float32, device=device)
                    vals_norm = net_next(inp_t).numpy()
                vals_flat = vals_norm * v_std + v_mean
                vals = vals_flat.reshape(cs, n_act, Q)

                # Handle bankruptcy
                bankrupt = w_next <= 0.001
                if np.any(bankrupt):
                    vals[bankrupt] = cara_utility(0.001, A)

            # Weighted sum over quadrature points: (cs, A)
            ev = np.einsum('saq,q->sa', vals, joint_weights)

            # Mask infeasible actions
            ev[~feas_c] = -1e20

            best_idx = np.argmax(ev, axis=1)  # (cs,)
            targets[s_start:s_end] = ev[np.arange(cs), best_idx]
            best_actions[s_start:s_end] = valid_actions[best_idx]

        # 4. Train V_t network with target normalization
        valid_mask = targets > -1e19
        if valid_mask.sum() < 10:
            print(f"  t={t}: WARNING only {valid_mask.sum()} valid states")
            net = ValueNet(1 + n, hidden_size).to(device)
            pol_net = PolicyNet(1 + n, n, hidden_size).to(device)
            v_nets[t] = (net, 0.0, 1.0)
            policy_nets[t] = pol_net
            continue

        states_np = np.column_stack([W_samples[valid_mask], p_samples[valid_mask]])
        target_vals = targets[valid_mask]

        v_mean = float(target_vals.mean())
        v_std = float(target_vals.std())
        if v_std < 1e-6:
            targets_norm = target_vals - v_mean
            v_std = 1.0
        else:
            targets_norm = (target_vals - v_mean) / v_std

        states_t = torch.tensor(states_np, dtype=torch.float32, device=device)
        targets_t = torch.tensor(targets_norm, dtype=torch.float32, device=device)

        net = ValueNet(1 + n, hidden_size).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        for _epoch in range(epochs):
            pred = net(states_t)
            loss = nn.functional.mse_loss(pred, targets_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        v_nets[t] = (net, v_mean, v_std)

        # 5. Train policy net: state -> action (normalized to [-1, 1])
        actions_norm = torch.tensor(
            best_actions[valid_mask] / action_max, dtype=torch.float32, device=device
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
        print(f"  t={t}: {valid_mask.sum()}/{S} valid, v_loss={loss.item():.6f}, "
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
    A = config['A']
    a_k, std_k = config['a_k'], config['std_k']
    leverage_factor = config['leverage_factor']
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
    alloc_history = np.zeros((num_episodes, T, n + 1))
    final_props = []
    short_steps = np.zeros(n, dtype=int)

    for ep in range(num_episodes):
        W = 1.0
        p = p_init.copy()
        wealth_paths[ep, 0] = W

        for t_step in range(T):
            p_risky = p[1:].copy()

            # Policy net: state -> action
            state = np.concatenate([[W], p_risky])
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = policy_nets[t_step](state_t).squeeze(0).numpy()
            action = action * action_max

            # Project: enforce turnover constraint
            d_cash = -action.sum()
            turnover = 0.5 * (np.abs(d_cash) + np.abs(action).sum())
            if turnover > max_turnover + 1e-10:
                action = action * (max_turnover / turnover)

            new_pr = p_risky + action
            new_cash = 1.0 - new_pr.sum()

            # Project: enforce leverage constraint
            gross = np.abs(new_cash) + np.abs(new_pr).sum()
            if gross > leverage_factor + 1e-10:
                scale = leverage_factor / gross
                new_pr = p_risky + action * scale
                new_cash = 1.0 - new_pr.sum()

            alloc_history[ep, t_step] = np.concatenate([[new_cash], new_pr])

            for k in range(n):
                if new_pr[k] < -1e-10:
                    short_steps[k] += 1

            if t_step == T - 1:
                final_props.append([new_cash] + list(new_pr))

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

        utilities.append(cara_utility(W, A))
        terminal_wealths.append(W)

    utilities = np.array(utilities)
    terminal_wealths = np.array(terminal_wealths)

    if verbose:
        avg_u = np.mean(utilities)
        std_err = np.std(utilities) / np.sqrt(num_episodes)
        print(f"\nADP Results ({num_episodes} episodes):")
        print(f"  Average utility:         {avg_u:.6f} +/- {std_err:.6f}")
        print(f"  Average terminal wealth: {np.mean(terminal_wealths):.4f}")
        print(f"  Median terminal wealth:  {np.median(terminal_wealths):.4f}")

    return dict(
        utilities=utilities,
        terminal_wealths=terminal_wealths,
        wealth_paths=wealth_paths,
        alloc_history=alloc_history,
        final_props=np.array(final_props) if final_props else np.array([]),
        short_steps=short_steps,
        avg_utility=float(np.mean(utilities)),
        std_err=float(np.std(utilities) / np.sqrt(num_episodes)),
        avg_wealth=float(np.mean(terminal_wealths)),
        med_wealth=float(np.median(terminal_wealths)),
    )


# ============================================================
# Default run when executed directly
# ============================================================
if __name__ == '__main__':
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
