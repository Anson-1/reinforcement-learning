"""
utils.py -- Shared utilities for portfolio allocation solvers.

Contains:
    - CARA utility function
    - Feasible action grid builder
    - Vectorized feasibility check
    - Reachable state sampling for ADP
    - Shared return generation for fair evaluation
    - Baseline policies (hold, heuristic)
"""

import itertools
import numpy as np


def cara_utility(W, A):
    """CARA (exponential) utility: u(W) = (1 - exp(-A*W)) / A."""
    return (1.0 - np.exp(-A * np.clip(W, -20, 100))) / A


def build_feasible_actions(n, action_max=0.10, action_step=0.025, max_turnover=0.10):
    """
    Build discrete action set with turnover pre-filtering.

    Each action is (delta_1, ..., delta_n) where delta_k is the change to the
    risky weight of asset k. Cash delta is implied: delta_cash = -sum(delta_risky).

    Turnover constraint: 0.5 * (|delta_cash| + sum|delta_k|) <= max_turnover

    Returns
    -------
    actions : ndarray (n_feasible, n)
    """
    vals = np.arange(-action_max, action_max + action_step * 0.5, action_step)
    vals = np.round(vals, 8)

    all_actions = np.array(list(itertools.product(vals, repeat=n)))
    delta_cash = -all_actions.sum(axis=1)
    turnover = 0.5 * (np.abs(delta_cash) + np.abs(all_actions).sum(axis=1))
    feasible = turnover <= max_turnover + 1e-10

    return all_actions[feasible]


def check_feasibility(p_risky, actions, leverage_factor):
    """
    Apply actions to states and check feasibility (leverage constraint).

    Parameters
    ----------
    p_risky : (S, n) current risky proportions
    actions : (A, n) candidate deltas
    leverage_factor : float, max gross exposure

    Returns
    -------
    p_new : (S, A, n) new risky proportions
    new_cash : (S, A) new cash proportions
    feasible : (S, A) bool mask
    """
    p_new = p_risky[:, None, :] + actions[None, :, :]  # (S, A, n)
    new_cash = 1.0 - p_new.sum(axis=2)                 # (S, A)

    gross = np.abs(new_cash) + np.abs(p_new).sum(axis=2)
    feasible = gross <= leverage_factor + 1e-10

    return p_new, new_cash, feasible


def sample_reachable_states(n, t, a_k, s_k, r, p_init_risky, actions,
                            leverage_factor, rng, n_samples):
    """
    Sample reachable (wealth, risky_proportions) at time t by rolling forward
    from p_init with random feasible actions. Vectorized.

    Returns
    -------
    W : (n_samples,)
    p_risky : (n_samples, n)
    """
    std_k = np.sqrt(s_k)
    W = np.ones(n_samples)
    p = np.tile(p_init_risky, (n_samples, 1))
    n_actions = len(actions)

    for step in range(t):
        idx = rng.integers(0, n_actions, size=n_samples)
        deltas = actions[idx]
        p_new = p + deltas
        new_cash = 1.0 - p_new.sum(axis=1)

        # Enforce leverage constraint
        gross = np.abs(new_cash) + np.abs(p_new).sum(axis=1)
        over = gross > leverage_factor + 1e-10
        if np.any(over):
            scale = leverage_factor / np.maximum(gross[over], 1e-12)
            p_new[over] = p[over] + deltas[over] * scale[:, None]
            new_cash = 1.0 - p_new.sum(axis=1)

        R = rng.normal(a_k, std_k, size=(n_samples, n))
        port_ret = new_cash * r + (p_new * R).sum(axis=1)
        W_new = W * (1.0 + port_ret)
        denom = 1.0 + port_ret
        p_new_drifted = p_new * (1.0 + R) / denom[:, None]

        bankrupt = W_new <= 0.001
        p_new_drifted[bankrupt] = p_init_risky
        W_new[bankrupt] = 0.001

        W = W_new
        p = p_new_drifted

    return W, p


def generate_shared_returns(n_episodes, T, n, a_k, s_k, seed=42):
    """
    Pre-generate return paths for fair cross-solver comparison.

    Returns
    -------
    returns : ndarray (n_episodes, T, n)
    """
    rng = np.random.default_rng(seed)
    return rng.normal(
        loc=np.array(a_k),
        scale=np.sqrt(np.array(s_k)),
        size=(n_episodes, T, n),
    )


def hold_policy(n):
    """Return a zero-action array (do nothing)."""
    return np.zeros(n)


def heuristic_policy(a_k, s_k, r, action_max=0.10):
    """
    Sharpe-ratio proportional heuristic: allocate proportional to
    each asset's excess return / std, normalized so the best asset
    gets the full action_max delta.

    Parameters
    ----------
    a_k : array-like, mean returns per asset
    s_k : array-like, variance per asset
    r : float, risk-free rate
    action_max : float, maximum delta per asset

    Returns
    -------
    action : ndarray (n,), the delta to apply
    """
    a_k = np.array(a_k)
    excess = a_k - r
    std = np.sqrt(np.array(s_k))
    scores = excess / np.maximum(std, 1e-8)
    max_abs = np.abs(scores).max()
    if max_abs < 1e-10:
        return np.zeros(len(a_k))
    return (scores / max_abs) * action_max
