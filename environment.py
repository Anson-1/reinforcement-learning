"""
Gymnasium environment for discrete-time asset allocation with rebalancing constraints.

Extends the single-asset CARA problem (Rao & Jelvis Section 8.4) to:
- Multiple risky assets (n >= 1)
- Portfolio weights with cash (p_0 = cash)
- 10% total rebalancing constraint per period
- CARA utility on terminal wealth
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class EnvParams:
    """Parameters for the multi-asset allocation environment.

    Uses assignment notation: a(k) = mean return, s(k) = variance of asset k.
    """
    n_assets: int = 3                                             # number of risky assets (n)
    means: list = field(default_factory=lambda: [0.10, 0.13, 0.08])  # a(k): expected returns
    variances: list = field(default_factory=lambda: [0.0225, 0.04, 0.0144])  # s(k): variances
    r: float = 0.07                                               # risk-free rate
    a: float = 1.0                                                # CARA risk aversion
    T: int = 4                                                    # horizon (periods)
    max_rebalance: float = 0.10                                   # max portfolio adjustment per period
    initial_weights: Optional[list] = None                        # p(k): initial portfolio portions
    initial_wealth: float = 1.0


class AssetAllocationEnv(gym.Env):
    """
    Multi-asset allocation environment with rebalancing constraints.

    State:  (p_0, p_1, ..., p_n, wealth, t_normalized)  — portfolio weights + wealth + normalized time
    Action: (delta_1, ..., delta_n) in [-1, 1]^n — raw weight changes (projected inside step)
    Reward: 0 for t < T, CARA utility U(W_T) = -exp(-a*W_T)/a at terminal time
    """

    metadata = {"render_modes": []}

    def __init__(self, params: EnvParams = None):
        super().__init__()
        if params is None:
            params = EnvParams()
        self.params = params
        self.n = params.n_assets
        self.means = np.array(params.means, dtype=np.float64)
        self.variances = np.array(params.variances, dtype=np.float64)
        self.stds = np.sqrt(self.variances)  # std devs for sampling returns

        if params.initial_weights is not None:
            self.init_weights = np.array(params.initial_weights, dtype=np.float64)
        else:
            # Equal weight across cash + n risky assets
            self.init_weights = np.ones(self.n + 1) / (self.n + 1)

        # State: portfolio weights (n+1) + wealth (1) + normalized time remaining (1)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(self.n + 3,), dtype=np.float64
        )
        # Action: weight changes for risky assets, raw in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n,), dtype=np.float64
        )

        self.weights = None
        self.wealth = None
        self.t = None

    def _get_obs(self):
        time_remaining = (self.params.T - self.t) / self.params.T # normalization, make the training more stable 
        return np.concatenate([self.weights, [self.wealth, time_remaining]])

    def _project_action(self, raw_action):
        """
        Project raw action to valid weight changes respecting:
        1. Total turnover <= max_rebalance
        2. All resulting weights >= 0 and sum to 1
        """
        # Scale raw action from [-1,1] to [-max_rebalance, max_rebalance] per asset
        delta_risky = raw_action * self.params.max_rebalance

        # Cash change is implicit: delta_0 = -sum(delta_k)
        delta_cash = -np.sum(delta_risky)
        delta_all = np.concatenate([[delta_cash], delta_risky])

        # Enforce total turnover constraint: sum(|delta_k|) <= max_rebalance
        total_turnover = np.sum(np.abs(delta_all))
        if total_turnover > self.params.max_rebalance:
            delta_all *= self.params.max_rebalance / total_turnover

        # Enforce non-negativity: clip so new weights >= 0
        new_weights = self.weights + delta_all
        for i in range(len(new_weights)):
            if new_weights[i] < 0:
                # Pull back the change to keep weight at 0
                delta_all[i] = -self.weights[i]
                new_weights[i] = 0.0

        # Re-normalize to sum to 1 (may drift slightly due to clipping)
        new_weights = np.maximum(new_weights, 0.0)
        weight_sum = np.sum(new_weights)
        if weight_sum > 0:
            new_weights /= weight_sum

        return new_weights

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.weights = self.init_weights.copy()
        self.wealth = self.params.initial_wealth
        self.t = 0
        return self._get_obs(), {}

    def step(self, action):
        # 1. Project action to valid weight changes
        new_weights = self._project_action(action)
        self.weights = new_weights
        chosen_weights = self.weights.copy()  # agent's allocation before returns
        wealth_before = self.wealth

        # 2. Sample returns for each risky asset: Y_k ~ N(a(k), s(k))
        Y = self.np_random.normal(self.means, self.stds)

        # 3. Update wealth: W_{t+1} = W_t * (p_0*(1+r) + sum(p_k*(1+Y_k)))
        portfolio_return = self.weights[0] * (1 + self.params.r) + np.sum(
            self.weights[1:] * (1 + Y)
        )
        self.wealth *= portfolio_return

        # 4. Weights drift based on realized returns
        new_vals = np.concatenate([
            [self.weights[0] * (1 + self.params.r)],
            self.weights[1:] * (1 + Y)
        ])
        self.weights = new_vals / np.sum(new_vals)

        # 5. Advance time
        self.t += 1
        terminated = self.t >= self.params.T

        # 6. Reward: 0 for intermediate steps, CARA utility at terminal
        if terminated:
            reward = -np.exp(-self.params.a * self.wealth) / self.params.a
        else:
            reward = 0.0

        return self._get_obs(), reward, terminated, False, {
            "wealth": self.wealth,
            "wealth_before": wealth_before,
            "weights": self.weights.copy(),
            "chosen_weights": chosen_weights,
        }


if __name__ == "__main__":
    # Quick sanity check
    params = EnvParams(n_assets=3, means=[0.10, 0.13, 0.08], variances=[0.0225, 0.04, 0.0144])
    env = AssetAllocationEnv(params)
    obs, _ = env.reset(seed=42)
    print(f"Initial obs: {obs}")
    print(f"Weights sum: {env.weights.sum():.6f}")
    print(f"Initial wealth: {env.wealth:.4f}")
    print()

    for step in range(params.T):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step+1}: reward={reward:.6f}, wealth={info['wealth']:.4f}, "
              f"weights={info['weights'].round(4)}, sum={info['weights'].sum():.6f}")

    print(f"\nTerminal wealth: {env.wealth:.4f}")
    print(f"CARA utility: {-np.exp(-params.a * env.wealth) / params.a:.6f}")

    # Verify constraints over many episodes
    print("\n--- Constraint verification (100 episodes) ---")
    env = AssetAllocationEnv(params)
    violations = 0
    for ep in range(100):
        obs, _ = env.reset(seed=ep)
        for step in range(params.T):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            w = info["weights"]
            if np.any(w < -1e-10) or abs(w.sum() - 1.0) > 1e-10:
                violations += 1
    print(f"Weight constraint violations: {violations}")
