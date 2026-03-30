"""
ppo.py -- Gymnasium environment + PPO/A2C solvers via stable-baselines3.

Model-free RL approaches: do not exploit known Gaussian return structure.
More sample-inefficient than DP methods but robust to model misspecification.

Uses gamma=1.0 (no discounting) since only the terminal reward matters.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from utils import cara_utility


class PortfolioEnv(gym.Env):
    """Gymnasium environment for constrained multi-asset portfolio allocation.

    Observation: [t/T, W, p_cash, p_1, ..., p_n]  (time normalized to [0,1])
    Action: [-1, 1]^n, scaled by action_max to get portfolio deltas
    Reward: 0 at non-terminal, CARA utility u(W_T) at terminal
    """

    def __init__(self, n=3, T=9, r=0.07, A=0.5, a_k=None, s_k=None, p_init=None,
                 leverage_factor=2.0, action_max=0.10, max_turnover=0.10):
        super().__init__()

        self.n = n
        self.T = T
        self.r = r
        self.A = A
        self.leverage_factor = leverage_factor
        self.action_max = action_max
        self.max_turnover = max_turnover

        self.a_k = np.array(a_k) if a_k is not None else np.array([0.08, 0.06, 0.10][:n])
        self.s_k = np.array(s_k) if s_k is not None else np.array([0.02, 0.015, 0.04][:n])

        if p_init is None:
            w = 0.60 / n
            self.p_init = np.array([1.0 - n * w] + [w] * n)
        else:
            self.p_init = np.array(p_init)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
        # [t/T, W, p_cash, p_1, ..., p_n]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n + 3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.wealth = 1.0
        self.p = self.p_init.copy()
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate(([self.t / self.T, self.wealth], self.p)).astype(np.float32)

    def step(self, action):
        actual_action = np.asarray(action, dtype=np.float64) * self.action_max
        p_risky = self.p[1:].copy()

        # Enforce turnover constraint
        delta_cash = -np.sum(actual_action)
        turnover = 0.5 * (np.abs(delta_cash) + np.sum(np.abs(actual_action)))
        if turnover > self.max_turnover + 1e-10:
            scale = self.max_turnover / turnover
            actual_action = actual_action * scale

        new_pr = p_risky + actual_action
        new_cash = 1.0 - np.sum(new_pr)

        # Enforce leverage constraint
        gross = np.abs(new_cash) + np.sum(np.abs(new_pr))
        if gross > self.leverage_factor + 1e-10:
            scale = self.leverage_factor / gross
            new_pr = p_risky + actual_action * scale
            new_cash = 1.0 - np.sum(new_pr)

        info = {
            'rebalanced_cash': new_cash,
            'rebalanced_risky': new_pr.copy(),
        }

        # Sample returns (or use injected shared returns)
        if hasattr(self, '_shared_return') and self._shared_return is not None:
            returns_risky = self._shared_return
            self._shared_return = None
        else:
            std_devs = np.sqrt(self.s_k)
            returns_risky = self.np_random.normal(self.a_k, std_devs)

        port_return = new_cash * self.r + np.sum(new_pr * returns_risky)
        W_new = self.wealth * (1.0 + port_return)

        if W_new <= 0.001:
            self.wealth = 0.001
            self.p = np.concatenate(([1.0], np.zeros(self.n)))
        else:
            self.wealth = W_new
            denom = 1.0 + port_return
            self.p = np.concatenate([
                [new_cash * (1.0 + self.r) / denom],
                new_pr * (1.0 + returns_risky) / denom
            ])

        self.t += 1
        done = bool(self.t >= self.T)

        reward = 0.0
        if done:
            reward = float(cara_utility(self.wealth, self.A))

        return self._get_obs(), reward, done, False, info


def train_ppo(env, timesteps=500_000, lr=0.001, seed=None, verbose=0):
    """Train a PPO agent on the portfolio environment."""
    from stable_baselines3 import PPO
    model = PPO(
        "MlpPolicy", env,
        learning_rate=lr,
        gamma=1.0,
        n_steps=2048,
        batch_size=64,
        verbose=verbose,
        seed=seed,
    )
    model.learn(total_timesteps=timesteps)
    return model


def train_a2c(env, timesteps=500_000, lr=0.001, n_steps=5, seed=None, verbose=0):
    """Train an A2C agent on the portfolio environment."""
    from stable_baselines3 import A2C
    model = A2C(
        "MlpPolicy", env,
        learning_rate=lr,
        gamma=1.0,
        n_steps=n_steps,
        verbose=verbose,
        seed=seed,
    )
    model.learn(total_timesteps=timesteps)
    return model


def train_sac(env, timesteps=500_000, lr=0.001, seed=None, verbose=0):
    """Train a SAC agent on the portfolio environment."""
    from stable_baselines3 import SAC
    model = SAC(
        "MlpPolicy", env,
        learning_rate=lr,
        gamma=1.0,
        batch_size=256,
        learning_starts=1000,
        verbose=verbose,
        seed=seed,
    )
    model.learn(total_timesteps=timesteps)
    return model


def train_td3(env, timesteps=500_000, lr=0.001, seed=None, verbose=0):
    """Train a TD3 agent on the portfolio environment."""
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )
    model = TD3(
        "MlpPolicy", env,
        learning_rate=lr,
        gamma=1.0,
        action_noise=action_noise,
        batch_size=256,
        learning_starts=1000,
        verbose=verbose,
        seed=seed,
    )
    model.learn(total_timesteps=timesteps)
    return model


# Keep backward-compatible alias
RelaxedMultiAssetEnv = PortfolioEnv
ConstrainedMultiAssetEnv = PortfolioEnv


if __name__ == "__main__":
    n_assets = 3
    time_horizon = 5

    env = PortfolioEnv(n=n_assets, T=time_horizon, r=0.03, A=0.5,
                       a_k=[0.08, 0.06, 0.10], s_k=[0.02, 0.015, 0.04],
                       p_init=[0.25, 0.25, 0.25, 0.25], leverage_factor=1.0)

    print("Training PPO...")
    import time
    t0 = time.time()
    model_ppo = train_ppo(env, timesteps=500_000)
    print(f"PPO trained in {time.time()-t0:.1f}s")

    print("\nTraining A2C...")
    t0 = time.time()
    model_a2c = train_a2c(env, timesteps=500_000)
    print(f"A2C trained in {time.time()-t0:.1f}s")

    # Quick evaluation
    num_test = 500
    for name, model in [("PPO", model_ppo), ("A2C", model_a2c)]:
        total_utility = 0
        for _ in range(num_test):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = env.step(action)
            total_utility += reward
        print(f"{name} avg utility: {total_utility/num_test:.5f}")
