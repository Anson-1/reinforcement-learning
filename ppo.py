import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

class RelaxedMultiAssetEnv(gym.Env):
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

        self.a_k = np.array(a_k) if a_k else np.array([0.08, 0.06, 0.10])
        self.s_k = np.array(s_k) if s_k else np.array([0.02, 0.015, 0.04])

        if p_init is None:
            w = 0.60 / n
            self.p_init = np.array([1.0 - n * w] + [w] * n)
        else:
            self.p_init = np.array(p_init)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
        # [Time, Wealth, p_cash, p_asset1, p_asset2, p_asset3]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n+3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.wealth = 1.0
        self.p = self.p_init.copy()
        return self._get_obs(), {}
        
    def _get_obs(self):
        return np.concatenate(([self.t, self.wealth], self.p)).astype(np.float32)
    
    def step(self, action):
        actual_action = action * self.action_max
        p_risky = self.p[1:].copy()
        
        delta_cash_initial = -np.sum(actual_action)
        turnover = 0.5 * (np.abs(delta_cash_initial) + np.sum(np.abs(actual_action)))
        if turnover > self.max_turnover + 1e-10:
            scale_turnover = self.max_turnover / turnover
            actual_action = actual_action * scale_turnover
            
        new_pr = p_risky + actual_action
        new_cash = 1.0 - np.sum(new_pr)

        max_lev = self.leverage_factor
        gross = np.abs(new_cash) + np.sum(np.abs(new_pr))
        
        if gross > max_lev + 1e-10:
            scale_lev = max_lev / gross
            new_pr = p_risky + actual_action * scale_lev
            new_cash = 1.0 - np.sum(new_pr)
            
        info = {
            'rebalanced_cash': new_cash,
            'rebalanced_risky': new_pr.copy() 
        }

        if hasattr(self, '_shared_return') and self._shared_return is not None:
            returns_risky = self._shared_return
            self._shared_return = None  # consume it
        else:
            std_devs = np.sqrt(self.s_k)
            returns_risky = np.random.normal(self.a_k, std_devs)

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
            reward = (1 - np.exp(-self.A * self.wealth)) / self.A

        return self._get_obs(), float(reward), done, False, info

# ==========================================
# TRAINING
# ==========================================
if __name__ == "__main__":
    n_assets = 3
    time_horizon = 9

    env = RelaxedMultiAssetEnv(n=n_assets, T=time_horizon, leverage_factor=2.0)

    print("Training the Relaxed (Short/Margin Allowed) PPO Agent...")
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.001)
    model.learn(total_timesteps=2000000)
    print("Training Complete!\n")

    num_test_episodes = 2000
    total_utility = 0
    short_steps = np.zeros(n_assets, dtype=int)
    total_steps = num_test_episodes * time_horizon

    for i in range(num_test_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            
            if env.p[1] < -1e-10: short_steps[0] += 1
            if env.p[2] < -1e-10: short_steps[1] += 1
            if env.p[3] < -1e-10: short_steps[2] += 1
        
        total_utility += reward
    
    print(f"Average Utility over {num_test_episodes} test runs: {total_utility/num_test_episodes:.5f}")
    
    print("\nShort-selling frequency:")
    for k in range(n_assets):
        pct = 100.0 * short_steps[k] / total_steps
        print(f"  Asset {k+1}: {short_steps[k]:,}/{total_steps:,} steps ({pct:.1f}%)")