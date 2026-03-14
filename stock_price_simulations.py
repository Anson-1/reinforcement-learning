import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

class ConstrainedMultiAssetEnv(gym.Env):
    def __init__(self, n=3, T=9, r=0.07, A=0.5, a_k=None, s_k=None, p_init=None):
        super().__init__()

        assert n < 5
        assert T < 10

        self.n = n 
        self.T = T 
        self.r = r
        self.A = A

        self.a_k = np.array(a_k) if a_k else np.array([0.08, 0.06, 0.10])
        self.s_k = np.array(s_k) if s_k else np.array([0.02, 0.015, 0.04])

        self.p_init = np.array(p_init) if p_init else np.array([0.40, 0.20, 0.20, 0.20])

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
        # 1. AI's raw desired change (scaled down by the 10% max step limit)
        desired_delta_risky = action * 0.10
        
        # --- THE LONG-ONLY MULTI-ASSET CONSTRAINT ---
        
        # RULE 1: No Short Selling (p[1], p[2], p[3] >= 0)
        # We cannot sell more of an asset than we currently own.
        delta_risky = np.maximum(desired_delta_risky, -self.p[1:])
        
        # RULE 2: No Margin Borrowing (p[0] >= 0)
        # Separate the intended buys (positive deltas) and sells (negative deltas)
        buys = np.maximum(delta_risky, 0)
        sells = np.minimum(delta_risky, 0)
        
        # How much cash do we actually have to spend?
        # Current Cash + Cash generated from selling other assets
        cash_available = self.p[0] - np.sum(sells)
        
        # If the total cost of our buys exceeds our available cash, scale the buys down
        total_buys = np.sum(buys)
        if total_buys > cash_available:
            scale_buys = cash_available / total_buys
            buys = buys * scale_buys
            
        # Recombine the fully safe, constrained buys and sells
        delta_risky = buys + sells
        delta_cash = -np.sum(delta_risky)

        # --- TURNOVER LIMIT ---
        turnover = 0.5 * (np.abs(delta_cash) + np.sum(np.abs(delta_risky)))

        if turnover > 0.10:
            scale = 0.10 / turnover
            delta_risky *= scale
            delta_cash *= scale

        # Apply the final safe actions
        self.p[1:] += delta_risky
        self.p[0] += delta_cash

        # Simulate Market
        std_devs = np.sqrt(self.s_k)
        returns_risky = np.random.normal(self.a_k, std_devs)

        port_return = self.p[0] * self.r + np.sum(self.p[1:] * returns_risky)
        self.wealth = self.wealth * (1 + port_return)

        # Update weights based on market movement
        self.p[0] = (self.p[0] * (1 + self.r)) / (1 + port_return)
        self.p[1:] = (self.p[1:] * (1 + returns_risky)) / (1 + port_return)

        self.t += 1
        done = bool(self.t >= self.T)

        reward = 0.0
        if done:
            reward = (1 - np.exp(-self.A * self.wealth)) / self.A

        return self._get_obs(), float(reward), done, False, {}
    
if __name__ == "__main__":
    n_assets = 3
    time_horizon = 9

    env = ConstrainedMultiAssetEnv(n=n_assets, T=time_horizon)

    print("Training the Constrained Multi-Asset RL Agent...")
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.001)
    model.learn(total_timesteps=500000)
    print("Training Complete!\n")

    num_test_episodes = 2000
    total_utility = 0

    for i in range(num_test_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
        
        total_utility += reward
    
    print(f"Average Utility over {num_test_episodes} test runs: {total_utility/num_test_episodes:.5f}")

    # --- Print Final Episode Weights ---
    print("\nFinal Portfolio Distribution (Last Test Run)")
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

    print(f"Final Cash Proportion p[0]:     {env.p[0] * 100:.2f}%")
    print(f"Final Asset 1 Proportion p[1]:  {env.p[1] * 100:.2f}%")
    print(f"Final Asset 2 Proportion p[2]:  {env.p[2] * 100:.2f}%")
    print(f"Final Asset 3 Proportion p[3]:  {env.p[3] * 100:.2f}%")
    print(f"Total Sum:                      {np.sum(env.p) * 100:.2f}%")
    print(f"Final Wealth:                   ${env.wealth:.3f}")