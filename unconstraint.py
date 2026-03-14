import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

class ConstrainedSingleAssetEnv(gym.Env):
    def __init__(self, n=1, T=5, r=0.03, A=0.5):
        super().__init__()
        self.n = n
        self.T = T
        self.r = r
        self.A = A
        
        # Match your analytical code: mu=0.08, sigma=0.15 -> variance=0.0225
        self.a_k = np.array([0.08])
        self.s_k = np.array([0.0225]) 
        
        # Start with 100% Cash, 0% Risky Asset
        self.p_init = np.array([1.0, 0.0])
        
        # Action space perfectly bounded for a max 100% shift
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
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
        # --- 1. THE LONG-ONLY CONSTRAINT ---
        desired_delta_risky = action[0] 
        desired_risky_proportion = self.p[1] + desired_delta_risky
        
        # INTERCEPT & CLIP: Force the risky proportion to stay between 0.0 and 1.0
        actual_risky_proportion = np.clip(desired_risky_proportion, 0.0, 1.0)
        
        # Calculate the actual delta we are allowed to execute
        actual_delta_risky = actual_risky_proportion - self.p[1]
        delta_cash = -actual_delta_risky
        
        # Apply the safe, constrained action
        self.p[1] += actual_delta_risky
        self.p[0] += delta_cash

        # --- 2. SIMULATE MARKET ---
        std_devs = np.sqrt(self.s_k)
        returns_risky = np.random.normal(self.a_k, std_devs)
        
        port_return = self.p[0] * self.r + np.sum(self.p[1:] * returns_risky)

        # --- 3. UPDATE WEALTH ---
        self.wealth = self.wealth * (1 + port_return)
        
        self.p[0] = (self.p[0] * (1 + self.r)) / (1 + port_return)
        self.p[1:] = (self.p[1:] * (1 + returns_risky)) / (1 + port_return)
        
        self.t += 1
        done = bool(self.t >= self.T)

        # --- 4. REWARD ---
        reward = 0.0
        if done:
            reward = (1 - np.exp(-self.A * self.wealth)) / self.A

        return self._get_obs(), float(reward), done, False, {}

# ==========================================
# TRAINING
# ==========================================
if __name__ == "__main__":
    env = ConstrainedSingleAssetEnv()
    
    print("Training the Constrained Long-Only Agent (500,000 steps)...")
    # Reduced learning rate slightly for more stable convergence in a constrained space
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0005)
    model.learn(total_timesteps=500000)
    print("Training complete!\n")
    
    num_test_episodes = 5000 
    total_utility = 0
    
    for i in range(num_test_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            
        total_utility += reward
        
    print(f"Average Utility over {num_test_episodes} test runs: {total_utility / num_test_episodes:.5f}")

    print("\nFinal Portfolio Distribution (Last Test Run)")

    obs, _ = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

    final_cash = env.p[0]
    final_risky = env.p[1]

    print(f"Final Cash Proportion: {final_cash:.3f}")
    print(f"Final Risky Asset Proportion: {final_risky:.3f}")
    print(f"Final Wealth: {env.wealth:.3f}")