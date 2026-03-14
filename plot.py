import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

class DynamicMarketEnv(gym.Env):
    def __init__(self, mu, variance, r, A=0.5, T=5):
        super().__init__()
        self.T, self.r, self.A = T, r, A
        self.a_k = np.array([mu])
        self.s_k = np.array([variance]) 
        self.p_init = np.array([1.0, 0.0]) # Start 100% cash
        
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t, self.wealth = 0, 1.0
        self.p = self.p_init.copy()
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate(([self.t, self.wealth], self.p)).astype(np.float32)

    def step(self, action):
        delta_risky = action 
        self.p[1:] += delta_risky
        self.p[0] -= np.sum(delta_risky) # Cash absorbs the difference

        returns_risky = np.random.normal(self.a_k, np.sqrt(self.s_k))
        port_return = self.p[0] * self.r + np.sum(self.p[1:] * returns_risky)

        self.wealth *= (1 + port_return)
        self.p[0] = (self.p[0] * (1 + self.r)) / (1 + port_return)
        self.p[1:] = (self.p[1:] * (1 + returns_risky)) / (1 + port_return)
        
        self.t += 1
        done = bool(self.t >= self.T)
        reward = (1 - np.exp(-self.A * self.wealth)) / self.A if done else 0.0

        return self._get_obs(), float(reward), done, False, {}

if __name__ == "__main__":
    # The parameters we want to test
    mu_values = [0.04, 0.06, 0.08, 0.10, 0.12]
    fixed_r = 0.03
    fixed_variance = 0.0225
    A = 0.5
    
    ai_weights = []
    analytical_weights = []
    
    print("Starting sensitivity analysis... This will train 5 models.\n")
    
    for mu in mu_values:
        print(f"Training market condition: mu = {mu*100:.1f}%, r = {fixed_r*100:.1f}%")
        env = DynamicMarketEnv(mu=mu, variance=fixed_variance, r=fixed_r)
        
        # Train a quick model (150k steps is enough to see the trend)
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.001)
        model.learn(total_timesteps=150000)
        
        # Test what the AI chooses on Day 1
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        chosen_weight = action[0] + env.p_init[1] # Action + starting risky weight (0)
        ai_weights.append(chosen_weight)
        
        # Calculate true mathematical optimum
        true_opt = (mu - fixed_r) / (A * fixed_variance)
        analytical_weights.append(true_opt)

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, analytical_weights, label='Analytical Optimal Weight (Merton)', linestyle='--', color='blue', marker='o')
    plt.plot(mu_values, ai_weights, label='AI Chosen Weight (PPO)', color='red', marker='x', markersize=8)
    
    plt.title('AI Portfolio Weight vs. Mathematical Optimal (Sensitivity Analysis)')
    plt.xlabel('Expected Return of Risky Asset (mu)')
    plt.ylabel('Proportion of Wealth in Risky Asset')
    plt.legend()
    plt.grid(True)
    plt.show()