import numpy as np
from dataclasses import dataclass
import math

np.random.seed(42)
@dataclass(frozen=True)
class DiscreteAssetAllocation:
    mu: float
    sigma: float
    r: float
    a: float
    T: int

    def get_optimal_allocation(self, t: int) -> float:

        excess_return = self.mu - self.r
        variance = self.sigma**2

        time_factor = (1+self.r)**(self.T - t -1)
        
        optimal_x = excess_return / (self.a * variance * time_factor)

        return optimal_x

class AssetEnv:
    def __init__(self, params: DiscreteAssetAllocation):
        self.p = params
        self.reset()

    def reset(self):
        self.wealth = 1
        self.t = 0
        return self.wealth
    
    def step(self, x_t):
        y_t = np.random.normal(self.p.mu, self.p.sigma)

        self.wealth = x_t*(1+y_t) + (self.wealth - x_t) * (1+self.p.r)
        
        self.t += 1
        done = self.t >= self.p.T

        reward = (1 - np.exp(-self.p.a * self.wealth)) / self.p.a if done else 0
        return self.wealth, reward, done
    
params = DiscreteAssetAllocation(mu=0.08, sigma=0.15, r=0.03, a=0.5, T=5)
env = AssetEnv(params)
total_utility = 0
num_episodes = 10000

for i in range(num_episodes):
    w = env.reset()
    done = False
    while not done:
        x_star = params.get_optimal_allocation(env.t)
        w, reward, done = env.step(x_star)
    
    total_utility += reward

print(f"Average Utility over {num_episodes} runs: {total_utility/ num_episodes}")