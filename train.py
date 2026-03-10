"""
PPO training pipeline for the asset allocation environment.

Uses Stable-Baselines3 PPO with configurable environment parameters.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from environment import AssetAllocationEnv, EnvParams


class EpisodeLogCallback(BaseCallback):
    """Logs episode returns during training for plotting."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True


def make_env(params: EnvParams):
    """Create the environment wrapped with Monitor for episode stats."""
    from stable_baselines3.common.monitor import Monitor
    env = AssetAllocationEnv(params)
    return Monitor(env)


def train(
    params: EnvParams = None,
    total_timesteps: int = 500_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    seed: int = 42,
    save_path: str = "results/ppo_model",
):
    """
    Train PPO on the asset allocation environment.

    Returns:
        model: trained PPO model
        callback: EpisodeLogCallback with training history
    """
    if params is None:
        params = EnvParams()

    env = make_env(params)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=1.0,  # no discounting — only terminal reward matters
        seed=seed,
        verbose=1,
        policy_kwargs={"net_arch": [64, 64]},
    )

    callback = EpisodeLogCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to {save_path}")

    return model, callback


def plot_training_curve(callback: EpisodeLogCallback, save_path: str = "results/training_curve.png"):
    """Plot episode returns over training."""
    rewards = callback.episode_rewards
    if not rewards:
        print("No episodes logged.")
        return

    # Smoothed curve (rolling average)
    window = min(100, len(rewards) // 5) if len(rewards) > 10 else 1
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.2, color="steelblue", label="Episode return")
    ax.plot(range(window - 1, window - 1 + len(smoothed)), smoothed,
            color="steelblue", linewidth=2, label=f"Rolling avg ({window} ep)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Return (CARA utility)")
    ax.set_title("PPO Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO for asset allocation")
    parser.add_argument("--n-assets", type=int, default=3)
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    params = EnvParams(
        n_assets=args.n_assets,
        means=[0.10, 0.13, 0.08][:args.n_assets],
        variances=[0.0225, 0.04, 0.0144][:args.n_assets],
        T=args.T,
    )
    print(f"Training with: {params}")
    print(f"Total timesteps: {args.timesteps}\n")

    model, callback = train(params, total_timesteps=args.timesteps, seed=args.seed)
    plot_training_curve(callback)

    # Quick evaluation
    print("\n--- Quick evaluation (100 episodes) ---")
    env = make_env(params)
    rewards = []
    for ep in range(100):
        obs, _ = env.reset(seed=1000 + ep)
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    print(f"Mean return: {np.mean(rewards):.6f}")
    print(f"Std return:  {np.std(rewards):.6f}")
