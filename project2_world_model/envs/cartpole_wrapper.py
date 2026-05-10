import gymnasium as gym
import numpy as np


class CartPoleWrapper:
    def __init__(self, seed: int = 42):
        self.env = gym.make("CartPole-v1")
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n
        self.seed = seed

    def reset(self):
        obs, _ = self.env.reset(seed=self.seed)
        self.seed += 1
        return obs.astype(np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs.astype(np.float32), float(reward), done, info

    def collect_random_data(self, n_episodes: int = 50):
        buffer = {"obs": [], "act": [], "rew": [], "next_obs": [], "done": []}
        for _ in range(n_episodes):
            obs = self.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_obs, reward, done, _ = self.step(action)
                buffer["obs"].append(obs)
                buffer["act"].append(action)
                buffer["rew"].append(reward)
                buffer["next_obs"].append(next_obs)
                buffer["done"].append(done)
                obs = next_obs
        return {k: np.array(v) for k, v in buffer.items()}

    def close(self):
        self.env.close()
