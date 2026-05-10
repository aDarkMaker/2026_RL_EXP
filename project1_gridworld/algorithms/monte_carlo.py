import numpy as np
from collections import defaultdict


class MonteCarloControl:
    def __init__(self, env, gamma: float = 0.99, epsilon: float = 0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((env.n_states, env.n_actions))
        self.returns_count = np.zeros((env.n_states, env.n_actions))
        self.policy = np.zeros(env.n_states, dtype=int)
        self.history = []

    def _epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.n_actions)
        return np.argmax(self.Q[state])

    def _generate_episode(self, max_steps=200):
        episode = []
        state = self.env.reset()
        for _ in range(max_steps):
            action = self._epsilon_greedy(state)
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        return episode

    def train(self, n_episodes: int = 5000) -> list:
        for ep in range(n_episodes):
            episode = self._generate_episode()
            G = 0
            visited = set()
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                if (state, action) not in visited:
                    visited.add((state, action))
                    self.returns_count[state, action] += 1
                    alpha = 1.0 / self.returns_count[state, action]
                    self.Q[state, action] += alpha * (G - self.Q[state, action])

            if (ep + 1) % 100 == 0:
                self.history.append({
                    "episode": ep + 1,
                    "avg_return": G,
                    "Q_mean": self.Q.mean()
                })

        self.policy = np.argmax(self.Q, axis=1)
        return self.history
