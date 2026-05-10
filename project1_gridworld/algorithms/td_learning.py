import numpy as np


class SARSA:
    def __init__(self, env, gamma: float = 0.99, alpha: float = 0.1, epsilon: float = 0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = np.zeros((env.n_states, env.n_actions))
        self.policy = np.zeros(env.n_states, dtype=int)
        self.history = []

    def _epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.n_actions)
        return np.argmax(self.Q[state])

    def train(self, n_episodes: int = 5000, max_steps: int = 200) -> list:
        for ep in range(n_episodes):
            state = self.env.reset()
            action = self._epsilon_greedy(state)
            total_reward = 0

            for _ in range(max_steps):
                next_state, reward, done, _ = self.env.step(action)
                next_action = self._epsilon_greedy(next_state)
                td_target = reward + self.gamma * self.Q[next_state, next_action] * (1 - done)
                self.Q[state, action] += self.alpha * (td_target - self.Q[state, action])
                total_reward += reward
                state, action = next_state, next_action
                if done:
                    break

            if (ep + 1) % 100 == 0:
                self.history.append({"episode": ep + 1, "total_reward": total_reward})

        self.policy = np.argmax(self.Q, axis=1)
        return self.history


class QLearning:
    def __init__(self, env, gamma: float = 0.99, alpha: float = 0.1, epsilon: float = 0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = np.zeros((env.n_states, env.n_actions))
        self.policy = np.zeros(env.n_states, dtype=int)
        self.history = []

    def _epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.n_actions)
        return np.argmax(self.Q[state])

    def train(self, n_episodes: int = 5000, max_steps: int = 200) -> list:
        for ep in range(n_episodes):
            state = self.env.reset()
            total_reward = 0

            for _ in range(max_steps):
                action = self._epsilon_greedy(state)
                next_state, reward, done, _ = self.env.step(action)
                td_target = reward + self.gamma * np.max(self.Q[next_state]) * (1 - done)
                self.Q[state, action] += self.alpha * (td_target - self.Q[state, action])
                total_reward += reward
                state = next_state
                if done:
                    break

            if (ep + 1) % 100 == 0:
                self.history.append({"episode": ep + 1, "total_reward": total_reward})

        self.policy = np.argmax(self.Q, axis=1)
        return self.history
