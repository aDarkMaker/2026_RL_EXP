import numpy as np


class LinearVFA:
    def __init__(self, env, gamma: float = 0.99, alpha: float = 0.01, epsilon: float = 0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.feature_dim = 4
        self.W = np.zeros((env.n_actions, self.feature_dim))
        self.policy = np.zeros(env.n_states, dtype=int)
        self.history = []

    def _get_q(self, state, action):
        features = self.env.get_state_features(state)
        return self.W[action] @ features

    def _get_all_q(self, state):
        features = self.env.get_state_features(state)
        return self.W @ features

    def _epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.n_actions)
        return np.argmax(self._get_all_q(state))

    def train(self, n_episodes: int = 5000, max_steps: int = 200) -> list:
        for ep in range(n_episodes):
            state = self.env.reset()
            total_reward = 0

            for _ in range(max_steps):
                action = self._epsilon_greedy(state)
                next_state, reward, done, _ = self.env.step(action)
                q_current = self._get_q(state, action)
                q_next = np.max(self._get_all_q(next_state)) if not done else 0
                td_error = reward + self.gamma * q_next - q_current
                features = self.env.get_state_features(state)
                self.W[action] += self.alpha * td_error * features
                total_reward += reward
                state = next_state
                if done:
                    break

            if (ep + 1) % 100 == 0:
                self.history.append({"episode": ep + 1, "total_reward": total_reward})

        for s in range(self.env.n_states):
            self.policy[s] = np.argmax(self._get_all_q(s))
        return self.history
