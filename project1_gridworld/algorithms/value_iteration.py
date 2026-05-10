import numpy as np


class ValueIteration:
    def __init__(self, env, gamma: float = 0.99, theta: float = 1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(env.n_states)
        self.policy = np.zeros(env.n_states, dtype=int)
        self.history = []

    def solve(self) -> int:
        iterations = 0
        while True:
            delta = 0
            for s in range(self.env.n_states):
                v = self.V[s]
                q_values = self._compute_q(s)
                self.V[s] = np.max(q_values)
                delta = max(delta, abs(v - self.V[s]))
            iterations += 1
            self.history.append({"iteration": iterations, "delta": delta, "V_mean": self.V.mean()})
            if delta < self.theta:
                break
        self._extract_policy()
        return iterations

    def _compute_q(self, s):
        q = np.zeros(self.env.n_actions)
        for a in range(self.env.n_actions):
            for prob, next_s, reward, done in self.env.P[s][a]:
                q[a] += prob * (reward + self.gamma * self.V[next_s] * (1 - done))
        return q

    def _extract_policy(self):
        for s in range(self.env.n_states):
            self.policy[s] = np.argmax(self._compute_q(s))
