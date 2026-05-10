import numpy as np


class PolicyIteration:
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
            self._policy_evaluation()
            stable = self._policy_improvement()
            iterations += 1
            self.history.append({"iteration": iterations, "stable": stable, "V_mean": self.V.mean()})
            if stable:
                break
        return iterations

    def _policy_evaluation(self):
        while True:
            delta = 0
            for s in range(self.env.n_states):
                v = self.V[s]
                a = self.policy[s]
                self.V[s] = sum(
                    prob * (reward + self.gamma * self.V[next_s] * (1 - done))
                    for prob, next_s, reward, done in self.env.P[s][a]
                )
                delta = max(delta, abs(v - self.V[s]))
            if delta < self.theta:
                break

    def _policy_improvement(self) -> bool:
        stable = True
        for s in range(self.env.n_states):
            old_action = self.policy[s]
            q = np.zeros(self.env.n_actions)
            for a in range(self.env.n_actions):
                for prob, next_s, reward, done in self.env.P[s][a]:
                    q[a] += prob * (reward + self.gamma * self.V[next_s] * (1 - done))
            self.policy[s] = np.argmax(q)
            if old_action != self.policy[s]:
                stable = False
        return stable
