import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    def __init__(self, n_features, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class REINFORCE:
    def __init__(self, env, gamma: float = 0.99, lr: float = 1e-3):
        self.env = env
        self.gamma = gamma
        self.feature_dim = 4
        self.net = PolicyNet(self.feature_dim, env.n_actions)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.policy = np.zeros(env.n_states, dtype=int)
        self.history = []

    def _select_action(self, state):
        features = torch.FloatTensor(self.env.get_state_features(state)).unsqueeze(0)
        probs = self.net(features)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def train(self, n_episodes: int = 3000, max_steps: int = 200) -> list:
        for ep in range(n_episodes):
            state = self.env.reset()
            log_probs, rewards = [], []

            for _ in range(max_steps):
                action, log_prob = self._select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                state = next_state
                if done:
                    break

            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.FloatTensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            loss = -sum(lp * G for lp, G in zip(log_probs, returns))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (ep + 1) % 100 == 0:
                self.history.append({"episode": ep + 1, "total_reward": sum(rewards)})

        self._extract_policy()
        return self.history

    @torch.no_grad()
    def _extract_policy(self):
        for s in range(self.env.n_states):
            features = torch.FloatTensor(self.env.get_state_features(s)).unsqueeze(0)
            probs = self.net(features)
            self.policy[s] = probs.argmax(dim=-1).item()
