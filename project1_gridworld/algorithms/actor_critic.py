import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorNet(nn.Module):
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


class CriticNet(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)


class ActorCritic:
    def __init__(self, env, gamma: float = 0.99, lr_actor: float = 1e-3, lr_critic: float = 5e-3):
        self.env = env
        self.gamma = gamma
        self.feature_dim = 4
        self.actor = ActorNet(self.feature_dim, env.n_actions)
        self.critic = CriticNet(self.feature_dim)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.policy = np.zeros(env.n_states, dtype=int)
        self.history = []

    def _select_action(self, state):
        features = torch.FloatTensor(self.env.get_state_features(state)).unsqueeze(0)
        probs = self.actor(features)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), self.critic(features)

    def train(self, n_episodes: int = 3000, max_steps: int = 200) -> list:
        for ep in range(n_episodes):
            state = self.env.reset()
            total_reward = 0

            for _ in range(max_steps):
                action, log_prob, value = self._select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                next_features = torch.FloatTensor(self.env.get_state_features(next_state)).unsqueeze(0)
                next_value = self.critic(next_features).detach()
                td_target = reward + self.gamma * next_value * (1 - done)
                advantage = td_target - value

                actor_loss = -log_prob * advantage.detach()
                critic_loss = advantage.pow(2)

                self.opt_actor.zero_grad()
                actor_loss.backward()
                self.opt_actor.step()

                self.opt_critic.zero_grad()
                critic_loss.backward()
                self.opt_critic.step()

                state = next_state
                if done:
                    break

            if (ep + 1) % 100 == 0:
                self.history.append({"episode": ep + 1, "total_reward": total_reward})

        self._extract_policy()
        return self.history

    @torch.no_grad()
    def _extract_policy(self):
        for s in range(self.env.n_states):
            features = torch.FloatTensor(self.env.get_state_features(s)).unsqueeze(0)
            probs = self.actor(features)
            self.policy[s] = probs.argmax(dim=-1).item()
