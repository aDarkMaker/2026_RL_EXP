import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden, act_dim)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)
        return logits, value

    def get_action(self, obs):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs, actions):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99, lam: float = 0.95,
                 clip_eps: float = 0.2, epochs_per_update: int = 10,
                 device: str = "cpu"):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs_per_update = epochs_per_update
        self.device = device
        self.net = ActorCriticNet(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.history = []

    def select_action(self, obs: np.ndarray):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.net.get_action(obs_t)
        return action.item(), log_prob.item(), value.item()

    def compute_gae(self, rewards, values, dones):
        advantages = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_val = values[t + 1] if t + 1 < len(values) else 0
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + np.array(values[:len(rewards)])
        return advantages, returns

    def update(self, batch: dict):
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        actions = torch.LongTensor(batch["act"]).to(self.device)
        old_log_probs = torch.FloatTensor(batch["log_prob"]).to(self.device)
        advantages = torch.FloatTensor(batch["advantages"]).to(self.device)
        returns = torch.FloatTensor(batch["returns"]).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs_per_update):
            log_probs, entropy, values = self.net.evaluate(obs, actions)
            ratio = (log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - values).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()

    def get_policy_action(self, obs: np.ndarray) -> int:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.net(obs_t)
        return logits.argmax(dim=-1).item()
