import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class DynamicsNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.act_embed = nn.Embedding(act_dim, 8)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 8, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.next_obs_head = nn.Linear(hidden, obs_dim)
        self.reward_head = nn.Linear(hidden, 1)
        self.done_head = nn.Linear(hidden, 1)

    def forward(self, obs, act):
        act_emb = self.act_embed(act)
        x = torch.cat([obs, act_emb], dim=-1)
        h = self.net(x)
        next_obs = self.next_obs_head(h)
        reward = self.reward_head(h).squeeze(-1)
        done_logit = self.done_head(h).squeeze(-1)
        return next_obs, reward, done_logit


class WorldModel:
    def __init__(self, obs_dim: int, act_dim: int, lr: float = 1e-3, device: str = "cpu"):
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.model = DynamicsNet(obs_dim, act_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_losses = []

    def train(self, data: dict, epochs: int = 50, batch_size: int = 256):
        obs = torch.FloatTensor(data["obs"]).to(self.device)
        act = torch.LongTensor(data["act"]).to(self.device)
        rew = torch.FloatTensor(data["rew"]).to(self.device)
        next_obs = torch.FloatTensor(data["next_obs"]).to(self.device)
        done = torch.FloatTensor(data["done"].astype(np.float32)).to(self.device)

        dataset = TensorDataset(obs, act, rew, next_obs, done)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in loader:
                b_obs, b_act, b_rew, b_next, b_done = batch
                pred_next, pred_rew, pred_done = self.model(b_obs, b_act)
                loss_obs = nn.MSELoss()(pred_next, b_next)
                loss_rew = nn.MSELoss()(pred_rew, b_rew)
                loss_done = nn.BCEWithLogitsLoss()(pred_done, b_done)
                loss = loss_obs + 0.5 * loss_rew + 0.1 * loss_done

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            self.train_losses.append(avg_loss)
        return self.train_losses

    @torch.no_grad()
    def predict(self, obs: np.ndarray, act: int):
        self.model.eval()
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        act_t = torch.LongTensor([act]).to(self.device)
        pred_next, pred_rew, pred_done = self.model(obs_t, act_t)
        next_obs = pred_next.squeeze(0).cpu().numpy()
        reward = pred_rew.item()
        done = torch.sigmoid(pred_done).item() > 0.5
        return next_obs, reward, done

    def imagine_rollout(self, start_obs: np.ndarray, policy_fn, horizon: int = 15):
        obs = start_obs.copy()
        trajectory = {"obs": [], "act": [], "rew": [], "done": []}
        for _ in range(horizon):
            action = policy_fn(obs)
            next_obs, reward, done = self.predict(obs, action)
            trajectory["obs"].append(obs)
            trajectory["act"].append(action)
            trajectory["rew"].append(reward)
            trajectory["done"].append(done)
            obs = next_obs
            if done:
                break
        return trajectory
