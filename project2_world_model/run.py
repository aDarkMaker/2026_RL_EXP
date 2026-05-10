import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from envs import CartPoleWrapper
from models import WorldModel, PPOAgent
from utils import plot_world_model_loss, plot_training_rewards, plot_imagination_comparison


def collect_ppo_rollout(env, agent, n_steps=2048):
    batch = {"obs": [], "act": [], "rew": [], "done": [], "log_prob": [], "value": []}
    obs = env.reset()
    episode_rewards = []
    ep_reward = 0

    for _ in range(n_steps):
        action, log_prob, value = agent.select_action(obs)
        next_obs, reward, done, _ = env.step(action)
        batch["obs"].append(obs)
        batch["act"].append(action)
        batch["rew"].append(reward)
        batch["done"].append(done)
        batch["log_prob"].append(log_prob)
        batch["value"].append(value)
        ep_reward += reward
        obs = next_obs
        if done:
            episode_rewards.append(ep_reward)
            ep_reward = 0
            obs = env.reset()

    advantages, returns = agent.compute_gae(
        np.array(batch["rew"]), batch["value"], np.array(batch["done"])
    )
    batch["obs"] = np.array(batch["obs"])
    batch["act"] = np.array(batch["act"])
    batch["log_prob"] = np.array(batch["log_prob"])
    batch["advantages"] = advantages
    batch["returns"] = returns
    return batch, episode_rewards


def collect_real_trajectory(env, agent, max_steps=200):
    traj = {"obs": [], "act": [], "rew": []}
    obs = env.reset()
    for _ in range(max_steps):
        action = agent.get_policy_action(obs)
        traj["obs"].append(obs)
        traj["act"].append(action)
        next_obs, reward, done, _ = env.step(action)
        traj["rew"].append(reward)
        obs = next_obs
        if done:
            break
    return traj


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Project 2: World Model + PPO (Dyna-style)")
    print(f"Device: {device}")
    print("=" * 60)

    env = CartPoleWrapper(seed=42)

    # Phase 1: Train World Model
    print("\n[Phase 1] Collecting random exploration data...")
    data = env.collect_random_data(n_episodes=100)
    print(f"  Collected {len(data['obs'])} transitions")

    print("\n[Phase 2] Training World Model...")
    wm = WorldModel(obs_dim=env.obs_dim, act_dim=env.act_dim, lr=1e-3, device=device)
    losses = wm.train(data, epochs=80, batch_size=256)
    plot_world_model_loss(losses)
    print(f"  Final loss: {losses[-1]:.4f}")

    # Phase 2: Train PPO with real environment
    print("\n[Phase 3] Training PPO in real environment...")
    agent_real = PPOAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                          lr=3e-4, gamma=0.99, device=device)
    all_rewards_real = []
    for iteration in range(80):
        batch, ep_rewards = collect_ppo_rollout(env, agent_real, n_steps=2048)
        agent_real.update(batch)
        all_rewards_real.extend(ep_rewards)
        if (iteration + 1) % 10 == 0:
            avg = np.mean(ep_rewards) if ep_rewards else 0
            print(f"  Iter {iteration+1:3d} | Avg Reward: {avg:.1f}")

    # Phase 3: Train PPO with Dyna (real + imagined)
    print("\n[Phase 4] Training PPO with Dyna (real + World Model imagination)...")
    agent_dyna = PPOAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                          lr=3e-4, gamma=0.99, device=device)
    all_rewards_dyna = []
    for iteration in range(80):
        batch, ep_rewards = collect_ppo_rollout(env, agent_dyna, n_steps=1024)

        # Augment with imagined data
        imag_obs, imag_act, imag_rew, imag_done = [], [], [], []
        for i in range(0, len(batch["obs"]), 64):
            start_obs = batch["obs"][i]
            traj = wm.imagine_rollout(start_obs, agent_dyna.get_policy_action, horizon=10)
            imag_obs.extend(traj["obs"])
            imag_act.extend(traj["act"])
            imag_rew.extend(traj["rew"])
            imag_done.extend(traj["done"])

        if imag_obs:
            imag_batch = {"obs": [], "act": [], "rew": [], "done": [], "log_prob": [], "value": []}
            for obs_i, act_i in zip(imag_obs, imag_act):
                _, lp, v = agent_dyna.select_action(obs_i)
                imag_batch["obs"].append(obs_i)
                imag_batch["act"].append(act_i)
                imag_batch["log_prob"].append(lp)
                imag_batch["value"].append(v)
            imag_batch["rew"] = imag_rew
            imag_batch["done"] = imag_done
            imag_adv, imag_ret = agent_dyna.compute_gae(
                np.array(imag_batch["rew"]), imag_batch["value"], np.array(imag_batch["done"], dtype=np.float32)
            )
            combined = {
                "obs": np.concatenate([batch["obs"], np.array(imag_batch["obs"])]),
                "act": np.concatenate([batch["act"], np.array(imag_batch["act"])]),
                "log_prob": np.concatenate([batch["log_prob"], np.array(imag_batch["log_prob"])]),
                "advantages": np.concatenate([batch["advantages"], imag_adv]),
                "returns": np.concatenate([batch["returns"], imag_ret]),
            }
            agent_dyna.update(combined)
        else:
            agent_dyna.update(batch)

        all_rewards_dyna.extend(ep_rewards)
        if (iteration + 1) % 10 == 0:
            avg = np.mean(ep_rewards) if ep_rewards else 0
            print(f"  Iter {iteration+1:3d} | Avg Reward: {avg:.1f}")

    # Visualization
    print("\n[Phase 5] Generating visualizations...")
    plot_training_rewards(all_rewards_real, all_rewards_dyna)

    real_traj = collect_real_trajectory(env, agent_real, max_steps=100)
    imag_traj = wm.imagine_rollout(
        real_traj["obs"][0], agent_real.get_policy_action, horizon=min(50, len(real_traj["obs"]))
    )
    plot_imagination_comparison(real_traj, imag_traj)

    print(f"\n{'='*60}")
    print("Results Summary:")
    print(f"  PPO (Real only)  - Final avg reward: {np.mean(all_rewards_real[-20:]):.1f}")
    print(f"  PPO (Dyna + WM)  - Final avg reward: {np.mean(all_rewards_dyna[-20:]):.1f}")
    print(f"\nAll outputs saved to: outputs/project2/")

    env.close()


if __name__ == "__main__":
    main()
