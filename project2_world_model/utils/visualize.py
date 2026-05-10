import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "project2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_world_model_loss(losses, filename="wm_loss.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("World Model Training Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()


def plot_training_rewards(rewards_real, rewards_dream=None, filename="training_rewards.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    window = max(1, len(rewards_real) // 20)
    smoothed = np.convolve(rewards_real, np.ones(window) / window, mode="valid")
    ax.plot(smoothed, label="Real Env", linewidth=1.5, color="steelblue")
    if rewards_dream is not None:
        smoothed_d = np.convolve(rewards_dream, np.ones(window) / window, mode="valid")
        ax.plot(smoothed_d, label="Dream Env (WM)", linewidth=1.5, color="coral")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("PPO Training Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()


def plot_imagination_comparison(real_traj, imag_traj, filename="imagination_vs_real.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    labels = ["Cart Pos", "Cart Vel", "Pole Angle", "Pole Vel"]
    real_obs = np.array(real_traj["obs"])
    imag_obs = np.array(imag_traj["obs"])
    min_len = min(len(real_obs), len(imag_obs))
    for i, ax in enumerate(axes.flat):
        ax.plot(real_obs[:min_len, i], label="Real", linewidth=1.5)
        ax.plot(imag_obs[:min_len, i], label="Imagined", linewidth=1.5, linestyle="--")
        ax.set_title(labels[i])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("World Model Imagination vs Real Trajectory", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()
