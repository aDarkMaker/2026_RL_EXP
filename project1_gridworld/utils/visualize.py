import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path


class Visualizer:
    ACTION_ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    def __init__(self, output_dir: str = "outputs/project1"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_value_function(self, V, env, title: str, filename: str):
        V_grid = V.reshape(env.rows, env.cols)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(V_grid, cmap="YlOrRd_r", interpolation="nearest")
        for i in range(env.rows):
            for j in range(env.cols):
                text_color = "white" if V_grid[i, j] < V_grid.mean() else "black"
                ax.text(j, i, f"{V_grid[i, j]:.1f}", ha="center", va="center",
                        fontsize=9, color=text_color)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_policy(self, policy, env, title: str, filename: str):
        fig, ax = plt.subplots(figsize=(6, 5))
        grid = np.zeros((env.rows, env.cols))
        for r in range(env.rows):
            for c in range(env.cols):
                s = r * env.cols + c
                if (r, c) == env.goal:
                    ax.text(c, r, "★", ha="center", va="center", fontsize=16, color="gold")
                elif (r, c) in env.obstacles:
                    grid[r, c] = -1
                    ax.text(c, r, "■", ha="center", va="center", fontsize=14, color="black")
                else:
                    ax.text(c, r, self.ACTION_ARROWS[policy[s]], ha="center", va="center", fontsize=14)
        ax.imshow(grid, cmap="coolwarm", alpha=0.1, interpolation="nearest")
        ax.set_xticks(range(env.cols))
        ax.set_yticks(range(env.rows))
        ax.grid(True, linewidth=0.5)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_training_curve(self, histories: dict, ylabel: str, title: str, filename: str):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, history in histories.items():
            episodes = [h["episode"] for h in history]
            rewards = [h["total_reward"] for h in history]
            ax.plot(episodes, rewards, label=name, linewidth=1.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_comparison_table(self, results: dict, filename: str):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        headers = ["Algorithm", "Converged Steps", "Optimal Policy Match", "Avg Return"]
        table_data = []
        for name, info in results.items():
            table_data.append([name, str(info.get("steps", "-")),
                              f"{info.get('match', 0)*100:.1f}%",
                              f"{info.get('avg_return', 0):.2f}"])
        table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        plt.title("Algorithm Comparison", fontsize=12, pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
