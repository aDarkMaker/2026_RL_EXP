import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from envs import GridWorld
from algorithms import (ValueIteration, PolicyIteration, MonteCarloControl,
                        SARSA, QLearning, LinearVFA, REINFORCE, ActorCritic)
from utils import Visualizer


def evaluate_policy(env, policy, n_episodes=100, max_steps=200):
    total = 0
    for _ in range(n_episodes):
        state = env.reset()
        for _ in range(max_steps):
            action = policy[state]
            state, reward, done, _ = env.step(action)
            total += reward
            if done:
                break
    return total / n_episodes


def main():
    np.random.seed(42)

    env = GridWorld(
        rows=5, cols=5, start=(0, 0), goal=(4, 4),
        obstacles=[(1, 1), (2, 3), (3, 1)],
        step_reward=-1.0, goal_reward=10.0, obstacle_reward=-5.0,
        stochastic=0.1
    )
    viz = Visualizer(output_dir=str(Path(__file__).resolve().parent.parent / "outputs" / "project1"))

    print("=" * 60)
    print("Project 1: Grid World Classic RL Algorithms")
    print("=" * 60)

    # --- Dynamic Programming ---
    print("\n[1/7] Value Iteration...")
    vi = ValueIteration(env, gamma=0.99)
    vi_steps = vi.solve()
    print(f"  Converged in {vi_steps} iterations")
    viz.plot_value_function(vi.V, env, "Value Iteration - V(s)", "vi_value.png")
    viz.plot_policy(vi.policy, env, "Value Iteration - Policy", "vi_policy.png")
    optimal_policy = vi.policy.copy()

    print("\n[2/7] Policy Iteration...")
    pi = PolicyIteration(env, gamma=0.99)
    pi_steps = pi.solve()
    print(f"  Converged in {pi_steps} iterations")
    viz.plot_policy(pi.policy, env, "Policy Iteration - Policy", "pi_policy.png")

    # --- Monte Carlo ---
    print("\n[3/7] Monte Carlo Control...")
    mc = MonteCarloControl(env, gamma=0.99, epsilon=0.1)
    mc.train(n_episodes=10000)
    viz.plot_policy(mc.policy, env, "Monte Carlo - Policy", "mc_policy.png")

    # --- TD Learning ---
    print("\n[4/7] SARSA...")
    sarsa = SARSA(env, gamma=0.99, alpha=0.1, epsilon=0.1)
    sarsa.train(n_episodes=10000)
    viz.plot_policy(sarsa.policy, env, "SARSA - Policy", "sarsa_policy.png")

    print("\n[5/7] Q-Learning...")
    ql = QLearning(env, gamma=0.99, alpha=0.1, epsilon=0.1)
    ql.train(n_episodes=10000)
    viz.plot_policy(ql.policy, env, "Q-Learning - Policy", "ql_policy.png")

    # --- Value Function Approximation ---
    print("\n[6/7] Linear Value Function Approximation...")
    vfa = LinearVFA(env, gamma=0.99, alpha=0.005, epsilon=0.1)
    vfa.train(n_episodes=10000)
    viz.plot_policy(vfa.policy, env, "Linear VFA - Policy", "vfa_policy.png")

    # --- Policy Gradient ---
    print("\n[7/7] REINFORCE & Actor-Critic...")
    reinforce = REINFORCE(env, gamma=0.99, lr=5e-3)
    reinforce.train(n_episodes=5000)
    viz.plot_policy(reinforce.policy, env, "REINFORCE - Policy", "reinforce_policy.png")

    ac = ActorCritic(env, gamma=0.99, lr_actor=5e-3, lr_critic=1e-2)
    ac.train(n_episodes=5000)
    viz.plot_policy(ac.policy, env, "Actor-Critic - Policy", "ac_policy.png")

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("Comparison & Evaluation")
    print("=" * 60)

    td_histories = {
        "SARSA": sarsa.history,
        "Q-Learning": ql.history,
        "Linear VFA": vfa.history,
    }
    viz.plot_training_curve(td_histories, "Total Reward", "TD Methods Training Curve", "td_comparison.png")

    pg_histories = {
        "REINFORCE": reinforce.history,
        "Actor-Critic": ac.history,
    }
    viz.plot_training_curve(pg_histories, "Total Reward", "Policy Gradient Training Curve", "pg_comparison.png")

    algorithms = {
        "Value Iteration": (vi.policy, vi_steps),
        "Policy Iteration": (pi.policy, pi_steps),
        "Monte Carlo": (mc.policy, 10000),
        "SARSA": (sarsa.policy, 10000),
        "Q-Learning": (ql.policy, 10000),
        "Linear VFA": (vfa.policy, 10000),
        "REINFORCE": (reinforce.policy, 5000),
        "Actor-Critic": (ac.policy, 5000),
    }

    results = {}
    for name, (policy, steps) in algorithms.items():
        match = np.mean(policy == optimal_policy)
        avg_return = evaluate_policy(env, policy)
        results[name] = {"steps": steps, "match": match, "avg_return": avg_return}
        print(f"  {name:20s} | Match: {match*100:5.1f}% | Avg Return: {avg_return:7.2f}")

    viz.plot_comparison_table(results, "comparison_table.png")
    print(f"\nAll outputs saved to: outputs/project1/")


if __name__ == "__main__":
    main()
