# 强化学习课程项目

> 张程 | U202311128 | BD2301

## 项目结构

```
.
├── project1_gridworld/          # 项目一：Grid World 经典 RL 算法
│   ├── envs/                    # 环境定义
│   ├── algorithms/              # 7 种经典算法实现
│   ├── utils/                   # 可视化工具
│   └── run.py                   # 一键运行入口
├── project2_world_model/        # 项目二：World Model + PPO (前沿探索)
│   ├── envs/                    # CartPole 环境封装
│   ├── models/                  # World Model & PPO
│   ├── utils/                   # 可视化工具
│   └── run.py                   # 一键运行入口
├── outputs/                     # 训练产出 (图表/日志)
├── pyproject.toml               # 项目配置 (uv 管理)
└── README.md
```

## 快速开始

```bash
# 安装依赖
uv sync

# 运行项目一
uv run project1_gridworld/run.py

# 运行项目二
uv run project2_world_model/run.py
```

## 项目一：Grid World 基础实验

在 5×5 Grid World 上实现并比较以下算法：

| 类别 | 算法 |
|------|------|
| 动态规划 | Value Iteration, Policy Iteration |
| 蒙特卡洛 | First-Visit MC Control |
| 时序差分 | SARSA, Q-Learning |
| 值函数近似 | Linear VFA (TD with feature) |
| 策略梯度 | REINFORCE, Actor-Critic |

输出包含：各算法策略可视化、值函数热力图、训练曲线对比、算法性能对比表。

## 项目二：World Model 前沿探索

基于 Dyna 架构，在 CartPole 环境中实现：

1. **World Model**：神经网络学习环境动力学 (状态转移 + 奖励 + 终止预测)
2. **PPO Agent**：在真实环境与 World Model 想象中联合训练
3. **对比实验**：纯真实环境 PPO vs. Dyna-PPO (加速收敛验证)
