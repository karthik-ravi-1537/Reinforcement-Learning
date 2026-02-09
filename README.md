# Reinforcement Learning Lab

A hands-on learning lab for **Reinforcement Learning** — from tabular fundamentals to the algorithms powering LLM alignment (RLHF, DPO, GRPO).

## Getting Started

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
cd Reinforcement-Learning
uv sync

# Verify everything works
uv run rl-lab info CartPole-v1
```

## The Learning Path

Work through the notebooks in order. Each one builds on the previous.

### Phase 1: RL Fundamentals

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 01 | [MDP Fundamentals](notebooks/01_mdp_fundamentals.ipynb) | Markov Decision Processes, Bellman equations, value iteration, policy iteration |
| 02 | [Q-Learning vs SARSA](notebooks/02_qlearning_vs_sarsa.ipynb) | Tabular RL, on-policy vs off-policy, the CliffWalking experiment |

**Key idea**: RL is about learning to make good sequential decisions by trial and error.

Start here if you're new to RL. These notebooks use simple grid environments where you can see every state and action.

### Phase 2: Deep RL

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 03 | [REINFORCE](notebooks/03_reinforce.ipynb) | Policy gradients, why we need them, variance reduction with baselines |
| 04 | [DQN Deep Dive](notebooks/04_dqn_deep_dive.ipynb) | Neural network Q-functions, experience replay, target networks |
| 05 | [PPO Deep Dive](notebooks/05_ppo_deep_dive.ipynb) | Clipped surrogate objective, GAE, the bridge from RL to RLHF |

**Key idea**: Replace tables with neural networks to handle complex environments.

These notebooks train agents on CartPole and LunarLander. PPO (notebook 05) is especially important — it's the backbone of RLHF.

### Phase 3: LLM Alignment

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 06 | [RLHF](notebooks/06_rlhf.ipynb) | Reward modeling from human preferences, PPO + KL penalty, reward hacking |
| 07 | [DPO](notebooks/07_dpo.ipynb) | Direct preference optimization — skip the reward model entirely |
| 08 | [GRPO](notebooks/08_grpo.ipynb) | Group-relative advantages, RLVR with verifiable rewards (DeepSeek-R1 style) |

**Key idea**: Use RL to align language models with human preferences.

These use toy "language models" to demonstrate the core mechanics without needing a GPU cluster. The math and algorithms are identical to real-world implementations.

## Running Notebooks

```bash
# Launch Jupyter
uv run jupyter lab notebooks/

# Start with notebook 01 and work through in order
```

Each notebook includes:
- **Theory** with equations and intuitive explanations
- **Implementation** from scratch (not just library calls)
- **Visualizations** of training, policies, and ablations
- **Exercises** to deepen understanding

## Running Experiments

```bash
# Train tabular agents on FrozenLake
uv run python rl_lab/experiments/train_frozenlake.py --agent all --episodes 10000

# Train a specific agent
uv run python rl_lab/experiments/train_frozenlake.py --agent q_learning

# Non-slippery version (easier)
uv run python rl_lab/experiments/train_frozenlake.py --no-slippery
```

## Project Structure

```
rl_lab/
├── agents/              # Agent implementations
│   ├── base.py          # BaseAgent, TabularAgent (ε-greedy, Q-table)
│   ├── q_learning.py    # Tabular Q-Learning
│   ├── sarsa.py         # SARSA + Expected SARSA
│   ├── dqn.py           # DQN (replay buffer, target network)
│   └── ppo.py           # PPO (clipped objective, GAE)
│
├── environments/        # Custom environments, wrappers
├── experiments/         # Training scripts
│   └── train_frozenlake.py
├── utils/
│   └── common.py        # Device detection, seeding, env helpers
└── cli.py               # CLI tool

notebooks/               # 8 interactive notebooks (the main learning path)
docs/                    # Project docs and status
```

## CLI Commands

```bash
uv run rl-lab list-envs          # List available environments
uv run rl-lab info CartPole-v1   # Show environment details
uv run rl-lab demo LunarLander-v3  # Watch random agent play
```

## Algorithm Cheat Sheet

```
         Tabular                    Deep RL                  LLM Alignment
    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
    │  Q-Learning  │    NN    │     DQN      │          │    RLHF      │
    │   (off-pol)  │ ──────→  │ (replay+tgt) │          │  (PPO+RM+KL) │
    └──────────────┘          └──────────────┘          └──────────────┘
    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
    │    SARSA     │  policy  │  REINFORCE   │  clip+   │     DPO      │
    │   (on-pol)   │  grad    │  (baseline)  │  GAE     │  (no RM)     │
    └──────────────┘ ──────→  └──────┬───────┘ ──────→  └──────────────┘
                                     │                  ┌──────────────┐
                                     ▼                  │    GRPO      │
                              ┌──────────────┐          │ (group adv)  │
                              │     PPO      │ ──────→  └──────────────┘
                              └──────────────┘
```

## Key Concepts

| Term | Meaning |
|------|---------|
| **Episode** | One complete run from reset to done |
| **Step** | One action taken |
| **Return** | Cumulative discounted reward: G = Σ γᵗrₜ |
| **Policy** π(a\|s) | Mapping from states to actions |
| **Value** V(s) | Expected return from state s |
| **Q-value** Q(s,a) | Expected return from state s taking action a |
| **Advantage** A(s,a) | Q(s,a) - V(s) — how much better is this action? |
| **GAE** | Generalized Advantage Estimation (bias-variance trade-off) |
| **KL Penalty** | Keeps policy close to reference (prevents reward hacking) |

## Prerequisites

- Python basics, comfort with NumPy
- Some calculus (gradients) helps for deep RL notebooks
- No prior RL knowledge needed — notebook 01 starts from scratch

## Hardware

Designed for **Apple M2 16GB** (local). All notebooks run in minutes, not hours. GPU not required.

## License

MIT
