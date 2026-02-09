# Codebase Direction & Status

**Last Updated**: 2026-02-02
**Status**: Initial setup complete, ready for development

---

## Project Vision

Build a **Reinforcement Learning learning lab** - from fundamentals to deep RL algorithms.

**Target Hardware**: Apple M2 16GB

---

## Current State

### Completed
- [x] uv-based package management
- [x] Directory structure
- [x] CLI with basic commands
- [x] Common utilities (device detection, seeding, env helpers)
- [x] Documentation structure
- [x] Phase 1: Tabular RL fundamentals
  - [x] MDP/Bellman notebook with value iteration
  - [x] Base agent class
  - [x] Q-Learning agent
  - [x] SARSA + Expected SARSA agents
  - [x] FrozenLake training script
  - [x] On-policy vs off-policy comparison notebook

- [x] Phase 2: Deep RL
  - [x] REINFORCE notebook (policy gradients intro)
  - [x] DQN agent + deep-dive notebook
  - [x] PPO agent + deep-dive notebook
- [x] Phase 3: LLM Alignment
  - [x] RLHF notebook (reward modeling + PPO for LMs)
  - [x] DPO notebook (direct preference optimization)
  - [x] GRPO notebook (group relative policy optimization + RLVR)

---

## Directory Structure

```
rl_lab/
├── agents/           # Agent implementations
│   ├── base.py       # Base agent class
│   ├── q_learning.py # Tabular Q-learning
│   ├── dqn.py        # Deep Q-Network
│   └── ppo.py        # Proximal Policy Optimization
│
├── environments/     # Custom environments
│   └── wrappers.py   # Common wrappers
│
├── experiments/      # Training scripts
│   ├── train_cartpole.py
│   └── train_lunarlander.py
│
└── utils/
    ├── common.py     # Device, seeding, env helpers
    ├── plotting.py   # Training curves, visualizations
    └── logging.py    # TensorBoard, metrics
```

---

## Learning Roadmap

### Phase 1: Fundamentals ✓
- [x] Understand MDPs, Bellman equations
- [x] Implement tabular Q-learning on FrozenLake
- [x] Implement SARSA
- [x] Compare on-policy vs off-policy

### Phase 2: Deep RL ✓
- [x] REINFORCE + baseline
- [x] DQN (experience replay, target networks)
- [x] PPO (clipped objective, GAE)

### Phase 3: LLM Alignment ✓
- [x] RLHF (reward modeling + PPO fine-tuning)
- [x] DPO (direct preference optimization)
- [x] GRPO + RLVR (group-relative policy optimization)

---

## Key Resources

**Courses:**
- [Spinning Up in Deep RL](https://spinningup.openai.com/) - OpenAI
- [Deep RL Course](https://huggingface.co/learn/deep-rl-course) - Hugging Face

**Books:**
- Sutton & Barto - Reinforcement Learning: An Introduction

**Libraries:**
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Docs](https://gymnasium.farama.org/)

---

## How to Resume

```bash
cd /Users/karthik/Projects/Personal/Reinforcement-Learning

# Activate
source .venv/bin/activate

# Or run with uv
uv run rl-lab list-envs
```
