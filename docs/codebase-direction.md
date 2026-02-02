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

### TODO
- [ ] Implement tabular Q-learning agent
- [ ] Implement DQN agent
- [ ] Implement PPO agent
- [ ] Add experiment scripts
- [ ] Add Jupyter notebooks for exploration
- [ ] Document research questions / learning goals

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

### Phase 1: Fundamentals
- [ ] Understand MDPs, Bellman equations
- [ ] Implement tabular Q-learning on FrozenLake
- [ ] Implement SARSA
- [ ] Compare on-policy vs off-policy

### Phase 2: Deep RL Basics
- [ ] Implement DQN from scratch
- [ ] Experience replay, target networks
- [ ] Train on CartPole, then Atari

### Phase 3: Policy Gradient
- [ ] Implement REINFORCE
- [ ] Implement Actor-Critic
- [ ] Implement PPO

### Phase 4: Advanced
- [ ] Continuous control with SAC
- [ ] Multi-agent RL
- [ ] Model-based RL

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
