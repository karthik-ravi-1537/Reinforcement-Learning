# Reinforcement Learning Lab

A personal learning lab for **Reinforcement Learning** - from fundamentals to deep RL.

## Quick Start

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup
cd Reinforcement-Learning
uv sync

# Test environment
uv run rl-lab info CartPole-v1

# Run a demo
uv run rl-lab demo CartPole-v1
```

## Project Structure

```
Reinforcement-Learning/
├── rl_lab/
│   ├── agents/           # Agent implementations
│   ├── environments/     # Custom envs, wrappers
│   ├── experiments/      # Training scripts
│   ├── utils/            # Helpers, plotting
│   └── cli.py            # CLI tool
├── notebooks/            # Interactive exploration
├── configs/              # Experiment configs
├── docs/                 # Documentation
└── tests/                # Unit tests
```

## Tool Stack

| Library | Purpose |
|---------|---------|
| **Gymnasium** | Environment API (successor to OpenAI Gym) |
| **Stable-Baselines3** | Production RL algorithms (PPO, DQN, SAC, etc.) |
| **TensorBoard** | Training visualization |
| **PyTorch** | Deep learning backend |

## Environments

### Classic Control
- `CartPole-v1` - Balance a pole on a cart
- `MountainCar-v0` - Drive car up a hill
- `Acrobot-v1` - Swing up a double pendulum
- `Pendulum-v1` - Swing up a pendulum (continuous)

### Box2D
- `LunarLander-v2` - Land a spacecraft
- `BipedalWalker-v3` - Walk with a 2D robot

### Atari (via ale-py)
- Pong, Breakout, Space Invaders, etc.

## Learning Path

1. **Fundamentals** - MDPs, value functions, Bellman equations
2. **Tabular Methods** - Q-learning, SARSA, Monte Carlo
3. **Function Approximation** - Linear, neural network
4. **Policy Gradient** - REINFORCE, Actor-Critic
5. **Deep RL** - DQN, PPO, SAC

## Commands

```bash
# Install dependencies
uv sync

# List environments
uv run rl-lab list-envs

# Environment info
uv run rl-lab info CartPole-v1

# Run demo with random actions
uv run rl-lab demo LunarLander-v2

# TensorBoard
uv run tensorboard --logdir runs/
```

## Key Concepts

- **Episode**: One complete run from reset to termination
- **Step**: One action taken in the environment
- **Return**: Cumulative (discounted) reward
- **Policy**: Mapping from states to actions
- **Value Function**: Expected return from a state

## License

MIT
