# Claude Rules for RL Lab

## Communication Style
- Be concise. Short answers are preferred over verbose explanations.
- Don't ask unnecessary clarifying questions - make reasonable decisions.
- When the user provides feedback or a preference, apply it without lengthy justification.
- "clear" means acknowledged, move on.

## Code Style
- Include docstrings for functions with Args/Returns sections.
- Use descriptive variable names.
- Add comments only where logic isn't self-evident.
- Use type hints for function signatures.

## Git Workflow
- Commit messages: short summary line, optional body for context.
- Do NOT include Co-Authored-By unless explicitly requested.
- Push when explicitly asked.

## Project Context
- This is a **Reinforcement Learning** learning lab.
- Goal: Build RL expertise from fundamentals to deep RL.
- Target hardware: Apple M2 16GB (local).

### Key Directories
- `rl_lab/agents/` - RL agent implementations
- `rl_lab/environments/` - Custom environments, wrappers
- `rl_lab/experiments/` - Training scripts, experiments
- `rl_lab/utils/` - Plotting, logging, helpers
- `notebooks/` - Interactive exploration
- `configs/` - Experiment configurations
- `docs/` - Research framing, findings

### Main Libraries
- **Gymnasium**: `gymnasium.make("CartPole-v1")`, standard env API
- **Stable-Baselines3**: `PPO`, `DQN`, `A2C`, `SAC` implementations
- **TensorBoard**: Training visualization

### Common Environments
- Classic Control: CartPole, MountainCar, Acrobot, Pendulum
- Box2D: LunarLander, BipedalWalker
- Atari: via ale-py
- MuJoCo: HalfCheetah, Hopper, Walker2d (if available)

## Package Management
- Uses **uv**, not pip.
- `uv sync` to install, `uv add <pkg>` to add dependencies.
- Lock file: `uv.lock` (do not edit manually).

## Running Experiments
- Training: `uv run python rl_lab/experiments/<script>.py`
- TensorBoard: `uv run tensorboard --logdir runs/`
- Videos saved to `videos/` directory

## RL Conventions
- Episode = one complete run from reset to done
- Step = one action taken
- Return = cumulative discounted reward
- Use `gymnasium` not `gym` (gym is deprecated)
