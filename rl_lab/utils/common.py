"""Common utilities for RL experiments."""

import os
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


def get_device() -> str:
    """Get the best available device for PyTorch."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int, env: gym.Env | None = None) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
        env: Optional Gymnasium environment to seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.reset(seed=seed)


def create_run_dir(base_dir: str = "runs", name: str | None = None) -> Path:
    """Create a timestamped directory for experiment runs.

    Args:
        base_dir: Base directory for runs.
        name: Optional name prefix for the run.

    Returns:
        Path to the created directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{name}_{timestamp}" if name else timestamp
    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_env(env_id: str, seed: int = 0, render_mode: str | None = None) -> gym.Env:
    """Create and configure a Gymnasium environment.

    Args:
        env_id: Environment ID (e.g., "CartPole-v1").
        seed: Random seed for the environment.
        render_mode: Render mode ("human", "rgb_array", or None).

    Returns:
        Configured Gymnasium environment.
    """
    env = gym.make(env_id, render_mode=render_mode)
    env.reset(seed=seed)
    return env


def print_env_info(env: gym.Env) -> None:
    """Print information about a Gymnasium environment.

    Args:
        env: Gymnasium environment.
    """
    print("=" * 50)
    print(f"Environment: {env.spec.id if env.spec else 'Unknown'}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    if hasattr(env, "reward_range"):
        print(f"Reward range: {env.reward_range}")

    if hasattr(env.spec, "max_episode_steps") and env.spec.max_episode_steps:
        print(f"Max episode steps: {env.spec.max_episode_steps}")

    print("=" * 50)
