#!/usr/bin/env python3
"""
Experiment 00: Verify Setup

Verifies that all RL tools are installed and working.
Run this first to ensure your environment is ready.

Usage:
    uv run python rl_lab/experiments/00_verify_setup.py
"""

import sys


def check_pytorch():
    """Check PyTorch and device availability."""
    print("\n[1/4] Checking PyTorch...")
    import torch

    print(f"  PyTorch version: {torch.__version__}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Using device: {device}")

    # Quick tensor test
    x = torch.randn(100, 100, device=device)
    y = x @ x.T
    print(f"  Tensor ops working: {y.shape}")
    return True


def check_gymnasium():
    """Check Gymnasium installation."""
    print("\n[2/4] Checking Gymnasium...")
    import gymnasium as gym

    print(f"  Gymnasium version: {gym.__version__}")

    # Test classic control
    env = gym.make("CartPole-v1")
    obs, info = env.reset()
    print(f"  CartPole-v1: obs shape = {obs.shape}")

    # Take a step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Step taken: reward = {reward}")
    env.close()

    # Count available envs
    all_envs = list(gym.envs.registry.keys())
    print(f"  Total registered environments: {len(all_envs)}")

    return True


def check_stable_baselines():
    """Check Stable-Baselines3 installation."""
    print("\n[3/4] Checking Stable-Baselines3...")
    import stable_baselines3 as sb3

    print(f"  SB3 version: {sb3.__version__}")

    # Check available algorithms
    from stable_baselines3 import A2C, DQN, PPO, SAC

    print("  Available algorithms: PPO, DQN, A2C, SAC")

    # Quick PPO test (don't train, just instantiate)
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=0)
    print(f"  PPO model created: {type(model).__name__}")
    env.close()

    return True


def check_visualization():
    """Check visualization tools."""
    print("\n[4/4] Checking visualization...")
    import matplotlib

    print(f"  matplotlib: {matplotlib.__version__}")

    import plotly

    print(f"  plotly: {plotly.__version__}")

    try:
        import tensorboard

        print(f"  tensorboard: installed")
    except ImportError:
        print("  tensorboard: not installed")

    return True


def main():
    print("=" * 60)
    print("REINFORCEMENT LEARNING LAB SETUP VERIFICATION")
    print("=" * 60)

    checks = [
        ("PyTorch + Device", check_pytorch),
        ("Gymnasium", check_gymnasium),
        ("Stable-Baselines3", check_stable_baselines),
        ("Visualization", check_visualization),
    ]

    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll checks passed! Your environment is ready for RL.")
        print("\nNext steps:")
        print("  1. Run: uv run rl-lab demo CartPole-v1")
        print("  2. Explore: uv run rl-lab list-envs")
        print("  3. Train your first agent!")
    else:
        print("\nSome checks failed. Install missing packages with:")
        print("  uv sync")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
