#!/usr/bin/env python3
"""CLI for RL Lab."""

import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"


def cmd_list_envs(args):
    """List available Gymnasium environments."""
    import gymnasium as gym

    print("\n=== Classic Control ===")
    for env_id in ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v1"]:
        print(f"  {env_id}")

    print("\n=== Box2D ===")
    for env_id in ["LunarLander-v2", "BipedalWalker-v3"]:
        print(f"  {env_id}")

    print("\n=== All registered environments ===")
    all_envs = gym.envs.registry.keys()
    print(f"  Total: {len(list(all_envs))} environments")
    print("  Use: gymnasium.make('EnvName-v0')")


def cmd_info(args):
    """Show info about an environment."""
    import gymnasium as gym

    try:
        env = gym.make(args.env_id)
        from rl_lab.utils.common import print_env_info

        print_env_info(env)
        env.close()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_demo(args):
    """Run a quick demo of an environment with random actions."""
    import gymnasium as gym

    env = gym.make(args.env_id, render_mode="human")
    obs, info = env.reset()

    total_reward = 0
    steps = 0

    print(f"\nRunning {args.env_id} with random actions...")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                print(f"Episode finished: {steps} steps, reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0
                steps = 0
    except KeyboardInterrupt:
        print("\nDemo stopped.")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="RL Lab CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # list-envs
    subparsers.add_parser("list-envs", help="List available environments")

    # info
    info_parser = subparsers.add_parser("info", help="Show environment info")
    info_parser.add_argument("env_id", help="Environment ID (e.g., CartPole-v1)")

    # demo
    demo_parser = subparsers.add_parser("demo", help="Run environment demo")
    demo_parser.add_argument("env_id", help="Environment ID (e.g., CartPole-v1)")

    args = parser.parse_args()

    if args.command == "list-envs":
        cmd_list_envs(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
