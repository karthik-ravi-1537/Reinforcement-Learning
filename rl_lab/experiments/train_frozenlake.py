"""Train tabular agents on FrozenLake."""

import argparse
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from rl_lab.agents import QLearningAgent, SarsaAgent, ExpectedSarsaAgent


def train_agent(
    agent,
    env,
    n_episodes: int = 10000,
    max_steps: int = 100,
) -> Tuple[List[float], List[float]]:
    """
    Train an agent on the environment.

    Args:
        agent: The agent to train
        env: Gymnasium environment
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode

    Returns:
        Tuple of (episode_rewards, running_success_rate)
    """
    episode_rewards = []
    successes = []
    running_success = []

    for episode in range(n_episodes):
        state, _ = env.reset()

        # Reset episode state for SARSA
        if hasattr(agent, "reset_episode"):
            agent.reset_episode()

        total_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        successes.append(1 if total_reward > 0 else 0)

        # Running success rate (last 100 episodes)
        window = successes[-100:]
        running_success.append(sum(window) / len(window))

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}: Success rate = {running_success[-1]:.1%}, ε = {agent.epsilon:.3f}")

    return episode_rewards, running_success


def evaluate_agent(agent, env, n_episodes: int = 1000) -> float:
    """
    Evaluate trained agent without exploration.

    Args:
        agent: Trained agent
        env: Gymnasium environment
        n_episodes: Number of evaluation episodes

    Returns:
        Success rate
    """
    successes = 0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if reward > 0:
                successes += 1
                break
    return successes / n_episodes


def main():
    parser = argparse.ArgumentParser(description="Train tabular agents on FrozenLake")
    parser.add_argument("--agent", type=str, default="q_learning",
                        choices=["q_learning", "sarsa", "expected_sarsa", "all"])
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--slippery", action="store_true", default=True)
    parser.add_argument("--no-slippery", dest="slippery", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--save-plot", type=str, default=None)
    args = parser.parse_args()

    # Create environment
    env = gym.make("FrozenLake-v1", is_slippery=args.slippery)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    print(f"FrozenLake-v1 (slippery={args.slippery})")
    print(f"States: {n_states}, Actions: {n_actions}")
    print(f"Training for {args.episodes} episodes\n")

    agent_classes = {
        "q_learning": QLearningAgent,
        "sarsa": SarsaAgent,
        "expected_sarsa": ExpectedSarsaAgent,
    }

    if args.agent == "all":
        agents_to_train = list(agent_classes.keys())
    else:
        agents_to_train = [args.agent]

    results = {}
    for agent_name in agents_to_train:
        print(f"\n{'='*50}")
        print(f"Training {agent_name}")
        print('='*50)

        AgentClass = agent_classes[agent_name]
        agent = AgentClass(
            n_states=n_states,
            n_actions=n_actions,
            learning_rate=args.lr,
            gamma=args.gamma,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.9995,
            seed=args.seed,
        )

        rewards, success_rates = train_agent(agent, env, n_episodes=args.episodes)
        eval_success = evaluate_agent(agent, env)
        results[agent_name] = {
            "rewards": rewards,
            "success_rates": success_rates,
            "eval_success": eval_success,
            "agent": agent,
        }

        print(f"\nFinal evaluation: {eval_success:.1%} success rate")

    # Plot results
    if len(results) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for name, data in results.items():
            ax1.plot(data["success_rates"], label=name, alpha=0.8)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Success Rate (100-ep rolling)")
        ax1.set_title("Training Progress")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bar chart of final performance
        names = list(results.keys())
        eval_scores = [results[n]["eval_success"] for n in names]
        ax2.bar(names, eval_scores)
        ax2.set_ylabel("Success Rate")
        ax2.set_title("Final Evaluation (1000 episodes)")
        ax2.set_ylim(0, 1)
        for i, v in enumerate(eval_scores):
            ax2.text(i, v + 0.02, f"{v:.1%}", ha="center")

        plt.tight_layout()

        if args.save_plot:
            plt.savefig(args.save_plot, dpi=150)
            print(f"\nPlot saved to {args.save_plot}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
