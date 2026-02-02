"""Tabular Q-Learning agent."""

from typing import Dict, Optional
import numpy as np

from rl_lab.agents.base import TabularAgent


class QLearningAgent(TabularAgent):
    """
    Q-Learning: Off-policy TD control.

    Update rule:
        Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]

    Key insight: Updates toward the BEST action in next state (greedy),
    regardless of what action was actually taken. This makes it off-policy.

    Reference: Watkins & Dayan (1992)
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ):
        """
        Initialize Q-Learning agent.

        Args:
            n_states: Size of state space
            n_actions: Size of action space
            learning_rate: Step size for Q updates (alpha)
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate for ε-greedy
            epsilon_min: Minimum exploration rate
            epsilon_decay: Multiplicative decay for epsilon
            seed: Random seed for reproducibility
        """
        super().__init__(
            n_states=n_states,
            n_actions=n_actions,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            seed=seed,
        )

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> Dict[str, float]:
        """
        Perform Q-learning update.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended

        Returns:
            Dict with td_error metric
        """
        # Current Q-value
        current_q = self.q_table[state, action]

        # Target: r + γ max_a' Q(s', a')
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # TD error
        td_error = target - current_q

        # Update Q-value
        self.q_table[state, action] += self.lr * td_error

        return {"td_error": abs(td_error)}
