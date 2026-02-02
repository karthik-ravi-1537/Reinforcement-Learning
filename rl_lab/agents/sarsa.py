"""SARSA agent - On-policy TD control."""

from typing import Dict, Optional
import numpy as np

from rl_lab.agents.base import TabularAgent


class SarsaAgent(TabularAgent):
    """
    SARSA: On-policy TD control.

    Update rule:
        Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]

    Key insight: Updates toward the ACTUAL next action taken (a'),
    not the best action. This makes it on-policy.

    Name comes from: (S, A, R, S', A') - the quintuple used in updates.

    Reference: Rummery & Niranjan (1994)
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
        Initialize SARSA agent.

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
        self._next_action: Optional[int] = None

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action, maintaining consistency for SARSA updates.

        For SARSA, we need to know the next action before the update.
        This method caches the action for use in the update step.

        Args:
            state: Current state
            training: If True, use epsilon-greedy

        Returns:
            Selected action
        """
        if self._next_action is not None:
            action = self._next_action
            self._next_action = None
            return action
        return super().select_action(state, training)

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> Dict[str, float]:
        """
        Perform SARSA update.

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

        # For SARSA, we need the actual next action
        if done:
            target = reward
        else:
            # Select next action using current policy (on-policy!)
            self._next_action = super().select_action(next_state, training=True)
            target = reward + self.gamma * self.q_table[next_state, self._next_action]

        # TD error
        td_error = target - current_q

        # Update Q-value
        self.q_table[state, action] += self.lr * td_error

        return {"td_error": abs(td_error)}

    def reset_episode(self) -> None:
        """Reset episode-specific state (call at start of each episode)."""
        self._next_action = None


class ExpectedSarsaAgent(TabularAgent):
    """
    Expected SARSA: Uses expected value over actions instead of sampled action.

    Update rule:
        Q(s, a) ← Q(s, a) + α [r + γ Σ_a' π(a'|s') Q(s', a') - Q(s, a)]

    Reduces variance compared to SARSA while maintaining on-policy flavor.
    With a greedy target policy, this becomes Q-learning.

    Reference: van Seijen et al. (2009)
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
        Initialize Expected SARSA agent.

        Args:
            n_states: Size of state space
            n_actions: Size of action space
            learning_rate: Step size for Q updates
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Multiplicative decay for epsilon
            seed: Random seed
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
        Perform Expected SARSA update.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended

        Returns:
            Dict with td_error metric
        """
        current_q = self.q_table[state, action]

        if done:
            target = reward
        else:
            # Compute expected value under ε-greedy policy
            q_next = self.q_table[next_state]
            best_action = np.argmax(q_next)

            # ε-greedy probabilities
            probs = np.ones(self.n_actions) * self.epsilon / self.n_actions
            probs[best_action] += 1 - self.epsilon

            # Expected value
            expected_q = np.sum(probs * q_next)
            target = reward + self.gamma * expected_q

        td_error = target - current_q
        self.q_table[state, action] += self.lr * td_error

        return {"td_error": abs(td_error)}
