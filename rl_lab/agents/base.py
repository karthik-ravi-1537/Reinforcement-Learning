"""Base agent class for all RL agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for RL agents.

    All agents should implement:
    - select_action: Choose an action given a state
    - update: Learn from experience
    - save/load: Persist learned parameters
    """

    def __init__(self, n_states: int, n_actions: int, seed: Optional[int] = None):
        """
        Initialize the agent.

        Args:
            n_states: Size of state space
            n_actions: Size of action space
            seed: Random seed for reproducibility
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select an action for the given state.

        Args:
            state: Current state
            training: If True, may explore; if False, act greedily

        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> Dict[str, float]:
        """
        Update the agent from a single transition.

        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended

        Returns:
            Dict of metrics (e.g., {"td_error": 0.5})
        """
        pass

    def save(self, path: str) -> None:
        """Save agent parameters to file."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load agent parameters from file."""
        raise NotImplementedError


class TabularAgent(BaseAgent):
    """
    Base class for tabular RL agents with Q-table.

    Provides:
    - Q-table initialization and access
    - Epsilon-greedy action selection
    - Save/load functionality
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
        Initialize tabular agent.

        Args:
            n_states: Size of state space
            n_actions: Size of action space
            learning_rate: Learning rate (alpha)
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon after each episode
            seed: Random seed
        """
        super().__init__(n_states, n_actions, seed)

        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialize Q-table with small random values to break ties
        self.q_table = self.rng.uniform(low=0, high=0.01, size=(n_states, n_actions))

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state: Current state
            training: If True, use epsilon-greedy; if False, act greedily

        Returns:
            Selected action
        """
        if training and self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def decay_epsilon(self) -> None:
        """Decay exploration rate (call at end of each episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_values(self, state: int) -> np.ndarray:
        """Get Q-values for all actions in a state."""
        return self.q_table[state].copy()

    def save(self, path: str) -> None:
        """Save Q-table and parameters to file."""
        np.savez(
            path,
            q_table=self.q_table,
            epsilon=self.epsilon,
            lr=self.lr,
            gamma=self.gamma,
        )

    def load(self, path: str) -> None:
        """Load Q-table and parameters from file."""
        data = np.load(path)
        self.q_table = data["q_table"]
        self.epsilon = float(data["epsilon"])
        self.lr = float(data["lr"])
        self.gamma = float(data["gamma"])
