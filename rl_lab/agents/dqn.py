"""Deep Q-Network (DQN) agent."""

import copy
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """MLP Q-network for discrete action spaces."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        """
        Args:
            obs_dim: Dimension of observation space
            n_actions: Number of discrete actions
            hidden_dim: Hidden layer size
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions."""
        return self.net(x)


class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning.

    Stores transitions (s, a, r, s', done) and samples random mini-batches.
    Breaking temporal correlations is key to stable DQN training.
    """

    def __init__(self, capacity: int = 100_000):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = torch.FloatTensor(np.array([t[0] for t in batch]))
        actions = torch.LongTensor([t[1] for t in batch])
        rewards = torch.FloatTensor([t[2] for t in batch])
        next_states = torch.FloatTensor(np.array([t[3] for t in batch]))
        dones = torch.BoolTensor([t[4] for t in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent.

    Key innovations over tabular Q-learning:
    1. Neural network function approximation for Q(s, a)
    2. Experience replay buffer (breaks correlation between consecutive samples)
    3. Target network (stabilizes training by providing fixed TD targets)

    Reference: Mnih et al. "Playing Atari with Deep Reinforcement Learning" (2013)
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        """
        Initialize DQN agent.

        Args:
            obs_dim: Dimension of observation space
            n_actions: Number of discrete actions
            hidden_dim: Hidden layer size
            lr: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay per episode
            buffer_size: Replay buffer capacity
            batch_size: Mini-batch size for updates
            target_update_freq: Steps between target network updates
            device: Device for PyTorch tensors
            seed: Random seed
        """
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Online network (the one we train)
        self.q_net = QNetwork(obs_dim, n_actions, hidden_dim).to(device)
        # Target network (provides stable TD targets)
        self.target_net = copy.deepcopy(self.q_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.step_count = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state: Current observation
            training: If True, use epsilon-greedy; if False, greedy

        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return q_values.argmax(dim=1).item()

    def update(self, state, action, reward, next_state, done) -> Dict[str, float]:
        """
        Store transition and learn from mini-batch.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended

        Returns:
            Dict with loss metric (empty if buffer too small)
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.step_count += 1

        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values: Q(s, a) for the actions taken
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + γ max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            target_q = rewards + self.gamma * next_q * ~dones

        # Huber loss (less sensitive to outliers than MSE)
        loss = nn.functional.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update target network periodically
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"loss": loss.item()}

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
        }, path)

    def load(self, path: str) -> None:
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.step_count = checkpoint["step_count"]
