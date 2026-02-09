"""Proximal Policy Optimization (PPO) agent."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """
    Shared-backbone actor-critic network.

    Actor (policy): π(a|s; θ) — outputs action probabilities
    Critic (value):  V(s; θ)   — estimates state value
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        """
        Args:
            obs_dim: Dimension of observation space
            n_actions: Number of discrete actions
            hidden_dim: Hidden layer size
        """
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action logits and state value.

        Args:
            x: Observation tensor

        Returns:
            Tuple of (action_logits, state_value)
        """
        features = self.shared(x)
        return self.actor(features), self.critic(features).squeeze(-1)

    def get_action_and_value(
        self, state: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.

        Args:
            state: Observation tensor
            action: If provided, compute log_prob for this action

        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value


class RolloutBuffer:
    """
    Buffer for storing on-policy rollout data.

    Stores complete episodes/rollouts, then computes advantages
    using Generalized Advantage Estimation (GAE).
    """

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []

    def push(self, state, action, reward, done, log_prob, value):
        """Store a single transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(
        self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE balances bias and variance in advantage estimation:
        - λ=0: One-step TD (low variance, high bias)
        - λ=1: Monte Carlo (high variance, low bias)
        - λ=0.95: Good default

        δ_t = r_t + γ V(s_{t+1}) - V(s_t)        (TD error)
        A_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}       (GAE)

        Args:
            last_value: V(s_T) for the final state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            Tuple of (advantages, returns) tensors
        """
        advantages = []
        gae = 0.0

        values = self.values + [last_value]

        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                next_value = 0.0
                gae = 0.0
            else:
                next_value = values[t + 1]

            delta = self.rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(self.values, dtype=torch.float32)

        return advantages, returns

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert buffer contents to tensors."""
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        return states, actions, old_log_probs

    def clear(self):
        """Reset the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) with clipped surrogate objective.

    Key ideas:
    1. Collect a batch of on-policy experience
    2. Compute advantages using GAE
    3. Update policy with clipped objective (prevents too-large updates)
    4. Multiple epochs of mini-batch updates on the same data

    The clipped objective:
        L^CLIP = E[min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]

    Where r_t(θ) = π_new(a|s) / π_old(a|s) is the probability ratio.

    Reference: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        batch_size: int = 64,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        """
        Initialize PPO agent.

        Args:
            obs_dim: Dimension of observation space
            n_actions: Number of discrete actions
            hidden_dim: Hidden layer size
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda (bias-variance trade-off)
            clip_eps: PPO clipping parameter (how far policy can move)
            n_epochs: Number of optimization epochs per rollout
            batch_size: Mini-batch size
            vf_coef: Value function loss coefficient
            ent_coef: Entropy bonus coefficient (encourages exploration)
            max_grad_norm: Gradient clipping threshold
            device: PyTorch device
            seed: Random seed
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)

        self.ac = ActorCritic(obs_dim, n_actions, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer()

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """
        Select action using current policy.

        Args:
            state: Current observation
            training: If True, sample; if False, take argmax

        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, _, value = self.ac.get_action_and_value(state_t)

            if not training:
                logits, _ = self.ac(state_t)
                action = logits.argmax(dim=-1)

            return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, done, log_prob, value):
        """Store a transition in the rollout buffer."""
        self.buffer.push(state, action, reward, done, log_prob, value)

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout data.

        Returns:
            Dict with training metrics
        """
        # Get last value for GAE computation
        with torch.no_grad():
            last_state = torch.FloatTensor(self.buffer.states[-1]).unsqueeze(0).to(self.device)
            _, last_value = self.ac(last_state)
            last_value = last_value.item()

        # Compute advantages and returns
        advantages, returns = self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)
        states, actions, old_log_probs = self.buffer.get_tensors()

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update: multiple epochs of mini-batch updates
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(self.n_epochs):
            # Random permutation for mini-batches
            indices = torch.randperm(len(self.buffer))

            for start in range(0, len(self.buffer), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                # Get current policy's action probs and values
                _, new_log_probs, entropy, new_values = self.ac.get_action_and_value(
                    states[batch_idx], actions[batch_idx]
                )

                # Probability ratio: r(θ) = π_new(a|s) / π_old(a|s)
                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])

                # Clipped surrogate objective
                batch_adv = advantages[batch_idx]
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(new_values, returns[batch_idx])

                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                n_updates += 1

        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save({
            "ac": self.ac.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(checkpoint["ac"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
