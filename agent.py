"""
agent.py - Advanced DQNAgent Integration

Upgraded with:
1. Dueling Double DQN support
2. Prioritized Experience Replay (PER) integration
3. Soft Network Updates (Polyak averaging)
4. Importance-Weighted Loss
5. Gradient Clipping
"""

import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any, List

from config import DQN_CONFIG, PATH_CONFIG
from model import QNetwork
from replay_buffer import ReplayBuffer

class DQNAgent:
    """
    Expert-level Dueling Double DQN Agent with PER and soft updates.
    """

    def __init__(self, state_size: int = None, action_size: int = None):
        self.config = DQN_CONFIG
        self.state_size = state_size or self.config.state_size
        self.action_size = action_size or self.config.action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Networks ---
        # Using the updated Dueling QNetwork from model.py
        self.online_network = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network = QNetwork(self.state_size, self.action_size).to(self.device)
        
        # Hard copy at start
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.config.learning_rate)
        
        # --- PER & Buffer ---
        # ReplayBuffer is now a PrioritizedReplayBuffer
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size, alpha=self.config.per_alpha)
        
        # PER Annealing
        self.beta = self.config.per_beta_start
        self.beta_increment = (1.0 - self.config.per_beta_start) / self.config.per_beta_frames

        # Epsilon-greedy
        self.epsilon = self.config.epsilon_start
        self.epsilon_min = self.config.epsilon_min
        self.epsilon_decay = self.config.epsilon_decay

        # Statistics
        self.training_steps = 0
        self.total_steps = 0
        self.last_loss = 0.0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy selection."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_network(state_t)
            return q_values.argmax(dim=1).item()

    def store_experience(self, state, action, reward, next_state, done):
        """Store in PER buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

    def train_step(self) -> Optional[float]:
        """
        Advanced Training Step:
        1. Sample from PER with Beta annealing
        2. Double DQN Bellman Update
        3. Importance-weighted Loss
        4. Priority Update
        5. Soft Network Sync
        """
        if not self.replay_buffer.can_sample(self.config.batch_size):
            return None

        # 1. Sample with Priority
        states, actions, rewards, next_states, dones, idxs, weights = \
            self.replay_buffer.sample(self.config.batch_size, beta=self.beta)
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Move to device
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        # 2. Current Q Values
        current_q = self.online_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # 3. Double DQN Target Calculation
        # Use Online network for action selection, Target network for evaluation
        with torch.no_grad():
            next_actions = self.online_network(next_states_t).argmax(dim=1, keepdim=True)
            max_next_q = self.target_network(next_states_t).gather(1, next_actions).squeeze(1)
            target_q = rewards_t + self.config.gamma * max_next_q * (1 - dones_t)

        # 4. TD Error and Priority Update
        td_errors = (current_q - target_q).detach().cpu().numpy()
        self.replay_buffer.update_priorities(idxs, td_errors)

        # 5. Weighted MSE Loss
        # Weights come from Importance Sampling
        loss = (weights_t * (current_q - target_q).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping for stability
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), self.config.grad_clip)
        
        self.optimizer.step()

        self.training_steps += 1
        self.last_loss = loss.item()

        # 6. Soft Network Update (instead of hard update)
        self.soft_update_target_network()

        return loss.item()

    def soft_update_target_network(self):
        """
        Softly update target network weights: 
        theta_target = tau * theta_online + (1 - tau) * theta_target
        """
        tau = self.config.tau
        for target_param, online_param in zip(self.target_network.parameters(), self.online_network.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def decay_epsilon(self):
        """Standard decay."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_checkpoint(self, filepath: str, episode: int = 0, extra: Dict = None):
        """Saves current model state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "episode": episode,
            "online_network_state": self.online_network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "total_steps": self.total_steps,
        }
        if extra: checkpoint.update(extra)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Loads model state."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.online_network.load_state_dict(checkpoint["online_network_state"])
        self.target_network.load_state_dict(checkpoint["target_network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon = checkpoint.get("epsilon", self.config.epsilon_min)
        self.training_steps = checkpoint.get("training_steps", 0)
        self.total_steps = checkpoint.get("total_steps", 0)
        return checkpoint

    def load_for_evaluation(self, filepath: str):
        """Ready for demo."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.online_network.load_state_dict(checkpoint.get("online_network_state", checkpoint.get("model_state_dict")))
        self.online_network.eval()
        self.epsilon = 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Stats for the Pygame HUD."""
        return {
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "total_steps": self.total_steps,
            "buffer_size": len(self.replay_buffer),
            "last_loss": self.last_loss,
            "device": str(self.device),
            "beta": getattr(self, 'beta', 0.4)
        }
