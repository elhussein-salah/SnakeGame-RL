"""
model.py - Dueling Deep Q-Network Architecture

Implements a Dueling DQN architecture which splits the network into two streams:
1. Value Stream V(s): Estimating the value of being in a state.
2. Advantage Stream A(s,a): Estimating the relative advantage of each action.

Formula: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class QNetwork(nn.Module):
    """
    Dueling DQN Architecture for Snake AI.
    Separates state value estimation from action advantage estimation.
    """

    def __init__(self, state_size: int, action_size: int, hidden_layers: Tuple[int, ...] = (256, 128, 64)):
        super(QNetwork, self).__init__()
        
        # --- Shared Feature Extractor ---
        # Learns general features from the 11-dim state vector
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU()
        )
        
        # --- Value Stream ---
        # Estimates V(s): The scalar value of being in the current state
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_layers[1], hidden_layers[2]),
            nn.ReLU(),
            nn.Linear(hidden_layers[2], 1)
        )
        
        # --- Advantage Stream ---
        # Estimates A(s,a): The relative advantage of each of the 3 actions
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_layers[1], hidden_layers[2]),
            nn.ReLU(),
            nn.Linear(hidden_layers[2], action_size)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining Value and Advantage streams.
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        """
        # 1. Extract shared features
        features = self.feature_layer(state)
        
        # 2. Compute Value and Advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # 3. Combine using the Dueling DQN formula
        # Subtracting the mean of advantages provides stability and addresses identifiability
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
