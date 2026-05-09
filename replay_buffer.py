"""
replay_buffer.py - Prioritized Experience Replay (PER)

Uses a SumTree data structure to efficiently sample transitions
based on their priorities (TD-error). Includes Importance Sampling
(IS) weights to correct for the sampling bias.
"""

import numpy as np
import random
from typing import Tuple, List, Optional

class SumTree:
    """
    A binary tree where each parent node is the sum of its children.
    Leaf nodes store the actual priorities.
    Allows O(log N) sampling and updates.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        # Tree nodes: 2 * capacity - 1
        self.tree = np.zeros(2 * capacity - 1)
        # Data storage: stores the actual transition indices
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx: int, change: float):
        """Update the sum up to the root."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find a leaf node based on a prefix sum s."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total_priority(self) -> float:
        return self.tree[0]

    def add(self, p: float, data: object):
        """Add a new priority and data to the tree."""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, p: float):
        """Update priority of an existing node."""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get_leaf(self, s: float) -> Tuple[int, float, object]:
        """Sample a transition from the tree."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class ReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    """
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha  # Priority exponent (0 = random, 1 = full prioritization)
        self.epsilon = 1e-6 # Small constant to avoid zero priority
        self.max_priority = 1.0 # Initial priority for new entries

    def __len__(self):
        return self.tree.n_entries

    def push(self, state, action, reward, next_state, done):
        """Store a new transition with max priority."""
        data = (state, action, reward, next_state, done)
        # New transitions get max priority to ensure they are sampled at least once
        self.tree.add(self.max_priority ** self.alpha, data)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        """
        Sample a batch based on priority and compute IS weights.
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idxs, priorities = [], []
        
        segment = self.tree.total_priority() / batch_size
        
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(s)
            
            p = max(self.epsilon, p) # Safety
            priorities.append(p)
            idxs.append(idx)
            
            s_val, a_val, r_val, ns_val, d_val = data
            states.append(s_val)
            actions.append(a_val)
            rewards.append(r_val)
            next_states.append(ns_val)
            dones.append(d_val)

        # Importance Sampling Weights: w = (1/N * 1/P(i))^beta
        sampling_probabilities = np.array(priorities) / self.tree.total_priority()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max() # Normalize for stability

        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), 
                idxs, is_weights)

    def update_priorities(self, idxs: List[int], errors: np.ndarray):
        """Update priorities based on TD-error."""
        for idx, error in zip(idxs, errors):
            # p = (|error| + epsilon) ^ alpha
            p = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, np.abs(error))

    def can_sample(self, batch_size: int) -> bool:
        return self.tree.n_entries >= batch_size
