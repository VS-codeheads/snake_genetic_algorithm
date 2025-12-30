"""
Experience Replay Buffer for DQN training.
"""

import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """
    
    def __init__(self, capacity):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.buffer, batch_size)
        
        states = np.array([e[0] for e in experiences], dtype=np.float32)
        actions = np.array([e[1] for e in experiences], dtype=np.int64)
        rewards = np.array([e[2] for e in experiences], dtype=np.float32)
        next_states = np.array([e[3] for e in experiences], dtype=np.float32)
        dones = np.array([e[4] for e in experiences], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)