# src/agents/random_agent.py
"""
Random baseline agent for Snake game.
"""

import random
import numpy as np


class RandomAgent:
    """
    Agent that selects actions uniformly at random.
    """
    
    def __init__(self, action_size=4, seed=None):
        """
        Initialize random agent.
        
        Args:
            action_size: Number of possible actions
            seed: Random seed for reproducibility
        """
        self.action_size = action_size
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def select_action(self, state, eval_mode=False):
        """
        Select random action.
        
        Args:
            state: Current state (unused)
            eval_mode: Evaluation mode (unused)
            
        Returns:
            Random action
        """
        return random.randrange(self.action_size)
    
    def store_transition(self, *args):
        """No-op for compatibility."""
        pass
    
    def update(self):
        """No-op for compatibility."""
        pass
    
    def update_epsilon(self):
        """No-op for compatibility."""
        pass