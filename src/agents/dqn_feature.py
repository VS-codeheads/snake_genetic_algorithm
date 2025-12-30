"""
DQN Agent with feature-based state representation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from src.replay_buffer import ReplayBuffer
import src.config as config


class FeatureNetwork(nn.Module):
    """
    Feed-forward neural network for feature-based state representation.
    Input: 10 features
    Output: Q-values for 4 actions
    """
    
    def __init__(self, state_size, action_size, hidden_layers=[128, 128]):
        """
        Initialize network.
        
        Args:
            state_size: Dimension of state (10 for feature-based)
            action_size: Number of actions (4)
            hidden_layers: List of hidden layer sizes
        """
        super(FeatureNetwork, self).__init__()
        
        layers = []
        input_size = state_size
        
        # Build hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through network."""
        return self.network(x)


class DQNFeatureAgent:
    """
    DQN Agent using feature-based state representation.
    """
    
    def __init__(self, state_size=10, action_size=4, device='cpu', seed=None):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            device: Device to run model on ('cpu' or 'cuda')
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Q-Networks
        self.policy_net = FeatureNetwork(
            state_size, 
            action_size, 
            config.FEATURE_HIDDEN_LAYERS
        ).to(device)
        
        self.target_net = FeatureNetwork(
            state_size, 
            action_size, 
            config.FEATURE_HIDDEN_LAYERS
        ).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=config.LEARNING_RATE
        )
        
        # Replay buffer
        self.memory = ReplayBuffer(config.REPLAY_BUFFER_SIZE)
        
        # Training parameters
        self.gamma = config.GAMMA
        self.batch_size = config.BATCH_SIZE
        
        # Exploration parameters
        self.epsilon = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        self.epsilon_decay_episodes = config.EPSILON_DECAY_EPISODES
        self.episode_count = 0
        
        # Loss tracking
        self.loss_history = []
    
    def select_action(self, state, eval_mode=False):
        """
        Select action using ε-greedy policy.
        
        Args:
            state: Current state
            eval_mode: If True, always select greedy action (ε=0)
            
        Returns:
            Selected action (int)
        """
        if eval_mode or random.random() > self.epsilon:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        else:
            # Random action
            return random.randrange(self.action_size)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.add(state, action, reward, next_state, done)
    
    def update(self):
        """
        Update policy network using a batch from replay buffer.
        
        Returns:
            Loss value if update performed, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def update_target_network(self):
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_epsilon(self):
        """Decay epsilon after each episode."""
        self.episode_count += 1
        
        if self.episode_count < self.epsilon_decay_episodes:
            # Linear decay
            decay_rate = (config.EPSILON_START - config.EPSILON_END) / self.epsilon_decay_episodes
            self.epsilon = config.EPSILON_START - decay_rate * self.episode_count
        else:
            self.epsilon = config.EPSILON_END
    
    def save(self, filepath):
        """Save model checkpoint."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
        }, filepath)
    
    def load(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']