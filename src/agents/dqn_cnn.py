import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from src.replay_buffer import ReplayBuffer
import src.config as config

class CNNNetwork(nn.Module):
    def __init__(self, grid_size, action_size=4):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, grid_size, grid_size)
            n_flat = self.conv(dummy).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(n_flat, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    

class DQNCNNAgent:
    """
    DQN Agent using CNN-based state representation.
    """

    def __init__(self, grid_size, action_size=4, device="cpu", seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.action_size = action_size
        self.device = device

        # Networks
        self.policy_net = CNNNetwork(grid_size, action_size).to(device)
        self.target_net = CNNNetwork(grid_size, action_size).to(device)
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
        if eval_mode or random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        else:
            return random.randrange(self.action_size)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_history.append(loss.item())
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        self.episode_count += 1

        if self.episode_count < self.epsilon_decay_episodes:
            decay = (config.EPSILON_START - config.EPSILON_END) / self.epsilon_decay_episodes
            self.epsilon = config.EPSILON_START - decay * self.episode_count
        else:
            self.epsilon = config.EPSILON_END

    def load(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]

        if "episode_count" in checkpoint:
            self.episode_count = checkpoint["episode_count"]

    def save(self, filepath):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", config.EPSILON_END)
        self.episode_count = checkpoint.get("episode_count", 0)