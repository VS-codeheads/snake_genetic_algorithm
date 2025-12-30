"""
Configuration parameters for DQN Snake training experiments.
"""

# Environment Configuration
GRID_SIZE = 30
MAX_STEPS_PER_EPISODE = 1000  # Prevent infinite loops

# Training Configuration
TOTAL_EPISODES = 1000
EVAL_INTERVAL = 1000  # Evaluate every 1000 episodes
EVAL_EPISODES = 100  # Number of episodes for evaluation
NUM_SEEDS = 3  # Number of random seeds to run
RANDOM_SEEDS = [42, 123, 456]

# DQN Hyperparameters
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 10000
LEARNING_RATE = 0.001
GAMMA = 0.95  # Discount factor
TARGET_UPDATE_FREQUENCY = 100  # Update target network every N episodes

# Exploration Strategy (ε-greedy)
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_EPISODES = 5000  # Decay over first 5000 episodes

# Neural Network Architecture (DQN-Feature)
FEATURE_HIDDEN_LAYERS = [128, 128]  # Two hidden layers with 128 neurons each

# Neural Network Architecture (DQN-CNN)
CNN_CHANNELS = [32, 64, 64]  # Convolutional channels
CNN_KERNELS = [8, 4, 3]  # Kernel sizes
CNN_STRIDES = [4, 2, 1]  # Strides
CNN_FC_LAYER = 512  # Fully connected layer after CNN

# Logging
LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"