# Snake Reinforcement Learning: DQN Implementation

**Comparative Analysis of State Representations for Deep Q-Learning in Snake Game**

This project investigates how different state representations (feature-based vs. CNN-based) affect the performance and training efficiency of Deep Q-Network (DQN) agents playing Snake.

---

## 🎯 Research Questions

1. **RQ1 (Primary)**: How do different state representations (feature-based vs. CNN-based) affect the performance and training efficiency of DQN agents in Snake?
2. **RQ2 (Baseline)**: How does DQN performance compare to human baseline performance?
3. **RQ3 (Progression)**: What qualitative strategies emerge at different training stages, and how do they differ between representations?

---

## 🧠 What's Implemented

### 1. Feature-Based DQN Agent (DQN-Feature)

**State Representation**: 11-feature vector capturing strategic information:

| Features | Description | Range |
|----------|-------------|-------|
| `[0-1]` food_dx, food_dy | Relative food position | [-2, 2] |
| `[2-4]` danger_left, danger_front, danger_right | Binary collision sensors | {0, 1} |
| `[5-8]` direction (L/R/U/D) | One-hot current direction | {0, 1} |
| `[9]` snake_length_normalized | Length / max_possible_length | [0, 1] |
| `[10]` steps_normalized | Episode progress | [0, 1] |

**Why this works**: 
- Abstracts spatial relationships into engineered features
- Enables fast learning (no need to learn spatial patterns from scratch)
- Markov property: 11 numbers contain sufficient information for optimal decisions

### 2. Deep Q-Network Architecture

**Neural Network**:
```
Input Layer (11 features)
    ↓
Dense Layer (128 neurons) + ReLU
    ↓
Dense Layer (128 neurons) + ReLU
    ↓
Output Layer (4 Q-values: LEFT, RIGHT, UP, DOWN)
```

**Key DQN Components**:

1. **Experience Replay Buffer**
   - Capacity: 10,000 experiences
   - Stores: (state, action, reward, next_state, done)
   - Sampling: Random batches of 64
   - **Why?** Breaks temporal correlations, stabilizes learning

2. **Target Network**
   - Frozen copy of policy network
   - Updated every 100 episodes
   - **Why?** Prevents "moving target" problem in Q-learning

3. **ε-Greedy Exploration**
   - Start: ε = 1.0 (100% random exploration)
   - Decay: Linear over 5,000 episodes
   - End: ε = 0.1 (10% exploration, 90% exploitation)
   - **Why?** Balances exploration vs. exploitation

4. **Q-Learning Update**
   ```python
   Target = reward + γ * max Q_target(next_state, action')
   Loss = MSE(Q_policy(state, action), Target)
   ```
   - Discount factor: γ = 0.95
   - Learning rate: 0.001 (Adam optimizer)

### 3. Reward Structure

| Event | Reward | Rationale |
|-------|--------|-----------|
| Food eaten | +10.0 | Strong positive reinforcement |
| Wall collision | -10.0 | Terminal penalty |
| Self collision | -10.0 | Terminal penalty |
| Each step | -0.01 | Encourages efficiency |

---

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.8
```

### Installation

1. **Clone and navigate**:
```bash
cd snake_genetic_algorithm
```

2. **Create virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows
```

3. **Install dependencies**:
```bash
pip install -r requirments.txt
```

Required packages:
- `pygame >= 2.5.0` - Game rendering
- `numpy >= 1.24.0` - Numerical operations
- `torch >= 2.0.0` - Neural networks
- `matplotlib >= 3.7.0` - Plotting (future)
- `scikit-learn >= 1.3.0` - Analysis (future)

---

## 🏃 Running Training

### Quick Test (1,000 episodes, ~5-10 minutes)

Current configuration trains for 1,000 episodes:

```bash
python -m src.train_dqn_feature
```

**Expected output**:
```
Using device: cpu
Starting training with seed 42
Total episodes: 1000
Epsilon decay: 1.0 -> 0.1 over 5000 episodes

Episode 100/1000 | Score: 0 | Avg Score (100): 0.07 | Epsilon: 0.982 | Loss: 0.1061
Episode 200/1000 | Score: 0 | Avg Score (100): 0.09 | Epsilon: 0.964 | Loss: 0.2496
...
Episode 1000/1000 | Score: 0 | Avg Score (100): 0.28 | Epsilon: 0.820 | Loss: 0.1104

=== Evaluation at episode 1000 ===
Mean Score: 9.03 ± 3.98
Max Score: 18
Mean Survival: 777.83
```

### Full Training (10,000 episodes, ~1-2 hours)

1. **Edit configuration**:
```bash
# In src/config.py
TOTAL_EPISODES = 10000
```

2. **Run training**:
```bash
python -m src.train_dqn_feature
```

### Configuration Options

Edit [`src/config.py`](src/config.py) to customize:

```python
# Environment
GRID_SIZE = 30  # Grid dimensions (30x30)
MAX_STEPS_PER_EPISODE = 1000  # Prevent infinite loops

# Training
TOTAL_EPISODES = 10000
EVAL_INTERVAL = 1000  # Evaluate every N episodes
EVAL_EPISODES = 100  # Games per evaluation
RANDOM_SEEDS = [42, 123, 456]  # Multiple runs for statistical validity

# Hyperparameters
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 10000
LEARNING_RATE = 0.001
GAMMA = 0.95
TARGET_UPDATE_FREQUENCY = 100

# Exploration
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_EPISODES = 5000
```

---

## 📊 Output and Results

### Directory Structure

```
results/
└── dqn_feature/
    ├── dqn_feature_seed42_results.json    # Training metrics
    ├── dqn_feature_seed123_results.json
    ├── dqn_feature_seed456_results.json
    └── checkpoints/
        ├── dqn_feature_seed42_ep1000.pt   # Model checkpoints
        ├── dqn_feature_seed123_ep1000.pt
        └── dqn_feature_seed456_ep1000.pt
```

### Results JSON Format

```json
{
  "seed": 42,
  "episode_scores": [0, 1, 0, 2, ...],  // Per-episode training scores
  "episode_steps": [23, 45, 12, ...],   // Steps per episode
  "evaluations": [
    {
      "episode": 1000,
      "mean_score": 9.03,
      "std_score": 3.98,
      "max_score": 18,
      "min_score": 0,
      "mean_survival": 777.83,
      "std_survival": 245.12
    }
  ],
  "config": {
    "batch_size": 64,
    "learning_rate": 0.001,
    "gamma": 0.95,
    ...
  }
}
```

### Interpreting Results

**Training Metrics** (logged every 100 episodes):
- **Score**: Number of food items eaten this episode
- **Avg Score (100)**: Rolling average over last 100 episodes
- **Epsilon**: Current exploration rate (1.0 → 0.1)
- **Loss**: Magnitude of TD-error (should stabilize over time)

**Evaluation Metrics** (every 1,000 episodes, ε=0):
- **Mean Score**: Average performance over 100 greedy test games
- **Max Score**: Best game during evaluation
- **Mean Survival**: Average steps before death (max 1,000)

**Expected Learning Trajectory**:
- Episodes 1-1000: Avg score 0.05 → 0.3, eval mean ~2-9
- Episodes 1000-5000: Rapid improvement as ε decays, eval mean ~10-20
- Episodes 5000-10000: Refinement with ε=0.1, eval mean ~15-25+

---

## 🧪 Testing

Verify installation and setup:

```bash
python test_dqn_setup.py
```

This runs 12 unit tests covering:
- Environment creation and state representation
- Replay buffer operations
- DQN agent initialization and action selection
- Mini training loop (20 episodes)

**Expected output**:
```
============================================================
Testing DQN Setup
============================================================

1. Testing environment import...
   ✓ SnakeEnvironment imported successfully

2. Testing environment creation...
   ✓ Environment created successfully
   - Grid size: 30x30
   - State shape: (11,)
   - Num actions: 4

...

============================================================
✓ ALL TESTS PASSED!
============================================================
```

---

## 📁 Project Structure

```
snake_genetic_algorithm/
├── README.md                 # This file
├── requirments.txt           # Python dependencies
├── test_dqn_setup.py        # Unit tests
│
├── src/
│   ├── __init__.py
│   ├── config.py            # Hyperparameters and settings
│   ├── replay_buffer.py     # Experience replay implementation
│   ├── train_dqn_feature.py # Training script (feature-based)
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── dqn_feature.py   # DQN agent with feature state
│   │   └── random_agent.py  # Baseline random agent
│   │
│   └── environment/
│       ├── __init__.py
│       ├── snake_env.py     # RL environment wrapper
│       └── snake_game.py    # Core Snake game logic
│
└── results/
    └── dqn_feature/         # Training outputs
        ├── *.json           # Metrics
        └── checkpoints/     # Model weights
```

---

## 📈 Current Results (1,000 Episodes)

**Seed 42**:
- Mean Score: **9.03 ± 3.98**
- Max Score: **18**
- Mean Survival: **777.83 steps**

**Seed 123**:
- Mean Score: **5.89 ± 4.22**
- Max Score: **17**
- Mean Survival: **869.78 steps**

**Seed 456**:
- Mean Score: **2.08 ± 2.61**
- Max Score: **12**
- Mean Survival: **949.14 steps**

**Analysis**:
- ✅ Learning is happening (training scores improved 0.07 → 0.28-0.41)
- ✅ Generalization works (evaluation with ε=0 shows non-trivial strategies)
- ✅ Variance across seeds demonstrates need for multi-seed protocol
- ⏳ Full performance pending 10K episodes (only 1K/10K complete, ε still 0.82)

---

## 🔬 Algorithm Details

### DQN Q-Learning Update

At each timestep:

1. **Observe** state `s`
2. **Select** action `a` using ε-greedy policy
3. **Execute** action, observe reward `r` and next state `s'`
4. **Store** transition `(s, a, r, s', done)` in replay buffer
5. **Sample** random batch of 64 transitions
6. **Compute** target values:
   ```
   y = r                           if done
   y = r + γ * max_a' Q_target(s', a')  otherwise
   ```
7. **Update** policy network:
   ```
   Loss = MSE(Q_policy(s, a), y)
   θ ← θ - α∇Loss
   ```
8. **Update** target network every 100 episodes:
   ```
   θ_target ← θ_policy
   ```

### Feature Engineering Rationale

Each feature serves a specific strategic purpose:

1. **Food Direction** (`food_dx`, `food_dy`): Navigate toward goal
2. **Danger Sensors** (left/front/right): Avoid immediate collisions
3. **Current Direction**: Prevent 180° turns, maintain momentum
4. **Snake Length**: Adjust strategy as snake grows (more cautious)
5. **Episode Progress**: Encourage time-efficient strategies

This abstraction allows the agent to learn policies like:
- "If food is ahead and no danger front, go forward"
- "If danger on all sides except left, turn left"
- "If snake is long, prioritize safety over aggressive food-seeking"

---

## 🚧 Next Steps

### Immediate (In Progress)
- [x] Implement feature-based DQN agent
- [x] Run training with 3 random seeds
- [x] Save checkpoints and evaluation metrics
- [ ] Complete full 10,000 episode training

### Research Implementation
- [ ] **DQN-CNN**: Implement CNN-based agent (RQ1)
  - 30×30×3 grid state (snake body, head, food channels)
  - Convolutional architecture (3 conv layers + FC)
  - Compare learning efficiency vs. feature-based
  
- [ ] **Human Baseline**: Collect human performance data (RQ2)
  - 5 players × 10 games each
  - Record scores and strategies
  - Establish performance benchmark

- [ ] **Random Agent**: Run baseline experiments
  - Control condition for comparison
  - Expected: ~0.05 mean score

- [ ] **Evaluation Suite**: Unified evaluation across all agents
  - 100 episodes per agent with ε=0
  - Statistical comparison (mean, variance, max score)
  - Qualitative strategy analysis

### Analysis and Paper
- [ ] Generate learning curves and comparison plots
- [ ] Statistical significance testing (t-tests, ANOVA)
- [ ] Qualitative strategy analysis at different training stages
- [ ] Write LaTeX research paper with results

---

## 🎓 Research Hypothesis

**Expected Results**:
1. **Performance Hierarchy**: DQN-CNN > DQN-Feature > Random
2. **Training Efficiency**: DQN-Feature learns faster initially, DQN-CNN achieves higher final performance
3. **Human Comparison**: DQN agents approach/surpass average human performance within training budget
4. **Emergent Behaviors**: 
   - DQN-CNN: Spatially-aware, context-dependent strategies
   - DQN-Feature: Rule-like, predictable behaviors

**Contingency Plans**:
- If training too slow: Reduce grid to 20×20
- If DQN struggles: Implement Double DQN variant
- If human baseline difficult: Use pre-recorded gameplay

---

## 📚 References

**Core Algorithm**:
- Mnih et al. (2015). "Human-level control through deep reinforcement learning." *Nature*
- Van Hasselt et al. (2016). "Deep Reinforcement Learning with Double Q-learning." *AAAI*

**Implementation**:
- PyTorch DQN Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*

---

## 🤝 Contributing

This is a research project. For questions or collaboration:
- Review code in `src/` directory
- Check `test_dqn_setup.py` for usage examples
- Refer to `src/config.py` for all tunable parameters

---

## 📝 License

Research project - December 2025