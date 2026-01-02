# Comparative Analysis of State Representations for Deep Q-Learning in Snake Game

A comprehensive empirical study comparing **feature-engineered** vs. **CNN-based** state representations for Deep Q-Network (DQN) agents in the Snake game.

**Research Question**: How do different state representations affect DQN performance, training efficiency, and learned strategies in fully-observable game environments?

**Key Finding**: Feature-engineered representations achieve **7.4× superhuman performance** (mean score 18.5 vs human 2.5), while CNN agents fail to learn (score 0.05), demonstrating that domain knowledge dramatically accelerates learning in small-scale environments.

---

## 📋 Contents

- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Implementation Details](#-implementation-details)
- [Recreating Findings](#-recreating-findings)
- [Results Summary](#-results-summary)

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
cd snake_genetic_algorithm

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements**:
- Python 3.8+
- PyTorch 2.0+
- NumPy 1.24+
- Pygame 2.5+
- Matplotlib 3.7+

### Train Feature-Based Agent (5,000 episodes)

```bash
python -m src.training.train_dqn_feature
```

### Train CNN Agent (5,000 episodes)

```bash
python -m src.training.train_dqn_cnn
```

### Evaluate All Agents

```bash
python -m src.evaluate_agents
```

### Generate Visualizations

```bash
python -m src.plot_results
```

---

## 📁 Project Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── config.py                    # Hyperparameter configuration
│   ├── evaluate_agents.py           # Unified evaluation script
│   ├── human_baseline.py            # Interactive human baseline
│   ├── plot_results.py              # Generate visualizations
│   ├── replay_buffer.py             # Experience replay
│   ├── agents/
│   │   ├── dqn_feature.py          # Feature-based DQN
│   │   ├── dqn_cnn.py              # CNN-based DQN
│   │   └── random_agent.py         # Random baseline
│   ├── environment/
│   │   ├── snake_env.py            # RL environment wrapper
│   │   └── snake_game.py           # Core game logic
│   ├── evaluation/
│   │   ├── play_random_agent.py
│   │   ├── play_trained_agent.py
│   │   └── play_trained_feature.py
│   └── training/
│       ├── train_dqn_feature.py    # Feature agent training
│       └── train_dqn_cnn.py        # CNN agent training
├── report/
│   ├── paper.tex                   # 5-page research paper
│   └── figures/
│       ├── learning_curves.png
│       ├── comparison.png
│       └── survival.png
├── results/
│   ├── dqn_feature/
│   ├── dqn_cnn/
│   └── eval/
└── notebooks/
    └── snake_game.ipynb
```

---

## 🧠 State Representations

### Feature-Based (DQN-Feature)

**10-dimensional vector**:

| # | Feature | Range |
|---|---------|-------|
| 1-2 | Relative food position (X, Y) | [-2, 2] |
| 3-5 | Collision threat (left, front, right) | {0, 1} |
| 6-9 | Direction one-hot (up, down, left, right) | {0, 1} |
| 10 | Normalized snake length | [0, 1] |

**Architecture**: 10 → 128 → 128 → 4 (18K parameters)

**Results**: Mean score **18.5** (7.4× human baseline)

### CNN-Based (DQN-CNN)

**3-channel 30×30 grid**:
- Channel 0: Snake body
- Channel 1: Snake head
- Channel 2: Food

**Architecture**: Conv layers + Flatten + FC (320K parameters)

**Results**: Mean score **0.05** (fails to learn)

---

## 🔬 Recreating Findings

### 1. Configure Training

Edit `src/config.py`:
```python
TOTAL_EPISODES = 5000
EVAL_INTERVAL = 1000
EVAL_EPISODES = 100
RANDOM_SEEDS = [42, 123, 456]
```

### 2. Train Feature Agent

```bash
python -m src.training.train_dqn_feature
```

**Expected**: 
- Seed 42: 21.4
- Seed 123: 21.0
- Seed 456: 13.0
- Mean: **18.5 ± 4.7**
- vs Human (2.5): **7.4×**

### 3. Train CNN Agent

```bash
python -m src.training.train_dqn_cnn
```

**Expected**: ~0.05 (no learning)

### 4. Evaluate & Visualize

```bash
python -m src.evaluate_agents
python -m src.plot_results
```

### 5. Build Paper

```bash
cd report && pdflatex paper.tex
```

---

## 📊 Results

| Metric | Feature | CNN | Human | Random |
|--------|---------|-----|-------|--------|
| Score | **18.5 ± 4.7** | 0.05 | 2.5 | 0.08 |
| Max | 46 | 1 | --- | 1 |
| Survival | 629 ± 143 | 408 | ~1000 | 380 |
| vs Human | **7.4×** | 0.02× | 1.0× | 0.03× |

### Learning Phases

| Phase | Episodes | Behavior | Score |
|-------|----------|----------|-------|
| Exploration | 0–1,500 | Random, no learning | < 0.5 |
| Rapid Learning | 1,500–3,500 | Food-seeking emerges | 2–3 |
| Mastery | 3,500–5,000 | Path-planning | > 10 |

---

## 🏗️ Implementation Details

### DQN Components

**Experience Replay**:
- Capacity: 10,000 experiences
- Batch: 64 samples
- Purpose: Decorrelates temporal data

**Target Network**:
- Frozen policy copy
- Update: Every 100 episodes
- Purpose: Stabilizes learning

**ε-Greedy Exploration**:
- Start: 1.0 → End: 0.1
- Decay: 5,000 episodes
- Purpose: Balance exploration/exploitation

**Q-Learning Update**:
```
Target = reward + γ × max Q_target(s', a')
Loss = MSE(Q_policy(s, a), Target)
```

### Reward Structure

| Event | Reward |
|-------|--------|
| Food | +10 |
| Collision | -10 |
| Step | -0.01 |

---

## 🔧 Configuration

Edit `src/config.py`:

```python
TOTAL_EPISODES = 5000
EVAL_INTERVAL = 1000
EVAL_EPISODES = 100
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 10000
LEARNING_RATE = 0.001
GAMMA = 0.95
TARGET_UPDATE_FREQUENCY = 100
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_EPISODES = 5000
```

---

## 🧪 Testing

```bash
python test_dqn_setup.py       # Feature based
python test_dqn_cnn_setup.py   # CNN tests
```

---

## 👥 Authors

**Sofie Amalie Roer Thorlund** & **Viktor Mekis Bach**

Academic Research Project, 2026

---

**Last Updated**: January 2, 2026