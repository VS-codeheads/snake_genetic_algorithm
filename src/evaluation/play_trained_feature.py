import torch
from src.environment.snake_env import SnakeEnvironment
from src.agents.dqn_feature import DQNFeatureAgent

CHECKPOINT_PATH = "results/dqn_feature/checkpoints/dqn_feature_seed42_ep1000.pt"

env = SnakeEnvironment(
    grid_size=30,
    render=True,
    seed=42
)

agent = DQNFeatureAgent(
    state_size=10,     # feature vector length
    action_size=4,
    device="cpu"
)

agent.load(CHECKPOINT_PATH)
agent.epsilon = 0.0  # no exploration

# Run episode 
state = env.reset()          # reset already returns feature state
done = False

while not done:
    action = agent.select_action(state, eval_mode=True)
    state, _, done = env.step(action)
