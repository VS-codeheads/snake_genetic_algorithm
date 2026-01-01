import torch
from src.environment.snake_env import SnakeEnvironment
from src.agents.dqn_cnn import DQNCNNAgent

CHECKPOINT_PATH = "results/dqn_cnn/checkpoints/dqn_cnn_seed42_ep1000.pt"

env = SnakeEnvironment(
    grid_size=30,
    render=True,
    seed=42
)

agent = DQNCNNAgent(
    grid_size=env.grid_size,
    action_size=4,
    device="cpu"
)

agent.load(CHECKPOINT_PATH)
agent.epsilon = 0.0  # no exploration


state = env.reset()
state = env.get_grid_state()

done = False
while not done:
    action = agent.select_action(state, eval_mode=True)
    _, _, done = env.step(action)
    state = env.get_grid_state()
