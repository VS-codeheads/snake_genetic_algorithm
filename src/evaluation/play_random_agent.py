import time
from src.environment.snake_env import SnakeEnvironment
from src.agents.random_agent import RandomAgent


def main():
    env = SnakeEnvironment(
        grid_size=30,
        render=True,
        seed=42
    )

    agent = RandomAgent(
        action_size=env.get_num_actions(),
        seed=42
    )

    state = env.reset()
    done = False

    print("Playing Random Agent... (close window to stop)")

    while not done:
        action = agent.select_action(state)
        state, reward, done = env.step(action)

        # Slow down rendering so it's watchable
        time.sleep(0.05)

    print("Game over!")
    print("Final score:", env.get_score())


if __name__ == "__main__":
    main()
