"""
Training script for DQN-CNN agent on Snake game.
"""

import os
import json
import numpy as np
import torch

from src.environment.snake_env import SnakeEnvironment
from src.agents.dqn_cnn import DQNCNNAgent
import src.config as config


def evaluate_agent(agent, env, num_episodes=100):
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    scores = []
    survival_times = []

    for _ in range(num_episodes):
        env.reset()
        state = env.get_grid_state()
        done = False
        steps = 0

        while not done and steps < config.MAX_STEPS_PER_EPISODE:
            action = agent.select_action(state, eval_mode=True)
            _, _, done = env.step(action)
            state = env.get_grid_state()
            steps += 1

        scores.append(env.get_score())
        survival_times.append(steps)

    agent.epsilon = old_epsilon

    return {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "max_score": int(np.max(scores)),
        "min_score": int(np.min(scores)),
        "mean_survival": float(np.mean(survival_times)),
        "std_survival": float(np.std(survival_times)),
    }


def train_dqn_cnn(seed=42, save_dir="results/dqn_cnn"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = SnakeEnvironment(
        grid_size=config.GRID_SIZE,
        render=False,
        seed=seed
    )

    agent = DQNCNNAgent(
        grid_size=env.grid_size,
        action_size=4,
        device=device,
        seed=seed
)

    episode_scores = []
    episode_steps = []
    evaluation_results = []

    print(f"Starting CNN training with seed {seed}")

    for episode in range(config.TOTAL_EPISODES):
        env.reset()
        state = env.get_grid_state()
        done = False
        steps = 0
        episode_loss = []

        while not done and steps < config.MAX_STEPS_PER_EPISODE:
            action = agent.select_action(state)
            _, reward, done = env.step(action)
            next_state = env.get_grid_state()

            agent.store_transition(state, action, reward, next_state, done)

            loss = agent.update()
            if loss is not None:
                episode_loss.append(loss)

            state = next_state
            steps += 1

        agent.update_epsilon()

        if (episode + 1) % config.TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_network()

        episode_scores.append(env.get_score())
        episode_steps.append(steps)

        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode+1}/{config.TOTAL_EPISODES} | "
                f"Avg Score (100): {np.mean(episode_scores[-100:]):.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

        if (episode + 1) % config.EVAL_INTERVAL == 0:
            print(f"\n=== Evaluation at episode {episode+1} ===")
            eval_env = SnakeEnvironment(
                grid_size=config.GRID_SIZE,
                render=False,
                seed=seed + 1000
            )

            eval_metrics = evaluate_agent(
                agent,
                eval_env,
                config.EVAL_EPISODES
            )

            eval_metrics["episode"] = episode + 1
            evaluation_results.append(eval_metrics)

            checkpoint_path = (
                f"{save_dir}/checkpoints/"
                f"dqn_cnn_seed{seed}_ep{episode+1}.pt"
            )
            agent.save(checkpoint_path)

    results = {
        "seed": seed,
        "episode_scores": episode_scores,
        "episode_steps": episode_steps,
        "evaluations": evaluation_results,
    }

    with open(f"{save_dir}/config.json", "w") as f:
        config_dict = {
        "GRID_SIZE": config.GRID_SIZE,
        "TOTAL_EPISODES": config.TOTAL_EPISODES,
        "BATCH_SIZE": config.BATCH_SIZE,
        "LEARNING_RATE": config.LEARNING_RATE,
        "GAMMA": config.GAMMA,
        "EPSILON_START": config.EPSILON_START,
        "EPSILON_END": config.EPSILON_END,
        "EPSILON_DECAY_EPISODES": config.EPSILON_DECAY_EPISODES,
        "REPLAY_BUFFER_SIZE": config.REPLAY_BUFFER_SIZE,
        "TARGET_UPDATE_FREQUENCY": config.TARGET_UPDATE_FREQUENCY,
        "EVAL_INTERVAL": config.EVAL_INTERVAL,
        "EVAL_EPISODES": config.EVAL_EPISODES,
        }
        json.dump(config_dict, f, indent=2)

    with open(f"{save_dir}/results_seed{seed}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    
    print("Training completed.")
    return results


if __name__ == "__main__":
    for seed in config.RANDOM_SEEDS:
        train_dqn_cnn(seed=seed)
