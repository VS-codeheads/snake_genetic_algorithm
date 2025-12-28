# src/train_dqn_feature.py
"""
Training script for DQN-Feature agent on Snake game.
"""

import os
import json
import numpy as np
import torch
from datetime import datetime
from src.environment.snake_env import SnakeEnvironment
from src.agents.dqn_feature import DQNFeatureAgent
import src.config as config


def evaluate_agent(agent, env, num_episodes=100):
    """
    Evaluate agent performance over multiple episodes.
    
    Args:
        agent: DQN agent
        env: Snake environment
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    scores = []
    survival_times = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < config.MAX_STEPS_PER_EPISODE:
            action = agent.select_action(state, eval_mode=True)
            state, reward, done = env.step(action)
            steps += 1
        scores.append(env.get_score())
        survival_times.append(steps)

    return {
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'max_score': int(np.max(scores)),
        'min_score': int(np.min(scores)),
        'mean_survival': float(np.mean(survival_times)),
        'std_survival': float(np.std(survival_times)),
    }


def train_dqn_feature(seed=42, save_dir='results'):
    """
    Train DQN-Feature agent.
    
    Args:
        seed: Random seed
        save_dir: Directory to save results
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    
    # Initialize environment and agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    env = SnakeEnvironment(grid_size=config.GRID_SIZE, render=False, seed=seed)
    agent = DQNFeatureAgent(
        state_size=11,
        action_size=4,
        device=device,
        seed=seed
    )
    
    # Training metrics
    episode_scores = []
    episode_steps = []
    evaluation_results = []
    
    print(f"Starting training with seed {seed}")
    print(f"Total episodes: {config.TOTAL_EPISODES}")
    print(f"Epsilon decay: {config.EPSILON_START} -> {config.EPSILON_END} over {config.EPSILON_DECAY_EPISODES} episodes")
    
    for episode in range(config.TOTAL_EPISODES):
        state = env.reset()
        done = False
        steps = 0
        episode_loss = []
        
        # Training loop
        while not done and steps < config.MAX_STEPS_PER_EPISODE:
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            loss = agent.update()
            if loss is not None:
                episode_loss.append(loss)
            
            state = next_state
            steps += 1
        
        # Update epsilon and target network
        agent.update_epsilon()
        
        if (episode + 1) % config.TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_network()
        
        # Record metrics
        episode_scores.append(env.get_score())
        episode_steps.append(steps)
        
        # Logging
        if (episode + 1) % 100 == 0:
            recent_scores = episode_scores[-100:]
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            print(f"Episode {episode + 1}/{config.TOTAL_EPISODES} | "
                  f"Score: {env.get_score()} | "
                  f"Avg Score (100): {np.mean(recent_scores):.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")
        
        # Evaluation
        if (episode + 1) % config.EVAL_INTERVAL == 0:
            print(f"\n=== Evaluation at episode {episode + 1} ===")
            eval_env = SnakeEnvironment(grid_size=config.GRID_SIZE, render=False, seed=seed+1000)
            eval_metrics = evaluate_agent(agent, eval_env, config.EVAL_EPISODES)
            eval_metrics['episode'] = episode + 1
            evaluation_results.append(eval_metrics)
            
            print(f"Mean Score: {eval_metrics['mean_score']:.2f} ± {eval_metrics['std_score']:.2f}")
            print(f"Max Score: {eval_metrics['max_score']}")
            print(f"Mean Survival: {eval_metrics['mean_survival']:.2f}")
            print("=" * 50 + "\n")
            
            # Save checkpoint
            checkpoint_path = f"{save_dir}/checkpoints/dqn_feature_seed{seed}_ep{episode+1}.pt"
            agent.save(checkpoint_path)
    
    # Save final results
    results = {
        'seed': seed,
        'episode_scores': episode_scores,
        'episode_steps': episode_steps,
        'evaluations': evaluation_results,
        'config': {
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'gamma': config.GAMMA,
            'epsilon_start': config.EPSILON_START,
            'epsilon_end': config.EPSILON_END,
            'replay_buffer_size': config.REPLAY_BUFFER_SIZE,
        }
    }
    
    results_path = f"{save_dir}/dqn_feature_seed{seed}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed! Results saved to {results_path}")
    
    return results


if __name__ == "__main__":
    # Train with all seeds
    for seed in config.RANDOM_SEEDS:
        print(f"\n{'='*60}")
        print(f"Training with seed {seed}")
        print(f"{'='*60}\n")
        train_dqn_feature(seed=seed, save_dir='results/dqn_feature')