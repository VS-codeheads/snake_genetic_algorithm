# src/evaluate_agents.py
import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch

from src.environment.snake_env import SnakeEnvironment
from src.agents.dqn_feature import DQNFeatureAgent
from src.agents.dqn_cnn import DQNCNNAgent
from src.agents.random_agent import RandomAgent
import src.config as config


def run_eval(agent, env, episodes, use_grid=False):
    old_eps = getattr(agent, "epsilon", None)
    if old_eps is not None:
        agent.epsilon = 0.0

    scores, steps_list = [], []
    for _ in range(episodes):
        env.reset()
        state = env.get_grid_state() if use_grid else env._get_state()
        done = False
        steps = 0
        while not done and steps < config.MAX_STEPS_PER_EPISODE:
            action = agent.select_action(state, eval_mode=True)
            _, _, done = env.step(action)
            state = env.get_grid_state() if use_grid else env._get_state()
            steps += 1
        scores.append(env.get_score())
        steps_list.append(steps)

    if old_eps is not None:
        agent.epsilon = old_eps

    return {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "max_score": int(np.max(scores)),
        "min_score": int(np.min(scores)),
        "mean_survival": float(np.mean(steps_list)),
        "std_survival": float(np.std(steps_list)),
    }


def eval_feature(seed, episodes, device, checkpoint):
    env = SnakeEnvironment(grid_size=config.GRID_SIZE, render=False, seed=seed)
    agent = DQNFeatureAgent(state_size=10, action_size=4, device=device, seed=seed)
    if checkpoint:
        agent.load(checkpoint)
    return run_eval(agent, env, episodes, use_grid=False)


def eval_cnn(seed, episodes, device, checkpoint):
    env = SnakeEnvironment(grid_size=config.GRID_SIZE, render=False, seed=seed)
    agent = DQNCNNAgent(grid_size=env.grid_size, action_size=4, device=device, seed=seed)
    if checkpoint:
        agent.load(checkpoint)
    return run_eval(agent, env, episodes, use_grid=True)


def eval_random(seed, episodes):
    env = SnakeEnvironment(grid_size=config.GRID_SIZE, render=False, seed=seed)
    agent = RandomAgent(action_size=4, seed=seed)
    return run_eval(agent, env, episodes, use_grid=False)


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation for Snake agents")
    parser.add_argument("--episodes", type=int, default=config.EVAL_EPISODES, help="Episodes per agent")
    parser.add_argument("--seeds", type=int, nargs="+", default=config.RANDOM_SEEDS, help="Seeds to evaluate")
    parser.add_argument("--feature-checkpoint", type=str, default=None, help="Path to feature DQN checkpoint")
    parser.add_argument("--cnn-checkpoint", type=str, default=None, help="Path to CNN DQN checkpoint")
    parser.add_argument("--outdir", type=str, default="results/eval", help="Directory to save results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "episodes": args.episodes,
        "seeds": args.seeds,
        "device": device,
        "feature_checkpoint": args.feature_checkpoint,
        "cnn_checkpoint": args.cnn_checkpoint,
        "config": {
            "GRID_SIZE": config.GRID_SIZE,
            "MAX_STEPS_PER_EPISODE": config.MAX_STEPS_PER_EPISODE,
        },
        "agents": [],
    }

    for seed in args.seeds:
        print(f"Evaluating seed {seed} on {device}")

        feat_metrics = eval_feature(seed, args.episodes, device, args.feature_checkpoint)
        cnn_metrics = eval_cnn(seed, args.episodes, device, args.cnn_checkpoint)
        rand_metrics = eval_random(seed, args.episodes)

        results["agents"].append({
            "seed": seed,
            "feature": feat_metrics,
            "cnn": cnn_metrics,
            "random": rand_metrics,
        })

        print(f"  Feature: {feat_metrics}")
        print(f"  CNN:     {cnn_metrics}")
        print(f"  Random:  {rand_metrics}")

    out_path = os.path.join(args.outdir, f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation to {out_path}")


if __name__ == "__main__":
    main()