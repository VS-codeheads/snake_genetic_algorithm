# human_baseline.py
import csv, os, datetime
import pygame
from src.environment.snake_env import SnakeEnvironment

PLAYER_ID = "player1"
RUNS = 10
OUT = "results/human_baseline.csv"

def play_one(env):
    state = env.reset()
    steps = 0
    done = False
    while not done and steps < env.max_steps:  # or MAX_STEPS_PER_EPISODE
        # poll pygame events for QUIT + arrow keys
        # map arrows to action {0:left,1:right,2:up,3:down}
        # env.step(action)
        steps += 1
    return env.get_score(), steps

def log(row):
    os.makedirs("results", exist_ok=True)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["player_id","run","grid","seed","score","steps","timestamp"])
        w.writerow(row)

if __name__ == "__main__":
    env = SnakeEnvironment(render=True, grid_size=30, seed=None)
    for r in range(RUNS):
        score, steps = play_one(env)
        log([PLAYER_ID, r+1, env.grid_size, None, score, steps, datetime.datetime.utcnow().isoformat()])