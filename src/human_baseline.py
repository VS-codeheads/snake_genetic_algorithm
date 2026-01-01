import csv, os, datetime
import pygame
from src.environment.snake_env import SnakeEnvironment
import src.config as config

PLAYER_ID = "player1"
RUNS = 10
OUT = "results/human_baseline.csv"

def play_one(env):
    """Play one game with human controls (arrow keys)"""
    state = env.reset()
    steps = 0
    done = False
    last_action = 1  # Start with right to prevent immediate reverse
    
    while not done and steps < config.MAX_STEPS_PER_EPISODE:
        # Poll pygame events FIRST
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return env.get_score(), steps
            
            if event.type == pygame.KEYDOWN:
                # Map arrow keys to actions
                if event.key == pygame.K_LEFT:
                    last_action = 0
                elif event.key == pygame.K_RIGHT:
                    last_action = 1
                elif event.key == pygame.K_UP:
                    last_action = 2
                elif event.key == pygame.K_DOWN:
                    last_action = 3
        
        # Step with last action
        _, _, done = env.step(last_action)
        steps += 1
        
        # Manual render (don't call _render() - it consumes events)
        env.game.screen.fill("black")
        for i, p in enumerate(env.snake.body):
            pygame.draw.rect(
                env.game.screen,
                (0, max(128, 255 - i * 8), 0),
                env.game.block(p))
        pygame.draw.rect(
            env.game.screen, 
            env.game.color_food, 
            env.game.block(env.food.p))
        pygame.display.flip()
        env.game.clock.tick(10)  # 10 FPS
    
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
        print(f"\n=== Round {r+1}/{RUNS} ===")
        print("Use arrow keys to control snake. Close window to finish game.")
        score, steps = play_one(env)
        log([PLAYER_ID, r+1, env.grid_size, None, score, steps, datetime.datetime.utcnow().isoformat()])
        print(f"Score: {score}, Steps: {steps}")
    
    print(f"\nResults saved to {OUT}")