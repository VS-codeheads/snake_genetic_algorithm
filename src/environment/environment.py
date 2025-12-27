import random
from collections import deque
import numpy as np
import pygame
from snake_game import SnakeGame, Snake, Food, Vector

class SnakeEnvironment:
    
    def __init__(self, render=False):
        self.render = render
        self.game = SnakeGame()
        self.snake = None
        self.food = None
        self.done = False

    def _render(self):
        self.game.screen.fill("black")

        for i, p in enumerate(self.snake.body):
            pygame.draw.rect(
                self.game.screen,
                (0, max(128, 255 - i * 8), 0),
                self.game.block(p))
            
        pygame.draw.rect(
            self.game.screen, 
            self.game.color_food, 
            self.game.block(self.food.p))
        
        pygame.display.flip()
        self.game.clock.tick(10)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True

    def reset(self):
        self.snake = Snake(game=self.game)
        self.food = Food(game=self.game)
        self.done = False
        return self._get_state()
    
    def _apply_action(self, action):
        # 0 = left, 1 = right, 2 = up, 3 = down
        if action == 0:
            self.snake.v = Vector(-1, 0)
        elif action == 1:
            self.snake.v = Vector(1, 0)
        elif action == 2:
            self.snake.v = Vector(0, -1)
        elif action == 3:
            self.snake.v = Vector(0, 1)

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has ended. Please reset the environment.")
        
        # Agent takes action
        self._apply_action(action)

        # Move snake
        self.snake.move()

        reward = 0.0

        # Checking for wall collision
        if not self.snake.p.within(self.game.grid):
            self.done = True
            reward = -1.0
            
        # Checking for self collision
        elif self.snake.cross_own_tail:
            self.done = True
            reward = -1.0
        
        elif self.snake.p == self.food.p:
            self.snake.add_score()
            self.food = Food(game=self.game)  # Spawn new food
            reward = 1.0

        if self.render:
            self._render()

        return self._get_state(), reward, self.done

    def _get_state(self):
        return np.array([0], dtype=np.float32)  # Placeholder for state representation
    # will be replaced with actual state representation logic (feature based state and CNN grid)


# Test with a random agent 
env = SnakeEnvironment(render=True)
state = env.reset()

done = False
while not done:
    action = np.random.randint(0, 4)
    state, reward, done = env.step(action)

print("Episode finished")