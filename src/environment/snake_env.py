import random
from collections import deque
import numpy as np
import pygame
from src.environment.snake_game import SnakeGame, Snake, Food, Vector


class SnakeEnvironment:
    """
    SnakeEnvironment wrapper for DQN training.
    Provides feature-based state representation with 0 features:
    [food_dx, food_dy, danger_left, danger_front, danger_right,
     direction_left, direction_right, direction_up, direction_down,
     snake_length_normalized]
    """

    ACTION_LEFT = 0
    ACTION_RIGHT = 1
    ACTION_UP = 2
    ACTION_DOWN = 3
    
    # Direction vectors
    DIRECTION_MAP = {
        ACTION_LEFT: Vector(-1, 0),
        ACTION_RIGHT: Vector(1, 0),
        ACTION_UP: Vector(0, -1),
        ACTION_DOWN: Vector(0, 1),
    }
    
    def __init__(self, grid_size=30, render=False, seed=None):
        """
        Initialize Snake environment.
        
        Args:
            grid_size: Size of the game grid (grid_size x grid_size)
            render: Whether to render the game
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.render = render
        self.grid_size = grid_size
        self.game = SnakeGame(xsize=grid_size, ysize=grid_size, scale=15)
        self.snake = None
        self.food = None
        self.done = False
        self.max_snake_length = grid_size * grid_size // 2  # Reasonable max length
        self.last_direction = Vector(1, 0)  # Track current direction
        # self.step_count = 0  # Track steps in episode ! Not used in 10-feature representation

    def _render(self):
        """Render the current game state."""
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
        """Reset environment for a new episode."""
        self.snake = Snake(game=self.game)
        self.food = Food(game=self.game)
        self.done = False
        # Initialize snake with a default direction (moving right)
        self.last_direction = Vector(1, 0)
        self.snake.v = Vector(1, 0)  # Set initial velocity
        # self.step_count = 0 - ! Not used in 10-feature representation
        return self._get_state()
    
    def _apply_action(self, action):
        """Apply action to update snake direction."""
        direction = self.DIRECTION_MAP[action]
        
        # Prevent 180-degree turns (going back into itself)
        if not self._is_opposite_direction(self.last_direction, direction):
            self.snake.v = direction
            self.last_direction = direction
        else:
            # Keep last direction if move is invalid
            self.snake.v = self.last_direction

    @staticmethod
    def _is_opposite_direction(current: Vector, next_dir: Vector) -> bool:
        """Check if next_dir is opposite to current direction."""
        return (current.x == -next_dir.x and current.y == -next_dir.y)

    def _get_danger_sensors(self):
        """
        Get binary danger sensors for left, front, right directions relative to snake's current direction.
        Returns: [danger_left, danger_front, danger_right]
        """
        head = self.snake.p
        direction = self.snake.v
        
        # If snake hasn't started moving yet, use last_direction
        if direction.x == 0 and direction.y == 0:
            direction = self.last_direction
        
        # Calculate perpendicular directions (left and right relative to movement)
        # If moving right (1, 0), left is up (0, -1), right is down (0, 1)
        left_dir = Vector(direction.y, -direction.x)
        right_dir = Vector(-direction.y, direction.x)
        
        # Check positions one step ahead in each direction
        left_pos = head + left_dir
        front_pos = head + direction
        right_pos = head + right_dir
        
        danger_left = not left_pos.within(self.game.grid) or left_pos in self.snake.body
        danger_front = not front_pos.within(self.game.grid) or front_pos in self.snake.body
        danger_right = not right_pos.within(self.game.grid) or right_pos in self.snake.body
        
        return [float(danger_left), float(danger_front), float(danger_right)]

    def _get_direction_onehot(self):
        """
        Get one-hot encoding of current snake direction.
        Returns: [is_left, is_right, is_up, is_down]
        """
        direction = self.snake.v
        
        # If snake hasn't started moving yet, use last_direction
        if direction.x == 0 and direction.y == 0:
            direction = self.last_direction
        
        direction_encoding = [0.0, 0.0, 0.0, 0.0]
        
        if direction.x == -1:  # Left
            direction_encoding[0] = 1.0
        elif direction.x == 1:  # Right
            direction_encoding[1] = 1.0
        elif direction.y == -1:  # Up
            direction_encoding[2] = 1.0
        elif direction.y == 1:  # Down
            direction_encoding[3] = 1.0
        
        return direction_encoding

    def _get_state(self):
        """
        Get feature-based state representation (10 features).
        
        Features:
        [0-1]: food_dx, food_dy - Normalized relative food position
        [2-4]: danger_left, danger_front, danger_right - Binary collision sensors
        [5-8]: direction_left, direction_right, direction_up, direction_down - One-hot direction
        [9-10]: snake_length_normalized
        
        Returns:
            np.ndarray: Feature vector of shape (10,) with dtype float32
        """
        head = self.snake.p
        
        # Feature 0-1: Relative food position (normalized to [-2, 2] range)
        food_dx = (self.food.p.x - head.x) / self.grid_size * 2
        food_dy = (self.food.p.y - head.y) / self.grid_size * 2
        
        # Feature 2-4: Danger sensors
        danger_sensors = self._get_danger_sensors()
        
        # Feature 5-8: Direction one-hot
        direction_onehot = self._get_direction_onehot()
        
        # Feature 9: Normalized snake length
        snake_length_normalized = len(self.snake.body) / self.max_snake_length
        
        # Feature 10: Steps normalized (encourages efficiency)
        #steps_normalized = min(self.step_count / 1000.0, 1.0)  ! This is an extra feature - introduces 11 featues which we do not work with in cnn
        
        # Combine all features
        state = np.array(
            [food_dx, food_dy] + 
            danger_sensors + 
            direction_onehot + 
            [snake_length_normalized], # + [steps_normalized]  ! This is the 11th feature - breaks the “representation-only difference” claim in synopsis
            dtype=np.float32
        )
        
        return state

    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Integer in [0, 3] representing direction
            
        Returns:
            tuple: (state, reward, done) where:
                - state: np.ndarray of shape (10,)
                - reward: float
                - done: bool
        """
        if self.done:
            raise RuntimeError("Episode has ended. Please reset the environment.")
        
        # self.step_count += 1 ! Not used in 10-feature representation
        
        # Agent takes action
        self._apply_action(action)

        # Move snake
        self.snake.move()

        reward = 0.0

        if not self.snake.p.within(self.game.grid):
            self.done = True
            reward = -1.0

        elif self.snake.cross_own_tail:
            self.done = True
            reward = -1.0

        elif self.snake.p == self.food.p:
            self.snake.add_score()
            self.food = Food(game=self.game)
            reward = +1.0

        if self.render:
            self._render()

        return self._get_state(), reward, self.done

    def get_state_shape(self):
        """Return the shape of the state representation."""
        return (10,)

    def get_num_actions(self):
        """Return the number of possible actions."""
        return 4

    def get_score(self):
        """Return current snake score."""
        return self.snake.score if self.snake else 0
    

    def get_grid_state(self):
        """
        This is the CNN based grid state representation.
        Shape: (3, grid_size, grid_size)
        """
        grid = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)

        # Snake body
        for segment in self.snake.body:
            if segment.within(self.game.grid):
                grid[0, segment.y, segment.x] = 1.0

        # Snake head
        head = self.snake.p
        if head.within(self.game.grid):
            grid[1, head.y, head.x] = 1.0

        # Food
        food = self.food.p
        if food.within(self.game.grid):
            grid[2, food.y, food.x] = 1.0

        return grid
    

env = SnakeEnvironment(render=False)
state = env.reset()

print(state)
print(state.shape)