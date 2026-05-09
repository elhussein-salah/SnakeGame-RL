"""
snake_game.py - Snake Game Environment for Reinforcement Learning

This module implements the Snake game as an RL environment with:
- 2D grid-based movement
- Relative action space (straight, left, right)
- Compact 11-dimensional binary state representation
- Shaped reward function with proximity bonuses and loop penalties
- Episode termination on wall/self collision or max steps

The environment is fully decoupled from rendering logic.
"""

import random
import numpy as np
from typing import List, Tuple, Optional
from enum import IntEnum

from config import GAME_CONFIG, REWARD_CONFIG


class Direction(IntEnum):
    """Cardinal directions for snake movement."""
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


# Direction vectors: maps Direction -> (dx, dy) in grid coordinates
DIRECTION_VECTORS = {
    Direction.RIGHT: (1, 0),
    Direction.DOWN:  (0, 1),
    Direction.LEFT:  (-1, 0),
    Direction.UP:    (0, -1),
}

# Relative turn mappings
# Action 0: straight (no turn), Action 1: turn left, Action 2: turn right
TURN_LEFT = {
    Direction.RIGHT: Direction.UP,
    Direction.UP:    Direction.LEFT,
    Direction.LEFT:  Direction.DOWN,
    Direction.DOWN:  Direction.RIGHT,
}

TURN_RIGHT = {
    Direction.RIGHT: Direction.DOWN,
    Direction.DOWN:  Direction.LEFT,
    Direction.LEFT:  Direction.UP,
    Direction.UP:    Direction.RIGHT,
}


class SnakeGame:
    """
    Snake Game Environment for DQN training.

    The snake moves on a grid_size x grid_size board.
    Actions are relative to the snake's current facing direction:
        0 -> Continue straight
        1 -> Turn left (relative)
        2 -> Turn right (relative)

    State is an 11-dimensional binary vector capturing:
        - Danger in 3 relative directions
        - Food direction (4 booleans)
        - Current movement direction (4 booleans, one-hot)
    """

    def __init__(self, grid_size: int = None, num_obstacles: int = None):
        self.grid_size = grid_size or GAME_CONFIG.grid_size
        self.max_steps = GAME_CONFIG.max_steps_per_episode
        self.num_obstacles = num_obstacles if num_obstacles is not None else GAME_CONFIG.num_obstacles
        self.obstacles: List[Tuple[int, int]] = []
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reset the game to initial state.

        Returns:
            Initial state vector (11-dim binary numpy array).
        """
        # Initialize snake at center of grid, facing right
        center = self.grid_size // 2
        self.direction = Direction.RIGHT

        # Snake body: list of (x, y) tuples, head is first element
        self.snake = [
            (center, center),           # Head
            (center - 1, center),       # Body segment 1
            (center - 2, center),       # Body segment 2 (tail)
        ]

        self.score = 0
        self.steps = 0
        self.total_reward = 0.0
        self.game_over = False

        # Track recent positions for loop detection
        self._recent_positions: List[Tuple[int, int]] = []
        self._loop_window = 20  # Check last N positions for loops

        # Spawn obstacles at random positions (not on snake)
        self.obstacles = self._spawn_obstacles()

        # Spawn food at random empty position
        self.food = self._spawn_food()

        return self.get_state()

    def _spawn_obstacles(self) -> List[Tuple[int, int]]:
        """
        Spawn obstacles at random positions not occupied by the snake.
        Obstacles stay fixed for the entire episode.
        """
        if self.num_obstacles <= 0:
            return []

        occupied = set(self.snake)
        obstacles = []
        all_cells = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in occupied
        ]
        random.shuffle(all_cells)
        for cell in all_cells[:self.num_obstacles]:
            obstacles.append(cell)
        return obstacles

    def _spawn_food(self) -> Tuple[int, int]:
        """
        Spawn food at a random position not occupied by the snake or obstacles.

        Returns:
            (x, y) position of the new food.
        """
        occupied = set(self.snake) | set(self.obstacles)
        empty_cells = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in occupied
        ]

        if not empty_cells:
            return self.snake[0]

        return random.choice(empty_cells)

    @property
    def head(self) -> Tuple[int, int]:
        """Get the current head position."""
        return self.snake[0]

    def _is_collision(self, point: Tuple[int, int]) -> bool:
        """
        Check if a point collides with walls or snake body.

        Args:
            point: (x, y) position to check.

        Returns:
            True if the point is a collision.
        """
        x, y = point
        # Wall collision
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        # Self collision (check against body, excluding head)
        if point in self.snake[1:]:
            return True
        # Obstacle collision
        if point in self.obstacles:
            return True
        return False

    def _get_next_position(self, direction: Direction) -> Tuple[int, int]:
        """
        Get the next position if moving in the given direction.

        Args:
            direction: The direction to move.

        Returns:
            (x, y) of the next position.
        """
        dx, dy = DIRECTION_VECTORS[direction]
        return (self.head[0] + dx, self.head[1] + dy)

    def _detect_loop(self) -> bool:
        """
        Detect if the snake is moving in repetitive loops.

        Uses a simple heuristic: if recent positions contain many
        repeated locations, the agent is likely looping.

        Returns:
            True if looping behavior is detected.
        """
        if len(self._recent_positions) < self._loop_window:
            return False

        recent = self._recent_positions[-self._loop_window:]
        unique_positions = len(set(recent))
        # If the snake visits fewer than 30% unique positions in the window,
        # it's likely stuck in a loop
        return unique_positions < self._loop_window * 0.3

    def get_state(self) -> np.ndarray:
        """
        Compute the 11-dimensional binary state vector.

        State representation:
            [0] danger_straight  - Collision if continuing straight
            [1] danger_left      - Collision if turning left
            [2] danger_right     - Collision if turning right
            [3] food_left        - Food is to the left of head
            [4] food_right       - Food is to the right of head
            [5] food_up          - Food is above head
            [6] food_down        - Food is below head
            [7] moving_left      - Currently moving left
            [8] moving_right     - Currently moving right
            [9] moving_up        - Currently moving up
            [10] moving_down     - Currently moving down

        Returns:
            numpy array of shape (11,) with binary values.
        """
        head_x, head_y = self.head
        food_x, food_y = self.food

        # Compute directions for relative actions
        dir_straight = self.direction
        dir_left = TURN_LEFT[self.direction]
        dir_right = TURN_RIGHT[self.direction]

        # Check danger in each relative direction
        danger_straight = int(self._is_collision(self._get_next_position(dir_straight)))
        danger_left = int(self._is_collision(self._get_next_position(dir_left)))
        danger_right = int(self._is_collision(self._get_next_position(dir_right)))

        # Food direction relative to head (absolute coordinates)
        food_left = int(food_x < head_x)
        food_right = int(food_x > head_x)
        food_up = int(food_y < head_y)      # y decreases going up
        food_down = int(food_y > head_y)

        # Current movement direction (one-hot encoding)
        moving_left = int(self.direction == Direction.LEFT)
        moving_right = int(self.direction == Direction.RIGHT)
        moving_up = int(self.direction == Direction.UP)
        moving_down = int(self.direction == Direction.DOWN)

        state = np.array([
            danger_straight,
            danger_left,
            danger_right,
            food_left,
            food_right,
            food_up,
            food_down,
            moving_left,
            moving_right,
            moving_up,
            moving_down,
        ], dtype=np.float32)

        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: 0 = straight, 1 = turn left, 2 = turn right

        Returns:
            Tuple of (next_state, reward, done, info_dict)
        """
        self.steps += 1
        reward = REWARD_CONFIG.step_penalty  # Base step penalty

        # --- Compute distance to food BEFORE moving ---
        old_distance = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])

        # --- Update direction based on relative action ---
        if action == 1:     # Turn left
            self.direction = TURN_LEFT[self.direction]
        elif action == 2:   # Turn right
            self.direction = TURN_RIGHT[self.direction]
        # action == 0: continue straight (no direction change)

        # --- Move snake ---
        new_head = self._get_next_position(self.direction)

        # --- Check collision ---
        if self._is_collision(new_head):
            self.game_over = True
            reward = REWARD_CONFIG.collision_penalty
            self.total_reward += reward
            info = {
                "score": self.score,
                "steps": self.steps,
                "cause": "collision",
            }
            return self.get_state(), reward, True, info

        # Move the snake: insert new head
        self.snake.insert(0, new_head)

        # --- Check if food is eaten ---
        if new_head == self.food:
            self.score += 1
            reward = REWARD_CONFIG.food_reward
            self.food = self._spawn_food()
            # Don't remove tail -> snake grows
        else:
            self.snake.pop()  # Remove tail -> snake moves without growing

            # --- Proximity reward shaping ---
            new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            if new_distance < old_distance:
                reward += REWARD_CONFIG.closer_to_food_reward
            elif new_distance > old_distance:
                reward += REWARD_CONFIG.farther_from_food_penalty

        # --- Track positions for loop detection ---
        self._recent_positions.append(new_head)
        if len(self._recent_positions) > self._loop_window * 2:
            self._recent_positions = self._recent_positions[-self._loop_window * 2:]

        # --- Apply loop penalty ---
        if self._detect_loop():
            reward += REWARD_CONFIG.loop_penalty

        # --- Check max steps (prevents infinite episodes) ---
        done = False
        if self.steps >= self.max_steps:
            done = True
            info_cause = "max_steps"
        else:
            info_cause = "running"

        self.total_reward += reward

        info = {
            "score": self.score,
            "steps": self.steps,
            "cause": info_cause,
        }

        return self.get_state(), reward, done, info

    def get_direction_name(self) -> str:
        """Get human-readable name of current direction."""
        names = {
            Direction.RIGHT: "RIGHT",
            Direction.DOWN: "DOWN",
            Direction.LEFT: "LEFT",
            Direction.UP: "UP",
        }
        return names.get(self.direction, "UNKNOWN")
