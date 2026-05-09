"""
config.py - Configuration and Hyperparameters for DQN Snake Game AI

Contains all tunable parameters for the DQN agent, game environment,
training loop, and visualization settings. Centralizing configuration
makes experimentation and hyperparameter tuning straightforward.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class GameConfig:
    """Configuration for the Snake game environment."""
    grid_size: int = 10                     # Grid dimensions (grid_size x grid_size)
    block_size: int = 70                    # Pixel size of each grid cell
    max_steps_per_episode: int = 200        # Prevent infinite loops
    num_obstacles: int = 0                  # Number of obstacles inside the grid
    speed_increase_per_food: int = 2        # FPS increase each time snake eats food

    @property
    def window_width(self) -> int:
        """Total window width including HUD panel."""
        return self.grid_size * self.block_size + self.hud_width

    @property
    def window_height(self) -> int:
        """Total window height."""
        return self.grid_size * self.block_size

    @property
    def game_area_width(self) -> int:
        """Width of the game area only."""
        return self.grid_size * self.block_size

    hud_width: int = 300                    # Width of the stats panel


@dataclass
class DQNConfig:
    """Configuration for the DQN agent and training."""
    # --- Neural Network ---
    state_size: int = 11                    # 11 binary features in state vector
    action_size: int = 3                    # straight, turn left, turn right
    hidden_layers: Tuple[int, ...] = (256, 128, 64)  # Hidden layer sizes

    # --- Training Hyperparameters ---
    gamma: float = 0.99                     # Discount factor for future rewards
    learning_rate: float = 0.001            # Adam optimizer learning rate
    batch_size: int = 64                    # Minibatch size for training
    replay_buffer_size: int = 100_000       # Maximum replay memory capacity

    # --- Exploration (Epsilon-Greedy) ---
    epsilon_start: float = 1.0              # Initial exploration rate
    epsilon_min: float = 0.01               # Minimum exploration rate
    epsilon_decay: float = 0.995            # Multiplicative decay per episode

    # --- Target Network & Stability ---
    target_update_frequency: int = 1000     # Steps between target network sync (if hard sync)
    tau: float = 0.005                      # Soft update parameter (if soft sync)
    grad_clip: float = 10.0                 # Gradient clipping max norm
    
    # --- Prioritized Experience Replay (PER) ---
    per_alpha: float = 0.6                  # Priority exponent
    per_beta_start: float = 0.4             # Initial importance sampling weight
    per_beta_frames: int = 100_000          # Frames over which to anneal beta to 1.0
    per_epsilon: float = 1e-6               # Small constant to avoid zero priority

    # --- Training Duration ---
    num_episodes: int = 500                 # Total training episodes
    save_frequency: int = 50                # Episodes between checkpoint saves


@dataclass
class RewardConfig:
    """Reward function configuration."""
    food_reward: float = 10.0               # Reward for eating food
    collision_penalty: float = -10.0        # Penalty for dying
    step_penalty: float = -0.1             # Small penalty per step (encourages efficiency)
    closer_to_food_reward: float = 0.5      # Reward for moving closer to food
    farther_from_food_penalty: float = -0.5 # Penalty for moving away from food
    loop_penalty: float = -1.0             # Penalty for repetitive looping movement


@dataclass
class RenderConfig:
    """Rendering and visualization configuration."""
    # --- Colors (RGB) ---
    bg_color: Tuple[int, int, int] = (18, 18, 24)           # Dark background
    grid_color: Tuple[int, int, int] = (30, 30, 42)         # Subtle grid lines
    snake_head_color: Tuple[int, int, int] = (0, 200, 120)  # Bright green head
    snake_body_color: Tuple[int, int, int] = (0, 160, 90)   # Slightly darker body
    snake_body_alt: Tuple[int, int, int] = (0, 140, 80)     # Alternating body color
    food_color: Tuple[int, int, int] = (255, 60, 80)        # Bright red food
    food_glow_color: Tuple[int, int, int] = (255, 100, 100) # Food glow effect
    obstacle_color: Tuple[int, int, int] = (100, 60, 180)   # Obstacle purple
    obstacle_border_color: Tuple[int, int, int] = (130, 80, 210) # Obstacle border
    hud_bg_color: Tuple[int, int, int] = (12, 12, 18)       # HUD background
    hud_text_color: Tuple[int, int, int] = (200, 200, 220)  # HUD text
    hud_accent_color: Tuple[int, int, int] = (0, 200, 120)  # HUD accent
    hud_warning_color: Tuple[int, int, int] = (255, 200, 50)# Warning color
    game_over_color: Tuple[int, int, int] = (255, 60, 80)   # Game over text
    border_color: Tuple[int, int, int] = (50, 50, 70)       # Border color
    eye_color: Tuple[int, int, int] = (255, 255, 255)       # Snake eye color
    pupil_color: Tuple[int, int, int] = (0, 0, 0)           # Snake pupil color

    # --- FPS Settings ---
    training_fps: int = 200                 # Fast training speed
    demo_fps: int = 15                      # Slow demo/watch speed
    min_fps: int = 5                        # Minimum FPS
    max_fps: int = 500                      # Maximum FPS

    # --- Game Over Display ---
    game_over_display_ms: int = 800         # How long to show game over screen

    # --- Font Sizes ---
    font_size_small: int = 16
    font_size_medium: int = 22
    font_size_large: int = 36
    font_size_title: int = 48


@dataclass
class PathConfig:
    """File paths for saving/loading."""
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir: str = ""
    plots_dir: str = ""
    assets_dir: str = ""
    report_dir: str = ""

    def __post_init__(self):
        self.checkpoints_dir = os.path.join(self.base_dir, "checkpoints")
        self.plots_dir = os.path.join(self.base_dir, "plots")
        self.assets_dir = os.path.join(self.base_dir, "assets")
        self.report_dir = os.path.join(self.base_dir, "report")

        # Create directories if they don't exist
        for d in [self.checkpoints_dir, self.plots_dir, self.assets_dir, self.report_dir]:
            os.makedirs(d, exist_ok=True)

    @property
    def best_model_path(self) -> str:
        return os.path.join(self.checkpoints_dir, "best_model.pth")

    def checkpoint_path(self, episode: int) -> str:
        return os.path.join(self.checkpoints_dir, f"checkpoint_ep{episode}.pth")

    def plot_path(self, name: str) -> str:
        return os.path.join(self.plots_dir, f"{name}.png")


# ============================================================
# Global Configuration Instances
# ============================================================

GAME_CONFIG = GameConfig()
DQN_CONFIG = DQNConfig()
REWARD_CONFIG = RewardConfig()
RENDER_CONFIG = RenderConfig()
PATH_CONFIG = PathConfig()
