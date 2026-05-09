"""
renderer.py - Pygame Visualization for Snake Game AI

Handles all rendering logic separately from game and RL logic:
- Game board with grid lines
- Snake (head with eyes, body segments)
- Food with glow effect
- HUD panel with live stats
- Game over screen
- Keyboard controls (ESC, SPACE, UP/DOWN arrows)
"""

import pygame
import sys
from typing import Optional, Dict, Any, List, Tuple

from config import GAME_CONFIG, RENDER_CONFIG
from snake_game import SnakeGame, Direction, DIRECTION_VECTORS


class Renderer:
    """
    Pygame renderer for the Snake Game AI visualization.
    Separates all rendering from game logic and RL logic.
    """

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("🐍 Snake Game AI — Deep Q-Network")

        self.game_cfg = GAME_CONFIG
        self.render_cfg = RENDER_CONFIG
        
        # --- Linear Scaling Logic ---
        # Fixed block size means window grows physically larger as grid increases
        self.block_size = 50
        
        # Update window dimensions based on block size
        self.game_width = self.game_cfg.grid_size * self.block_size
        self.window_height = self.game_cfg.grid_size * self.block_size
        self.window_width = self.game_width + self.game_cfg.hud_width

        # Create display
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        self.current_fps = self.render_cfg.training_fps

        # Fonts
        self.font_small = pygame.font.SysFont("consolas", self.render_cfg.font_size_small)
        self.font_medium = pygame.font.SysFont("consolas", self.render_cfg.font_size_medium, bold=True)
        self.font_large = pygame.font.SysFont("consolas", self.render_cfg.font_size_large, bold=True)
        self.font_title = pygame.font.SysFont("consolas", self.render_cfg.font_size_title, bold=True)

        # State
        self.paused = False
        self.frame_count = 0
        self.base_fps = self.render_cfg.training_fps  # Stored for speed-increase calc

    def handle_events(self) -> str:
        """
        Handle keyboard events.
        Returns: 'quit', 'pause', 'resume', or 'continue'
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    return "pause" if self.paused else "resume"
                elif event.key == pygame.K_UP:
                    self.base_fps = min(self.base_fps + 10, self.render_cfg.max_fps)
                    self.current_fps = self.base_fps
                elif event.key == pygame.K_DOWN:
                    self.base_fps = max(self.base_fps - 10, self.render_cfg.min_fps)
                    self.current_fps = self.base_fps
        return "continue"

    def render_frame(self, game: SnakeGame, stats: Dict[str, Any]):
        """Render one complete frame: background, grid, obstacles, food, snake, HUD."""
        self.frame_count += 1
        self.screen.fill(self.render_cfg.bg_color)

        self._draw_grid()
        self._draw_obstacles(game.obstacles)
        self._draw_food(game.food)
        self._draw_snake(game.snake, game.direction)
        self._draw_hud(game, stats)
        self._draw_border()

        pygame.display.flip()
        self.clock.tick(self.current_fps)

    def _draw_grid(self):
        """Draw subtle grid lines on the game area."""
        for x in range(0, self.game_width + 1, self.block_size):
            pygame.draw.line(self.screen, self.render_cfg.grid_color, (x, 0), (x, self.window_height))
        for y in range(0, self.window_height + 1, self.block_size):
            pygame.draw.line(self.screen, self.render_cfg.grid_color, (0, y), (self.game_width, y))

    def _draw_food(self, food_pos: Tuple[int, int]):
        """Draw food with a pulsing glow effect."""
        x, y = food_pos
        px = x * self.block_size
        py = y * self.block_size
        bs = self.block_size

        # Outer glow (pulsing)
        import math
        pulse = abs(math.sin(self.frame_count * 0.08)) * 0.4 + 0.6
        glow_size = int(bs * 0.15 * pulse)

        glow_rect = pygame.Rect(px - glow_size, py - glow_size, bs + glow_size * 2, bs + glow_size * 2)
        glow_surf = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.render_cfg.food_glow_color, 60), glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, glow_rect.topleft)

        # Food body
        margin = 3
        food_rect = pygame.Rect(px + margin, py + margin, bs - margin * 2, bs - margin * 2)
        pygame.draw.rect(self.screen, self.render_cfg.food_color, food_rect, border_radius=6)

        # Inner highlight
        hl_rect = pygame.Rect(px + bs // 4, py + bs // 4, bs // 4, bs // 4)
        pygame.draw.rect(self.screen, (255, 140, 140), hl_rect, border_radius=3)

    def _draw_snake(self, snake: List[Tuple[int, int]], direction: Direction):
        """Draw snake with distinct head (with eyes) and alternating body."""
        bs = self.block_size

        # Draw body segments (from tail to neck, skip head)
        for i, (x, y) in enumerate(reversed(snake[1:])):
            px, py = x * bs, y * bs
            margin = 2
            color = self.render_cfg.snake_body_color if i % 2 == 0 else self.render_cfg.snake_body_alt
            rect = pygame.Rect(px + margin, py + margin, bs - margin * 2, bs - margin * 2)
            pygame.draw.rect(self.screen, color, rect, border_radius=5)

        # Draw head
        hx, hy = snake[0]
        head_rect = pygame.Rect(hx * bs + 1, hy * bs + 1, bs - 2, bs - 2)
        pygame.draw.rect(self.screen, self.render_cfg.snake_head_color, head_rect, border_radius=7)

        # Draw eyes based on direction
        eye_size = max(2, self.block_size // 7)
        pupil_size = max(1, self.block_size // 12)
        cx, cy = hx * bs + bs // 2, hy * bs + bs // 2
        
        # Proportional offsets
        off = bs // 4
        off_inner = bs // 6
        
        offsets = {
            Direction.RIGHT: [(off_inner, -off), (off_inner, off)],
            Direction.LEFT:  [(-off_inner, -off), (-off_inner, off)],
            Direction.UP:    [(-off, -off_inner), (off, -off_inner)],
            Direction.DOWN:  [(-off, off_inner), (off, off_inner)],
        }
        for ox, oy in offsets.get(direction, [(0, 0), (0, 0)]):
            ex, ey = cx + ox, cy + oy
            pygame.draw.circle(self.screen, self.render_cfg.eye_color, (ex, ey), eye_size)
            pygame.draw.circle(self.screen, self.render_cfg.pupil_color, (ex, ey), pupil_size)

    def _draw_hud(self, game: SnakeGame, stats: Dict[str, Any]):
        """Draw the HUD stats panel on the right side."""
        hud_x = self.game_width
        hud_rect = pygame.Rect(hud_x, 0, self.game_cfg.hud_width, self.window_height)
        pygame.draw.rect(self.screen, self.render_cfg.hud_bg_color, hud_rect)
        pygame.draw.line(self.screen, self.render_cfg.border_color, (hud_x, 0), (hud_x, self.window_height), 2)

        x_pad = hud_x + 18
        y = 18

        # Title
        title = self.font_medium.render("SNAKE AI", True, self.render_cfg.hud_accent_color)
        self.screen.blit(title, (x_pad, y))
        y += 42

        # Separator
        pygame.draw.line(self.screen, self.render_cfg.border_color, (x_pad, y), (hud_x + self.game_cfg.hud_width - 18, y))
        y += 18

        # Agent Name (Optional)
        if stats.get("agent_name"):
            name_surf = self.font_medium.render(stats["agent_name"], True, self.render_cfg.hud_warning_color)
            self.screen.blit(name_surf, (x_pad, y))
            y += 35

        # Stats
        eps_val = stats.get("epsilon", 0)
        eps_str = f"{eps_val:.3f}" if isinstance(eps_val, (int, float)) else str(eps_val)

        stat_items = [
            ("Score", str(game.score), self.render_cfg.hud_accent_color),
            ("High Score", str(stats.get("high_score", 0)), self.render_cfg.hud_warning_color),
            ("Episode", str(stats.get("episode", 0)), self.render_cfg.hud_text_color),
            ("Epsilon", eps_str, self.render_cfg.hud_text_color),
            ("Train Step", str(stats.get("training_steps", 0)), self.render_cfg.hud_text_color),
            ("FPS", str(self.current_fps), self.render_cfg.hud_text_color),
            ("Direction", game.get_direction_name(), self.render_cfg.hud_text_color),
            ("Steps", str(game.steps), self.render_cfg.hud_text_color),
        ]

        if stats.get("last_loss", 0) > 0:
            stat_items.append(("Loss", f"{stats['last_loss']:.4f}", self.render_cfg.hud_text_color))

        if stats.get("avg_score") is not None:
            stat_items.append(("Avg Score", f"{stats['avg_score']:.1f}", self.render_cfg.hud_accent_color))

        for label, value, color in stat_items:
            label_surf = self.font_small.render(f"{label}:", True, (130, 130, 160))
            self.screen.blit(label_surf, (x_pad, y))
            y += 24
            value_surf = self.font_small.render(f"  {value}", True, color)
            self.screen.blit(value_surf, (x_pad, y))
            y += 32

        # Controls hint at bottom
        y = self.window_height - 110
        pygame.draw.line(self.screen, self.render_cfg.border_color, (x_pad, y), (hud_x + self.game_cfg.hud_width - 18, y))
        y += 14
        controls = ["ESC: Quit", "SPACE: Pause", "UP/DN: FPS"]
        for ctrl in controls:
            surf = self.font_small.render(ctrl, True, (100, 100, 130))
            self.screen.blit(surf, (x_pad, y))
            y += 26

        # Paused overlay
        if self.paused:
            pause_surf = self.font_large.render("PAUSED", True, self.render_cfg.hud_warning_color)
            px = self.game_width // 2 - pause_surf.get_width() // 2
            py = self.window_height // 2 - pause_surf.get_height() // 2
            bg = pygame.Surface((pause_surf.get_width() + 40, pause_surf.get_height() + 20), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 180))
            self.screen.blit(bg, (px - 20, py - 10))
            self.screen.blit(pause_surf, (px, py))

    def _draw_border(self):
        """Draw border around game area."""
        pygame.draw.rect(self.screen, self.render_cfg.border_color,
                         (0, 0, self.game_width, self.window_height), 2)

    def show_game_over(self, score: int):
        """Display game over screen briefly."""
        overlay = pygame.Surface((self.game_width, self.window_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.screen.blit(overlay, (0, 0))

        go_text = self.font_large.render("GAME OVER", True, self.render_cfg.game_over_color)
        score_text = self.font_medium.render(f"Score: {score}", True, self.render_cfg.hud_text_color)

        cx = self.game_width // 2
        self.screen.blit(go_text, (cx - go_text.get_width() // 2, self.window_height // 2 - 30))
        self.screen.blit(score_text, (cx - score_text.get_width() // 2, self.window_height // 2 + 20))

        pygame.display.flip()
        pygame.time.wait(self.render_cfg.game_over_display_ms)

    def set_fps(self, fps: int):
        """Set the rendering FPS."""
        self.current_fps = max(self.render_cfg.min_fps, min(fps, self.render_cfg.max_fps))
        self.base_fps = self.current_fps

    def increase_speed_for_score(self, score: int):
        """
        Increase FPS as the snake grows (eats more food).
        Speed = base_fps + score * speed_increase_per_food
        """
        from config import GAME_CONFIG
        extra = score * GAME_CONFIG.speed_increase_per_food
        self.current_fps = min(self.base_fps + extra, self.render_cfg.max_fps)

    def _draw_obstacles(self, obstacles: List[Tuple[int, int]]):
        """Draw obstacles as purple blocks with a border."""
        bs = self.block_size
        for (x, y) in obstacles:
            px, py = x * bs, y * bs
            margin = 2
            rect = pygame.Rect(px + margin, py + margin, bs - margin * 2, bs - margin * 2)
            pygame.draw.rect(self.screen, self.render_cfg.obstacle_color, rect, border_radius=4)
            pygame.draw.rect(self.screen, self.render_cfg.obstacle_border_color, rect, width=2, border_radius=4)

    def close(self):
        """Clean up pygame resources."""
        pygame.quit()
