"""
main.py - Playful Main Menu for Snake Game AI

A fancy Pygame dashboard to launch training and evaluation commands.
Features:
- Animated snake decoration
- Interactive buttons with hover effects
- Direct command execution via subprocess
"""

import pygame
import sys
import subprocess
import math
import random
from typing import List, Tuple, Callable

# Reuse colors from the project config if possible, or define them here for autonomy
COLORS = {
    "bg": (18, 18, 24),
    "accent": (0, 200, 120),    # Neon Green
    "accent_dim": (0, 150, 90),
    "danger": (255, 60, 80),    # Red
    "text": (220, 220, 240),
    "text_dim": (130, 130, 160),
    "border": (50, 50, 70),
}

class Button:
    """Interactive button with hover animation."""
    def __init__(self, x: int, y: int, width: int, height: int, text: str, 
                 callback: Callable, color: Tuple[int, int, int] = COLORS["accent"]):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.callback = callback
        self.color = color
        self.hovered = False
        self.animation_offset = 0.0
        
    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        # Hover animation: grow slightly
        target_offset = 5 if self.hovered else 0
        self.animation_offset += (target_offset - self.animation_offset) * 0.2
        
        draw_rect = self.rect.inflate(self.animation_offset * 2, self.animation_offset * 2)
        
        # Shadow/Glow
        if self.hovered:
            glow_rect = draw_rect.inflate(10, 10)
            glow_surf = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*self.color, 40), glow_surf.get_rect(), border_radius=12)
            screen.blit(glow_surf, glow_rect.topleft)
            
        # Button body
        color = self.color if self.hovered else COLORS["border"]
        pygame.draw.rect(screen, color, draw_rect, border_radius=10, width=2)
        
        # Text
        text_color = COLORS["text"] if self.hovered else COLORS["text_dim"]
        text_surf = font.render(self.text, True, text_color)
        text_rect = text_surf.get_rect(center=draw_rect.center)
        screen.blit(text_surf, text_rect)

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.hovered:
                self.callback()

class InputField:
    """A field for numerical/text input with label."""
    def __init__(self, x: int, y: int, width: int, height: int, label: str, initial_value: str):
        self.rect = pygame.Rect(x, y, width, height)
        self.label = label
        self.value = initial_value
        self.active = False
        self.cursor_timer = 0
        
    def draw(self, screen: pygame.Surface, font_label: pygame.font.Font, font_input: pygame.font.Font):
        # Draw Label
        label_surf = font_label.render(self.label, True, COLORS["text_dim"])
        screen.blit(label_surf, (self.rect.x, self.rect.y - 25))
        
        # Draw Box
        color = COLORS["accent"] if self.active else COLORS["border"]
        pygame.draw.rect(screen, color, self.rect, border_radius=6, width=2)
        if self.active:
            glow_rect = self.rect.inflate(4, 4)
            glow_surf = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*COLORS["accent"], 30), glow_surf.get_rect(), border_radius=8)
            screen.blit(glow_surf, glow_rect.topleft)

        # Draw Text
        text_surf = font_input.render(self.value, True, COLORS["text"])
        screen.blit(text_surf, (self.rect.x + 10, self.rect.y + (self.rect.height // 2 - text_surf.get_height() // 2)))
        
        # Cursor
        if self.active:
            self.cursor_timer += 1
            if (self.cursor_timer // 30) % 2 == 0:
                cursor_x = self.rect.x + 12 + text_surf.get_width()
                pygame.draw.line(screen, COLORS["accent"], (cursor_x, self.rect.y + 10), 
                                 (cursor_x, self.rect.y + self.rect.height - 10), 2)

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
            else:
                self.active = False
        
        if self.active and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.value = self.value[:-1]
            elif event.unicode.isdigit() or (not self.label.endswith(":") and event.unicode.isalnum()):
                if len(self.value) < 8: # Limit input length
                    self.value += event.unicode

class DecorativeSnake:
    """An animated snake that slithers in the background."""
    def __init__(self, screen_width: int, screen_height: int):
        self.width = screen_width
        self.height = screen_height
        self.points = [(100, 100)] * 20
        self.angle = 0.0
        
    def update(self):
        self.angle += 0.05
        head_x = (math.sin(self.angle * 0.7) * 0.4 + 0.5) * self.width
        head_y = (math.cos(self.angle * 1.1) * 0.3 + 0.5) * self.height
        
        self.points.insert(0, (head_x, head_y))
        if len(self.points) > 40:
            self.points.pop()
            
    def draw(self, screen: pygame.Surface):
        for i, (x, y) in enumerate(reversed(self.points)):
            # Tapering size and fading color
            size = int(10 + (len(self.points) - i) * 0.5)
            alpha = int(100 * (i / len(self.points)))
            color = (*COLORS["accent"], alpha)
            
            surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (size, size), size)
            screen.blit(surf, (x - size, y - size))

def run_cmd(cmd: List[str]):
    """Execute a shell command in a new process."""
    print(f"Executing: {' '.join(cmd)}")
    subprocess.Popen([sys.executable] + cmd)

def main():
    pygame.init()
    
    # Screen setup
    WIDTH, HEIGHT = 900, 750
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("🐍 Snake AI Dashboard")
    clock = pygame.time.Clock()
    
    # Fonts
    try:
        font_title = pygame.font.SysFont("consolas", 64, bold=True)
        font_button = pygame.font.SysFont("consolas", 20, bold=True)
        font_footer = pygame.font.SysFont("consolas", 16)
    except:
        font_title = pygame.font.Font(None, 64)
        font_button = pygame.font.Font(None, 24)
        font_footer = pygame.font.Font(None, 18)
        
    # Decorative elements
    snake = DecorativeSnake(WIDTH, HEIGHT)
    
    # Inputs
    input_w, input_h = 100, 45
    inputs_y = 190
    input_spacing = 160
    input_start_x = WIDTH // 2 - (input_spacing * 3) // 2 + 30
    
    field_grid = InputField(input_start_x, inputs_y, input_w, input_h, "GRID SIZE", "10")
    field_obs = InputField(input_start_x + input_spacing, inputs_y, input_w, input_h, "OBSTACLES", "0")
    field_eps = InputField(input_start_x + input_spacing * 2, inputs_y, input_w, input_h, "EPISODES", "500")
    
    all_inputs = [field_grid, field_obs, field_eps]

    def get_args():
        return [
            "--grid-size", field_grid.value if field_grid.value else "10",
            "--obstacles", field_obs.value if field_obs.value else "0",
            "--episodes", field_eps.value if field_eps.value else "500"
        ]

    # Buttons
    btn_w, btn_h = 420, 52
    center_x = WIDTH // 2 - btn_w // 2
    start_y = 265
    spacing = 65
    
    buttons = [
        Button(center_x, start_y, btn_w, btn_h, "1. Train AI (Visual)", 
               lambda: run_cmd(["train.py"] + get_args())),
        Button(center_x, start_y + spacing, btn_w, btn_h, "2. Train AI (Fast / No-Render)", 
               lambda: run_cmd(["train.py", "--no-render"] + get_args())),
        Button(center_x, start_y + spacing*2, btn_w, btn_h, "3. Watch Trained Agent", 
               lambda: run_cmd(["evaluate.py"] + ["--grid-size", field_grid.value, "--obstacles", field_obs.value])),
        Button(center_x, start_y + spacing*3, btn_w, btn_h, "4. Visual Benchmark (DQN vs Random)", 
               lambda: run_cmd(["evaluate.py", "--visual-compare"] + ["--episodes", field_eps.value, "--grid-size", field_grid.value, "--obstacles", field_obs.value])),
        Button(center_x, start_y + spacing*4, btn_w, btn_h, "5. Benchmark AI vs Random (Stats)", 
               lambda: run_cmd(["evaluate.py", "--compare"] + ["--episodes", field_eps.value])),
        Button(center_x, start_y + spacing*5, btn_w, btn_h, "6. Run Benchmark (Headless)", 
               lambda: run_cmd(["evaluate.py", "--benchmark"] + ["--episodes", field_eps.value])),
    ]
    
    running = True
    while running:
        screen.fill(COLORS["bg"])
        
        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            for btn in buttons:
                btn.handle_event(event)
            for inp in all_inputs:
                inp.handle_event(event)
                
        # Update
        snake.update()
        
        # Draw Background Decor
        snake.draw(screen)
        
        # Draw Title
        title_surf = font_title.render("SNAKE AI", True, COLORS["accent"])
        title_rect = title_surf.get_rect(center=(WIDTH // 2, 80))
        # Title Glow
        glow = font_title.render("SNAKE AI", True, (*COLORS["accent"], 50))
        for offset in [(2,2), (-2,-2), (2,-2), (-2,2)]:
            screen.blit(glow, (title_rect.x + offset[0], title_rect.y + offset[1]))
        screen.blit(title_surf, title_rect)
        
        subtitle_surf = font_button.render("DQN REINFORCEMENT LEARNING DASHBOARD", True, COLORS["text_dim"])
        screen.blit(subtitle_surf, subtitle_surf.get_rect(center=(WIDTH // 2, 130)))
        
        # Draw Inputs
        for inp in all_inputs:
            inp.draw(screen, font_footer, font_button)
            
        # Draw Buttons
        for btn in buttons:
            btn.draw(screen, font_button)
            
        # Footer
        footer_text = "Select a mode to launch. Parameters above will be passed to scripts."
        footer_surf = font_footer.render(footer_text, True, COLORS["text_dim"])
        screen.blit(footer_surf, footer_surf.get_rect(center=(WIDTH // 2, HEIGHT - 40)))
        
        pygame.display.flip()
        clock.tick(60)
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
