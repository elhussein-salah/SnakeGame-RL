"""
evaluate.py - Evaluation and Demo Mode for DQN Snake AI

Provides:
1. Demo/Watch mode: Watch the trained AI play at slow FPS
2. DQN vs Random Agent comparison with statistical analysis
3. Benchmark evaluation over many episodes

Usage:
    python evaluate.py                          # Watch AI play (demo mode)
    python evaluate.py --model path/to/model    # Specify model path
    python evaluate.py --compare --episodes 100 # Compare DQN vs Random
    python evaluate.py --benchmark --episodes 200  # Benchmark evaluation
"""

import argparse
import sys
import numpy as np

from config import GAME_CONFIG, DQN_CONFIG, RENDER_CONFIG, PATH_CONFIG
from snake_game import SnakeGame
from agent import DQNAgent
from renderer import Renderer
from utils import RandomAgent, plot_comparison, MetricsTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DQN Snake AI Agent")
    parser.add_argument("--model", type=str, default=PATH_CONFIG.best_model_path,
                        help="Path to trained model checkpoint")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--fps", type=int, default=RENDER_CONFIG.demo_fps,
                        help="Demo mode FPS")
    parser.add_argument("--compare", action="store_true",
                        help="Run DQN vs Random agent comparison")
    parser.add_argument("--visual-compare", action="store_true",
                        help="Watch DQN vs Random comparison in real-time")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark (no rendering)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering")
    parser.add_argument("--grid-size", type=int, default=GAME_CONFIG.grid_size,
                        help="Grid size (e.g. 10, 15, 20)")
    parser.add_argument("--obstacles", type=int, default=GAME_CONFIG.num_obstacles,
                        help="Number of obstacles inside the grid")
    return parser.parse_args()


def run_demo(args):
    """Watch the trained AI play Snake in real-time."""
    print("=" * 60)
    print("  DQN Snake AI — Demo Mode")
    print("=" * 60)

    game = SnakeGame(grid_size=args.grid_size, num_obstacles=args.obstacles)
    agent = DQNAgent()

    # Load trained model
    try:
        agent.load_for_evaluation(args.model)
        print(f"[Model] Loaded: {args.model}")
    except FileNotFoundError:
        print(f"[Error] Model not found: {args.model}")
        print("[Info] Train first with: python train.py")
        return

    GAME_CONFIG.grid_size = args.grid_size
    renderer = Renderer()
    renderer.set_fps(args.fps)
    print(f"[Demo] FPS: {args.fps}")
    print(f"[Demo] Press ESC to quit, SPACE to pause, UP/DOWN for FPS")
    print("-" * 60)

    episode = 0
    high_score = 0
    running = True

    while running:
        episode += 1
        state = game.reset()
        done = False

        while not done:
            event_result = renderer.handle_events()
            if event_result == "quit":
                running = False
                break

            while renderer.paused:
                event_result = renderer.handle_events()
                if event_result == "quit":
                    running = False
                    break
                if event_result == "resume":
                    break
                stats = {"episode": episode, "high_score": high_score,
                         "epsilon": 0.0, "training_steps": 0, "avg_score": None}
                renderer.render_frame(game, stats)

            if not running:
                break

            # Agent selects greedy action (no exploration)
            action = agent.select_action(state, training=False)
            state, reward, done, info = game.step(action)

            stats = {"episode": episode, "high_score": high_score,
                     "epsilon": 0.0, "training_steps": 0, "avg_score": None}
            renderer.increase_speed_for_score(game.score)
            renderer.render_frame(game, stats)

        if running:
            high_score = max(high_score, game.score)
            print(f"Episode {episode:3d} | Score: {game.score:3d} | "
                  f"Steps: {game.steps:4d} | High Score: {high_score}")
            renderer.show_game_over(game.score)

    renderer.close()


def run_comparison(args):
    """Compare DQN agent vs Random agent."""
    print("=" * 60)
    print("  DQN vs Random Agent Comparison")
    print("=" * 60)

    game = SnakeGame(grid_size=args.grid_size, num_obstacles=args.obstacles)

    # DQN Agent
    dqn_agent = DQNAgent()
    try:
        dqn_agent.load_for_evaluation(args.model)
        print(f"[DQN] Model loaded: {args.model}")
    except FileNotFoundError:
        print(f"[Error] Model not found: {args.model}")
        return

    # Random Agent
    random_agent = RandomAgent()

    # Evaluate DQN
    print(f"\n[DQN Agent] Running {args.episodes} episodes...")
    dqn_scores = []
    dqn_steps = []
    for ep in range(args.episodes):
        state = game.reset()
        done = False
        while not done:
            action = dqn_agent.select_action(state, training=False)
            state, _, done, _ = game.step(action)
        dqn_scores.append(game.score)
        dqn_steps.append(game.steps)

    # Evaluate Random
    print(f"[Random Agent] Running {args.episodes} episodes...")
    random_scores = []
    random_steps = []
    for ep in range(args.episodes):
        state = game.reset()
        done = False
        while not done:
            action = random_agent.select_action()
            state, _, done, _ = game.step(action)
        random_scores.append(game.score)
        random_steps.append(game.steps)

    # Print results
    print("\n" + "=" * 60)
    print(f"{'Metric':<25} {'DQN Agent':>15} {'Random Agent':>15}")
    print("-" * 60)
    print(f"{'Average Score':<25} {np.mean(dqn_scores):>15.2f} {np.mean(random_scores):>15.2f}")
    print(f"{'Max Score':<25} {max(dqn_scores):>15d} {max(random_scores):>15d}")
    print(f"{'Min Score':<25} {min(dqn_scores):>15d} {min(random_scores):>15d}")
    print(f"{'Std Dev':<25} {np.std(dqn_scores):>15.2f} {np.std(random_scores):>15.2f}")
    print(f"{'Avg Survival (steps)':<25} {np.mean(dqn_steps):>15.1f} {np.mean(random_steps):>15.1f}")
    print("=" * 60)

    # Generate comparison plot
    plot_comparison(dqn_scores, random_scores)


def run_benchmark(args):
    """Benchmark DQN agent without rendering."""
    print("=" * 60)
    print("  DQN Snake AI — Benchmark")
    print("=" * 60)

    game = SnakeGame(grid_size=args.grid_size, num_obstacles=args.obstacles)
    agent = DQNAgent()

    try:
        agent.load_for_evaluation(args.model)
        print(f"[Model] Loaded: {args.model}")
    except FileNotFoundError:
        print(f"[Error] Model not found: {args.model}")
        return

    scores = []
    steps_list = []

    for ep in range(args.episodes):
        state = game.reset()
        done = False
        while not done:
            action = agent.select_action(state, training=False)
            state, _, done, _ = game.step(action)
        scores.append(game.score)
        steps_list.append(game.steps)
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{args.episodes} — Avg: {np.mean(scores):.1f}")

    print(f"\n{'Benchmark Results':=^60}")
    print(f"  Episodes:       {args.episodes}")
    print(f"  Average Score:  {np.mean(scores):.2f}")
    print(f"  Max Score:      {max(scores)}")
    print(f"  Std Dev:        {np.std(scores):.2f}")
    print(f"  Avg Steps:      {np.mean(steps_list):.1f}")
    print("=" * 60)


def run_visual_comparison(args):
    """Watch DQN and Random agents play side-by-side (sequentially)."""
    print("=" * 60)
    print("  DQN vs Random — Visual Comparison")
    print("=" * 60)

    game = SnakeGame(grid_size=args.grid_size, num_obstacles=args.obstacles)
    dqn_agent = DQNAgent()
    random_agent = RandomAgent()

    try:
        dqn_agent.load_for_evaluation(args.model)
        print(f"[DQN] Model loaded: {args.model}")
    except FileNotFoundError:
        print(f"[Error] Model not found: {args.model}")
        return

    GAME_CONFIG.grid_size = args.grid_size
    renderer = Renderer()
    renderer.set_fps(args.fps)
    
    dqn_scores = []
    random_scores = []
    
    running = True
    for ep in range(args.episodes):
        if not running: break
        
        # Determine which agent plays this episode
        is_dqn = (ep % 2 == 0)
        current_agent = dqn_agent if is_dqn else random_agent
        agent_name = "DQN AGENT" if is_dqn else "RANDOM AGENT"
        accent_color = RENDER_CONFIG.hud_accent_color if is_dqn else RENDER_CONFIG.hud_warning_color
        
        print(f"\n[Episode {ep+1}/{args.episodes}] {agent_name} is playing...")
        
        state = game.reset()
        done = False
        
        while not done:
            event_result = renderer.handle_events()
            if event_result == "quit":
                running = False
                break
                
            # Agent action
            if is_dqn:
                action = dqn_agent.select_action(state, training=False)
            else:
                action = random_agent.select_action()
                
            state, _, done, _ = game.step(action)
            
            # Custom stats for comparison HUD
            stats = {
                "agent_name": agent_name,
                "episode": f"{ep+1}/{args.episodes}",
                "high_score": f"DQN:{max(dqn_scores) if dqn_scores else 0} Rnd:{max(random_scores) if random_scores else 0}",
                "epsilon": 0.0,
                "training_steps": "COMPARING...",
                "avg_score": np.mean(dqn_scores) if is_dqn and dqn_scores else (np.mean(random_scores) if random_scores else 0)
            }
            
            renderer.render_frame(game, stats)
            
        if not running: break
        
        if is_dqn: dqn_scores.append(game.score)
        else: random_scores.append(game.score)
        
        renderer.show_game_over(game.score)

    renderer.close()
    print("\nVisual Comparison Finished.")
    if dqn_scores and random_scores:
        print(f"DQN Avg: {np.mean(dqn_scores):.1f} | Random Avg: {np.mean(random_scores):.1f}")


if __name__ == "__main__":
    args = parse_args()

    if args.compare:
        run_comparison(args)
    elif args.visual_compare:
        run_visual_comparison(args)
    elif args.benchmark:
        run_benchmark(args)
    else:
        run_demo(args)
