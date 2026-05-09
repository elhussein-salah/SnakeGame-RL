"""
train.py - DQN Training Loop with Real-Time Pygame Visualization

Main training script that:
1. Creates the Snake game environment
2. Initializes the DQN agent
3. Runs the training loop with live Pygame rendering
4. Tracks metrics and saves checkpoints
5. Generates training plots on completion

Usage:
    python train.py                    # Train with visualization
    python train.py --episodes 1000    # Train for 1000 episodes
    python train.py --no-render        # Train without visualization (faster)

Controls during training:
    ESC     - Quit training
    SPACE   - Pause/Resume
    UP/DOWN - Adjust FPS
"""

import argparse
import sys
import time
import numpy as np

from config import GAME_CONFIG, DQN_CONFIG, RENDER_CONFIG, PATH_CONFIG
from snake_game import SnakeGame
from agent import DQNAgent
from renderer import Renderer
from utils import MetricsTracker, plot_training_results


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN Snake AI Agent")
    parser.add_argument("--episodes", type=int, default=DQN_CONFIG.num_episodes,
                        help="Number of training episodes")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable Pygame visualization for faster training")
    parser.add_argument("--fps", type=int, default=RENDER_CONFIG.training_fps,
                        help="Initial training FPS")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--grid-size", type=int, default=GAME_CONFIG.grid_size,
                        help="Grid size (e.g. 10, 15, 20)")
    parser.add_argument("--obstacles", type=int, default=GAME_CONFIG.num_obstacles,
                        help="Number of obstacles inside the grid")
    return parser.parse_args()


def train(args):
    """Main training loop."""
    print("=" * 60)
    print("  DQN Snake AI — Training")
    print("=" * 60)

    # Initialize components
    game = SnakeGame(grid_size=args.grid_size, num_obstacles=args.obstacles)
    agent = DQNAgent()
    metrics = MetricsTracker()

    # Initialize renderer (unless --no-render)
    renderer = None
    if not args.no_render:
        # Update GAME_CONFIG so renderer picks up correct grid size
        GAME_CONFIG.grid_size = args.grid_size
        renderer = Renderer()
        renderer.set_fps(args.fps)

    # Resume from checkpoint if specified
    start_episode = 0
    if args.resume:
        print(f"[Resume] Loading checkpoint: {args.resume}")
        checkpoint = agent.load_checkpoint(args.resume)
        start_episode = checkpoint.get("episode", 0)
        print(f"[Resume] Resuming from episode {start_episode}, epsilon={agent.epsilon:.3f}")

    print(f"[Config] Episodes: {args.episodes}")
    print(f"[Config] Device: {agent.device}")
    print(f"[Config] Grid: {game.grid_size}x{game.grid_size}")
    print(f"[Config] Obstacles: {game.num_obstacles}")
    print(f"[Config] Render: {'ON' if renderer else 'OFF'}")
    print("-" * 60)

    best_score = 0
    training_active = True

    try:
        for episode in range(start_episode, args.episodes):
            if not training_active:
                break

            # Reset environment for new episode
            state = game.reset()
            episode_loss = 0.0
            loss_count = 0
            done = False

            while not done:
                # Handle Pygame events
                if renderer:
                    event_result = renderer.handle_events()
                    if event_result == "quit":
                        training_active = False
                        break

                    # Handle pause
                    while renderer.paused:
                        event_result = renderer.handle_events()
                        if event_result == "quit":
                            training_active = False
                            break
                        if event_result == "resume":
                            break
                        # Render paused frame
                        stats = {**agent.get_stats(),
                                 "episode": episode + 1,
                                 "high_score": best_score,
                                 "avg_score": metrics.current_avg}
                        renderer.render_frame(game, stats)

                    if not training_active:
                        break

                # Agent selects action using epsilon-greedy
                action = agent.select_action(state, training=True)

                # Environment step
                next_state, reward, done, info = game.step(action)

                # Store experience in replay buffer
                agent.store_experience(state, action, reward, next_state, done)

                # Train on minibatch from replay buffer
                loss = agent.train_step()
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1

                state = next_state

                # Render frame
                if renderer:
                    # Increase speed as snake grows
                    renderer.increase_speed_for_score(game.score)
                    stats = {
                        **agent.get_stats(),
                        "episode": episode + 1,
                        "high_score": best_score,
                        "avg_score": metrics.current_avg,
                    }
                    renderer.render_frame(game, stats)

            if not training_active:
                break

            # Episode complete
            score = game.score
            avg_loss = episode_loss / max(loss_count, 1)

            # Decay exploration rate
            agent.decay_epsilon()

            # Record metrics
            metrics.record_episode(score, agent.epsilon, avg_loss,
                                   game.total_reward, game.steps)

            # Update best score and save best model
            if score > best_score:
                best_score = score
                agent.save_checkpoint(PATH_CONFIG.best_model_path, episode + 1,
                                      extra={"best_score": best_score})

            # Show game over screen
            if renderer and done:
                renderer.show_game_over(score)

            # Periodic checkpoint
            if (episode + 1) % DQN_CONFIG.save_frequency == 0:
                path = PATH_CONFIG.checkpoint_path(episode + 1)
                agent.save_checkpoint(path, episode + 1,
                                      extra={"best_score": best_score})

            # Console logging
            if (episode + 1) % 10 == 0 or episode == 0:
                print(f"Ep {episode+1:4d} | Score: {score:3d} | "
                      f"Avg: {metrics.current_avg:6.1f} | Best: {best_score:3d} | "
                      f"Eps: {agent.epsilon:.3f} | Loss: {avg_loss:.4f} | "
                      f"Steps: {game.steps:4d} | Buffer: {len(agent.replay_buffer)}")

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")

    # Save final checkpoint
    final_path = PATH_CONFIG.checkpoint_path(episode + 1 if 'episode' in dir() else 0)
    agent.save_checkpoint(final_path, episode + 1 if 'episode' in dir() else 0,
                          extra={"best_score": best_score})

    # Generate training plots
    print("\n[Plot] Generating training plots...")
    plot_training_results(metrics)

    # Print summary
    summary = metrics.get_summary()
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Episodes trained: {summary['episodes']}")
    print(f"  Best Score:       {summary['best_score']}")
    print(f"  Final Avg Score:  {summary['avg_score']:.1f}")
    print(f"  Best model saved: {PATH_CONFIG.best_model_path}")
    print("=" * 60)

    if renderer:
        renderer.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
