"""
utils.py - Utility Functions and Metrics Tracking

Provides:
- MetricsTracker: records and plots training metrics
- plot_training_results: generates publication-quality plots
- Random agent for comparison baseline
"""

import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from collections import deque

from config import PATH_CONFIG


class MetricsTracker:
    """
    Tracks training metrics across episodes:
    - Episode scores, rewards, steps, losses
    - Running averages
    - Best score tracking
    """

    def __init__(self, avg_window: int = 50):
        self.scores: List[int] = []
        self.avg_scores: List[float] = []
        self.epsilons: List[float] = []
        self.losses: List[float] = []
        self.total_rewards: List[float] = []
        self.survival_times: List[int] = []
        self.best_score: int = 0
        self.avg_window = avg_window
        self._recent_scores: deque = deque(maxlen=avg_window)

    def record_episode(self, score: int, epsilon: float, loss: float,
                       total_reward: float, steps: int):
        """Record metrics for one completed episode."""
        self.scores.append(score)
        self.epsilons.append(epsilon)
        self.losses.append(loss)
        self.total_rewards.append(total_reward)
        self.survival_times.append(steps)

        self._recent_scores.append(score)
        avg = np.mean(self._recent_scores)
        self.avg_scores.append(avg)

        if score > self.best_score:
            self.best_score = score

    @property
    def current_avg(self) -> float:
        if not self._recent_scores:
            return 0.0
        return np.mean(self._recent_scores)

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "episodes": len(self.scores),
            "best_score": self.best_score,
            "avg_score": self.current_avg,
            "latest_score": self.scores[-1] if self.scores else 0,
        }


def plot_training_results(metrics: MetricsTracker, save_dir: str = None):
    """Generate and save training result plots."""
    save_dir = save_dir or PATH_CONFIG.plots_dir
    os.makedirs(save_dir, exist_ok=True)

    episodes = range(1, len(metrics.scores) + 1)

    # Set style
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DQN Snake AI — Training Results", fontsize=16, fontweight="bold",
                 color="#00c878")

    # 1. Scores
    ax = axes[0][0]
    ax.plot(episodes, metrics.scores, alpha=0.3, color="#00c878", linewidth=0.8, label="Score")
    ax.plot(episodes, metrics.avg_scores, color="#00e88a", linewidth=2, label=f"Avg (last {metrics.avg_window})")
    ax.set_title("Episode Scores", color="white")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(alpha=0.2)

    # 2. Epsilon
    ax = axes[0][1]
    ax.plot(episodes, metrics.epsilons, color="#ffc832", linewidth=2)
    ax.set_title("Epsilon Decay", color="white")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.grid(alpha=0.2)

    # 3. Training Loss
    ax = axes[1][0]
    valid_losses = [l for l in metrics.losses if l > 0]
    if valid_losses:
        ax.plot(range(1, len(valid_losses) + 1), valid_losses, color="#ff3c50", alpha=0.5, linewidth=0.8)
        # Smoothed loss
        if len(valid_losses) > 10:
            window = min(50, len(valid_losses) // 5)
            smoothed = np.convolve(valid_losses, np.ones(window) / window, mode="valid")
            ax.plot(range(window, len(valid_losses) + 1), smoothed, color="#ff6070", linewidth=2, label="Smoothed")
            ax.legend()
    ax.set_title("Training Loss", color="white")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.2)

    # 4. Survival Time
    ax = axes[1][1]
    ax.plot(episodes, metrics.survival_times, alpha=0.3, color="#50a0ff", linewidth=0.8)
    if len(metrics.survival_times) > 10:
        window = min(50, len(metrics.survival_times) // 5)
        smoothed = np.convolve(metrics.survival_times, np.ones(window) / window, mode="valid")
        ax.plot(range(window, len(metrics.survival_times) + 1), smoothed, color="#70b0ff", linewidth=2)
    ax.set_title("Survival Time (Steps)", color="white")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    filepath = os.path.join(save_dir, "training_results.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] Training results saved to {filepath}")


def plot_comparison(dqn_scores: List[int], random_scores: List[int],
                    save_dir: str = None):
    """Plot comparison between DQN agent and random agent."""
    save_dir = save_dir or PATH_CONFIG.plots_dir
    os.makedirs(save_dir, exist_ok=True)

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DQN Agent vs Random Agent", fontsize=16, fontweight="bold", color="#00c878")

    # Box plot
    ax1.boxplot([dqn_scores, random_scores], labels=["DQN Agent", "Random Agent"],
                patch_artist=True, boxprops=dict(facecolor="#00c87850", color="#00c878"),
                medianprops=dict(color="#ffc832"), whiskerprops=dict(color="white"),
                capprops=dict(color="white"), flierprops=dict(markeredgecolor="white"))
    ax1.set_title("Score Distribution", color="white")
    ax1.set_ylabel("Score")
    ax1.grid(alpha=0.2)

    # Bar chart of averages
    means = [np.mean(dqn_scores), np.mean(random_scores)]
    stds = [np.std(dqn_scores), np.std(random_scores)]
    bars = ax2.bar(["DQN Agent", "Random Agent"], means, yerr=stds,
                   color=["#00c878", "#ff3c50"], alpha=0.8, capsize=5)
    ax2.set_title("Average Score Comparison", color="white")
    ax2.set_ylabel("Average Score")
    ax2.grid(alpha=0.2)
    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{mean:.1f}", ha="center", va="bottom", color="white", fontweight="bold")

    plt.tight_layout()
    filepath = os.path.join(save_dir, "agent_comparison.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] Comparison saved to {filepath}")


class RandomAgent:
    """Baseline random agent for comparison."""

    def __init__(self, action_size: int = 3):
        self.action_size = action_size

    def select_action(self, state=None, training=False) -> int:
        return random.randint(0, self.action_size - 1)
