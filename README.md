# 🐍 Snake Game AI — Deep Q-Network (DQN)

A complete Deep Reinforcement Learning project that trains an AI agent to play the classic Snake game using a Deep Q-Network (DQN) implemented from scratch with PyTorch.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.5+-green.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Reinforcement Learning Background](#reinforcement-learning-background)
- [DQN Algorithm](#dqn-algorithm)
- [Project Architecture](#project-architecture)
- [Setup & Installation](#setup--installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Hyperparameters](#hyperparameters)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)

---

## 🎯 Overview

This project implements a **Deep Q-Network (DQN)** agent that learns to play Snake through reinforcement learning. The agent receives a compact 11-dimensional binary state representation, takes actions relative to its current direction, and learns an optimal policy through experience replay and target network stabilization.

**Key Features:**
- ✅ Complete DQN implementation from scratch (no RL libraries)
- ✅ Real-time Pygame visualization during training
- ✅ Demo/watch mode for trained agents
- ✅ Comprehensive metrics tracking and plotting
- ✅ Model checkpointing and best model saving
- ✅ DQN vs Random agent comparison
- ✅ Adjustable training speed (FPS control)
- ✅ Clean modular architecture

---

## 🧠 Reinforcement Learning Background

Reinforcement Learning (RL) is a paradigm where an **agent** learns to make decisions by interacting with an **environment**. At each time step:

1. The agent observes the current **state** `s`
2. Selects an **action** `a` based on its **policy**
3. Receives a **reward** `r` and transitions to a new **state** `s'`
4. The goal is to maximize the **cumulative discounted reward**

The agent learns from trial and error, discovering which actions lead to higher long-term rewards through the Bellman equation.

---

## 🔬 DQN Algorithm

The **Deep Q-Network** combines Q-learning with deep neural networks:

### Core Components

| Component | Description |
|-----------|-------------|
| **Online Q-Network** | Neural network that predicts Q-values for each action |
| **Target Q-Network** | Stabilized copy of the online network for computing targets |
| **Experience Replay** | Buffer of past transitions, randomly sampled for training |
| **Epsilon-Greedy** | Exploration strategy that decays over time |

### Bellman Update Equation

```
Q(s, a) = r + γ × max_a'[Q_target(s', a')]
```

Where:
- `Q(s, a)` is the estimated value of taking action `a` in state `s`
- `r` is the immediate reward
- `γ` (gamma) is the discount factor for future rewards
- `Q_target` uses the target network for stable targets

### State Representation (11 binary features)

```
[danger_straight, danger_left, danger_right,
 food_left, food_right, food_up, food_down,
 moving_left, moving_right, moving_up, moving_down]
```

### Actions (3 relative actions)

| Action | Description |
|--------|-------------|
| 0 | Move Straight |
| 1 | Turn Left |
| 2 | Turn Right |

---

## 🏗️ Project Architecture

```
Input State (11) → FC(256) → ReLU → FC(128) → ReLU → FC(64) → ReLU → Q-values (3)
```

The architecture separates concerns into distinct modules:
- **Game Logic** (`snake_game.py`) — Environment dynamics
- **RL Logic** (`agent.py`, `model.py`, `replay_buffer.py`) — DQN algorithm
- **Rendering** (`renderer.py`) — Pygame visualization
- **Training/Evaluation** (`train.py`, `evaluate.py`) — Orchestration

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Clone or navigate to the project directory
cd "RL Project"

# Install dependencies
pip install -r requirements.txt
```

---

## 🏋️ Training

### Basic Training (with visualization)
```bash
python train.py
```

### Custom Training
```bash
# Train for 1000 episodes
python train.py --episodes 1000

# Train without rendering (faster)
python train.py --no-render --episodes 1000

# Set custom training FPS
python train.py --fps 300

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_ep500.pth
```

### Keyboard Controls During Training
| Key | Action |
|-----|--------|
| `ESC` | Quit training |
| `SPACE` | Pause/Resume |
| `↑` | Increase FPS |
| `↓` | Decrease FPS |

---

## 📊 Evaluation

### Watch AI Play (Demo Mode)
```bash
python evaluate.py
```

### Compare DQN vs Random Agent
```bash
python evaluate.py --compare --episodes 100
```

### Benchmark (No Rendering)
```bash
python evaluate.py --benchmark --episodes 200
```

### Custom Model Path
```bash
python evaluate.py --model checkpoints/checkpoint_ep500.pth --fps 20
```

---

## 🎛️ Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `gamma` | 0.99 | Discount factor |
| `learning_rate` | 0.001 | Adam optimizer LR |
| `batch_size` | 64 | Minibatch size |
| `replay_buffer_size` | 100,000 | Max replay memory |
| `epsilon_start` | 1.0 | Initial exploration |
| `epsilon_min` | 0.01 | Min exploration |
| `epsilon_decay` | 0.995 | Decay per episode |
| `target_update_freq` | 1,000 | Target sync interval |
| `hidden_layers` | (256, 128, 64) | Network architecture |

### Reward Function

| Event | Reward |
|-------|--------|
| Eat food | +10.0 |
| Collision (game over) | -10.0 |
| Each step | -0.1 |
| Move closer to food | +0.5 |
| Move away from food | -0.5 |
| Looping movement | -1.0 |

---

## 📁 Project Structure

```
RL Project/
│
├── config.py          # All hyperparameters and configuration
├── snake_game.py      # Snake game environment (RL interface)
├── model.py           # Q-Network architecture (PyTorch)
├── replay_buffer.py   # Experience replay buffer
├── agent.py           # DQN agent (training + inference)
├── renderer.py        # Pygame visualization (HUD, snake, food)
├── utils.py           # Metrics tracking, plotting, random agent
├── train.py           # Training loop with live visualization
├── evaluate.py        # Demo mode, comparison, benchmarking
├── requirements.txt   # Python dependencies
├── README.md          # This file
│
├── assets/            # (Optional) game assets
├── checkpoints/       # Saved model checkpoints
├── plots/             # Training result plots
└── report/
    └── report.md      # Detailed technical report
```

---

## 📸 Screenshots

> Training plots and game screenshots are automatically saved to the `plots/` directory after training completes.

- **Training Results**: `plots/training_results.png` — Scores, epsilon decay, loss, survival time
- **Agent Comparison**: `plots/agent_comparison.png` — DQN vs Random agent

---

## 🚀 Future Improvements

- **Double DQN**: Reduce Q-value overestimation
- **Dueling DQN**: Separate state-value and advantage streams
- **Prioritized Experience Replay**: Sample important transitions more often
- **CNN-based State**: Use raw pixel input instead of hand-crafted features
- **Curriculum Learning**: Gradually increase grid size
- **Multi-agent**: Multiple snakes competing
- **Noisy Networks**: Parameter-space exploration instead of epsilon-greedy
- **Human Play Mode**: Allow human to play and compare with AI

---

## 📄 License

This project is developed for educational and academic purposes.

---

*Built with PyTorch, Pygame, and a passion for Reinforcement Learning* 🎮🤖
