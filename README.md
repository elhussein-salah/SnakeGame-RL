# 🐍 Snake Game AI — Advanced Deep Q-Network

An expert-level Deep Reinforcement Learning project that trains an AI agent to play Snake using a **Dueling Double Deep Q-Network (Dueling DDQN)** with **Prioritized Experience Replay (PER)**.

This implementation is built from scratch with PyTorch and features a full Pygame UI for real-time visualization and control.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.5+-green.svg)
![RL Algorithm](https://img.shields.io/badge/Algorithm-Dueling_DDQN_+_PER-orange.svg)

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Architecture](#-project-architecture)
- [Setup & Installation](#-setup--installation)
- [How to Use the Dashboard](#-how-to-use-the-dashboard)
- [Hyperparameters](#-hyperparameters)
- [Project Structure](#-project-structure)

---

## 🎯 Overview

This project implements a state-of-the-art **Dueling Double DQN** agent that learns to play Snake through reinforcement learning. It goes beyond a basic DQN by incorporating modern techniques that significantly improve training speed, stability, and peak performance.

The agent observes an 11-dimensional state vector, takes actions relative to its current direction, and learns an optimal policy by sampling from a prioritized buffer of past experiences.

---

## ✨ Key Features

- ✅ **Advanced RL Algorithm:** Dueling DDQN with Prioritized Experience Replay, built from scratch.
- ✅ **Interactive Dashboard (`main.py`):** A full-featured UI to launch all training and evaluation modes.
- ✅ **Dynamic Inputs:** Configure grid size, obstacles, and episodes directly from the UI.
- ✅ **Live Visualization:** Watch the AI train or play in a Pygame window with a detailed HUD.
- ✅ **Visual & Statistical Benchmarking:** Compare the trained AI against a random agent visually or through statistical reports.
- ✅ **Smart Window Scaling:** The game window automatically scales to a comfortable size for any grid layout.
- ✅ **Full Control:** Adjust FPS in real-time (`UP`/`DOWN` arrows) or pause (`SPACE`) during any visual mode.

---

## 🏗️ Project Architecture

The agent's "brain" is a **Dueling Q-Network**, which separates state valuation and action advantage calculation for more robust learning.

```
Input State (11) → Shared Layers → | → Value Stream V(s)
                               | → Advantage Stream A(s,a)
```
Combined via: `Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))`

The project's code is highly modular:
- **Game Logic:** `snake_game.py`
- **RL Algorithm:** `agent.py`, `model.py`, `replay_buffer.py`
- **Dashboard & UI:** `main.py`, `renderer.py`
- **Entrypoints:** `train.py`, `evaluate.py` (launched by the dashboard)
- **Configuration:** `config.py`

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8 or higher
- `pip` package manager

### Installation
```bash
# 1. Clone or navigate to the project directory
cd "snackGameAI"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the main dashboard
python main.py
```

---

## 🚀 How to Use the Dashboard

The `main.py` dashboard is the central hub for the project.

### Configurable Parameters
- **GRID SIZE:** The dimensions of the game board (e.g., `10` for a 10x10 grid).
- **OBSTACLES:** The number of static obstacles placed on the board.
- **EPISODES:** The number of games to play for training or benchmarking.

### Dashboard Buttons

| Button | Description |
| :--- | :--- |
| **1. Train AI (Visual)** | Trains the AI while showing the live game window. Great for observing the learning process. |
| **2. Train AI (Fast / No-Render)** | Trains the AI without rendering. Use this for maximum speed to build a strong model quickly. |
| **3. Watch Trained Agent** | Loads your best-trained model and lets it play. This is the "Showcase" mode. |
| **4. Visual Benchmark** | Watch the DQN and a Random agent play alternating episodes to visually compare their intelligence. |
| **5. Benchmark (Stats)** | Runs a headless comparison and generates a professional plot (`agent_comparison.png`) of the results. |
| **6. Run Benchmark (Headless)** | Stress-tests the final AI model over many episodes and prints a statistical summary. |

---

## 🎛️ Hyperparameters

Key parameters are configured in `config.py`.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `gamma` | 0.99 | Discount factor for future rewards. |
| `learning_rate` | 0.001 | Adam optimizer learning rate. |
| `batch_size` | 64 | Number of transitions sampled per training step. |
| `per_alpha` | 0.6 | PER priority exponent (how much to prioritize). |
| `per_beta_start` | 0.4 | Initial IS weight for correcting PER bias. |
| `tau` | 0.005 | Soft update factor for the target network. |
| `grad_clip` | 10.0 | Gradient clipping norm to prevent exploding gradients. |

---

## 📁 Project Structure

```
snackGameAI/
│
├── main.py            # The main dashboard application
├── config.py          # All hyperparameters
├── snake_game.py      # Game environment
├── model.py           # Dueling Q-Network architecture
├── replay_buffer.py   # Prioritized Experience Replay (PER)
├── agent.py           # Dueling DDQN agent logic
├── renderer.py        # Pygame visualization
├── utils.py           # Metrics tracking and plotting
├── train.py           # Training loop script
├── evaluate.py        # Evaluation and benchmark script
├── requirements.txt   # Dependencies
├── README.md          # This file
│
├── checkpoints/       # Saved model checkpoints
└── plots/             # Training/benchmark result plots
```
