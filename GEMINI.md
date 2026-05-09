# Project Instructions: Snake Game AI (DQN)

This project is a Deep Reinforcement Learning implementation of the classic Snake game using Deep Q-Networks (DQN) with PyTorch and Pygame.

## Project Overview

- **Core Technology:** Python 3.8+, PyTorch 2.0+, Pygame 2.5+.
- **Architecture:** Modular design separating game logic, RL algorithms, rendering, and training/evaluation loops.
- **State Representation:** 11-dimensional binary vector representing danger and food relative to the snake's head and its current direction.
- **Actions:** 3 relative actions (Straight, Left, Right).

## Key Components

- `main.py`: Central dashboard to launch all training and evaluation modes visually.
- `config.py`: Centralized configuration using `dataclasses` for game, DQN, rewards, and rendering.
- `snake_game.py`: The environment logic following a standard RL step-interface.
- `agent.py`: `DQNAgent` class managing the online/target networks and training steps.
- `model.py`: Neural network architecture (PyTorch).
- `renderer.py`: Visualization logic using Pygame.
- `train.py`: Main entry point for training the agent with live visualization.
- `evaluate.py`: Entry point for watching the trained agent or benchmarking performance.

## Building and Running

### Setup
```bash
pip install -r requirements.txt
```

### Dashboard (Recommended)
```bash
python main.py
```

### Training
```bash
# Standard training with visualization
python train.py

# Fast training without rendering
python train.py --no-render --episodes 1000
```

### Evaluation
```bash
# Watch the best trained agent
python evaluate.py

# Benchmark performance (no rendering)
python evaluate.py --benchmark --episodes 100
```

## Development Conventions

- **Configuration:** Always use or extend `config.py` for hyperparameters and settings. Avoid hardcoding values in other modules.
- **Modularity:** Keep game logic (`snake_game.py`), AI logic (`agent.py`), and visualization (`renderer.py`) decoupled.
- **Type Hinting:** Use Python type hints for better code clarity and IDE support.
- **Documentation:** Maintain descriptive module-level docstrings and function documentation.
- **Checkpoints:** Models are saved to the `checkpoints/` directory. Use `--resume` in `train.py` to continue training.
- **Metrics:** Training metrics are tracked and plotted in the `plots/` directory after training sessions.
