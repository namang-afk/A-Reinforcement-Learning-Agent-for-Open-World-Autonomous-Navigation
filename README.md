# 🧭 Reinforcement Learning Agent for Open-World Autonomous Navigation

A reinforcement learning agent trained via **Proximal Policy Optimization (PPO)** to navigate procedurally generated 2D grid environments, avoid static and dynamic obstacles, and generalize to unseen maps.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-pytest-yellow?logo=pytest)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)

---

## Overview

This project demonstrates end-to-end reinforcement learning applied to autonomous navigation:

1. **Custom Gymnasium Environment** — Procedurally generated 20×20 grids with walls, a target, and moving obstacles. Every map is validated for solvability via BFS.
2. **Partial Observability** — The agent observes only a 5×5 local patch plus its normalized position and relative target direction (29-dim vector), forcing it to learn reactive navigation rather than memorize maps.
3. **Shaped Reward Function** — Combines goal bonus (+100), collision penalty (−10), Manhattan-distance progress shaping, and a per-step penalty (−0.1) to provide dense learning signal without exploitable shortcuts.
4. **PPO Training** — Uses Stable-Baselines3 with vectorized environments, periodic evaluation checkpoints, and TensorBoard logging.
5. **Baselines** — Compared against a random policy (lower bound) and an A* oracle (approximate upper bound with full grid knowledge).

---

## Architecture

```
┌─────────────────────────────────────────┐
│         Procedural Map Generator         │
│  (walls, start, goal, dynamic obstacles) │
└──────────────────┬──────────────────────┘
                   │ reset()
                   ▼
┌─────────────────────────────────────────┐
│        GridNavigationEnv (Gymnasium)     │
│  Obs: [5×5 local view | pos | target]   │
│  Act: {Up, Down, Left, Right}            │
│  Reward: goal + progress − step − coll   │
└──────────┬──────────────┬───────────────┘
           │ obs, reward  │ action
           ▼              │
┌──────────────────┐      │
│   PPO Agent      │◄─────┘
│  (SB3 + PyTorch) │
│  MLP [128, 128]  │
└──────────────────┘
```

---

## Features

| Feature | Description |
|---------|-------------|
| 🗺️ Procedural maps | Every episode generates a unique, solvable grid |
| 👁️ Partial observability | 5×5 local view — no full map access |
| 🚧 Dynamic obstacles | Moving obstacles that require reactive avoidance |
| 📊 Reward shaping | Dense progress signal + sparse goal reward |
| 🔄 Reproducibility | Seeded experiments, pinned dependencies |
| 📈 TensorBoard | Real-time training monitoring |
| 🧪 Test suite | 20+ automated tests via pytest |
| 🏁 Baselines | Random + A* for meaningful comparison |
| 📉 Visualization | Trajectories, training curves, comparison charts |

---

## Setup

### Prerequisites
- Python 3.9 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rl-autonomous-navigation.git
cd rl-autonomous-navigation

# Create a virtual environment (recommended)
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Run Tests
```bash
pytest tests/ -v
```

### 2. Train the Agent
```bash
# Default training (500K timesteps)
python train.py

# Quick test run
python train.py --timesteps 50000

# Custom configuration
python train.py --timesteps 1000000 --seed 123 --grid-size 25

# Resume from checkpoint
python train.py --resume models/ppo_nav_checkpoint_100000_steps
```

### 3. Monitor Training (TensorBoard)
```bash
tensorboard --logdir logs/
```

### 4. Evaluate
```bash
# Evaluate all policies (PPO, Random, A*)
python evaluate.py

# Custom evaluation
python evaluate.py --model models/ppo_navigation_final --episodes 200
```

### 5. Generate Visualizations
```bash
python visualize.py

# With interactive display
python visualize.py --show
```

---

## Project Structure

```
rl-autonomous-navigation/
├── config.py              # Centralized configuration (hyperparameters, paths)
├── environment.py         # Custom Gymnasium grid navigation environment
├── train.py               # PPO training pipeline with checkpoints & logging
├── evaluate.py            # Evaluation metrics: success rate, collisions, efficiency
├── baselines.py           # Random and A* baseline policies
├── visualize.py           # Trajectory plots, training curves, comparisons
├── requirements.txt       # Pinned dependencies
├── README.md              # This file
├── tests/
│   ├── __init__.py
│   └── test_environment.py  # 20+ automated tests
├── models/                # Saved model checkpoints (created during training)
├── logs/                  # TensorBoard logs (created during training)
├── results/               # Evaluation results JSON (created during evaluation)
└── figures/               # Saved visualization plots (created during visualization)
```

---

## Results

> **Note:** The results below represent *expected* performance based on the reward structure and environment design. Actual results will vary based on training duration and random seeds. Train the agent yourself to obtain empirical results.

### Expected Performance (after ~500K timesteps)

| Metric | PPO Agent | Random | A* Oracle |
|--------|-----------|--------|-----------|
| Success Rate | 60-80% | ~1-3% | ~95%+ |
| Avg Episode Length | 30-60 | ~50 (truncated) | Optimal |
| Collision Rate | 10-25% | ~80%+ | Low* |
| Path Efficiency | 0.4-0.7 | N/A | 1.0 |

*\*A\* has full grid knowledge but can still collide with dynamic obstacles that move after path computation.*

### Improving Results
- **More timesteps** (1-2M) significantly improves success rate
- **Larger networks** (`net_arch: [256, 256]`) for harder maps
- **Curriculum learning**: start with small grids, increase over time
- **Lower wall density** first, then increase for harder generalization

---

## Configuration

All parameters are in [`config.py`](config.py). Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_size` | 20 | Grid dimensions (NxN) |
| `view_size` | 5 | Local observation window |
| `wall_density` | 0.15 | Fraction of interior cells that are walls |
| `num_dynamic_obstacles` | 3 | Number of moving obstacles |
| `max_steps` | 200 | Episode length limit |
| `total_timesteps` | 500,000 | PPO training duration |
| `learning_rate` | 3e-4 | PPO learning rate |
| `n_envs` | 4 | Parallel training environments |
| `reward_goal` | +100.0 | Reward for reaching target |
| `reward_collision` | -10.0 | Penalty for hitting obstacles |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Ensure virtual environment is activated and `pip install -r requirements.txt` was run |
| Training is very slow | Reduce `total_timesteps`, `grid_size`, or `n_envs`. Use GPU if available |
| Agent doesn't learn | Increase `total_timesteps` to 1M+. Check TensorBoard for reward trends |
| Success rate stays at 0% | Lower `wall_density` to 0.05 for initial experiments |
| `RuntimeError: Failed to generate valid map` | Reduce `wall_density` or `min_start_goal_distance` |
| CUDA out of memory | Set `device="cpu"` in train.py or reduce `batch_size` |

---

## Technical Details

### Observation Space (29-dimensional float vector)
- **Local view** (25 values): Flattened 5×5 grid centered on agent. Out-of-bounds cells are encoded as walls (1). Cell values: 0=empty, 1=wall, 2=dynamic obstacle, 3=target.
- **Normalized position** (2 values): Agent's (row, col) divided by grid_size.
- **Relative target** (2 values): (target_row − agent_row, target_col − agent_col) divided by grid_size.

### Reward Function
```
R = −0.1 (step penalty)
  + 1.0 × (prev_manhattan − curr_manhattan)   [progress shaping]
  + 100.0  [if goal reached, terminal]
  − 10.0   [if collision, terminal]
```

### PPO Configuration
- Policy: MLP with 2 hidden layers of 128 units each (ReLU)
- Rollout: 2048 steps per update, 4 parallel environments = 8192 transitions per update
- Optimization: 10 epochs per update, batch size 64, clip range 0.2
- Entropy bonus: 0.01 (encourages exploration)
- Discount: γ = 0.99, GAE λ = 0.95

---

## License

This project is released under the MIT License.

---

## Acknowledgments

- [Gymnasium](https://gymnasium.farama.org/) — Environment API
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — RL algorithms
- [PyTorch](https://pytorch.org/) — Deep learning framework
