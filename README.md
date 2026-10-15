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
git clone https://github.com/namang-afk/A-Reinforcement-Learning-Agent-for-Open-World-Autonomous-Navigation.git
cd A-Reinforcement-Learning-Agent-for-Open-World-Autonomous-Navigation

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

### 1. Launch Interactive Web Dashboard (Recommended)
Watch the trained PPO agent navigate in real time, compare against A* and Random baselines, or take manual keyboard control!

```bash
# Start the web dashboard (auto-launches your browser at http://localhost:8000)
python dashboard.py

# Custom port
python dashboard.py --port 8080
```

### 2. Run Tests
```bash
pytest tests/ -v
```

### 3. Train the Agent
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

### 4. Monitor Training (TensorBoard)
```bash
tensorboard --logdir logs/
```

### 5. Evaluate
```bash
# Evaluate all policies (PPO, Random, A*)
python evaluate.py

# Custom evaluation
python evaluate.py --model models/ppo_navigation_final --episodes 200
```

### 6. Generate Visualizations
```bash
python visualize.py

# With interactive display
python visualize.py --show
```

---

## Project Structure

```
A-Reinforcement-Learning-Agent-for-Open-World-Autonomous-Navigation/
├── dashboard.py           # Interactive Web/GUI dashboard backend server
├── web/                   # Frontend assets for live dashboard
│   ├── index.html         # Glassmorphic UI with canvas viewport & HUD
│   ├── style.css          # Dark slate/cyan styling system
│   └── app.js             # Canvas rendering, telemetry & playback engine
├── config.py              # Centralized configuration (hyperparameters, paths)
├── environment.py         # Custom Gymnasium grid navigation environment
├── train.py               # PPO training pipeline with checkpoints & logging
├── evaluate.py            # Evaluation metrics: success rate, collisions, efficiency
├── baselines.py           # Random and A* baseline policies
├── visualize.py           # Trajectory plots, training curves, comparisons
├── requirements.txt       # Pinned dependencies
├── README.md              # Documentation and interview guide
├── tests/
│   ├── __init__.py
│   └── test_environment.py  # 31 automated pytest tests
├── models/                # Saved model checkpoints
├── logs/                  # TensorBoard logs & evaluation logs
├── results/               # Evaluation results JSON
└── figures/               # Saved visualization plots
```

---

## Results & Empirical Benchmarks

The benchmark metrics below were measured across **100 procedurally generated, unseen maps** (seed 123) on 20×20 grids with static wall density 0.10 and dynamic moving obstacles:

### Empirical Performance Comparison (100 Unseen Maps)

| Metric | PPO Agent (Ours) | Uniform Random | A* Oracle (Full Grid) |
|---|---|---|---|
| **Success Rate** | **81.0%** | 14.0% | 91.0% |
| **Collision Rate** | **1.0%** (Superhuman avoidance) | 34.0% | 9.0%* |
| **Path Efficiency** | **88.2%** | 11.8% | 100.0% |
| **Avg Episode Length** | **51.5 ± 72.4** | 146.6 ± 68.3 | 12.3 ± 5.3 |
| **Avg Reward** | **+63.81 ± 101.3** | -41.43 ± 54.1 | +105.42 ± 34.6 |

*\*Note on Dynamic Obstacles:* The A* Oracle possesses complete global map knowledge and plans optimal static paths, but does not anticipate future random walks of moving hazards—colliding in 9.0% of episodes. In contrast, the trained PPO agent has learned reactive dynamic obstacle evasion using its local egocentric sensor view, achieving a superior **1.0% collision rate**.

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
