"""
Centralized configuration for the RL Navigation project.

All hyperparameters, environment settings, training parameters,
and file paths are defined here for reproducibility and easy tuning.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class EnvConfig:
    """Environment configuration parameters."""

    # Grid dimensions
    grid_size: int = 20

    # Local observation window (must be odd)
    view_size: int = 5

    # Obstacle density: fraction of free cells to fill with walls
    wall_density: float = 0.10

    # Number of dynamic (moving) obstacles
    num_dynamic_obstacles: int = 2

    # Dynamic obstacle movement probability per step
    dynamic_move_prob: float = 0.3

    # Maximum steps per episode before truncation
    max_steps: int = 200

    # Reward parameters
    reward_goal: float = 100.0
    reward_collision: float = -10.0
    reward_step: float = -0.1
    reward_progress_scale: float = 1.5

    # Minimum Manhattan distance between start and goal
    min_start_goal_distance: int = 5

    def __post_init__(self):
        assert self.grid_size >= 10, "Grid size must be at least 10"
        assert self.view_size % 2 == 1, "View size must be odd"
        assert 0.0 <= self.wall_density <= 0.4, "Wall density must be in [0, 0.4]"
        assert self.num_dynamic_obstacles >= 0, "Cannot have negative obstacles"
        assert self.max_steps > 0, "Max steps must be positive"


@dataclass
class TrainConfig:
    """PPO training configuration."""

    # PPO hyperparameters
    learning_rate: float = 5e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.04
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Neural network architecture
    net_arch: List[int] = field(default_factory=lambda: [128, 128])

    # Training duration
    total_timesteps: int = 500_000

    # Checkpointing
    checkpoint_freq: int = 50_000  # Save every N timesteps
    log_interval: int = 10  # Log every N rollouts

    # Reproducibility
    seed: int = 42

    # Number of parallel environments for training
    n_envs: int = 4

    # Paths
    model_dir: str = "models"
    log_dir: str = "logs"
    final_model_name: str = "ppo_navigation_final"


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    # Number of evaluation episodes
    n_episodes: int = 100

    # Whether to use deterministic actions during evaluation
    deterministic: bool = True

    # Seed for evaluation (different from training for true generalization)
    seed: int = 123

    # Output directory for results
    results_dir: str = "results"


@dataclass
class VisualConfig:
    """Visualization configuration."""

    # Figure sizes
    fig_size_trajectory: tuple = (8, 8)
    fig_size_curves: tuple = (12, 6)
    fig_size_comparison: tuple = (10, 6)

    # Output directory
    output_dir: str = "figures"

    # DPI for saved figures
    dpi: int = 150

    # Color scheme
    color_empty: str = "#FFFFFF"
    color_wall: str = "#2C3E50"
    color_agent: str = "#E74C3C"
    color_target: str = "#27AE60"
    color_dynamic_obstacle: str = "#F39C12"
    color_path: str = "#3498DB"
    color_start: str = "#9B59B6"


# ---------------------------------------------------------------------------
# Convenience: default instances
# ---------------------------------------------------------------------------

ENV_CONFIG = EnvConfig()
TRAIN_CONFIG = TrainConfig()
EVAL_CONFIG = EvalConfig()
VISUAL_CONFIG = VisualConfig()


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [TRAIN_CONFIG.model_dir, TRAIN_CONFIG.log_dir,
              EVAL_CONFIG.results_dir, VISUAL_CONFIG.output_dir]:
        os.makedirs(d, exist_ok=True)
