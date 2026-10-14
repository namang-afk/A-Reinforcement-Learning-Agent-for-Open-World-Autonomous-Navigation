"""
Visualization module for the RL Navigation project.

Produces:
  1. Environment layout heatmaps
  2. Agent trajectory overlays
  3. Training reward / episode-length curves (from SB3 Monitor logs)
  4. Comparative bar charts (PPO vs Random vs A*)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.collections import LineCollection

from environment import GridNavigationEnv
from baselines import AStarPolicy
from config import (
    EnvConfig,
    VisualConfig,
    VISUAL_CONFIG,
    ENV_CONFIG,
    ensure_dirs,
)


def plot_environment(
    env: GridNavigationEnv,
    title: str = "Environment Layout",
    trajectory: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[str] = None,
    vc: Optional[VisualConfig] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot the grid environment with optional trajectory overlay.

    Args:
        env: The environment to visualize (must have been reset)
        title: Plot title
        trajectory: List of (row, col) positions to draw as a path
        save_path: If provided, save the figure to this path
        vc: Visual configuration
        show: Whether to display the plot

    Returns:
        matplotlib Figure
    """
    vc = vc or VISUAL_CONFIG

    fig, ax = plt.subplots(1, 1, figsize=vc.fig_size_trajectory)

    grid = env.grid.copy().astype(float)

    # Custom colormap: 0=white, 1=dark, 2=orange, 3=green
    cmap = mcolors.ListedColormap([
        vc.color_empty, vc.color_wall, vc.color_dynamic_obstacle, vc.color_target
    ])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(grid, cmap=cmap, norm=norm, origin="upper")

    # Draw agent
    ar, ac = env.agent_pos
    ax.plot(ac, ar, "o", color=vc.color_agent, markersize=12, zorder=5,
            markeredgecolor="white", markeredgewidth=2)

    # Draw trajectory
    if trajectory and len(trajectory) > 1:
        rows = [p[0] for p in trajectory]
        cols = [p[1] for p in trajectory]
        ax.plot(cols, rows, "-", color=vc.color_path, linewidth=2, alpha=0.7, zorder=4)
        # Start marker
        ax.plot(cols[0], rows[0], "s", color=vc.color_start, markersize=10,
                zorder=6, markeredgecolor="white", markeredgewidth=2)

    # Grid lines
    ax.set_xticks(np.arange(-0.5, env.grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.grid_size, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.3, alpha=0.5)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks(np.arange(0, env.grid_size, 5))
    ax.set_yticks(np.arange(0, env.grid_size, 5))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=vc.color_wall, label="Wall"),
        mpatches.Patch(facecolor=vc.color_dynamic_obstacle, label="Dynamic Obstacle"),
        mpatches.Patch(facecolor=vc.color_target, label="Target"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=vc.color_agent,
                    markersize=10, label="Agent"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=vc.color_start,
                    markersize=10, label="Start"),
        plt.Line2D([0], [0], color=vc.color_path, linewidth=2, label="Path"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.9)

    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=vc.dpi, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_training_curves(
    log_dir: str = "logs",
    save_path: Optional[str] = None,
    vc: Optional[VisualConfig] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot training reward and episode length curves from SB3 evaluations.

    Reads the evaluations.npz file produced by EvalCallback.

    Args:
        log_dir: Directory containing evaluations.npz
        save_path: If provided, save figure
        vc: Visual config
        show: Whether to display

    Returns:
        matplotlib Figure or None if no data found
    """
    vc = vc or VISUAL_CONFIG
    eval_file = os.path.join(log_dir, "evaluations.npz")

    if not os.path.exists(eval_file):
        print(f"  Warning: {eval_file} not found. Train the agent first.")
        return None

    data = np.load(eval_file)
    timesteps = data["timesteps"]
    results = data["results"]        # shape: (n_evals, n_episodes)
    ep_lengths = data["ep_lengths"]   # shape: (n_evals, n_episodes)

    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)
    mean_lengths = np.mean(ep_lengths, axis=1)
    std_lengths = np.std(ep_lengths, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=vc.fig_size_curves)

    # --- Reward curve ---
    ax = axes[0]
    ax.plot(timesteps, mean_rewards, color="#3498DB", linewidth=2, label="Mean Reward")
    ax.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
        color="#3498DB",
    )
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("Training Reward Curve", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Episode length curve ---
    ax = axes[1]
    ax.plot(timesteps, mean_lengths, color="#E74C3C", linewidth=2, label="Mean Length")
    ax.fill_between(
        timesteps,
        mean_lengths - std_lengths,
        mean_lengths + std_lengths,
        alpha=0.2,
        color="#E74C3C",
    )
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Episode Length", fontsize=12)
    ax.set_title("Episode Length Curve", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("PPO Training Progress", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=vc.dpi, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_comparison_bars(
    results_path: str = "results/evaluation_results.json",
    save_path: Optional[str] = None,
    vc: Optional[VisualConfig] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot comparative bar charts for PPO vs baselines.

    Args:
        results_path: Path to the JSON evaluation results
        save_path: If provided, save the figure
        vc: Visual config
        show: Whether to display

    Returns:
        matplotlib Figure or None if no data found
    """
    vc = vc or VISUAL_CONFIG

    if not os.path.exists(results_path):
        print(f"  Warning: {results_path} not found. Run evaluation first.")
        return None

    with open(results_path) as f:
        data = json.load(f)

    policy_names = list(data.keys())
    colors = ["#3498DB", "#95A5A6", "#27AE60"]

    metrics = {
        "Success Rate": [data[p]["success_rate"] for p in policy_names],
        "Avg Reward": [data[p]["avg_reward"] for p in policy_names],
        "Collision Rate": [data[p]["collision_rate"] for p in policy_names],
        "Avg Episode Length": [data[p]["avg_episode_length"] for p in policy_names],
    }

    fig, axes = plt.subplots(2, 2, figsize=vc.fig_size_comparison)
    axes = axes.flatten()

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        bars = ax.bar(policy_names, values, color=colors[:len(policy_names)],
                      edgecolor="white", linewidth=1.5)
        ax.set_title(metric_name, fontsize=12, fontweight="bold")
        ax.set_ylabel(metric_name)

        # Value labels
        for bar, val in zip(bars, values):
            fmt = f"{val:.1%}" if "Rate" in metric_name else f"{val:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                fmt,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Policy Comparison: PPO vs Baselines",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=vc.dpi, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def visualize_episode(
    policy: Any,
    env_config: Optional[EnvConfig] = None,
    seed: int = 42,
    title: str = "Agent Trajectory",
    save_path: Optional[str] = None,
    is_astar: bool = False,
    show: bool = True,
) -> plt.Figure:
    """
    Run one episode with a policy and visualize the trajectory.

    Args:
        policy: Policy with .predict(obs) -> (action, _)
        env_config: Environment config
        seed: Random seed for reproducibility
        title: Plot title
        save_path: Save path
        is_astar: Whether policy is A*
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    ec = env_config or ENV_CONFIG
    env = GridNavigationEnv(config=ec)
    obs, info = env.reset(seed=seed)

    if is_astar:
        policy.set_env(env)

    done = False
    while not done:
        if hasattr(policy, "predict"):
            if is_astar or isinstance(policy, (type(None),)):
                action, _ = policy.predict(obs)
            else:
                action, _ = policy.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if is_astar:
            policy.set_env(env)

    success_str = "✓ Success" if info.get("is_success", False) else "✗ Failed"
    full_title = f"{title} ({success_str}, {env.steps} steps)"

    return plot_environment(
        env,
        title=full_title,
        trajectory=env.trajectory,
        save_path=save_path,
        show=show,
    )


def generate_all_figures(
    model_path: str = "models/ppo_navigation_final",
    log_dir: str = "logs",
    results_path: str = "results/evaluation_results.json",
    output_dir: str = "figures",
    show: bool = False,
) -> None:
    """
    Generate all visualization figures and save them.

    Args:
        model_path: Path to trained PPO model
        log_dir: Directory with training logs
        results_path: Path to evaluation results JSON
        output_dir: Directory to save figures
        show: Whether to display figures interactively
    """
    ensure_dirs()
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print("  Generating Visualizations")
    print("=" * 50)

    # 1. Sample environment layout
    print("\n  [1/5] Environment layout...")
    env = GridNavigationEnv()
    env.reset(seed=42)
    plot_environment(
        env,
        title="Sample Procedural Environment",
        save_path=os.path.join(output_dir, "environment_layout.png"),
        show=show,
    )

    # 2. Training curves
    print("  [2/5] Training curves...")
    plot_training_curves(
        log_dir=log_dir,
        save_path=os.path.join(output_dir, "training_curves.png"),
        show=show,
    )

    # 3. PPO trajectory
    print("  [3/5] PPO agent trajectory...")
    try:
        from stable_baselines3 import PPO as PPO_
        model = PPO_.load(model_path)
        visualize_episode(
            model,
            seed=99,
            title="PPO Agent Trajectory",
            save_path=os.path.join(output_dir, "ppo_trajectory.png"),
            show=show,
        )
    except Exception as e:
        print(f"    Skipped PPO trajectory: {e}")

    # 4. A* trajectory
    print("  [4/5] A* baseline trajectory...")
    astar = AStarPolicy()
    visualize_episode(
        astar,
        seed=99,
        title="A* Baseline Trajectory",
        save_path=os.path.join(output_dir, "astar_trajectory.png"),
        is_astar=True,
        show=show,
    )

    # 5. Comparison bars
    print("  [5/5] Comparison bar charts...")
    plot_comparison_bars(
        results_path=results_path,
        save_path=os.path.join(output_dir, "comparison_bars.png"),
        show=show,
    )

    print(f"\n  All figures saved to: {output_dir}/")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate visualization figures")
    parser.add_argument(
        "--model", type=str, default="models/ppo_navigation_final",
        help="Path to trained PPO model"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display figures interactively"
    )
    args = parser.parse_args()

    generate_all_figures(model_path=args.model, show=args.show)


if __name__ == "__main__":
    main()
