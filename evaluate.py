"""
Evaluation pipeline for the trained PPO agent and baselines.

Runs episodes on unseen procedurally generated maps and computes:
  - Success rate (fraction reaching the goal)
  - Average episode length
  - Collision rate
  - Path efficiency (optimal A* length / actual length)
  - Average cumulative reward
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
from stable_baselines3 import PPO

from environment import GridNavigationEnv
from baselines import RandomPolicy, AStarPolicy
from config import EnvConfig, EvalConfig, EVAL_CONFIG, ENV_CONFIG, ensure_dirs


@dataclass
class EpisodeResult:
    """Metrics from a single evaluation episode."""
    success: bool
    episode_length: int
    total_reward: float
    collided: bool
    trajectory: list
    optimal_path_length: Optional[int]

    @property
    def path_efficiency(self) -> Optional[float]:
        """Ratio of optimal to actual path length (1.0 = perfect)."""
        if self.optimal_path_length and self.success and self.episode_length > 0:
            return self.optimal_path_length / self.episode_length
        return None


@dataclass
class EvalResults:
    """Aggregated evaluation metrics."""
    policy_name: str
    n_episodes: int
    success_rate: float
    avg_episode_length: float
    avg_reward: float
    collision_rate: float
    avg_path_efficiency: Optional[float]
    std_reward: float
    std_episode_length: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"\n{'=' * 50}",
            f"  Evaluation Results: {self.policy_name}",
            f"{'=' * 50}",
            f"  Episodes:            {self.n_episodes}",
            f"  Success rate:        {self.success_rate:.1%}",
            f"  Avg episode length:  {self.avg_episode_length:.1f} ± {self.std_episode_length:.1f}",
            f"  Avg reward:          {self.avg_reward:.2f} ± {self.std_reward:.2f}",
            f"  Collision rate:      {self.collision_rate:.1%}",
        ]
        if self.avg_path_efficiency is not None:
            lines.append(
                f"  Avg path efficiency: {self.avg_path_efficiency:.2%}"
            )
        lines.append(f"{'=' * 50}")
        return "\n".join(lines)


def compute_optimal_path_length(env: GridNavigationEnv) -> Optional[int]:
    """Use A* to compute the shortest path length on the current map."""
    astar = AStarPolicy()
    path = astar._astar(env.grid, env.agent_pos, env.target_pos, env.grid_size)
    if path is not None:
        return len(path) - 1  # number of steps = nodes - 1
    return None


def run_episode(
    env: GridNavigationEnv,
    policy: Any,
    is_astar: bool = False,
    deterministic: bool = True,
) -> EpisodeResult:
    """
    Run a single evaluation episode.

    Args:
        env: The navigation environment
        policy: A policy with .predict(obs) -> (action, _)
        is_astar: Whether the policy is A* (needs env binding)
        deterministic: Whether to use deterministic actions (PPO)

    Returns:
        EpisodeResult with metrics
    """
    obs, info = env.reset()

    # Compute optimal path before the episode begins
    optimal_len = compute_optimal_path_length(env)

    if is_astar:
        policy.set_env(env)

    total_reward = 0.0
    done = False
    collided = False

    while not done:
        if hasattr(policy, "predict"):
            if is_astar or isinstance(policy, RandomPolicy):
                action, _ = policy.predict(obs)
            else:
                action, _ = policy.predict(obs, deterministic=deterministic)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Detect collision (negative reward beyond step penalty)
        if terminated and not info.get("is_success", False):
            collided = True

        # Update A* env reference after step
        if is_astar:
            policy.set_env(env)

    return EpisodeResult(
        success=info.get("is_success", False),
        episode_length=info.get("steps", env.steps),
        total_reward=total_reward,
        collided=collided,
        trajectory=list(env.trajectory),
        optimal_path_length=optimal_len,
    )


def evaluate_policy(
    policy: Any,
    policy_name: str,
    env_config: Optional[EnvConfig] = None,
    eval_config: Optional[EvalConfig] = None,
    is_astar: bool = False,
) -> EvalResults:
    """
    Evaluate a policy over multiple episodes on unseen maps.

    Args:
        policy: Policy with .predict(obs) interface
        policy_name: Human-readable name
        env_config: Environment configuration
        eval_config: Evaluation parameters
        is_astar: Whether policy is A*

    Returns:
        EvalResults with aggregated metrics
    """
    ec = eval_config or EVAL_CONFIG
    env_c = env_config or ENV_CONFIG

    env = GridNavigationEnv(config=env_c)

    results: List[EpisodeResult] = []

    print(f"\n  Evaluating {policy_name} over {ec.n_episodes} episodes...")

    for i in range(ec.n_episodes):
        # Use a different seed for each episode to ensure unseen maps
        env.reset(seed=ec.seed + i)
        # Re-reset to apply the seed (first reset sets the RNG)
        result = run_episode(
            env, policy, is_astar=is_astar, deterministic=ec.deterministic
        )
        results.append(result)

        if (i + 1) % 20 == 0:
            successes = sum(1 for r in results if r.success)
            print(f"    [{i + 1}/{ec.n_episodes}] Success so far: {successes}/{i + 1}")

    # --- Aggregate metrics ----------------------------------------------------
    successes = [r.success for r in results]
    lengths = [r.episode_length for r in results]
    rewards = [r.total_reward for r in results]
    collisions = [r.collided for r in results]
    efficiencies = [r.path_efficiency for r in results if r.path_efficiency is not None]

    return EvalResults(
        policy_name=policy_name,
        n_episodes=ec.n_episodes,
        success_rate=np.mean(successes),
        avg_episode_length=np.mean(lengths),
        avg_reward=np.mean(rewards),
        collision_rate=np.mean(collisions),
        avg_path_efficiency=np.mean(efficiencies) if efficiencies else None,
        std_reward=np.std(rewards),
        std_episode_length=np.std(lengths),
    )


def evaluate_all(
    model_path: str,
    env_config: Optional[EnvConfig] = None,
    eval_config: Optional[EvalConfig] = None,
) -> Dict[str, EvalResults]:
    """
    Evaluate PPO, Random, and A* policies and return results.

    Args:
        model_path: Path to saved PPO model (.zip)
        env_config: Environment config
        eval_config: Eval config

    Returns:
        Dict mapping policy name to EvalResults
    """
    ensure_dirs()
    results = {}

    # --- PPO Agent ---
    print("\n  Loading PPO model...")
    model = PPO.load(model_path)
    results["PPO Agent"] = evaluate_policy(
        model, "PPO Agent", env_config, eval_config
    )
    print(results["PPO Agent"].summary())

    # --- Random Baseline ---
    random_policy = RandomPolicy(seed=42)
    results["Random"] = evaluate_policy(
        random_policy, "Random", env_config, eval_config
    )
    print(results["Random"].summary())

    # --- A* Baseline ---
    astar_policy = AStarPolicy()
    results["A*"] = evaluate_policy(
        astar_policy, "A*", env_config, eval_config, is_astar=True
    )
    print(results["A*"].summary())

    # --- Save results to JSON ------------------------------------------------
    ec = eval_config or EVAL_CONFIG
    results_path = os.path.join(ec.results_dir, "evaluation_results.json")
    os.makedirs(ec.results_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(
            {name: r.to_dict() for name, r in results.items()},
            f,
            indent=2,
        )
    print(f"\n  Results saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate navigation policies")
    parser.add_argument(
        "--model", type=str, default="models/ppo_navigation_final",
        help="Path to trained PPO model (without .zip extension)"
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Evaluation seed"
    )
    args = parser.parse_args()

    eval_cfg = EvalConfig()
    if args.episodes:
        eval_cfg.n_episodes = args.episodes
    if args.seed:
        eval_cfg.seed = args.seed

    evaluate_all(args.model, eval_config=eval_cfg)


if __name__ == "__main__":
    main()
