"""
PPO training pipeline using Stable-Baselines3.

Features:
  - Vectorized environments (SubprocVecEnv) for parallel data collection
  - Periodic model checkpoints
  - TensorBoard logging
  - Reproducible seeds
  - Configurable hyperparameters via config.py
"""

import os
import sys
import argparse
import time
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from environment import GridNavigationEnv
from config import EnvConfig, TrainConfig, TRAIN_CONFIG, ENV_CONFIG, ensure_dirs


def make_env(
    config: EnvConfig, rank: int, seed: int
):
    """
    Factory function for creating a seeded, monitored environment.

    Args:
        config: Environment configuration
        rank: Index of this environment in the vectorized set
        seed: Base random seed
    """
    def _init():
        env = GridNavigationEnv(config=config)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(
    train_config: Optional[TrainConfig] = None,
    env_config: Optional[EnvConfig] = None,
    resume_from: Optional[str] = None,
) -> PPO:
    """
    Train a PPO agent on the GridNavigationEnv.

    Args:
        train_config: Training hyperparameters (defaults to TRAIN_CONFIG)
        env_config: Environment config (defaults to ENV_CONFIG)
        resume_from: Path to a saved model to resume training from

    Returns:
        The trained PPO model
    """
    tc = train_config or TRAIN_CONFIG
    ec = env_config or ENV_CONFIG
    ensure_dirs()

    print("=" * 60)
    print("  PPO Navigation Training")
    print("=" * 60)
    print(f"  Grid size:       {ec.grid_size}x{ec.grid_size}")
    print(f"  Wall density:    {ec.wall_density}")
    print(f"  Dynamic obs:     {ec.num_dynamic_obstacles}")
    print(f"  Total timesteps: {tc.total_timesteps:,}")
    print(f"  Parallel envs:   {tc.n_envs}")
    print(f"  Seed:            {tc.seed}")
    print(f"  Device:          {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)

    # --- Reproducibility ------------------------------------------------------
    set_random_seed(tc.seed)

    # --- Vectorized environments ----------------------------------------------
    env_fns = [make_env(ec, i, tc.seed) for i in range(tc.n_envs)]

    # Use DummyVecEnv on Windows (SubprocVecEnv can have pickling issues)
    if sys.platform == "win32":
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)

    # --- Evaluation environment -----------------------------------------------
    eval_env = DummyVecEnv([make_env(ec, 0, tc.seed + 1000)])

    # --- Callbacks ------------------------------------------------------------
    checkpoint_cb = CheckpointCallback(
        save_freq=max(tc.checkpoint_freq // tc.n_envs, 1),
        save_path=tc.model_dir,
        name_prefix="ppo_nav_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Adaptive evaluation frequency based on total_timesteps
    target_eval_interval = min(tc.checkpoint_freq, max(tc.total_timesteps // 5, 1000))
    eval_freq = max(target_eval_interval // tc.n_envs, 1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=tc.model_dir,
        log_path=tc.log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
    )

    callbacks = CallbackList([checkpoint_cb, eval_cb])

    # --- PPO model ------------------------------------------------------------
    if resume_from and os.path.exists(resume_from):
        print(f"\n  Resuming from: {resume_from}")
        model = PPO.load(resume_from, env=vec_env)
        model.learning_rate = tc.learning_rate
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=tc.learning_rate,
            n_steps=tc.n_steps,
            batch_size=tc.batch_size,
            n_epochs=tc.n_epochs,
            gamma=tc.gamma,
            gae_lambda=tc.gae_lambda,
            clip_range=tc.clip_range,
            ent_coef=tc.ent_coef,
            vf_coef=tc.vf_coef,
            max_grad_norm=tc.max_grad_norm,
            policy_kwargs={"net_arch": tc.net_arch},
            tensorboard_log=tc.log_dir,
            seed=tc.seed,
            verbose=1,
            device="auto",
        )

    # Check progress bar support
    use_progress_bar = False
    try:
        import rich
        import tqdm
        use_progress_bar = True
    except ImportError:
        use_progress_bar = False

    # --- Train ----------------------------------------------------------------
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=tc.total_timesteps,
            callback=callbacks,
            log_interval=tc.log_interval,
            progress_bar=use_progress_bar,
        )
    except Exception:
        # Fallback without progress bar in non-interactive environments
        model.learn(
            total_timesteps=tc.total_timesteps,
            callback=callbacks,
            log_interval=tc.log_interval,
            progress_bar=False,
        )
    elapsed = time.time() - start_time

    # --- Save final model -----------------------------------------------------
    final_path = os.path.join(tc.model_dir, tc.final_model_name)
    model.save(final_path)
    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Final model saved to: {final_path}.zip")

    vec_env.close()
    eval_env.close()

    return model


def main():
    parser = argparse.ArgumentParser(description="Train PPO navigation agent")
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total training timesteps"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed"
    )
    parser.add_argument(
        "--grid-size", type=int, default=None,
        help="Override grid size"
    )
    parser.add_argument(
        "--n-envs", type=int, default=None,
        help="Override number of parallel environments"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to model checkpoint to resume from"
    )
    args = parser.parse_args()

    tc = TrainConfig()
    ec = EnvConfig()

    if args.timesteps is not None:
        tc.total_timesteps = args.timesteps
    if args.seed is not None:
        tc.seed = args.seed
    if args.grid_size is not None:
        ec.grid_size = args.grid_size
    if args.n_envs is not None:
        tc.n_envs = args.n_envs

    train(train_config=tc, env_config=ec, resume_from=args.resume)


if __name__ == "__main__":
    main()
