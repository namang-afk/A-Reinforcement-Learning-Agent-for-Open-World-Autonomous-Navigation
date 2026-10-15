"""
Comprehensive Model Analyzer for RL Open-World Autonomous Navigation.

Inspects all trained checkpoints and models in models/:
  1. Architecture & parameter count analysis (Policy net, Value net, Feature Extractor)
  2. Hyperparameters & training configuration inspection
  3. Empirical benchmarking across identical test seeds:
     - Success rate (%)
     - Collision rate (%)
     - Timeout rate (%)
     - Average episode length
     - Average cumulative reward
  4. Generates comparison tables and saves results/model_analysis.json
"""

import os
import sys
import json
import zipfile
import time
from typing import Dict, Any, List
import numpy as np

# Auto-relaunch inside local venv if executed from system Python
def _ensure_venv():
    if hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix):
        return
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = (
        os.path.join(base_dir, "venv", "Scripts", "python.exe")
        if sys.platform == "win32"
        else os.path.join(base_dir, "venv", "bin", "python")
    )
    if os.path.exists(venv_python) and os.path.abspath(sys.executable) != os.path.abspath(venv_python):
        import subprocess
        try:
            res = subprocess.run([venv_python] + sys.argv)
            sys.exit(res.returncode)
        except Exception:
            pass

_ensure_venv()

import torch
from stable_baselines3 import PPO
from environment import GridNavigationEnv
from config import EnvConfig, ensure_dirs


def get_model_metadata(zip_path: str) -> Dict[str, Any]:
    """Extract metadata, parameters, and network architecture from an SB3 zip file."""
    meta = {
        "filename": os.path.basename(zip_path),
        "size_kb": round(os.path.getsize(zip_path) / 1024, 1),
        "last_modified": time.ctime(os.path.getmtime(zip_path)),
        "net_arch": None,
        "total_params": 0,
        "policy_params": 0,
        "value_params": 0,
        "gamma": None,
        "learning_rate": None,
        "ent_coef": None,
        "clip_range": None,
    }
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            if 'data' in z.namelist():
                data = json.loads(z.read('data').decode('utf-8'))
                meta["gamma"] = data.get("gamma")
                meta["learning_rate"] = data.get("learning_rate")
                meta["ent_coef"] = data.get("ent_coef")
                meta["clip_range"] = data.get("clip_range")
                policy_kwargs = data.get("policy_kwargs", {})
                if policy_kwargs and "net_arch" in policy_kwargs:
                    meta["net_arch"] = policy_kwargs["net_arch"]
    except Exception as e:
        meta["error_meta"] = str(e)

    # Inspect PyTorch weights
    try:
        model = PPO.load(zip_path, device="cpu")
        policy = model.policy
        total_p = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        meta["total_params"] = total_p
        
        # Policy & Value params
        p_count = sum(p.numel() for p in policy.mlp_extractor.policy_net.parameters())
        p_count += sum(p.numel() for p in policy.action_net.parameters())
        v_count = sum(p.numel() for p in policy.mlp_extractor.value_net.parameters())
        v_count += sum(p.numel() for p in policy.value_net.parameters())
        meta["policy_params"] = p_count
        meta["value_params"] = v_count
        
        # If net_arch was not found in data JSON, infer from weights
        if not meta["net_arch"]:
            arch = []
            for layer in policy.mlp_extractor.policy_net:
                if isinstance(layer, torch.nn.Linear):
                    arch.append(layer.out_features)
            meta["net_arch"] = arch
    except Exception as e:
        meta["load_error"] = str(e)

    return meta


def benchmark_model(model_path: str, n_episodes: int = 30, seed: int = 999) -> Dict[str, Any]:
    """Benchmark a model checkpoint on unseen procedural environments using evaluate.py's pipeline."""
    from evaluate import run_episode
    env_config = EnvConfig(grid_size=20, wall_density=0.10, num_dynamic_obstacles=2, max_steps=200)
    env = GridNavigationEnv(config=env_config)
    
    try:
        model = PPO.load(model_path, env=env, device="cpu")
    except Exception as e:
        return {"error": f"Failed to load: {e}"}

    successes = 0
    collisions = 0
    timeouts = 0
    ep_lengths = []
    ep_rewards = []
    efficiencies = []

    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        res = run_episode(env, model, is_astar=False, deterministic=True)
        
        ep_lengths.append(res.episode_length)
        ep_rewards.append(res.total_reward)
        if res.path_efficiency is not None:
            efficiencies.append(res.path_efficiency)

        if res.success:
            successes += 1
        elif res.collided:
            collisions += 1
        else:
            timeouts += 1

    avg_eff = round(float(np.mean(efficiencies)) * 100, 1) if efficiencies else 0.0

    return {
        "episodes": n_episodes,
        "success_rate": round(successes / n_episodes * 100, 1),
        "collision_rate": round(collisions / n_episodes * 100, 1),
        "timeout_rate": round(timeouts / n_episodes * 100, 1),
        "path_efficiency": avg_eff,
        "avg_length": round(float(np.mean(ep_lengths)), 1),
        "std_length": round(float(np.std(ep_lengths)), 1),
        "avg_reward": round(float(np.mean(ep_rewards)), 1),
        "std_reward": round(float(np.std(ep_rewards)), 1),
    }


def analyze_all_models(models_dir: str = "models", n_eval_episodes: int = 20):
    """Scan models_dir, extract metadata, benchmark each, and output comparison."""
    ensure_dirs()
    if not os.path.exists(models_dir):
        print(f"Error: {models_dir} not found.")
        return

    files = [f for f in os.listdir(models_dir) if f.endswith(".zip")]
    if not files:
        print("No .zip model files found in models/")
        return

    # Sort files logically: checkpoints by step number, then best_model and final
    def sort_key(name):
        if "checkpoint" in name:
            try:
                num = int(name.split("_")[3])
                return (1, num)
            except Exception:
                return (1, 0)
        elif "best" in name:
            return (2, 0)
        elif "final" in name:
            return (3, 0)
        return (0, name)

    files = sorted(files, key=sort_key)

    print("=" * 80)
    print("  MODEL DIRECTORY & CHECKPOINT ANALYSIS")
    print(f"  Scanning directory: {os.path.abspath(models_dir)}")
    print(f"  Total models found: {len(files)}")
    print(f"  Evaluation episodes per model: {n_eval_episodes} (Seed: 999)")
    print("=" * 80)

    results = []

    for idx, f in enumerate(files, 1):
        zip_path = os.path.join(models_dir, f)
        print(f"\n[{idx}/{len(files)}] Inspecting {f}...")
        meta = get_model_metadata(zip_path)
        print(f"   Size: {meta['size_kb']} KB | Net Arch: {meta.get('net_arch')} | Trainable Params: {meta.get('total_params'):,}")
        
        print(f"   Benchmarking on {n_eval_episodes} unseen maps...")
        perf = benchmark_model(zip_path, n_episodes=n_eval_episodes, seed=999)
        
        if "error" in perf:
            print(f"   Benchmark error: {perf['error']}")
        else:
            print(f"   -> Success: {perf['success_rate']}% | Collisions: {perf['collision_rate']}% | Timeouts: {perf['timeout_rate']}% | Avg Rew: {perf['avg_reward']}")

        combined = {**meta, "performance": perf}
        results.append(combined)

    # Save to JSON
    out_json = os.path.join("results", "model_analysis.json")
    with open(out_json, "w") as jf:
        json.dump(results, jf, indent=2)
    print(f"\nSaved analysis results to: {out_json}")

    # Print summary markdown table
    print("\n" + "=" * 110)
    print("  SUMMARY BENCHMARK TABLE")
    print("=" * 110)
    print(f"{'Model Checkpoint':<35} | {'Arch':<10} | {'Params':<8} | {'Success':<8} | {'Collision':<9} | {'Timeouts':<8} | {'Path Eff':<9} | {'Avg Steps':<10} | {'Avg Rew':<8}")
    print("-" * 110)
    for r in results:
        p = r.get("performance", {})
        arch_str = str(r.get("net_arch")) if r.get("net_arch") else "N/A"
        if len(arch_str) > 10:
            arch_str = arch_str[:9] + "…"
        succ_str = f"{p.get('success_rate', 0)}%" if "success_rate" in p else "ERR"
        coll_str = f"{p.get('collision_rate', 0)}%" if "collision_rate" in p else "ERR"
        tout_str = f"{p.get('timeout_rate', 0)}%" if "timeout_rate" in p else "ERR"
        eff_str = f"{p.get('path_efficiency', 0)}%" if "path_efficiency" in p else "N/A"
        step_str = f"{p.get('avg_length', 0)}" if "avg_length" in p else "ERR"
        rew_str = f"{p.get('avg_reward', 0)}" if "avg_reward" in p else "ERR"
        params_str = f"{r.get('total_params', 0):,}"
        print(f"{r['filename']:<35} | {arch_str:<10} | {params_str:<8} | {succ_str:<8} | {coll_str:<9} | {tout_str:<8} | {eff_str:<9} | {step_str:<10} | {rew_str:<8}")
    print("=" * 110)


if __name__ == "__main__":
    analyze_all_models()
