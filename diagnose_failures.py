"""
Diagnostic script to inspect failure cases of best_model.zip
"""
import os
import sys
import numpy as np
from stable_baselines3 import PPO
from environment import GridNavigationEnv
from config import EnvConfig

def diagnose():
    model = PPO.load("models/best_model.zip", device="cpu")
    env = GridNavigationEnv(config=EnvConfig(grid_size=20, wall_density=0.10, num_dynamic_obstacles=2, max_steps=200))
    
    print("Running diagnostic on 10 episodes...")
    for ep in range(10):
        obs, _ = env.reset(seed=999 + ep)
        done = False
        steps = 0
        wall_bumps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            prev_pos = env.agent_pos
            obs, rew, term, trunc, info = env.step(int(action))
            steps += 1
            if env.agent_pos == prev_pos:
                wall_bumps += 1
            done = term or trunc
            
        success = info.get("is_success", False)
        status = "SUCCESS" if success else ("COLLISION" if term else "TIMEOUT")
        unique_cells = len(set(env.trajectory))
        print(f"Ep {ep+1:02d}: {status:<9} | Steps: {steps:3d} | Unique cells: {unique_cells:2d} | Wall bumps: {wall_bumps:3d} | Start: {env.trajectory[0]} -> End: {env.agent_pos} -> Goal: {env.target_pos}")

if __name__ == "__main__":
    diagnose()
