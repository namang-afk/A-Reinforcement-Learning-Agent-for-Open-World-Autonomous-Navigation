"""
Test deterministic vs stochastic vs wall-bump breakout
"""
import numpy as np
from stable_baselines3 import PPO
from environment import GridNavigationEnv
from config import EnvConfig

def run_test(mode_name, get_action_fn):
    model = PPO.load("models/best_model.zip", device="cpu")
    env = GridNavigationEnv(config=EnvConfig(grid_size=20, wall_density=0.10, num_dynamic_obstacles=2, max_steps=200))
    successes = 0
    total_steps = []
    
    for ep in range(100):
        obs, _ = env.reset(seed=999 + ep)
        done = False
        steps = 0
        last_pos = None
        bumped = False
        
        while not done:
            action = get_action_fn(model, obs, env, last_pos)
            last_pos = env.agent_pos
            obs, rew, term, trunc, info = env.step(action)
            steps += 1
            done = term or trunc
            
        if info.get("is_success", False):
            successes += 1
        total_steps.append(steps)
        
    print(f"[{mode_name:<25}] Success: {successes}/100 ({successes:.1f}%) | Avg Steps: {np.mean(total_steps):.1f}")

if __name__ == "__main__":
    # 1. Pure deterministic
    run_test("1. Pure Deterministic", lambda m, obs, env, lp: int(m.predict(obs, deterministic=True)[0]))
    
    # 2. Stochastic
    run_test("2. Stochastic Sampling", lambda m, obs, env, lp: int(m.predict(obs, deterministic=False)[0]))
    
    # 3. Deterministic with Bump Breakout (if bumped last step, sample stochastically to escape)
    def bump_breakout(m, obs, env, lp):
        if lp is not None and env.agent_pos == lp:
            return int(m.predict(obs, deterministic=False)[0])
        return int(m.predict(obs, deterministic=True)[0])
    run_test("3. Bump Breakout", bump_breakout)
    
    # 4. Action Masking (never pick an action that leads into a static wall)
    def masked_policy(m, obs, env, lp):
        # Local view is 5x5. Center is (2,2).
        # ACTION_DELTAS = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}
        r, c = env.agent_pos
        valid_actions = []
        for a, (dr, dc) in env.ACTION_DELTAS.items():
            nr, nc = r + dr, c + dc
            if env._in_bounds(nr, nc) and env.grid[nr, nc] != 1:
                valid_actions.append(a)
        
        # Get policy action distribution
        import torch
        obs_tensor = torch.as_tensor(obs).unsqueeze(0)
        dist = m.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().numpy()[0]
        
        # Mask invalid actions
        for a in range(4):
            if a not in valid_actions:
                probs[a] = -1e9
        return int(np.argmax(probs))
        
    run_test("4. Masked Wall Policy", masked_policy)
