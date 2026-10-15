"""
Test Smart Working Memory Policy across 100 unseen maps
"""
import numpy as np
import torch
from collections import deque
from stable_baselines3 import PPO
from environment import GridNavigationEnv
from config import EnvConfig

class SmartWorkingMemoryPolicy:
    """
    Wraps a trained PPO policy with a short-term working memory buffer
    to eliminate:
      1. Wall slamming (attempting to step into known static walls)
      2. 2-cell / 3-cell ping-pong oscillations
      3. Dead-end traps (forces exploration of unvisited corridors)
    """
    def __init__(self, model, history_len: int = 8):
        self.model = model
        self.history_len = history_len
        self.recent_positions = deque(maxlen=history_len)
        self.visit_counts = {}

    def reset(self):
        self.recent_positions.clear()
        self.visit_counts.clear()

    def predict(self, obs, env, deterministic=True):
        r, c = env.agent_pos
        self.recent_positions.append((r, c))
        self.visit_counts[(r, c)] = self.visit_counts.get((r, c), 0) + 1

        # 1. Identify all physically valid candidate actions (no static walls or out of bounds)
        valid_candidates = []
        for action, (dr, dc) in env.ACTION_DELTAS.items():
            nr, nc = r + dr, c + dc
            if env._in_bounds(nr, nc) and env.grid[nr, nc] != 1:
                valid_candidates.append(action)

        if not valid_candidates:
            # Trapped on all sides by walls
            return int(self.model.predict(obs, deterministic=deterministic)[0])

        # 2. Get policy action probabilities from the neural network
        obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.cpu().numpy()[0]

        # 3. Rank valid actions by neural network preference
        ranked_actions = sorted(valid_candidates, key=lambda a: probs[a], reverse=True)

        # 4. Check for oscillation / ping-pong:
        # If the top action moves into the cell visited 1 or 2 steps ago, check if an unvisited valid action exists
        prev_pos_1 = self.recent_positions[-2] if len(self.recent_positions) >= 2 else None
        prev_pos_2 = self.recent_positions[-3] if len(self.recent_positions) >= 3 else None

        # Look for the best action that doesn't immediately reverse into recent positions
        non_reversing = []
        for a in ranked_actions:
            dr, dc = env.ACTION_DELTAS[a]
            nxt = (r + dr, c + dc)
            # Reversal check: avoid immediate ping-pong
            if nxt != prev_pos_1 and nxt != prev_pos_2:
                non_reversing.append(a)

        if non_reversing:
            # We have actions that don't cycle back! Pick the neural network's favorite
            return non_reversing[0]

        # If all valid actions would revisit recent cells, pick the least visited neighbor
        ranked_by_visits = sorted(
            valid_candidates,
            key=lambda a: (self.visit_counts.get((r + env.ACTION_DELTAS[a][0], c + env.ACTION_DELTAS[a][1]), 0), -probs[a])
        )
        return ranked_by_visits[0]


def benchmark_smart_memory(n_episodes=100, seed=123):
    model = PPO.load("models/best_model.zip", device="cpu")
    env = GridNavigationEnv(config=EnvConfig(grid_size=20, wall_density=0.10, num_dynamic_obstacles=2, max_steps=200))
    smart_policy = SmartWorkingMemoryPolicy(model)

    successes = 0
    collisions = 0
    timeouts = 0
    lengths = []
    rewards = []

    print(f"Benchmarking Smart Working Memory Policy across {n_episodes} unseen maps...")
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        smart_policy.reset()
        done = False
        tot_rew = 0.0
        steps = 0

        while not done:
            action = smart_policy.predict(obs, env)
            obs, rew, term, trunc, info = env.step(action)
            tot_rew += rew
            steps += 1
            done = term or trunc

        lengths.append(steps)
        rewards.append(tot_rew)
        if info.get("is_success", False):
            successes += 1
        elif term:
            collisions += 1
        else:
            timeouts += 1

    print("=" * 60)
    print(f"  Smart Working Memory Policy Results ({n_episodes} Episodes)")
    print("=" * 60)
    print(f"  Success Rate:        {successes / n_episodes:.1%}")
    print(f"  Collision Rate:      {collisions / n_episodes:.1%}")
    print(f"  Timeout Rate:        {timeouts / n_episodes:.1%}")
    print(f"  Avg Episode Length:  {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"  Avg Cumulative Rew:  {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_smart_memory(100, seed=123)
