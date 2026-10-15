"""
Baseline policies for comparison against the trained PPO agent.

Provides:
  1. RandomPolicy — uniformly random actions
  2. AStarPolicy — optimal pathfinding using A* on the current grid
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np

from environment import GridNavigationEnv


class RandomPolicy:
    """
    Uniformly random baseline: selects actions from {0,1,2,3} with equal probability.

    This establishes a lower bound on performance.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)

    def predict(self, obs: np.ndarray, **kwargs) -> Tuple[int, None]:
        """Return a random action (matching SB3's model.predict interface)."""
        action = self.rng.randint(0, 4)
        return action, None

    def __repr__(self):
        return "RandomPolicy"


class AStarPolicy:
    """
    A* pathfinding baseline: computes the shortest path on the *static* grid
    and follows it step by step.

    Notes:
        - Uses full grid knowledge (not partial observations), so this is an
          *oracle* baseline, not a fair comparison for generalization.
        - Recomputes the path when dynamic obstacles invalidate the current plan.
        - Serves as an approximate upper bound on path efficiency.
    """

    def __init__(self):
        self.path: List[Tuple[int, int]] = []
        self.path_index: int = 0
        self.current_env: Optional[GridNavigationEnv] = None

    def set_env(self, env: GridNavigationEnv) -> None:
        """Bind the policy to an environment instance for grid access."""
        self.current_env = env
        self._compute_path()

    def _compute_path(self) -> None:
        """Run A* from agent_pos to target_pos on the current grid."""
        env = self.current_env
        if env is None or env.grid is None:
            self.path = []
            self.path_index = 0
            return

        start = env.agent_pos
        goal = env.target_pos
        grid = env.grid

        path = self._astar(grid, start, goal, env.grid_size)
        if path is not None:
            self.path = path
            self.path_index = 1  # index 0 is the starting position
        else:
            self.path = []
            self.path_index = 0

    def predict(self, obs: np.ndarray, **kwargs) -> Tuple[int, None]:
        """
        Return the next action along the A* path.

        Falls back to a random action if no path exists or the path is exhausted.
        """
        env = self.current_env

        # Recompute if path is invalid or exhausted
        if (
            not self.path
            or self.path_index >= len(self.path)
            or (env is not None and env.agent_pos != self.path[self.path_index - 1])
        ):
            self._compute_path()

        if not self.path or self.path_index >= len(self.path):
            # Fallback: random action
            return np.random.randint(0, 4), None

        next_pos = self.path[self.path_index]
        self.path_index += 1

        # Convert position delta to action
        if env is not None:
            dr = next_pos[0] - env.agent_pos[0]
            dc = next_pos[1] - env.agent_pos[1]
            action = self._delta_to_action(dr, dc)
            return action, None

        return np.random.randint(0, 4), None

    @staticmethod
    def _delta_to_action(dr: int, dc: int) -> int:
        """Convert (row_delta, col_delta) to action index."""
        if dr == -1 and dc == 0:
            return 0  # Up
        elif dr == 1 and dc == 0:
            return 1  # Down
        elif dr == 0 and dc == -1:
            return 2  # Left
        elif dr == 0 and dc == 1:
            return 3  # Right
        else:
            return 0  # fallback

    @staticmethod
    def _astar(
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid_size: int,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        A* pathfinding on the grid.

        Args:
            grid: 2D numpy array (0=free, 1=wall, 2=dynamic obs, 3=target)
            start: (row, col) start position
            goal: (row, col) goal position
            grid_size: size of the grid

        Returns:
            List of (row, col) positions from start to goal, or None if unreachable
        """
        # Cells that block movement: walls only. Dynamic obstacles are avoided
        # but we treat them as passable (they move, and A* recomputes).
        blocked = {1}  # wall cells block; 2 (dynamic) treated as passable for planning

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}

        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for dr, dc in deltas:
                nr, nc = current[0] + dr, current[1] + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    cell_val = grid[nr, nc]
                    if cell_val in blocked:
                        continue

                    tentative_g = g_score[current] + 1
                    if tentative_g < g_score.get((nr, nc), float("inf")):
                        came_from[(nr, nc)] = current
                        g_score[(nr, nc)] = tentative_g
                        f_score = tentative_g + abs(nr - goal[0]) + abs(nc - goal[1])
                        heapq.heappush(open_set, (f_score, (nr, nc)))

        return None  # No path found

    def __repr__(self):
        return "AStarPolicy"


class SmartNavigationPolicy:
    """
    Augments the trained PPO policy with short-term working memory to eliminate:
      1. Static wall bumping (filters out actions into known static walls)
      2. 2-cell / 3-cell ping-pong oscillations (prevents immediate reversal into recent cells)
      3. Dead-end traps (forces exploration of alternative open corridors)
    """

    def __init__(self, model: Any, history_len: int = 16):
        self.model = model
        self.history_len = history_len
        from collections import deque
        self.recent_positions: deque = deque(maxlen=history_len)
        self.visit_counts: Dict[Tuple[int, int], int] = {}
        self.env: Optional[GridNavigationEnv] = None

    def set_env(self, env: GridNavigationEnv):
        self.env = env

    def reset(self):
        self.recent_positions.clear()
        self.visit_counts.clear()

    def predict(
        self, obs: np.ndarray, deterministic: bool = True, **kwargs
    ) -> Tuple[int, None]:
        env = kwargs.get("env", self.env)
        if env is None or not hasattr(env, "agent_pos") or env.grid is None:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            return int(action), None

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
            action, _ = self.model.predict(obs, deterministic=deterministic)
            return int(action), None

        # 2. Extract policy action probabilities from the neural network
        try:
            import torch
            obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.cpu().numpy()[0]
        except Exception:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            return int(action), None

        # 3. Separate candidate actions into unvisited (fresh) vs already in recent history
        unvisited = []
        for a in valid_candidates:
            dr, dc = env.ACTION_DELTAS[a]
            nxt = (r + dr, c + dc)
            if nxt not in self.recent_positions:
                unvisited.append(a)

        # 4. If any unvisited direction exists, choose the one preferred by the neural network!
        # This completely breaks 2-cell, 3-cell, 4-cell square loops or cycles of ANY shape!
        if unvisited:
            unvisited.sort(key=lambda a: probs[a], reverse=True)
            return int(unvisited[0]), None

        # 5. If all neighbors have been visited recently (e.g. backtracking out of a dead-end),
        # prioritize the cell with the lowest total visits across the episode, breaking ties with PPO probability!
        ranked_by_visits = sorted(
            valid_candidates,
            key=lambda a: (
                self.visit_counts.get((r + env.ACTION_DELTAS[a][0], c + env.ACTION_DELTAS[a][1]), 0),
                -probs[a],
            ),
        )
        return int(ranked_by_visits[0]), None

    def __repr__(self):
        return "SmartNavigationPolicy"
