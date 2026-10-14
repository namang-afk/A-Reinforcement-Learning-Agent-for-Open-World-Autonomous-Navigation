"""
Automated test suite for the RL Navigation project.

Tests cover:
  - Environment reset and step API compliance
  - Observation space correctness
  - Reward function behavior
  - Termination conditions (goal, collision, truncation)
  - Map generation and solvability
  - Baseline policies
  - Basic training integration
"""

import numpy as np
import pytest

from environment import GridNavigationEnv
from config import EnvConfig
from baselines import RandomPolicy, AStarPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_env():
    """Create a default environment."""
    env = GridNavigationEnv()
    env.reset(seed=42)
    return env


@pytest.fixture
def small_env():
    """Create a small environment for fast testing."""
    config = EnvConfig(
        grid_size=10,
        wall_density=0.1,
        num_dynamic_obstacles=1,
        max_steps=50,
        min_start_goal_distance=5,
    )
    env = GridNavigationEnv(config=config)
    env.reset(seed=42)
    return env


# ---------------------------------------------------------------------------
# Environment Reset Tests
# ---------------------------------------------------------------------------

class TestEnvironmentReset:
    """Test environment initialization and reset behavior."""

    def test_reset_returns_correct_types(self, default_env):
        """Reset should return (observation, info) tuple."""
        obs, info = default_env.reset(seed=123)
        assert isinstance(obs, np.ndarray), "Observation must be a numpy array"
        assert isinstance(info, dict), "Info must be a dictionary"

    def test_observation_shape(self, default_env):
        """Observation shape must match observation_space."""
        obs, _ = default_env.reset(seed=1)
        expected_shape = default_env.observation_space.shape
        assert obs.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {obs.shape}"
        )

    def test_observation_in_bounds(self, default_env):
        """Observation values must be within observation_space bounds."""
        obs, _ = default_env.reset(seed=1)
        assert default_env.observation_space.contains(obs), (
            "Observation out of bounds"
        )

    def test_agent_on_free_cell(self, default_env):
        """Agent must start on a free cell (not wall or obstacle)."""
        env = default_env
        r, c = env.agent_pos
        assert env.grid[r, c] == 0 or env.grid[r, c] == 3, (
            f"Agent started on cell type {env.grid[r, c]} at ({r}, {c})"
        )

    def test_target_on_grid(self, default_env):
        """Target must be placed on the grid and marked as 3."""
        env = default_env
        r, c = env.target_pos
        assert env.grid[r, c] == 3, (
            f"Target cell should be 3, got {env.grid[r, c]}"
        )

    def test_different_seeds_produce_different_maps(self):
        """Different seeds should produce different maps."""
        env = GridNavigationEnv()
        env.reset(seed=1)
        grid1 = env.grid.copy()
        env.reset(seed=999)
        grid2 = env.grid.copy()
        assert not np.array_equal(grid1, grid2), (
            "Different seeds should produce different maps"
        )

    def test_same_seed_produces_same_map(self):
        """Same seed should produce identical maps."""
        env = GridNavigationEnv()
        env.reset(seed=42)
        grid1 = env.grid.copy()
        pos1 = env.agent_pos
        env.reset(seed=42)
        grid2 = env.grid.copy()
        pos2 = env.agent_pos
        assert np.array_equal(grid1, grid2), "Same seed should produce same map"
        assert pos1 == pos2, "Same seed should produce same start position"

    def test_trajectory_initialized(self, default_env):
        """Trajectory should contain the starting position after reset."""
        assert len(default_env.trajectory) == 1
        assert default_env.trajectory[0] == default_env.agent_pos


# ---------------------------------------------------------------------------
# Step API Tests
# ---------------------------------------------------------------------------

class TestEnvironmentStep:
    """Test step function behavior."""

    def test_step_returns_correct_types(self, small_env):
        """Step should return (obs, reward, terminated, truncated, info)."""
        obs, reward, terminated, truncated, info = small_env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_observation_shape(self, small_env):
        """Observation after step must have correct shape."""
        obs, _, _, _, _ = small_env.step(0)
        assert obs.shape == small_env.observation_space.shape

    def test_valid_actions(self, small_env):
        """All four actions should be accepted."""
        for action in range(4):
            env = GridNavigationEnv(config=EnvConfig(
                grid_size=10, wall_density=0.05, num_dynamic_obstacles=0,
                max_steps=50, min_start_goal_distance=3
            ))
            env.reset(seed=42)
            obs, reward, term, trunc, info = env.step(action)
            assert isinstance(reward, float)

    def test_step_increments_counter(self, small_env):
        """Step counter should increment with each step."""
        initial_steps = small_env.steps
        small_env.step(0)
        assert small_env.steps == initial_steps + 1


# ---------------------------------------------------------------------------
# Reward Function Tests
# ---------------------------------------------------------------------------

class TestRewardFunction:
    """Test reward function behavior."""

    def test_step_penalty_applied(self):
        """Each step should include a step penalty."""
        config = EnvConfig(
            grid_size=10, wall_density=0.0, num_dynamic_obstacles=0,
            max_steps=100, min_start_goal_distance=3,
        )
        env = GridNavigationEnv(config=config)
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(1)  # Down
        # Reward should include the step penalty (-0.1) plus possible progress
        assert reward <= 5.0, "Reward should not be unreasonably high on a normal step"

    def test_goal_reward(self):
        """Reaching the goal should give a large positive reward."""
        config = EnvConfig(
            grid_size=10, wall_density=0.0, num_dynamic_obstacles=0,
            max_steps=100, min_start_goal_distance=3,
        )
        env = GridNavigationEnv(config=config)
        env.reset(seed=42)

        # Manually navigate agent next to goal
        tr, tc_ = env.target_pos
        # Place agent one cell above the target
        candidate_pos = (tr - 1, tc_)
        if (0 <= candidate_pos[0] < env.grid_size and
                env.grid[candidate_pos[0], candidate_pos[1]] == 0):
            env.agent_pos = candidate_pos
            env.prev_dist = env._manhattan_distance(env.agent_pos, env.target_pos)

            # Step down into the target
            _, reward, terminated, _, info = env.step(1)  # Down
            if info.get("is_success", False):
                assert reward > 50.0, "Goal reward should be large and positive"
                assert terminated, "Episode should terminate on goal"


# ---------------------------------------------------------------------------
# Termination Tests
# ---------------------------------------------------------------------------

class TestTermination:
    """Test termination and truncation conditions."""

    def test_truncation_at_max_steps(self):
        """Episode should truncate after max_steps."""
        config = EnvConfig(
            grid_size=10, wall_density=0.0, num_dynamic_obstacles=0,
            max_steps=5, min_start_goal_distance=3,
        )
        env = GridNavigationEnv(config=config)
        env.reset(seed=42)

        for step in range(10):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break

        # Should have been truncated or terminated within max_steps
        assert step <= config.max_steps, (
            f"Episode ran for {step + 1} steps without terminating"
        )

    def test_wall_collision_terminates(self):
        """Hitting a wall should terminate the episode."""
        config = EnvConfig(
            grid_size=10, wall_density=0.0, num_dynamic_obstacles=0,
            max_steps=100, min_start_goal_distance=3,
        )
        env = GridNavigationEnv(config=config)
        env.reset(seed=42)

        # Move agent to the border
        env.agent_pos = (0, 5)  # Top border (wall)
        env.prev_dist = env._manhattan_distance(env.agent_pos, env.target_pos)

        # Try to move up (out of bounds)
        _, reward, terminated, _, _ = env.step(0)  # Up
        assert terminated, "Moving into wall should terminate episode"
        assert reward < 0, "Collision should give negative reward"


# ---------------------------------------------------------------------------
# Map Generation Tests
# ---------------------------------------------------------------------------

class TestMapGeneration:
    """Test procedural map generation properties."""

    def test_border_walls(self, default_env):
        """Borders should all be walls."""
        grid = default_env.grid
        n = default_env.grid_size
        assert np.all(grid[0, :] == 1), "Top border not all walls"
        assert np.all(grid[-1, :] == 1), "Bottom border not all walls"
        assert np.all(grid[:, 0] == 1), "Left border not all walls"
        assert np.all(grid[:, -1] == 1), "Right border not all walls"

    def test_map_contains_target(self, default_env):
        """Grid should contain exactly one target cell."""
        target_count = np.sum(default_env.grid == 3)
        assert target_count == 1, f"Expected 1 target, found {target_count}"

    def test_start_goal_minimum_distance(self):
        """Start and goal should be at least min_distance apart."""
        config = EnvConfig(min_start_goal_distance=8)
        env = GridNavigationEnv(config=config)

        for seed in range(5):
            env.reset(seed=seed)
            dist = env._manhattan_distance(env.agent_pos, env.target_pos)
            assert dist >= config.min_start_goal_distance, (
                f"Seed {seed}: distance {dist} < min {config.min_start_goal_distance}"
            )

    def test_map_is_solvable(self):
        """Generated maps should have a path from start to goal."""
        env = GridNavigationEnv()
        for seed in range(10):
            env.reset(seed=seed)
            # Use BFS to verify
            reachable = env._bfs_reachable(
                env.grid, env.agent_pos, env.target_pos
            )
            assert reachable, f"Map with seed {seed} is not solvable"

    def test_dynamic_obstacles_placed(self, default_env):
        """Dynamic obstacles should be placed on the grid."""
        n_dyn = np.sum(default_env.grid == 2)
        assert n_dyn == len(default_env.dynamic_obstacles), (
            f"Grid has {n_dyn} dynamic obstacles but list has "
            f"{len(default_env.dynamic_obstacles)}"
        )

    def test_high_wall_density_still_solvable(self):
        """Even with high wall density, maps should be solvable."""
        config = EnvConfig(wall_density=0.3, min_start_goal_distance=5)
        env = GridNavigationEnv(config=config)
        for seed in range(5):
            env.reset(seed=seed)
            assert env._bfs_reachable(env.grid, env.agent_pos, env.target_pos)


# ---------------------------------------------------------------------------
# Baseline Tests
# ---------------------------------------------------------------------------

class TestBaselines:
    """Test baseline policy behavior."""

    def test_random_policy_returns_valid_actions(self):
        """Random policy should return actions in {0, 1, 2, 3}."""
        policy = RandomPolicy(seed=0)
        obs = np.zeros(29)  # dummy observation
        for _ in range(100):
            action, _ = policy.predict(obs)
            assert 0 <= action <= 3, f"Invalid action: {action}"

    def test_astar_finds_path(self):
        """A* should find a path on a solvable map."""
        config = EnvConfig(
            grid_size=10, wall_density=0.1, num_dynamic_obstacles=0,
            min_start_goal_distance=3,
        )
        env = GridNavigationEnv(config=config)
        env.reset(seed=42)

        astar = AStarPolicy()
        path = astar._astar(env.grid, env.agent_pos, env.target_pos, env.grid_size)
        assert path is not None, "A* should find a path on a solvable map"
        assert path[0] == env.agent_pos, "Path should start at agent position"
        assert path[-1] == env.target_pos, "Path should end at target position"

    def test_astar_policy_reaches_goal(self):
        """A* policy should reach the goal on a static map."""
        config = EnvConfig(
            grid_size=10, wall_density=0.1, num_dynamic_obstacles=0,
            max_steps=200, min_start_goal_distance=3,
        )
        env = GridNavigationEnv(config=config)
        obs, _ = env.reset(seed=42)

        astar = AStarPolicy()
        astar.set_env(env)

        done = False
        steps = 0
        while not done and steps < 200:
            action, _ = astar.predict(obs)
            obs, _, terminated, truncated, info = env.step(action)
            astar.set_env(env)
            done = terminated or truncated
            steps += 1

        assert info.get("is_success", False), (
            f"A* should reach the goal (steps={steps})"
        )


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests verifying end-to-end behavior."""

    def test_full_episode_runs(self):
        """A full episode should run without errors."""
        env = GridNavigationEnv()
        obs, info = env.reset(seed=42)

        total_reward = 0
        done = False
        steps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        assert steps > 0, "Episode should run at least one step"
        assert isinstance(total_reward, float)

    def test_gymnasium_check_env(self):
        """Environment should pass Gymnasium's built-in checker."""
        from gymnasium.utils.env_checker import check_env
        env = GridNavigationEnv()
        # check_env will raise if something is wrong
        try:
            check_env(env.unwrapped, skip_render_check=True)
        except Exception as e:
            pytest.fail(f"check_env failed: {e}")

    def test_render_returns_array(self):
        """Render should return an RGB numpy array."""
        env = GridNavigationEnv(render_mode="rgb_array")
        env.reset(seed=42)
        img = env.render()
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3  # RGB


# ---------------------------------------------------------------------------
# Observation Tests
# ---------------------------------------------------------------------------

class TestObservations:
    """Test observation construction."""

    def test_local_view_size(self, default_env):
        """Local view portion of observation should be view_size^2."""
        obs, _ = default_env.reset(seed=1)
        view_elements = default_env.view_size ** 2
        local_view = obs[:view_elements]
        assert len(local_view) == view_elements

    def test_normalized_position_range(self, default_env):
        """Normalized positions should be in [0, 1]."""
        obs, _ = default_env.reset(seed=1)
        view_elements = default_env.view_size ** 2
        norm_pos = obs[view_elements:view_elements + 2]
        assert np.all(norm_pos >= 0.0) and np.all(norm_pos <= 1.0), (
            f"Normalized position out of range: {norm_pos}"
        )

    def test_relative_target_reasonable(self, default_env):
        """Relative target direction should be within [-1, 1]."""
        obs, _ = default_env.reset(seed=1)
        view_elements = default_env.view_size ** 2
        rel_target = obs[view_elements + 2:view_elements + 4]
        assert np.all(rel_target >= -1.0) and np.all(rel_target <= 1.0), (
            f"Relative target out of range: {rel_target}"
        )
