"""
Interactive Web/GUI Dashboard Backend Server for RL Autonomous Navigation.

Features:
  - Serves sleek glassmorphic frontend UI
  - Real-time step-by-step stepping with PPO, A*, Random, or Manual (WASD) control
  - Dynamic procedural map generation and obstacle parameter adjustments
  - Full-episode simulation for client-side animated playback
  - Zero heavy web framework dependencies (uses standard library ThreadingHTTPServer)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

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

import argparse
import json
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

try:
    import numpy as np
    import torch
    from stable_baselines3 import PPO

    from config import EnvConfig, ENV_CONFIG
    from environment import GridNavigationEnv
    from baselines import RandomPolicy, AStarPolicy
except ImportError as err:
    print("\n" + "=" * 65)
    print("  [!] Missing Dependencies Error")
    print("=" * 65)
    print(f"  Failed to import: {err}")
    print("\n  Please activate your virtual environment first:")
    if sys.platform == "win32":
        print("    .\\venv\\Scripts\\activate")
    else:
        print("    source venv/bin/activate")
    print("\n  Or run directly using the environment's python:")
    if sys.platform == "win32":
        print("    .\\venv\\Scripts\\python.exe dashboard.py")
    else:
        print("    ./venv/bin/python dashboard.py")
    print("=" * 65 + "\n")
    sys.exit(1)


# Determine paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "ppo_navigation_final.zip")


class SimulationManager:
    """Manages active environment instances and policy predictions."""

    ACTION_NAMES = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.ppo_model: Optional[PPO] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load PPO model if available
        self._load_ppo_model()

        # Baselines
        self.astar_policy = AStarPolicy()
        self.random_policy = RandomPolicy()

        # Environment
        self.current_config = EnvConfig()
        self.env = GridNavigationEnv(config=self.current_config)
        self.current_obs: Optional[np.ndarray] = None
        self.cum_reward: float = 0.0
        self.seed: int = 42

    def _load_ppo_model(self):
        # Prefer best_model.zip if available, fallback to final or default
        best_path = os.path.join(BASE_DIR, "models", "best_model.zip")
        load_path = best_path if os.path.exists(best_path) else self.model_path

        if os.path.exists(load_path):
            try:
                self.ppo_model = PPO.load(load_path, device="cpu")
                print(f"[Dashboard] Loaded PPO model from: {load_path}")
            except Exception as e:
                print(f"[Dashboard] Warning: Could not load PPO model: {e}")
                self.ppo_model = None
        else:
            print(f"[Dashboard] Notice: No model found at {load_path}. Baselines will be available.")
            self.ppo_model = None

    def reset(
        self,
        grid_size: int = 20,
        wall_density: float = 0.15,
        num_dynamic_obstacles: int = 3,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Reset environment with given parameters."""
        self.current_config = EnvConfig(
            grid_size=grid_size,
            wall_density=wall_density,
            num_dynamic_obstacles=num_dynamic_obstacles,
        )
        self.env = GridNavigationEnv(config=self.current_config)
        self.seed = seed
        self.cum_reward = 0.0
        self._last_agent_pos = None

        self.current_obs, info = self.env.reset(seed=seed)
        self.astar_policy.set_env(self.env)

        return self._format_state(0.0, False, False, False, None)

    def step(self, policy_type: str, manual_action: Optional[int] = None) -> Dict[str, Any]:
        """Take a single step using the specified policy."""
        action = 0

        if policy_type == "manual" and manual_action is not None:
            action = int(manual_action)
        elif policy_type == "astar":
            self.astar_policy.set_env(self.env)
            action, _ = self.astar_policy.predict(self.current_obs)
        elif policy_type == "random":
            action, _ = self.random_policy.predict(self.current_obs)
        else:
            # PPO
            if self.ppo_model is not None:
                # If agent was blocked by a wall in the last step, sample stochastically to detour
                use_det = True
                if self._last_agent_pos is not None and self._last_agent_pos == self.env.agent_pos:
                    use_det = False
                action, _ = self.ppo_model.predict(self.current_obs, deterministic=use_det)
            else:
                # Fallback to random if no PPO model is loaded
                action, _ = self.random_policy.predict(self.current_obs)

        # Convert action to int
        if hasattr(action, "item"):
            action = int(action.item())
        else:
            action = int(action)

        self._last_agent_pos = self.env.agent_pos
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_obs = obs
        self.cum_reward += reward

        # Update A* env binding
        self.astar_policy.set_env(self.env)

        is_success = info.get("is_success", False)
        return self._format_state(reward, terminated, truncated, is_success, action)

    def run_full_episode(
        self,
        policy_type: str,
        grid_size: int = 20,
        wall_density: float = 0.15,
        num_dynamic_obstacles: int = 3,
        seed: int = 42,
        max_steps: int = 200,
    ) -> Dict[str, Any]:
        """Run an entire episode and return all frames."""
        start_state = self.reset(grid_size, wall_density, num_dynamic_obstacles, seed)
        frames = [start_state]

        done = False
        steps = 0
        while not done and steps < max_steps:
            state = self.step(policy_type)
            frames.append(state)
            done = state["terminated"] or state["truncated"]
            steps += 1

        return {"frames": frames}

    def _format_state(
        self,
        reward: float,
        terminated: bool,
        truncated: bool,
        is_success: bool,
        action: Optional[int],
    ) -> Dict[str, Any]:
        """Format current environment state for JSON serialization."""
        view_size = self.env.view_size
        local_view_raw = self.current_obs[: view_size * view_size].tolist() if self.current_obs is not None else []

        action_name = self.ACTION_NAMES.get(action, "--") if action is not None else "--"

        return {
            "grid": self.env.grid.tolist() if self.env.grid is not None else [],
            "grid_size": self.env.grid_size,
            "agent_pos": list(self.env.agent_pos) if self.env.agent_pos else [0, 0],
            "target_pos": list(self.env.target_pos) if self.env.target_pos else [0, 0],
            "dynamic_obstacles": [list(p) for p in self.env.dynamic_obstacles],
            "trajectory": [list(p) for p in self.env.trajectory],
            "local_view": local_view_raw,
            "steps": self.env.steps,
            "reward": float(reward),
            "total_reward": float(self.cum_reward),
            "distance": float(self.env._manhattan_distance(self.env.agent_pos, self.env.target_pos)) if self.env.agent_pos and self.env.target_pos else 0.0,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "is_success": bool(is_success),
            "action": action,
            "action_name": action_name,
        }


# Global Manager
MANAGER = SimulationManager()


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    """Custom request handler supporting REST API endpoints and static files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        """Route GET requests."""
        if self.path == "/api/status":
            self._send_json({
                "status": "online",
                "ppo_available": MANAGER.ppo_model is not None,
                "device": MANAGER.device,
            })
            return

        # Serve static web frontend
        if self.path == "/" or self.path == "":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        """Route POST API requests."""
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"

        try:
            data = json.loads(post_body.decode("utf-8")) if post_body else {}
        except Exception:
            data = {}

        if self.path == "/api/reset":
            grid_size = int(data.get("grid_size", 20))
            wall_density = float(data.get("wall_density", 0.15))
            num_dynamic = int(data.get("num_dynamic_obstacles", 3))
            seed = int(data.get("seed", 42))

            result = MANAGER.reset(grid_size, wall_density, num_dynamic, seed)
            self._send_json(result)

        elif self.path == "/api/step":
            policy = str(data.get("policy", "ppo")).lower()
            action = data.get("action", None)
            result = MANAGER.step(policy, action)
            self._send_json(result)

        elif self.path == "/api/run_episode":
            policy = str(data.get("policy", "ppo")).lower()
            grid_size = int(data.get("grid_size", 20))
            wall_density = float(data.get("wall_density", 0.15))
            num_dynamic = int(data.get("num_dynamic_obstacles", 3))
            seed = int(data.get("seed", 42))

            result = MANAGER.run_full_episode(
                policy_type=policy,
                grid_size=grid_size,
                wall_density=wall_density,
                num_dynamic_obstacles=num_dynamic,
                seed=seed,
            )
            self._send_json(result)

        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def _send_json(self, data: Dict[str, Any], status: int = 200):
        """Send JSON response with appropriate headers."""
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response_bytes)

    def log_message(self, format, *args):
        """Suppress noisy request logs to keep terminal clean."""
        pass


def run_dashboard(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True):
    """Start the dashboard server."""
    server_address = (host, port)
    server = ThreadingHTTPServer(server_address, DashboardRequestHandler)
    # Safe stdout encoding
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    url = f"http://{host}:{port}"
    print("=" * 65)
    print("  [*] Autonomous Navigation RL Dashboard")
    print("=" * 65)
    print(f"  Live Server URL:   {url}")
    print(f"  PPO Model Loaded:  {MANAGER.ppo_model is not None}")
    print(f"  Press Ctrl+C to stop the dashboard server.")
    print("=" * 65)

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Dashboard] Shutting down server gracefully...")
        server.shutdown()
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Run the interactive RL Navigation Dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    parser.add_argument("--model", type=str, default=None, help="Custom path to PPO model .zip")
    args = parser.parse_args()

    if args.model:
        MANAGER.model_path = args.model
        MANAGER._load_ppo_model()

    run_dashboard(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
