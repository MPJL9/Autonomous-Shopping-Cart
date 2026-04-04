"""
Gym environment for the Autonomous Shopping Cart.

Supports two backends:
  - "sim"    : Uses the built-in physics model (MockRobotBackend style).
               No hardware needed. Fast. Good for initial training.
  - "real"   : Sends commands to the Pi agent via HTTP.
               Requires ArUco vision for distance feedback.

Usage:
    import gymnasium as gym
    from cart_stack.rl.env import CartFollowEnv

    env = CartFollowEnv(backend="sim")        # sim training
    env = CartFollowEnv(backend="real",        # real Pi
                        pi_url="http://172.20.10.4:8001")
"""

from __future__ import annotations

import math
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import httpx
except ImportError:
    httpx = None


class CartFollowEnv(gym.Env):
    """
    Cart person-following environment.

    State:  [distance_m, heading_rad, linear_vel, angular_vel]
    Action: [left_motor, right_motor]  in [-1, 1]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        backend: str = "sim",
        pi_url: str = "http://172.20.10.4:8001",
        target_gap_m: float = 0.65,
        max_steps: int = 500,
        dt: float = 0.1,
    ):
        super().__init__()

        self.backend = backend
        self.pi_url = pi_url.rstrip("/")
        self.target_gap_m = target_gap_m
        self.max_steps = max_steps
        self.dt = dt

        # State: [distance, heading, linear_vel, angular_vel]
        self.observation_space = spaces.Box(
            low=np.array([0.25, -math.pi, -1.0, -5.0], dtype=np.float32),
            high=np.array([3.5, math.pi, 1.0, 5.0], dtype=np.float32),
        )

        # Action: [left_motor, right_motor]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # Sim state
        self._distance = 1.5
        self._heading = 0.0
        self._linear_vel = 0.0
        self._angular_vel = 0.0
        self._step_count = 0
        self._prev_action = np.zeros(2, dtype=np.float32)

        # Sim physics params
        self._max_speed = float(0.9)
        self._track_width = float(0.58)

        # Real backend client
        self._client = None
        if backend == "real":
            if httpx is None:
                raise ImportError("httpx is required for real backend: pip install httpx")
            self._client = httpx.Client(base_url=self.pi_url, timeout=2.0)

    def _get_obs(self) -> np.ndarray:
        return np.array([
            self._distance,
            self._heading,
            self._linear_vel,
            self._angular_vel,
        ], dtype=np.float32)

    def _sim_step(self, left: float, right: float) -> None:
        """Advance the simple physics simulation."""
        target_linear = self._max_speed * (left + right) / 2.0
        target_angular = self._max_speed * (right - left) / max(self._track_width, 0.1)

        # Smooth blending
        blend = min(self.dt * 2.5, 1.0)
        self._linear_vel += (target_linear - self._linear_vel) * blend
        self._angular_vel += (target_angular - self._angular_vel) * blend

        # Update heading and distance
        self._heading += self._angular_vel * self.dt
        self._heading = math.atan2(math.sin(self._heading), math.cos(self._heading))
        self._distance += -self._linear_vel * 0.22 * self.dt
        self._distance = max(0.25, min(3.5, self._distance))

        # Simulate person moving slightly (adds noise/challenge)
        self._distance += np.random.normal(0, 0.005)
        self._heading += np.random.normal(0, 0.01)
        self._distance = max(0.25, min(3.5, self._distance))
        self._heading = math.atan2(math.sin(self._heading), math.cos(self._heading))

    def _real_step(self, left: float, right: float) -> None:
        """Send drive command to Pi and read back state."""
        resp = self._client.post("/api/drive", json={"left": left, "right": right})
        snap = resp.json()
        self._distance = snap["distance_m"]
        self._heading = snap["heading_rad"]
        self._linear_vel = snap["linear_velocity_mps"]
        self._angular_vel = snap["angular_velocity_rad_s"]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._step_count = 0
        self._prev_action = np.zeros(2, dtype=np.float32)

        if self.backend == "sim":
            # Randomize starting conditions
            self._distance = self.np_random.uniform(0.5, 2.5)
            self._heading = self.np_random.uniform(-0.5, 0.5)
            self._linear_vel = 0.0
            self._angular_vel = 0.0
        else:
            # Reset real cart: estop, then arm
            self._client.post("/api/estop", json={"enabled": True})
            time.sleep(0.3)
            self._client.post("/api/estop", json={"enabled": False})
            self._client.post("/api/mode", json={"mode": "manual"})
            self._client.post("/api/vision/tracking", json={"enabled": True})
            time.sleep(0.5)
            # Read initial state
            snap = self._client.get("/api/status").json()
            self._distance = snap["distance_m"]
            self._heading = snap["heading_rad"]
            self._linear_vel = snap["linear_velocity_mps"]
            self._angular_vel = snap["angular_velocity_rad_s"]

        return self._get_obs(), {}

    def step(self, action):
        left = float(np.clip(action[0], -1.0, 1.0))
        right = float(np.clip(action[1], -1.0, 1.0))

        if self.backend == "sim":
            self._sim_step(left, right)
        else:
            self._real_step(left, right)
            time.sleep(self.dt)

        self._step_count += 1

        # --- Reward ---
        dist_error = abs(self._distance - self.target_gap_m)
        heading_error = abs(self._heading)
        action_arr = np.array([left, right], dtype=np.float32)
        jerk = float(np.sum(np.abs(action_arr - self._prev_action)))
        self._prev_action = action_arr

        reward = (
            -2.0 * dist_error          # penalize distance error
            - 1.0 * heading_error       # penalize heading error
            - 0.3 * jerk               # penalize jerky commands
        )

        # Bonus for being close to target gap
        if dist_error < 0.1:
            reward += 0.5
        if dist_error < 0.05 and heading_error < 0.1:
            reward += 1.0

        # --- Termination ---
        terminated = False
        truncated = self._step_count >= self.max_steps

        # End episode if cart gets too close or too far
        if self._distance < 0.28 or self._distance > 3.4:
            terminated = True
            reward -= 5.0

        return self._get_obs(), reward, terminated, truncated, {}

    def close(self):
        if self._client is not None:
            # Stop motors
            try:
                self._client.post("/api/stop")
            except Exception:
                pass
            self._client.close()
