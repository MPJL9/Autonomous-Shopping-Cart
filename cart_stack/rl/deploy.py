"""
Load a trained RL policy and run inference for cart control.

Used by the agent to replace the proportional controller in auto mode.

Usage (standalone test):
    python -m cart_stack.rl.deploy --model rl_output/cart_follow_policy.zip
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
except ImportError:
    PPO = None
    VecNormalize = None


class RLController:
    """Wraps a trained PPO policy for inference."""

    def __init__(self, model_path: str | None = None, normalize_path: str | None = None):
        if PPO is None:
            self.ready = False
            return

        if model_path is None:
            model_path = os.getenv("CART_RL_MODEL_PATH", "")
        if normalize_path is None:
            normalize_path = os.getenv("CART_RL_NORMALIZE_PATH", "")

        self.ready = False
        self._model = None
        self._obs_mean = None
        self._obs_var = None
        self._clip_obs = 10.0

        if not model_path or not Path(model_path).exists():
            return

        self._model = PPO.load(model_path)
        self.ready = True

        # Load observation normalization stats if available
        if normalize_path and Path(normalize_path).exists():
            import pickle
            with open(normalize_path, "rb") as f:
                vec_norm = pickle.load(f)
            if hasattr(vec_norm, "obs_rms"):
                self._obs_mean = vec_norm.obs_rms.mean.astype(np.float32)
                self._obs_var = vec_norm.obs_rms.var.astype(np.float32)

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_mean is not None:
            obs = (obs - self._obs_mean) / np.sqrt(self._obs_var + 1e-8)
            obs = np.clip(obs, -self._clip_obs, self._clip_obs)
        return obs

    def predict(self, distance_m: float, heading_rad: float,
                linear_vel: float, angular_vel: float) -> tuple[float, float]:
        """Returns (left_command, right_command) in [-1, 1]."""
        if not self.ready:
            return 0.0, 0.0

        obs = np.array([distance_m, heading_rad, linear_vel, angular_vel], dtype=np.float32)
        obs = self._normalize_obs(obs)
        action, _ = self._model.predict(obs, deterministic=True)
        left = float(np.clip(action[0], -1.0, 1.0))
        right = float(np.clip(action[1], -1.0, 1.0))
        return left, right
