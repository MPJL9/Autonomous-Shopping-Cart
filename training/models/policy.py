"""
policy.py — MLP policy module shared by BC and PPO training.

Observation (4D): [distance_m, heading_rad, linear_vel_mps, angular_vel_rad_s]
Action space:
  1D case:  forward_speed ∈ [-1, 1]          applied symmetrically as [L, R]=(v, v)
  2D case:  [left, right] ∈ [-1, 1]^2        the raw dashboard joystick format

Observations are normalized using dataset statistics stored with the checkpoint,
so inference at deploy time reproduces the training input distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PolicyConfig:
    obs_dim: int = 4
    action_dim: int = 1          # 1D (forward) or 2D ([L, R])
    hidden_sizes: tuple = (64, 64)
    activation: str = "tanh"     # tanh keeps activations bounded; good for small nets


def _act(name: str) -> nn.Module:
    return {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}[name]()


class PolicyMLP(nn.Module):
    """Deterministic MLP policy. Output is squashed to [-1, 1] by a final Tanh."""

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        sizes = [cfg.obs_dim, *cfg.hidden_sizes, cfg.action_dim]
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(_act(cfg.activation))
        layers.append(nn.Tanh())  # squash to [-1, 1]
        self.net = nn.Sequential(*layers)

        # Normalization stats filled in by training; default identity.
        self.register_buffer("obs_mean", torch.zeros(cfg.obs_dim))
        self.register_buffer("obs_std", torch.ones(cfg.obs_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.obs_mean) / (self.obs_std + 1e-6)
        return self.net(x)

    def set_obs_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.obs_mean.copy_(mean)
        self.obs_std.copy_(std)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Inference helper: accepts a 1D obs tensor, returns 1D action tensor."""
        if obs.ndim == 1:
            return self.forward(obs.unsqueeze(0)).squeeze(0)
        return self.forward(obs)


class StochasticPolicyMLP(nn.Module):
    """Gaussian policy used by PPO. Mean comes from PolicyMLP; log-std is a
    state-independent parameter (standard for continuous-control PPO)."""

    def __init__(self, cfg: PolicyConfig, init_log_std: float = -0.5):
        super().__init__()
        self.mean_net = PolicyMLP(cfg)
        self.log_std = nn.Parameter(torch.full((cfg.action_dim,), init_log_std))

    @property
    def obs_mean(self): return self.mean_net.obs_mean

    @property
    def obs_std(self): return self.mean_net.obs_std

    def set_obs_stats(self, mean, std):
        self.mean_net.set_obs_stats(mean, std)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean_net(obs)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def forward(self, obs):
        return self.mean_net(obs)

    def sample(self, obs, deterministic: bool = False):
        d = self.dist(obs)
        if deterministic:
            a = d.mean
        else:
            a = d.rsample()
        log_prob = d.log_prob(a).sum(dim=-1)
        a = torch.clamp(a, -1.0, 1.0)
        return a, log_prob

    @classmethod
    def from_bc(cls, bc_policy: PolicyMLP, init_log_std: float = -0.5) -> "StochasticPolicyMLP":
        out = cls(bc_policy.cfg, init_log_std=init_log_std)
        out.mean_net.load_state_dict(bc_policy.state_dict())
        return out


class ValueMLP(nn.Module):
    """Value head for PPO. Separate network, same obs normalization as the policy."""

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        sizes = [cfg.obs_dim, *cfg.hidden_sizes, 1]
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(_act(cfg.activation))
        self.net = nn.Sequential(*layers)
        self.register_buffer("obs_mean", torch.zeros(cfg.obs_dim))
        self.register_buffer("obs_std", torch.ones(cfg.obs_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.obs_mean) / (self.obs_std + 1e-6)
        return self.net(x).squeeze(-1)

    def set_obs_stats(self, mean, std):
        self.obs_mean.copy_(mean)
        self.obs_std.copy_(std)
