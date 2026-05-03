#!/usr/bin/env python3
"""
export_policy.py — convert a trained torch checkpoint into a numpy-only npz
that the Pi can load and run without torch.

The Pi's rl_session.py has a `bc_1d` policy that expects the following keys:

  obs_mean  (obs_dim,)      float32
  obs_std   (obs_dim,)      float32
  W0, b0, W1, b1, ..., Wn, bn   weight+bias tensors per Linear layer
  activations   bytes       e.g. b"tanh" per hidden layer
  action_dim    int
  final        bytes        b"tanh" (output squash) or b"none"

The runtime applies: y = Wn @ (... act(W1 @ act(W0 @ x + b0) + b1) ...) + bn
with a final tanh if `final == b"tanh"`.

Usage:
  python3 export_policy.py training/runs/table/bc.pt -o training/runs/table/bc_policy.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models.policy import PolicyConfig, PolicyMLP


def export(ckpt_path: Path, out_path: Path):
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = PolicyConfig(**ck["policy_cfg"])
    policy = PolicyMLP(cfg)
    policy.load_state_dict(ck["policy_state"])
    policy.eval()

    # Walk policy.net in order. Expected pattern:
    # Linear, Act, Linear, Act, ..., Linear, Tanh (output)
    linears = []
    activations = []
    modules = list(policy.net.children())
    for m in modules:
        if isinstance(m, torch.nn.Linear):
            linears.append(m)
        else:
            activations.append(type(m).__name__.lower())

    final_squash = activations[-1] if activations else "none"
    hidden_acts = activations[:-1] if activations else []

    arrs = {
        "obs_mean": policy.obs_mean.detach().cpu().numpy().astype(np.float32),
        "obs_std":  policy.obs_std.detach().cpu().numpy().astype(np.float32),
        "action_dim": np.int32(cfg.action_dim),
        "obs_dim": np.int32(cfg.obs_dim),
        "n_layers": np.int32(len(linears)),
        "final": np.array(final_squash.encode("utf-8")),
        "activations": np.array("|".join(hidden_acts).encode("utf-8")),
    }
    for i, lin in enumerate(linears):
        arrs[f"W{i}"] = lin.weight.detach().cpu().numpy().astype(np.float32)
        arrs[f"b{i}"] = lin.bias.detach().cpu().numpy().astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrs)
    print(f"exported -> {out_path}")
    print(f"  obs_dim={cfg.obs_dim}  action_dim={cfg.action_dim}  layers={len(linears)}")
    print(f"  hidden activations = {hidden_acts}  final = {final_squash}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="torch .pt from train_bc.py / train_ppo_1d.py")
    ap.add_argument("-o", "--out", required=True, help="output .npz path")
    args = ap.parse_args()
    export(Path(args.checkpoint), Path(args.out))


if __name__ == "__main__":
    main()
