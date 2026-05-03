"""Behavior cloning from RL session JSONL logs.

Trains a small MLP to map observation -> action using supervised learning
(MSE loss). Skips steps where the human wasn't actively commanding a drive
(action in stop deadband).

Usage:
    # Train on one file
    python -m cart_stack.rl.train_bc rl_logs/rl_manual_20260418.jsonl

    # Train on all manual logs in a directory
    python -m cart_stack.rl.train_bc rl_logs/

    # With options
    python -m cart_stack.rl.train_bc rl_logs/ --epochs 80 --hidden 128 \
        --out rl_output/bc_policy.pt

Output:
    rl_output/bc_policy.pt       state_dict + arch info
    rl_output/bc_norm.npz        obs mean/std for normalization

Load at inference with `BCController` from cart_stack.rl.deploy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


OBS_DIM = 4
ACT_DIM = 2


@dataclass
class Dataset:
    obs: np.ndarray   # (N, 4)
    act: np.ndarray   # (N, 2)


def load_jsonl_dataset(paths: list[Path], action_deadband: float = 0.02) -> Dataset:
    obs_list: list[list[float]] = []
    act_list: list[list[float]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("type") != "step":
                    continue
                action = record["action"]
                # Skip near-zero actions — they're mostly idle frames where
                # the human wasn't actually commanding anything. Including
                # them biases BC toward "do nothing."
                if max(abs(action[0]), abs(action[1])) < action_deadband:
                    continue
                obs_list.append(record["obs"])
                act_list.append(action)
    if not obs_list:
        raise ValueError("No usable (obs, action) pairs found. Did you record with policy=manual?")
    return Dataset(
        obs=np.asarray(obs_list, dtype=np.float32),
        act=np.asarray(act_list, dtype=np.float32),
    )


class BCPolicy(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, ACT_DIM),
            nn.Tanh(),  # clip into [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def resolve_inputs(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted(p.glob("rl_*.jsonl")))
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(item)
    if not out:
        raise FileNotFoundError("No JSONL files found.")
    return out


def train(args: argparse.Namespace) -> None:
    paths = resolve_inputs(args.inputs)
    print(f"Loading {len(paths)} file(s)...")
    ds = load_jsonl_dataset(paths, action_deadband=args.action_deadband)
    print(f"  {len(ds.obs)} (obs, action) pairs after filtering.")

    # Normalize observations.
    obs_mean = ds.obs.mean(axis=0)
    obs_std = ds.obs.std(axis=0) + 1e-6
    obs_norm = (ds.obs - obs_mean) / obs_std

    # 90/10 train/val split.
    n = len(obs_norm)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    n_val = max(1, n // 10)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    tr_x = torch.from_numpy(obs_norm[tr_idx])
    tr_y = torch.from_numpy(ds.act[tr_idx])
    va_x = torch.from_numpy(obs_norm[val_idx])
    va_y = torch.from_numpy(ds.act[val_idx])

    loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=args.batch_size, shuffle=True)

    model = BCPolicy(hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(f"Training {args.epochs} epochs, hidden={args.hidden}, lr={args.lr}, batch={args.batch_size}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(xb)
        tr_loss = total / len(tr_x)

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(va_x), va_y))

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  train_loss={tr_loss:.5f}  val_loss={val_loss:.5f}")

    out_model = Path(args.out)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden": args.hidden,
            "obs_dim": OBS_DIM,
            "act_dim": ACT_DIM,
        },
        out_model,
    )
    norm_path = out_model.with_suffix(".norm.npz")
    np.savez(norm_path, obs_mean=obs_mean, obs_std=obs_std)

    print(f"\nSaved model → {out_model}")
    print(f"Saved normalization → {norm_path}")
    print("\nLoad for inference with:")
    print("    from cart_stack.rl.deploy import BCController")
    print(f"    ctrl = BCController('{out_model}', '{norm_path}')")
    print("    ctrl.predict(distance_m, heading_rad, linear_vel, angular_vel)")


def main() -> None:
    p = argparse.ArgumentParser(description="Behavior-cloning trainer for cart RL logs.")
    p.add_argument("inputs", nargs="+", help="JSONL file(s) or directory containing rl_*.jsonl")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--action-deadband", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="rl_output/bc_policy.pt")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
