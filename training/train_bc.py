#!/usr/bin/env python3
"""
train_bc.py — behavior cloning on processed RL sessions.

Reads one or more .npz files produced by process_session.py, trains an MLP
policy to map obs -> action via MSE, and saves a checkpoint that includes
observation normalization statistics so inference reproduces the training
distribution.

Config file (YAML) controls everything. Example (training/configs/table.yaml):

    action_dim: 1           # 1 = forward-only (symmetric drive), 2 = [L, R]
    obs_dim: 4
    hidden_sizes: [64, 64]
    data_glob: "training/data/processed/*.npz"
    out_dir: "training/runs/table"
    checkpoint_name: "bc.pt"
    seed: 0
    batch_size: 128
    lr: 0.001
    epochs: 200
    val_frac: 0.15
    device: "auto"          # auto, cpu, mps, cuda

Usage:
    python3 train_bc.py --config configs/table.yaml

For 1D mode the 2D dashboard action [L, R] is projected to forward = (L+R)/2,
which is exactly what the cart would experience on straight-line drives.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models.policy import PolicyConfig, PolicyMLP


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def _seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cap_action_runs_mask(actions: np.ndarray,
                          stop_threshold: float,
                          stop_max_per_run: int | None,
                          action_max_per_run: int | None,
                          rng) -> np.ndarray:
    """Cap contiguous runs of (quantized) identical actions.

    A "run" is a maximal sequence of consecutive frames where the action
    quantized to 0.1 increments is the same across both wheels. For each run:
    - If the run consists of stop frames (max|action| <= stop_threshold),
      keep at most `stop_max_per_run` randomly-chosen frames.
    - Otherwise, keep at most `action_max_per_run` randomly-chosen frames.

    Pass None for either cap to disable (keeps all frames in those runs).
    Operates per session so runs don't cross session boundaries.
    """
    n = len(actions)
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep
    # Quantize to 0.1 so micro-noise doesn't split a sustained command into
    # many tiny runs. (0.4, 0.4) and (0.401, 0.399) collapse to the same run.
    quant = np.round(actions * 10).astype(np.int32)
    is_stop_row = np.max(np.abs(actions), axis=1) <= stop_threshold
    i = 0
    while i < n:
        j = i + 1
        while j < n and np.array_equal(quant[j], quant[i]):
            j += 1
        run_len = j - i
        cap = stop_max_per_run if is_stop_row[i] else action_max_per_run
        if cap is None or cap >= run_len:
            keep[i:j] = True
        else:
            chosen = rng.choice(run_len, size=cap, replace=False) + i
            keep[chosen] = True
        i = j
    return keep


def _load_dataset(data_glob: str, action_dim: int,
                  stop_max_per_run: int | None = None,
                  action_max_per_run: int | None = None,
                  stop_threshold: float = 0.04,
                  mirror_augment: bool = False,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (obs, action, sources). 1D mode projects [L, R] -> (L+R)/2.

    Block-cap downsampling: a "run" is a maximal stretch of frames that share
    the same (quantized-to-0.1) action. For each run:
    - stop runs (max|action| <= stop_threshold) are capped to `stop_max_per_run`
    - active runs are capped to `action_max_per_run`
    Pass None for either cap to disable that side. Runs are computed per
    session so they don't cross session boundaries.

    Capping active runs is what makes a 10-frame "held spin_right" event
    contribute the same number of examples as a 1-frame "spin_left" tap —
    rebalancing the dataset across heading without losing any *distinct*
    correction event the operator demonstrated.
    """
    # data_glob may be a single pattern, a list, or a comma-separated string.
    if isinstance(data_glob, (list, tuple)):
        patterns = list(data_glob)
    elif isinstance(data_glob, str) and "," in data_glob:
        patterns = [p.strip() for p in data_glob.split(",") if p.strip()]
    else:
        patterns = [data_glob]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit(f"No npz files matched: {data_glob}")

    rng = np.random.default_rng(seed)
    do_cap = (
        (isinstance(stop_max_per_run, int) and stop_max_per_run > 0)
        or (isinstance(action_max_per_run, int) and action_max_per_run > 0)
    )

    obs_list, act_list, sources = [], [], []
    counts_in = {"stop": 0, "active": 0}
    counts_out = {"stop": 0, "active": 0}

    for p in paths:
        d = np.load(p, allow_pickle=True)
        o = d["obs"].astype(np.float32)
        raw_a = d["action"].astype(np.float32)

        n_in = len(o)
        is_stop_in = np.max(np.abs(raw_a), axis=1) <= stop_threshold
        counts_in["stop"] += int(is_stop_in.sum())
        counts_in["active"] += int((~is_stop_in).sum())

        if do_cap:
            keep_mask = _cap_action_runs_mask(
                raw_a,
                stop_threshold=stop_threshold,
                stop_max_per_run=stop_max_per_run,
                action_max_per_run=action_max_per_run,
                rng=rng,
            )
            o = o[keep_mask]
            raw_a = raw_a[keep_mask]

        is_stop_out = np.max(np.abs(raw_a), axis=1) <= stop_threshold
        counts_out["stop"] += int(is_stop_out.sum())
        counts_out["active"] += int((~is_stop_out).sum())

        if action_dim == 1:
            a = ((raw_a[:, 0] + raw_a[:, 1]) * 0.5).reshape(-1, 1)
        elif raw_a.shape[1] != action_dim:
            raise ValueError(f"{p} has action dim {raw_a.shape[1]}, config wants {action_dim}")
        else:
            a = raw_a

        obs_list.append(o)
        act_list.append(a)
        sources.append(f"{os.path.basename(p)} (N={len(o)}, was {n_in})")

    obs = np.concatenate(obs_list, axis=0)
    act = np.concatenate(act_list, axis=0)

    if do_cap:
        sources.append(
            f"[stop_max_per_run={stop_max_per_run}, action_max_per_run={action_max_per_run}] "
            f"stop rows: {counts_out['stop']}/{counts_in['stop']}, "
            f"active rows: {counts_out['active']}/{counts_in['active']}, "
            f"final N={len(obs)}"
        )

    if mirror_augment:
        # Cart is left-right symmetric: at heading h, action (L,R) is the right
        # call; by symmetry at heading -h, action (R,L) is the right call. Adding
        # the mirrored copy of every sample doubles the dataset and forces the
        # model to be heading-symmetric, regardless of whether the operator
        # happened to walk to the left or right side more often during recording.
        obs_mirror = obs.copy()
        obs_mirror[:, 1] *= -1.0   # heading
        obs_mirror[:, 3] *= -1.0   # angular velocity
        if action_dim == 2:
            act_mirror = act[:, [1, 0]].copy()  # swap L and R
        else:
            # 1D action is (L+R)/2; symmetric forward speed unchanged under mirror.
            act_mirror = act.copy()
        n_orig = len(obs)
        obs = np.concatenate([obs, obs_mirror], axis=0)
        act = np.concatenate([act, act_mirror], axis=0)
        sources.append(
            f"[mirror_augment=True] doubled {n_orig} -> {len(obs)} rows "
            f"(every example has a heading-flipped twin)"
        )

    return obs, act, sources


def _compute_stats(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = obs.mean(axis=0)
    std = obs.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)  # guard degenerate columns
    return mean.astype(np.float32), std.astype(np.float32)


def train(cfg: dict) -> dict:
    _seed_all(int(cfg.get("seed", 0)))
    device = _resolve_device(cfg.get("device", "auto"))
    action_dim = int(cfg["action_dim"])
    obs_dim = int(cfg["obs_dim"])

    print(f"[bc] device = {device}   obs_dim={obs_dim}  action_dim={action_dim}")

    smpr_raw = cfg.get("stop_max_per_run", None)
    ampr_raw = cfg.get("action_max_per_run", None)
    smpr = int(smpr_raw) if smpr_raw is not None else None
    ampr = int(ampr_raw) if ampr_raw is not None else None
    obs_np, act_np, sources = _load_dataset(
        cfg["data_glob"],
        action_dim,
        stop_max_per_run=smpr,
        action_max_per_run=ampr,
        stop_threshold=float(cfg.get("stop_threshold", 0.04)),
        mirror_augment=bool(cfg.get("mirror_augment", False)),
        seed=int(cfg.get("seed", 0)),
    )
    print("[bc] data sources:")
    for s in sources:
        print(f"       {s}")
    print(f"[bc] total rows: {len(obs_np)}")

    mean, std = _compute_stats(obs_np)
    print(f"[bc] obs mean = {np.round(mean, 3)}")
    print(f"[bc] obs std  = {np.round(std, 3)}")

    obs_t = torch.from_numpy(obs_np)
    act_t = torch.from_numpy(act_np)
    n_total = len(obs_t)
    n_val = max(1, int(n_total * float(cfg.get("val_frac", 0.15))))
    n_train = n_total - n_val
    dataset = TensorDataset(obs_t, act_t)
    gen = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    train_ld = DataLoader(train_ds, batch_size=int(cfg.get("batch_size", 128)), shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=int(cfg.get("batch_size", 128)))

    policy_cfg = PolicyConfig(
        obs_dim=obs_dim, action_dim=action_dim,
        hidden_sizes=tuple(cfg.get("hidden_sizes", [64, 64])),
        activation=cfg.get("activation", "tanh"),
    )
    policy = PolicyMLP(policy_cfg).to(device)
    policy.set_obs_stats(torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device))

    opt = torch.optim.Adam(policy.parameters(), lr=float(cfg.get("lr", 1e-3)))
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = cfg.get("checkpoint_name", "bc.pt")
    best_ckpt = out_dir / ckpt_name
    snapshots_dir = out_dir / "checkpoints"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(cfg.get("epochs", 200))
    # Default: keep ~10 snapshots during training, min every-5 epochs, always last.
    save_every = int(cfg.get("save_every", max(1, epochs // 10)))

    def _ckpt_dict(epoch: int, val_mse: float) -> dict:
        return {
            "policy_state": policy.state_dict(),
            "policy_cfg": asdict(policy_cfg),
            "obs_mean": mean.tolist(),
            "obs_std": std.tolist(),
            "val_mse": val_mse,
            "epoch": epoch,
            "config": cfg,
        }

    history = []
    for epoch in range(1, epochs + 1):
        policy.train()
        tr_loss, tr_n = 0.0, 0
        for xb, yb in train_ld:
            xb = xb.to(device); yb = yb.to(device)
            pred = policy(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_n += xb.size(0)
        tr_mse = tr_loss / max(tr_n, 1)

        policy.eval()
        vl_loss, vl_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_ld:
                xb = xb.to(device); yb = yb.to(device)
                vl_loss += loss_fn(policy(xb), yb).item() * xb.size(0)
                vl_n += xb.size(0)
        vl_mse = vl_loss / max(vl_n, 1)
        history.append({"epoch": epoch, "train_mse": tr_mse, "val_mse": vl_mse})

        improved = vl_mse < best_val
        if improved:
            best_val = vl_mse
            torch.save(_ckpt_dict(epoch, vl_mse), str(best_ckpt))

        is_snapshot = (epoch % save_every == 0) or epoch == epochs
        if is_snapshot:
            snap = snapshots_dir / f"epoch_{epoch:04d}.pt"
            torch.save(_ckpt_dict(epoch, vl_mse), str(snap))

        if epoch % max(1, epochs // 20) == 0 or epoch == 1 or epoch == epochs:
            flags = []
            if improved: flags.append("*")
            if is_snapshot: flags.append("snap")
            print(f"[bc] epoch {epoch:4d}  train_mse={tr_mse:.5f}  val_mse={vl_mse:.5f}"
                  f"  {' '.join(flags)}")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    _plot_loss_curve(history, out_dir / "loss.png", title=f"BC — {out_dir.name}")
    print(f"[bc] best val_mse = {best_val:.5f}  -> {best_ckpt}")
    print(f"[bc] snapshots -> {snapshots_dir}")
    print(f"[bc] loss plot -> {out_dir / 'loss.png'}")
    return {"best_val": best_val, "checkpoint": str(best_ckpt)}


def _plot_loss_curve(history: list[dict], out_path: Path, *, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[bc] matplotlib not installed; skipping loss plot")
        return
    if not history:
        return
    xs = [h["epoch"] for h in history]
    tr = [h["train_mse"] for h in history]
    vl = [h["val_mse"] for h in history]
    best_val = min(vl)
    best_epoch = xs[vl.index(best_val)]

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=110)
    ax.plot(xs, tr, label="train MSE", linewidth=1.5, color="#1f77b4")
    ax.plot(xs, vl, label="val MSE", linewidth=1.5, color="#d62728")
    ax.axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.0,
               label=f"best val ep{best_epoch}  ({best_val:.4f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    # log scale if dynamic range is big
    try:
        if min(tr + vl) > 0 and max(tr + vl) / min(tr + vl) > 50:
            ax.set_yscale("log")
    except ValueError:
        pass
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[],
                    help="Ad-hoc overrides in key=value form, e.g. epochs=50 lr=5e-4.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for kv in args.override:
        k, _, v = kv.partition("=")
        # cast v: try int, float, else str
        for caster in (int, float):
            try:
                cfg[k] = caster(v); break
            except ValueError:
                continue
        else:
            cfg[k] = v

    # Resolve paths relative to the repo root (parent of training/).
    # data_glob may be a comma-separated list of patterns; resolve each.
    repo = Path(__file__).resolve().parent.parent
    if "out_dir" in cfg and not os.path.isabs(cfg["out_dir"]):
        cfg["out_dir"] = str(repo / cfg["out_dir"])
    if "data_glob" in cfg:
        v = cfg["data_glob"]
        if isinstance(v, str):
            parts = [p.strip() for p in v.split(",")]
            cfg["data_glob"] = ", ".join(
                p if os.path.isabs(p) else str(repo / p) for p in parts
            )
        elif isinstance(v, (list, tuple)):
            cfg["data_glob"] = [
                p if os.path.isabs(p) else str(repo / p) for p in v
            ]

    train(cfg)


if __name__ == "__main__":
    main()
