#!/usr/bin/env python3
"""train_awr.py — Advantage-Weighted Regression on a recorded RL session.

Pipeline (offline RL, single-trajectory variant):
  1. Load a session JSONL (Pi-side log) -> (obs, action) per step.
  2. Compute per-step reward using the cost formula in EVALUATION.md
     (negated): r = -[w_d·miss² + w_θ·θ² + w_j·||Δa||²]. Band defaults to
     the 4/29-era [1.26, 1.56] m, matching the runtime deadband bc_2d_pivot
     was trained against.
  3. Compute Monte-Carlo returns G_t = Σ_{k=0..N-t} γ^k r_{t+k}.
  4. Fit a small value MLP V_φ(s) on (obs_t, G_t) via MSE.
  5. Advantage A_t = G_t - V_φ(obs_t). Weight w_t = min(exp(A_t/β), w_max),
     then normalize so mean(w) = 1.
  6. Warm-start a policy MLP from a BC checkpoint (default: bc_2d) and
     fine-tune via per-row-weighted MSE on (obs_t, action_t).

Output: a torch checkpoint shaped like train_bc.py's bc.pt, plus a
matching exportable bc_policy.npz (call training/export_policy.py to make
the runtime-loadable npz).

Usage:
  python3 training/train_awr.py \
      --session "RL data/5:2/bc_2d_tier_1/rl_bc_2d_pivot_20260502-165737.jsonl" \
      --warm-start training/runs/4_29_2d_no_mirror/bc.pt \
      --out-dir training/runs/awr_bc_2d
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models.policy import PolicyConfig, PolicyMLP

# ---- reward formula (matches evaluate_policy.py / EVALUATION.md) ----
W_DISTANCE = 1.0
W_HEADING  = 0.5
W_JERK     = 0.1
DEADBAND_LO_M = 1.26   # 4/29-era; bc_2d / bc_2d_pivot's training equilibrium
DEADBAND_HI_M = 1.56


def load_session(jsonl_path: str) -> dict:
    obs, act = [], []
    header = None
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "header":
                header = r
            elif r.get("type") == "step":
                obs.append(r["obs"])
                act.append(r["action"])
    obs = np.asarray(obs, dtype=np.float32)
    act = np.asarray(act, dtype=np.float32)
    return {"header": header, "obs": obs, "action": act}


def compute_rewards(obs: np.ndarray, action: np.ndarray,
                    band_lo: float, band_hi: float,
                    w_d: float, w_h: float, w_j: float) -> np.ndarray:
    """Per-step cost from EVALUATION.md, negated."""
    d = obs[:, 0]; theta = obs[:, 1]
    miss = np.maximum.reduce([np.zeros_like(d), band_lo - d, d - band_hi])
    distance_term = w_d * (miss ** 2)
    heading_term = w_h * (theta ** 2)
    da = np.zeros_like(action)
    da[1:] = action[1:] - action[:-1]
    jerk_term = w_j * np.sum(da * da, axis=1)
    return -(distance_term + heading_term + jerk_term)


def mc_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """G_t = Σ γ^k r_{t+k} from t to T-1 (single-trajectory)."""
    n = len(rewards)
    g = np.zeros(n, dtype=np.float32)
    running = 0.0
    for t in range(n - 1, -1, -1):
        running = float(rewards[t]) + gamma * running
        g[t] = running
    return g


class ValueMLP(nn.Module):
    def __init__(self, obs_dim: int = 4, hidden: tuple = (32, 32),
                 obs_mean: np.ndarray | None = None,
                 obs_std: np.ndarray | None = None):
        super().__init__()
        sizes = [obs_dim, *hidden, 1]
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        if obs_mean is None: obs_mean = np.zeros(obs_dim, dtype=np.float32)
        if obs_std  is None: obs_std  = np.ones(obs_dim,  dtype=np.float32)
        self.register_buffer("obs_mean", torch.from_numpy(obs_mean.astype(np.float32)))
        self.register_buffer("obs_std",  torch.from_numpy(obs_std.astype(np.float32)))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.obs_mean) / (self.obs_std + 1e-6)
        return self.net(x).squeeze(-1)


def fit_value(obs: np.ndarray, returns: np.ndarray,
              obs_mean: np.ndarray, obs_std: np.ndarray,
              epochs: int = 200, lr: float = 1e-3, batch_size: int = 64,
              device: torch.device = torch.device("cpu"),
              verbose: bool = True) -> ValueMLP:
    v = ValueMLP(obs.shape[1], (32, 32), obs_mean, obs_std).to(device)
    opt = torch.optim.Adam(v.parameters(), lr=lr)
    obs_t = torch.from_numpy(obs).to(device)
    g_t = torch.from_numpy(returns.astype(np.float32)).to(device)
    n = len(obs_t)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred = v(obs_t[idx])
            loss = ((pred - g_t[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n
        if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
            print(f"  [v] epoch {epoch:3d}  mse={epoch_loss:.5f}")
    return v


def awr_weights(advantages: np.ndarray, beta: float, w_max: float
                ) -> np.ndarray:
    a = advantages.astype(np.float64)
    w = np.exp(a / max(beta, 1e-6))
    w = np.minimum(w, w_max)
    w = w / max(w.mean(), 1e-9)
    return w.astype(np.float32)


def fit_policy(obs: np.ndarray, action: np.ndarray, weights: np.ndarray,
               warm_ckpt_path: Path | None,
               obs_mean: np.ndarray, obs_std: np.ndarray,
               epochs: int = 100, lr: float = 5e-5, batch_size: int = 64,
               device: torch.device = torch.device("cpu"),
               verbose: bool = True) -> tuple[PolicyMLP, dict]:
    cfg = PolicyConfig(obs_dim=obs.shape[1], action_dim=action.shape[1],
                       hidden_sizes=(64, 64), activation="tanh")
    policy = PolicyMLP(cfg).to(device)
    if warm_ckpt_path is not None and warm_ckpt_path.exists():
        ck = torch.load(str(warm_ckpt_path), map_location=device, weights_only=False)
        policy.load_state_dict(ck["policy_state"])
        if verbose:
            print(f"  [π] warm-started from {warm_ckpt_path}")
    policy.set_obs_stats(torch.from_numpy(obs_mean.astype(np.float32)).to(device),
                         torch.from_numpy(obs_std.astype(np.float32)).to(device))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    obs_t = torch.from_numpy(obs).to(device)
    act_t = torch.from_numpy(action).to(device)
    w_t = torch.from_numpy(weights).to(device)
    n = len(obs_t)
    history = []
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred = policy(obs_t[idx])
            sq = ((pred - act_t[idx]) ** 2).sum(dim=1)  # per-row L2
            loss = (w_t[idx] * sq).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n
        history.append({"epoch": epoch, "weighted_mse": epoch_loss})
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(f"  [π] epoch {epoch:3d}  weighted_mse={epoch_loss:.5f}")
    return policy, {"history": history, "policy_cfg": dict(
        obs_dim=cfg.obs_dim, action_dim=cfg.action_dim,
        hidden_sizes=list(cfg.hidden_sizes), activation=cfg.activation)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", action="append", default=[],
                    help="Path to rl_*.jsonl session log. Pass multiple times "
                         "for multi-session AWR; MC returns are computed "
                         "per-session (don't cross episode boundaries).")
    ap.add_argument("--session-glob", default=None,
                    help="Glob pattern that expands to one or more session jsonls "
                         "(e.g. 'RL data/5:2/.../*.jsonl'). Combined with --session.")
    ap.add_argument("--warm-start", default="training/runs/4_29_2d_no_mirror/bc.pt",
                    help="BC checkpoint to warm-start the policy from.")
    ap.add_argument("--out-dir", default="training/runs/awr_bc_2d")
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--beta", type=float, default=1.0,
                    help="AWR temperature. Higher = closer to BC, lower = more aggressive.")
    ap.add_argument("--w-max", type=float, default=20.0)
    ap.add_argument("--band-lo", type=float, default=DEADBAND_LO_M)
    ap.add_argument("--band-hi", type=float, default=DEADBAND_HI_M)
    ap.add_argument("--w-distance", type=float, default=W_DISTANCE)
    ap.add_argument("--w-heading",  type=float, default=W_HEADING)
    ap.add_argument("--w-jerk",     type=float, default=W_JERK)
    ap.add_argument("--v-epochs", type=int, default=200)
    ap.add_argument("--pi-epochs", type=int, default=100)
    ap.add_argument("--pi-lr", type=float, default=5e-5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "auto"
                          else ("mps" if torch.backends.mps.is_available()
                                else "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[awr] device = {device}")

    # 1. Load all sessions, compute per-session rewards + MC returns,
    #    then concatenate.
    import glob as _glob
    session_paths: list[str] = list(args.session)
    if args.session_glob:
        session_paths.extend(sorted(_glob.glob(args.session_glob)))
    if not session_paths:
        raise SystemExit("No --session paths provided (also try --session-glob).")

    obs_list, act_list, r_list, g_list = [], [], [], []
    for sp in session_paths:
        sess = load_session(sp)
        o = sess["obs"]; a = sess["action"]
        r = compute_rewards(o, a, args.band_lo, args.band_hi,
                            args.w_distance, args.w_heading, args.w_jerk)
        g = mc_returns(r, args.gamma)
        obs_list.append(o); act_list.append(a); r_list.append(r); g_list.append(g)
        print(f"[awr] {os.path.basename(sp):45s} N={len(o):4d}  "
              f"r̄={r.mean():+.4f}  Ḡ={g.mean():+.3f}  policy={sess['header'].get('policy')}")
    obs = np.concatenate(obs_list, axis=0)
    act = np.concatenate(act_list, axis=0)
    rewards = np.concatenate(r_list, axis=0)
    G = np.concatenate(g_list, axis=0)
    print(f"[awr] combined: N={len(obs)} from {len(session_paths)} session(s)")
    print(f"[awr] reward: mean={rewards.mean():+.4f}  std={rewards.std():.4f}  "
          f"q10/q50/q90 = {np.quantile(rewards,0.1):+.4f}/{np.quantile(rewards,0.5):+.4f}/{np.quantile(rewards,0.9):+.4f}")
    print(f"[awr] returns G: mean={G.mean():+.3f}  std={G.std():.3f}  "
          f"range=[{G.min():+.3f}, {G.max():+.3f}]")

    # Use BC's obs_mean/obs_std for consistency (don't recompute on the session,
    # which is a different distribution than the demonstrations).
    warm = Path(args.warm_start)
    if warm.exists():
        ck = torch.load(str(warm), map_location="cpu", weights_only=False)
        # Pull obs stats from the BC checkpoint
        warm_policy = PolicyMLP(PolicyConfig(**ck["policy_cfg"]))
        warm_policy.load_state_dict(ck["policy_state"])
        obs_mean = warm_policy.obs_mean.detach().cpu().numpy().astype(np.float32)
        obs_std  = warm_policy.obs_std.detach().cpu().numpy().astype(np.float32)
        print(f"[awr] using BC obs stats from {warm}")
    else:
        obs_mean = obs.mean(axis=0).astype(np.float32)
        obs_std  = np.where(obs.std(axis=0) < 1e-6, 1.0, obs.std(axis=0)).astype(np.float32)
        print(f"[awr] no warm-start checkpoint at {warm}; recomputing obs stats from session")
    print(f"[awr] obs_mean = {obs_mean.round(3).tolist()}")
    print(f"[awr] obs_std  = {obs_std.round(3).tolist()}")

    # 4. Fit value function
    print("[awr] fitting value function V_φ(s)…")
    v_net = fit_value(obs, G, obs_mean, obs_std,
                      epochs=args.v_epochs, lr=1e-3, batch_size=64,
                      device=device)

    # 5. Advantages + weights
    with torch.no_grad():
        v_pred = v_net(torch.from_numpy(obs).to(device)).cpu().numpy()
    advantages = G - v_pred
    print(f"[awr] advantage: mean={advantages.mean():+.3f}  std={advantages.std():.3f}  "
          f"q10/q50/q90 = {np.quantile(advantages,0.1):+.3f}/{np.quantile(advantages,0.5):+.3f}/{np.quantile(advantages,0.9):+.3f}")

    weights = awr_weights(advantages, args.beta, args.w_max)
    print(f"[awr] weights (β={args.beta}, w_max={args.w_max}):")
    print(f"      mean={weights.mean():.3f}  std={weights.std():.3f}  "
          f"min={weights.min():.3f}  max={weights.max():.3f}  "
          f"q10/q50/q90 = {np.quantile(weights,0.1):.3f}/{np.quantile(weights,0.5):.3f}/{np.quantile(weights,0.9):.3f}")
    print(f"      frac at w_max ({args.w_max}): {(weights >= args.w_max - 1e-3).mean():.1%}")

    # 6. Fit policy
    print("[awr] fitting policy π_θ via weighted MSE…")
    policy, info = fit_policy(obs, act, weights, warm if warm.exists() else None,
                              obs_mean, obs_std,
                              epochs=args.pi_epochs, lr=args.pi_lr, batch_size=64,
                              device=device)

    # Save
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    bc_pt = out_dir / "bc.pt"
    torch.save({
        "policy_state": policy.state_dict(),
        "policy_cfg":   info["policy_cfg"],
        "history":      info["history"],
        "awr": {
            "session": str(args.session),
            "gamma": args.gamma, "beta": args.beta, "w_max": args.w_max,
            "band_lo": args.band_lo, "band_hi": args.band_hi,
            "w_distance": args.w_distance, "w_heading": args.w_heading,
            "w_jerk": args.w_jerk,
        },
    }, str(bc_pt))
    print(f"[awr] saved -> {bc_pt}")
    # Diagnostic plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(11, 7))
        ax[0,0].plot(rewards, lw=0.7); ax[0,0].set_title("per-step reward"); ax[0,0].axhline(0, color="k", lw=0.3)
        ax[0,1].plot(G, lw=0.7); ax[0,1].plot(v_pred, lw=0.7); ax[0,1].set_title("MC return G (blue) vs V_φ (orange)")
        ax[1,0].plot(advantages, lw=0.7); ax[1,0].set_title("advantage = G - V"); ax[1,0].axhline(0, color="k", lw=0.3)
        ax[1,1].plot(weights, lw=0.7); ax[1,1].set_title(f"weights (β={args.beta})"); ax[1,1].axhline(1, color="k", lw=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "awr_diag.png", dpi=120)
        print(f"[awr] saved diag -> {out_dir / 'awr_diag.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
