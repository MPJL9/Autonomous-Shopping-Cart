#!/usr/bin/env python3
"""analyze_rewards.py — compute per-step rewards on processed sessions and
report the distribution.

Reward formula matches evaluate_policy.py / EVALUATION.md (negated cost):

    r_t = -[ w_d * miss_t^2 + w_h * theta_t^2 + w_j * ||a_t - a_{t-1}||^2 ]
    miss_t = max(0, d_lo - d_t, d_t - d_hi)

Frames with visible=False or NaN aruco_distance are dropped from the
histogram (they're untrustworthy supervision targets for advantage).

Purpose: sanity-check whether the per-step reward signal is informative
before we commit to fitting a value function. If most steps are at r ~= 0
(operator already in band, smooth control), AWR will collapse to plain BC.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

DEADBAND_LO_M = 1.01
DEADBAND_HI_M = 1.32
W_DISTANCE = 1.0
W_HEADING  = 0.5
W_JERK     = 0.1


def per_step_rewards(npz_path: str, *, d_lo: float, d_hi: float,
                     w_d: float, w_h: float, w_j: float) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    obs = data["obs"]
    action = data["action"]
    visible = data["visible"].astype(bool)
    aruco_d = data["aruco_distance_m"]
    n = len(obs)
    if n < 2:
        return {"path": npz_path, "n": 0}

    d_t = obs[:, 0]
    th_t = obs[:, 1]

    miss = np.maximum.reduce([np.zeros(n), d_lo - d_t, d_t - d_hi])
    distance_term = w_d * (miss ** 2)
    heading_term = w_h * (th_t ** 2)

    da = np.zeros((n, action.shape[1]), dtype=np.float32)
    da[1:] = action[1:] - action[:-1]
    jerk_term = w_j * np.sum(da * da, axis=1)

    cost = distance_term + heading_term + jerk_term
    reward = -cost

    # `visible` is authoritative for obs trust (see process_session.py:
    # True iff offline ArUco hit OR session reprocessed with --use-live-obs).
    valid = visible

    return {
        "path": npz_path,
        "session": os.path.basename(npz_path).replace(".npz", ""),
        "n": int(n),
        "n_valid": int(valid.sum()),
        "reward": reward,
        "valid": valid,
        "distance_term": distance_term,
        "heading_term": heading_term,
        "jerk_term": jerk_term,
        "miss": miss,
        "in_band": (d_t >= d_lo) & (d_t <= d_hi),
    }


def summarize(rows: list[dict]) -> None:
    all_r = np.concatenate([r["reward"][r["valid"]] for r in rows])
    all_d = np.concatenate([r["distance_term"][r["valid"]] for r in rows])
    all_h = np.concatenate([r["heading_term"][r["valid"]] for r in rows])
    all_j = np.concatenate([r["jerk_term"][r["valid"]] for r in rows])
    all_in = np.concatenate([r["in_band"][r["valid"]] for r in rows])

    print(f"\n=== aggregate over {len(rows)} sessions, "
          f"{len(all_r)} valid steps ===")
    print(f"reward    : mean={all_r.mean():+.5f}  std={all_r.std():.5f}  "
          f"min={all_r.min():+.5f}  max={all_r.max():+.5f}")
    print(f"           q10={np.quantile(all_r, 0.10):+.5f}  "
          f"q50={np.quantile(all_r, 0.50):+.5f}  "
          f"q90={np.quantile(all_r, 0.90):+.5f}  "
          f"q99={np.quantile(all_r, 0.99):+.5f}")

    print(f"\ncomponent contributions to cost (-reward):")
    print(f"  distance term: mean={all_d.mean():.5f}  share={all_d.mean()/(-all_r.mean() + 1e-9):.1%}")
    print(f"  heading  term: mean={all_h.mean():.5f}  share={all_h.mean()/(-all_r.mean() + 1e-9):.1%}")
    print(f"  jerk     term: mean={all_j.mean():.5f}  share={all_j.mean()/(-all_r.mean() + 1e-9):.1%}")

    print(f"\nin-band fraction: {all_in.mean():.1%}")

    near_zero = (np.abs(all_r) < 1e-3).mean()
    print(f"steps with |r| < 1e-3 (effectively zero): {near_zero:.1%}")
    print(f"steps with r < -0.1  (clearly bad)      : {(all_r < -0.1).mean():.1%}")
    print(f"steps with r < -0.5  (very bad)         : {(all_r < -0.5).mean():.1%}")

    return all_r, all_d, all_h, all_j


def plot_hist(all_r, all_d, all_h, all_j, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available, skipping plot")
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(all_r, bins=80, color="#3a7", edgecolor="k", linewidth=0.3)
    axes[0, 0].set_title("per-step reward (= -cost)")
    axes[0, 0].set_xlabel("reward")
    axes[0, 0].set_ylabel("# steps")
    axes[0, 0].axvline(0, color="k", lw=0.6, alpha=0.5)
    axes[0, 0].set_yscale("log")

    axes[0, 1].hist(all_d, bins=80, color="#36c", edgecolor="k", linewidth=0.3)
    axes[0, 1].set_title("distance term  w_d · miss²")
    axes[0, 1].set_yscale("log")

    axes[1, 0].hist(all_h, bins=80, color="#c63", edgecolor="k", linewidth=0.3)
    axes[1, 0].set_title("heading term  w_θ · θ²")
    axes[1, 0].set_yscale("log")

    axes[1, 1].hist(all_j, bins=80, color="#963", edgecolor="k", linewidth=0.3)
    axes[1, 1].set_title("jerk term  w_j · ||Δa||²")
    axes[1, 1].set_yscale("log")

    fig.suptitle("Per-step reward distribution (combined 4/29 + 5/1)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nsaved histogram to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+",
                    help="glob patterns for processed npz files")
    ap.add_argument("--d-lo", type=float, default=DEADBAND_LO_M)
    ap.add_argument("--d-hi", type=float, default=DEADBAND_HI_M)
    ap.add_argument("--w-distance", type=float, default=W_DISTANCE)
    ap.add_argument("--w-heading",  type=float, default=W_HEADING)
    ap.add_argument("--w-jerk",     type=float, default=W_JERK)
    ap.add_argument("--out-png", type=str,
                    default="training/analysis/reward_hist.png")
    args = ap.parse_args()

    paths: list[str] = []
    for pat in args.globs:
        paths.extend(sorted(glob.glob(pat)))
    paths = [p for p in paths if p.endswith(".npz")]
    if not paths:
        print("no npz files matched")
        return

    print(f"loaded {len(paths)} sessions")
    rows = []
    for p in paths:
        r = per_step_rewards(
            p, d_lo=args.d_lo, d_hi=args.d_hi,
            w_d=args.w_distance, w_h=args.w_heading, w_j=args.w_jerk,
        )
        if r.get("n", 0) > 0:
            rows.append(r)
            mean_r = float(r["reward"][r["valid"]].mean()) if r["n_valid"] > 0 else 0.0
            print(f"  {r['session']:40s}  N={r['n']:4d}  valid={r['n_valid']:4d}  "
                  f"mean_r={mean_r:+.4f}")

    all_r, all_d, all_h, all_j = summarize(rows)

    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    plot_hist(all_r, all_d, all_h, all_j, args.out_png)


if __name__ == "__main__":
    main()
