#!/usr/bin/env python3
"""evaluate_policy.py — score a policy session bundle (already processed
into an npz) against the deadband-aware metric defined in EVALUATION.md.

Usage:
  python3 training/evaluate_policy.py training/data/processed/<dir>/<session>.npz [more...]

Output: a one-row-per-npz table with the scalar J and its components, so
multiple policies can be diffed at a glance.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np

# ---- defaults from EVALUATION.md --------------------------------------------
# Band derived from combined 4/29 + 5/1 training Q25-Q75. See
# "Band derivation" section in EVALUATION.md for the data behind it.
DEADBAND_LO_M = 1.01
DEADBAND_HI_M = 1.32
W_DISTANCE = 1.0      # m^-2
W_HEADING  = 0.5      # rad^-2
W_JERK     = 0.1


def _outside_band(d: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Distance from band: 0 inside [lo,hi], (lo-d) below, (d-hi) above."""
    out = np.maximum(0.0, lo - d)
    out = np.maximum(out, d - hi)
    return out


def evaluate(npz_path: str, *, w_d: float, w_h: float, w_j: float,
             d_lo: float, d_hi: float, ignore_lost_lock_in_heading: bool = True
             ) -> dict:
    """Score one processed-session npz. Returns a dict of metrics."""
    data = np.load(npz_path, allow_pickle=True)

    # `obs` columns are offline ArUco truth (rebuilt by process_session.py)
    obs    = data["obs"]                  # (N, 4) [d, theta, lin_v, ang_v]
    action = data["action"]               # (N, 2)
    visible = data["visible"].astype(bool)
    aruco_d = data["aruco_distance_m"]    # NaN where the offline solve missed
    n = len(obs)
    if n == 0:
        return {"path": npz_path, "n_steps": 0}

    d_t = obs[:, 0]
    th_t = obs[:, 1]

    # `visible` is the authoritative "obs[0] is trustworthy" flag set by
    # process_session.py — it is True iff the offline ArUco hit, OR the
    # session was reprocessed with --use-live-obs (live Pi-side tracker
    # was reliable, but offline ArUco can't see through burned-in overlay).
    valid_d = visible
    od = _outside_band(d_t, d_lo, d_hi)
    distance_term = w_d * (od ** 2)
    distance_term_valid = distance_term[valid_d]

    # Heading penalty: ignore lost-lock frames if asked.
    if ignore_lost_lock_in_heading:
        valid_h = valid_d
    else:
        valid_h = np.ones(n, dtype=bool)
    heading_term = w_h * (th_t ** 2)
    heading_term_valid = heading_term[valid_h]

    # Jerk: norm of consecutive action diffs. First step has no predecessor,
    # so we drop t=0.
    if n >= 2:
        da = action[1:] - action[:-1]
        jerk_per_step = np.sum(da * da, axis=1)
    else:
        jerk_per_step = np.zeros(0, dtype=np.float32)
    jerk_term = w_j * jerk_per_step

    # Compose mean-of-components, each averaged over its own valid mask, then
    # combine. This is the per-step cost averaged over the session.
    def _safe_mean(a):
        return float(a.mean()) if len(a) else 0.0

    J_distance = _safe_mean(distance_term_valid)
    J_heading  = _safe_mean(heading_term_valid)
    J_jerk     = _safe_mean(jerk_term)
    J_total    = J_distance + J_heading + J_jerk

    # Auxiliary stats
    in_band = (d_t >= d_lo) & (d_t <= d_hi)
    pct_in_band = float((in_band & valid_d).sum()) / max(int(valid_d.sum()), 1)

    track_loss = 1.0 - (float(visible.sum()) / max(n, 1))

    rms_d_err_outside = float(np.sqrt(np.mean(od[valid_d] ** 2))) if valid_d.any() else 0.0
    rms_heading_deg = float(np.degrees(np.sqrt(np.mean(th_t[valid_h] ** 2)))) if valid_h.any() else 0.0
    rms_jerk = float(np.sqrt(np.mean(jerk_per_step))) if len(jerk_per_step) else 0.0

    sat_count = int(((np.abs(action[:, 0]) >= 0.999) | (np.abs(action[:, 1]) >= 0.999)).sum())
    sat_frac = sat_count / max(n, 1)

    return {
        "path": npz_path,
        "session": os.path.basename(npz_path).replace(".npz", ""),
        "policy": str(data["policy"]) if "policy" in data.files else "unknown",
        "n_steps": int(n),
        # Real wall-clock duration: t_sec[-1] is video time of last frame.
        "duration_s": round(float(data["t_sec"][-1]) if "t_sec" in data.files and n > 0 else 0.0, 2),
        "J_total": J_total,
        "J_distance": J_distance,
        "J_heading":  J_heading,
        "J_jerk":     J_jerk,
        "pct_in_band": pct_in_band,
        "track_loss": track_loss,
        "rms_d_outside_m": rms_d_err_outside,
        "rms_heading_deg": rms_heading_deg,
        "rms_jerk": rms_jerk,
        "action_saturation_pct": sat_frac,
    }


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("(no rows)")
        return
    headers = [
        ("session",          "policy session",            "{:s}"),
        ("policy",           "policy",                     "{:s}"),
        ("duration_s",       "T (s)",                      "{:6.1f}"),
        ("track_loss",       "track_loss",                 "{:6.0%}"),
        ("J_total",          "J",                          "{:7.4f}"),
        ("J_distance",       "  J_d",                      "{:7.4f}"),
        ("J_heading",        "  J_θ",                      "{:7.4f}"),
        ("J_jerk",           "  J_a",                      "{:7.4f}"),
        ("pct_in_band",      "in-band%",                   "{:7.0%}"),
        ("rms_d_outside_m",  "RMS d_out",                  "{:7.3f}"),
        ("rms_heading_deg",  "RMS θ°",                     "{:6.1f}"),
        ("rms_jerk",         "RMS jerk",                   "{:7.4f}"),
    ]
    label_row = "  ".join(f"{lbl:>20s}" if k == "session" else f"{lbl:>10s}" for k, lbl, _ in headers)
    print(label_row)
    print("-" * len(label_row))
    for r in rows:
        cells = []
        for k, _, fmt in headers:
            v = r.get(k, "")
            if isinstance(v, str) and k == "session":
                cells.append(f"{v:>20s}")
            elif isinstance(v, (int, float)):
                cells.append(f"{fmt.format(v):>10s}")
            else:
                cells.append(f"{str(v):>10s}")
        print("  ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+", help="processed-session npz files (one or more)")
    ap.add_argument("--w-distance", type=float, default=W_DISTANCE)
    ap.add_argument("--w-heading",  type=float, default=W_HEADING)
    ap.add_argument("--w-jerk",     type=float, default=W_JERK)
    ap.add_argument("--d-lo", type=float, default=DEADBAND_LO_M)
    ap.add_argument("--d-hi", type=float, default=DEADBAND_HI_M)
    ap.add_argument("--no-skip-lost-lock", action="store_true",
                    help="Count lost-lock frames in the heading term too "
                         "(default: skip).")
    args = ap.parse_args()

    rows = []
    for path in args.npz:
        if not os.path.exists(path):
            print(f"[warn] missing: {path}", file=sys.stderr)
            continue
        rows.append(evaluate(
            path,
            w_d=args.w_distance,
            w_h=args.w_heading,
            w_j=args.w_jerk,
            d_lo=args.d_lo,
            d_hi=args.d_hi,
            ignore_lost_lock_in_heading=not args.no_skip_lost_lock,
        ))

    _print_table(rows)
    print()
    print(f"weights: w_d={args.w_distance}  w_θ={args.w_heading}  w_jerk={args.w_jerk}   "
          f"band: [{args.d_lo}, {args.d_hi}] m   "
          f"lost-lock heading: {'skipped' if not args.no_skip_lost_lock else 'counted'}")


if __name__ == "__main__":
    main()
