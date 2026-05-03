#!/usr/bin/env python3
"""eval_jsonl.py — score a recorded RL session's JSONL log against the
EVALUATION.md cost metric, without going through process_session.py.

Useful when the recorded video has the live overlay burned in (offline
ArUco can't re-detect the marker), so we have to trust the Pi-side live
obs instead.

Reports the same columns as `evaluate_policy.py`:
  J, J_d, J_θ, J_a, in-band%, RMS d_outside, RMS θ°, RMS jerk, sat%

Usage:
  python3 training/eval_jsonl.py "RL data/.../*.jsonl" [more...] \\
      [--band-lo 1.26 --band-hi 1.56]
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys
import numpy as np

W_DISTANCE = 1.0
W_HEADING  = 0.5
W_JERK     = 0.1

# V2 additions for "decisiveness". Weights chosen so a moderately-bad
# instance contributes ~0.01 per step, comparable to the J_θ scale.
W_HEAD_ACC   = 0.05    # heading second-difference (anti-oscillation)
W_REV        = 0.05    # spin direction-reversal indicator
W_COMMIT     = 0.5     # weak-spin-at-large-heading commitment penalty
SPIN_NOISE   = 0.05    # |spin| threshold below which we treat it as "no command"
THETA_BIG    = 0.30    # |θ| above this triggers the commitment check (~17°)
K_EXPECTED   = 0.6     # expected spin magnitude per rad of heading (Tier-1 K)


def load_session(path: str) -> dict:
    obs, act = [], []
    header = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "header":
                header = r
            elif r.get("type") == "step":
                obs.append(r["obs"]); act.append(r["action"])
    return {
        "header": header,
        "obs": np.asarray(obs, dtype=np.float32),
        "action": np.asarray(act, dtype=np.float32),
        "path": path,
    }


def score(sess: dict, band_lo: float, band_hi: float,
          w_d: float, w_h: float, w_j: float,
          w_head_acc: float = W_HEAD_ACC, w_rev: float = W_REV,
          w_commit: float = W_COMMIT,
          spin_noise: float = SPIN_NOISE, theta_big: float = THETA_BIG,
          k_expected: float = K_EXPECTED) -> dict:
    obs = sess["obs"]; act = sess["action"]
    n = len(obs)
    if n == 0:
        return {"path": sess["path"], "n": 0}
    d = obs[:, 0]; theta = obs[:, 1]
    spin = act[:, 1] - act[:, 0]

    # ---- V1 components --------------------------------------------------
    miss = np.maximum.reduce([np.zeros(n), band_lo - d, d - band_hi])
    distance_term = w_d * (miss ** 2)
    heading_term  = w_h * (theta ** 2)
    da = np.zeros_like(act); da[1:] = act[1:] - act[:-1]
    jerk_per_step = np.sum(da * da, axis=1)
    jerk_term     = w_j * jerk_per_step

    # ---- V2 additions ---------------------------------------------------
    # (a) heading acceleration (second-difference): θ_t - 2θ_{t-1} + θ_{t-2}
    head_acc = np.zeros(n)
    if n >= 3:
        head_acc[2:] = theta[2:] - 2.0 * theta[1:-1] + theta[:-2]
    head_acc_term = w_head_acc * (head_acc ** 2)

    # (b) spin sign-reversal indicator (only counts above noise floor)
    rev_indicator = np.zeros(n)
    if n >= 2:
        cur = spin[1:]; prev = spin[:-1]
        sign_flip = (np.sign(cur) != np.sign(prev)) & (np.sign(cur) != 0) & (np.sign(prev) != 0)
        big_enough = (np.abs(cur) > spin_noise) & (np.abs(prev) > spin_noise)
        rev_indicator[1:] = (sign_flip & big_enough).astype(np.float32)
    rev_term = w_rev * rev_indicator

    # (c) commitment penalty: when |θ| is big, the expected spin magnitude is
    # k_expected * |θ|. If actual is weaker than expected, charge the
    # squared shortfall.
    is_big = np.abs(theta) > theta_big
    expected_spin = k_expected * np.abs(theta)
    shortfall = np.maximum(0.0, expected_spin - np.abs(spin))
    commit_term = w_commit * is_big * (shortfall ** 2)

    # ---- aggregates -----------------------------------------------------
    in_band = (d >= band_lo) & (d <= band_hi)
    rms_d_out = float(np.sqrt(np.mean(miss ** 2))) if n else 0.0
    rms_th_deg = float(math.degrees(np.sqrt(np.mean(theta ** 2)))) if n else 0.0
    rms_jerk = float(np.sqrt(np.mean(jerk_per_step))) if n >= 2 else 0.0
    sat = ((np.abs(act[:, 0]) >= 0.999) | (np.abs(act[:, 1]) >= 0.999)).mean()
    rev_rate = float(rev_indicator.sum()) / max(n - 1, 1)
    rms_head_acc = float(np.sqrt(np.mean(head_acc ** 2))) if n >= 3 else 0.0

    j_v1 = float(distance_term.mean() + heading_term.mean() + jerk_term.mean())
    j_v2_extra = float(head_acc_term.mean() + rev_term.mean() + commit_term.mean())
    j_v2 = j_v1 + j_v2_extra

    return {
        "path": sess["path"],
        "session": os.path.basename(sess["path"]).replace(".jsonl", ""),
        "policy": (sess["header"] or {}).get("policy", "unknown"),
        "n": n,
        "duration_s": round(n / max((sess["header"] or {}).get("step_hz", 5.0), 1e-6), 1),
        "J":         j_v1,
        "J_d":       float(distance_term.mean()),
        "J_h":       float(heading_term.mean()),
        "J_a":       float(jerk_term.mean()),
        "J_v2":      j_v2,
        "J_acc":     float(head_acc_term.mean()),    # heading acceleration
        "J_rev":     float(rev_term.mean()),         # spin direction-reversal
        "J_commit":  float(commit_term.mean()),      # weak-at-large-heading
        "in_band":      float(in_band.mean()),
        "rms_d_out":    rms_d_out,
        "rms_h_deg":    rms_th_deg,
        "rms_jerk":     rms_jerk,
        "rms_head_acc": rms_head_acc,
        "rev_rate":     rev_rate,
        "action_sat":   float(sat),
    }


def print_table(rows: list[dict]) -> None:
    cols = [
        ("session",       "session",      "{:s}", 35),
        ("policy",        "policy",       "{:s}", 16),
        ("duration_s",    "T (s)",        "{:6.1f}", 7),
        # V1 metric and components
        ("J",             "J(v1)",        "{:7.4f}", 8),
        ("J_d",           "  J_d",        "{:7.4f}", 8),
        ("J_h",           "  J_θ",        "{:7.4f}", 8),
        ("J_a",           "  J_a",        "{:7.4f}", 8),
        # V2 metric and decisiveness components
        ("J_v2",          "J(v2)",        "{:7.4f}", 8),
        ("J_acc",         "  J_acc",      "{:7.4f}", 8),
        ("J_rev",         "  J_rev",      "{:7.4f}", 8),
        ("J_commit",      "  J_cmt",      "{:7.4f}", 8),
        # diagnostics
        ("rev_rate",      "rev/step",     "{:6.1%}", 8),
        ("in_band",       "in-band%",     "{:6.0%}", 9),
        ("rms_jerk",      "RMS jerk",     "{:7.4f}", 9),
    ]
    head = "  ".join(f"{lbl:>{w}}" for _, lbl, _, w in cols)
    print(head); print("-" * len(head))
    for r in rows:
        cells = []
        for k, _, fmt, w in cols:
            v = r.get(k, "")
            if isinstance(v, str):
                cells.append(f"{v:>{w}.{w}}")
            elif isinstance(v, (int, float)):
                cells.append(f"{fmt.format(v):>{w}}")
            else:
                cells.append(f"{str(v):>{w}}")
        print("  ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+", help="paths or globs to *.jsonl session logs")
    ap.add_argument("--band-lo", type=float, default=1.26)
    ap.add_argument("--band-hi", type=float, default=1.56)
    ap.add_argument("--w-distance", type=float, default=W_DISTANCE)
    ap.add_argument("--w-heading",  type=float, default=W_HEADING)
    ap.add_argument("--w-jerk",     type=float, default=W_JERK)
    args = ap.parse_args()

    paths: list[str] = []
    for pat in args.globs:
        paths.extend(sorted(glob.glob(pat)))
    paths = sorted({p for p in paths if p.endswith(".jsonl")})
    if not paths:
        print("(no sessions matched)", file=sys.stderr); sys.exit(1)

    rows = [score(load_session(p), args.band_lo, args.band_hi,
                  args.w_distance, args.w_heading, args.w_jerk) for p in paths]
    print(f"weights: w_d={args.w_distance} w_θ={args.w_heading} w_j={args.w_jerk}   "
          f"band: [{args.band_lo}, {args.band_hi}] m\n")
    print_table(rows)

    # Group-level aggregate by policy
    print()
    by_policy: dict[str, list[dict]] = {}
    for r in rows:
        by_policy.setdefault(r["policy"], []).append(r)
    if len(by_policy) > 1:
        agg_rows = []
        for policy, group in by_policy.items():
            n_total = sum(g["n"] for g in group)
            agg = {
                "session":     f"<MEAN of {len(group)} runs>",
                "policy":      policy,
                "duration_s":  sum(g["duration_s"] for g in group),
                "J":           float(np.mean([g["J"] for g in group])),
                "J_d":         float(np.mean([g["J_d"] for g in group])),
                "J_h":         float(np.mean([g["J_h"] for g in group])),
                "J_a":         float(np.mean([g["J_a"] for g in group])),
                "J_v2":        float(np.mean([g["J_v2"] for g in group])),
                "J_acc":       float(np.mean([g["J_acc"] for g in group])),
                "J_rev":       float(np.mean([g["J_rev"] for g in group])),
                "J_commit":    float(np.mean([g["J_commit"] for g in group])),
                "rev_rate":    float(np.mean([g["rev_rate"] for g in group])),
                "in_band":     float(np.mean([g["in_band"] for g in group])),
                "rms_d_out":   float(np.mean([g["rms_d_out"] for g in group])),
                "rms_h_deg":   float(np.mean([g["rms_h_deg"] for g in group])),
                "rms_jerk":    float(np.mean([g["rms_jerk"] for g in group])),
                "rms_head_acc":float(np.mean([g["rms_head_acc"] for g in group])),
                "action_sat":  float(np.mean([g["action_sat"] for g in group])),
                "n":           n_total,
            }
            agg_rows.append(agg)
        print_table(agg_rows)


if __name__ == "__main__":
    main()
