#!/usr/bin/env python3
"""compare_policies.py — sweep obs space and compare two .npz policy bundles.

Loads each policy as numpy weights (the format the Pi runs), and prints:
  - response to distance sweep (heading=0, vel=0)
  - response to heading sweep at d=1.16m (band center)
  - obs normalization stats and weight norms

Useful when a "better val_mse" policy behaves worse on hardware -> tells us
whether the policy has wildly different output structure.
"""
from __future__ import annotations
import argparse, sys
import numpy as np


def load_policy(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    keys = list(z.files)
    n_layers = int(z["n_layers"])
    layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(n_layers)]
    final = z["final"].item().decode("utf-8")
    acts = z["activations"].item().decode("utf-8").split("|") if "activations" in keys else []
    return {
        "path": path, "obs_mean": z["obs_mean"], "obs_std": z["obs_std"],
        "layers": layers, "final": final, "activations": acts,
        "obs_dim": int(z["obs_dim"]), "action_dim": int(z["action_dim"]),
    }


def forward(p, obs):
    x = (obs - p["obs_mean"]) / (p["obs_std"] + 1e-6)
    for i, (W, b) in enumerate(p["layers"]):
        x = x @ W.T + b
        if i < len(p["layers"]) - 1:
            x = np.tanh(x)  # train_bc.py uses tanh
    if p["final"] == "tanh":
        x = np.tanh(x)
    return x


def header(name): return f"\n=== {name} ==="


def stats(p):
    print(f"  obs_mean = {p['obs_mean'].round(3).tolist()}")
    print(f"  obs_std  = {p['obs_std'].round(3).tolist()}")
    for i, (W, b) in enumerate(p["layers"]):
        print(f"  layer {i}: W{W.shape}  ||W||F={np.linalg.norm(W):.3f}  ||b||={np.linalg.norm(b):.3f}")


def sweep_d(p, ds, theta=0.0, lin_v=0.0, ang_v=0.0):
    obs = np.array([[d, theta, lin_v, ang_v] for d in ds], dtype=np.float32)
    a = np.stack([forward(p, o) for o in obs])
    forward_speed = 0.5 * (a[:, 0] + a[:, 1])  # L+R/2
    spin = a[:, 1] - a[:, 0]                     # R-L
    return forward_speed, spin, a


def sweep_h(p, hs, d=1.16, lin_v=0.0, ang_v=0.0):
    obs = np.array([[d, h, lin_v, ang_v] for h in hs], dtype=np.float32)
    a = np.stack([forward(p, o) for o in obs])
    forward_speed = 0.5 * (a[:, 0] + a[:, 1])
    spin = a[:, 1] - a[:, 0]
    return forward_speed, spin, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policies", nargs="+", help="Policy .npz files")
    args = ap.parse_args()

    pols = [(p.split("/")[-1].replace(".npz", ""), load_policy(p)) for p in args.policies]

    print(header("policy stats"))
    for name, p in pols:
        print(f"\n[{name}]"); stats(p)

    print(header("distance sweep at heading=0"))
    ds = np.array([0.5, 0.7, 0.85, 1.0, 1.16, 1.32, 1.5, 1.7, 2.0, 2.5], dtype=np.float32)
    head = "    d (m) " + "  ".join(f"{n:>22s}" for n, _ in pols)
    print(head); print("-" * len(head))
    for i, d in enumerate(ds):
        cells = []
        for name, p in pols:
            fwd, spin, a = sweep_d(p, [float(d)])
            cells.append(f"L={a[0,0]:+5.2f} R={a[0,1]:+5.2f}  fwd{fwd[0]:+5.2f}")
        print(f"  {d:5.2f}  " + "  ".join(f"{c:>22s}" for c in cells))

    print(header("heading sweep at d=1.16m"))
    hs = np.linspace(-0.4, 0.4, 9)
    head = "  θ (rad) " + "  ".join(f"{n:>22s}" for n, _ in pols)
    print(head); print("-" * len(head))
    for h in hs:
        cells = []
        for name, p in pols:
            fwd, spin, a = sweep_h(p, [float(h)])
            cells.append(f"L={a[0,0]:+5.2f} R={a[0,1]:+5.2f}  spin{spin[0]:+5.2f}")
        print(f"  {h:+5.2f}  " + "  ".join(f"{c:>22s}" for c in cells))

    print(header("center-band response (heading=0, d=1.16, vel=0)"))
    for name, p in pols:
        a = forward(p, np.array([1.16, 0, 0, 0], dtype=np.float32))
        print(f"  {name:30s}  L={a[0]:+.3f}  R={a[1]:+.3f}  fwd={(a[0]+a[1])/2:+.3f}  spin={a[1]-a[0]:+.3f}")


if __name__ == "__main__":
    main()
