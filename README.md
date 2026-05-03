# Autonomous Shopping Cart

Vision-based follower cart. The cart sees a person via a Pi camera, infers
distance and bearing from an ArUco marker, and commands four
continuous-rotation servos through a behavior-cloning policy fine-tuned
with offline RL (AWR).

> CSCI3387.01 – Topics in Computational Intelligence — Tyler Potsiadlo,
> Tianxiang Liu, Vedarsh Mishra.

## Repo layout

```
cart_stack/        # the runtime: agent on the Pi, dashboard on the Mac
  agent/             camera, motors, vision, RL session manager
  dashboard/         FastAPI server + static UI; relays to the Pi
  shared/            wire-format models and command parsing
  dashboard/policies/  deployed BC / AWR weights (npz)

training/          # offline pipeline (BC + AWR)
  process_session.py   raw video -> training-ready npz
  train_bc.py          MLP behavior cloning
  train_awr.py         offline RL refinement on top of BC
  evaluate_policy.py   closed-loop scoring against EVALUATION.md metric
  configs/             YAML training recipes
  data/processed/      processed npzs (4/29)
  runs/                trained checkpoints (4_29_2d, 4_29_2d_no_mirror, awr_bc_2d)

RL data/4:29/      # raw demonstration sessions (6 sessions, ~7.5 min)
aruco/calibration/ # camera intrinsics from ChArUco calibration

tests/             # pytest checks for camera, runtime, motion math
scripts/           # Pi-side install + deploy helpers
```

## Quick start — local dashboard + Pi agent

Install once:
```bash
python -m pip install -e .[dev]
```

Configure target Pi (copy + edit):
```bash
cp cart.config.example cart.config
# edit CART_PI_IP and CART_PI_USER
```

Run the cart:
```bash
./deploy_to_pi.sh        # rsync code + bundled policies, restart agent
./start.sh               # start the dashboard at http://localhost:8000
```

Open the dashboard, connect to the Pi, click **Install Bundled Policies**,
then run an RL session under `bc_2d_pivot` (4/29 BC + heading P-controller)
or `bc_2d_awr_pivot` (AWR-refined version of the same).

## Quick start — train a new policy from scratch

The full training pipeline is reproducible from the bundled 4/29 data:
```bash
# raw demos -> training-ready npzs
python training/process_session.py "RL data/4:29/152648" --no-yolo --keep-lost
# (repeat for each session, or pass multiple folders)

# behavior cloning (300 epochs, ~1 min on CPU)
python training/train_bc.py --config training/configs/4_29_2d.yaml

# export to runtime npz
python training/export_policy.py training/runs/4_29_2d/bc.pt \
    -o cart_stack/dashboard/policies/bc_2d_mirror.npz
```

For the AWR refinement step (offline RL on previously-recorded rollouts):
```bash
python training/train_awr.py \
    --session-glob "RL data/<run-folder>/*.jsonl" \
    --warm-start training/runs/4_29_2d_no_mirror/bc.pt
python training/export_policy.py training/runs/awr_bc_2d/bc.pt \
    -o cart_stack/dashboard/policies/bc_2d_awr.npz
```

## Tests

```bash
pytest
```
