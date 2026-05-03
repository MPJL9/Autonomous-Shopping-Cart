# Full Training Instructions

End-to-end pipeline: record demos → process → train BC → deploy to cart → test.
All commands assume you're at `/Users/tliu/Desktop/Workspace/ML_Project` with the
cart_stack venv (`/Users/tliu/Desktop/Workspace/Autonomous-Shopping-Cart/.venv`).

## TL;DR

```
# 1. Record a session through the dashboard's RL Training tab (policy=manual)
#    Download Session Bundle → 3 files land in ~/Downloads.

# 2. Move them into a session folder:
mv ~/Downloads/rl_manual_*.{webm,jsonl,meta.json} \
   "RL data/<date>/<session_name>/"

# 3. Process video → npz with ArUco/YOLO/action per frame:
PY=/Users/tliu/Desktop/Workspace/Autonomous-Shopping-Cart/.venv/bin/python
$PY training/process_session.py "RL data/<date>/<session_name>"

# 4. Train BC (do both 1D and 2D so you have both options on hardware):
$PY training/train_bc.py --config training/configs/<your_config>_1d.yaml
$PY training/train_bc.py --config training/configs/<your_config>_2d.yaml

# 5. Export to numpy npz for the Pi runtime:
$PY training/export_policy.py training/runs/<run>/bc.pt -o training/runs/<run>/bc_policy.npz

# 6. Open dashboard → RL Training tab → Upload Policy → pick the .npz file.
#    Dropdown picks policy: bc_1d / bc_2d / bc_2d_pivot / bc_2d_translate.
#    Set step_hz=15, action_scale=0.25 for first hardware test.
#    Click Start Exploration + Record.
```

---

## 0. Prerequisites

- Pi running, dashboard connected, camera streaming (verify on the Live tab).
- ArUco marker `single_marker_id1_10cm.pdf` (10 cm side, dict 4×4_50) printed
  and worn at chest height on the operator's shirt.
- Camera intrinsics calibrated. `aruco/calibration/calib.npz` should exist
  (it does — captured 2026-03-31, image_size 640×480, reproj 0.37 px).
  If you change camera mount or stream resolution, redo it with
  `aruco/calibration/calibrate_camera.py` against the charuco video.
- The cart_stack venv has cv2, ultralytics, torch, numpy, scikit-learn,
  matplotlib, pyyaml.

---

## 1. Record demonstrations

In the dashboard:

1. Open the **Live** tab first to confirm the camera feed is streaming. **Stay on the Live tab while recording** — if you switch away, MediaRecorder may stop capturing frames (we hit this on day 1, lost a session).
2. Open the **RL Training** tab and configure:
   - `policy = manual`
   - `step_hz = 15` (denser action labels than the default 5; helps BC quality)
   - `max_steps = 600` (~40 s at 15 Hz; or longer if you want)
   - `action_scale` is unused in manual; leave at 0.35
3. Click **Start Exploration + Record**. Status badge shows "Recording (RL)".
4. Drive the cart with the joystick or quick buttons (both are now in the RL panel itself, plus the Manual mode panel — same controls). The session log captures whatever the joystick is sending each tick.
5. Click **Stop** when finished, or let `max_steps` end the session.
6. Click **Download Session Bundle**. Three files download:
   - `rl_manual_<timestamp>.webm` (browser video)
   - `rl_manual_<timestamp>.jsonl` (Pi action log)
   - `rl_manual_<timestamp>.meta.json` (alignment sidecar)

**Recording quality check.** After every session, before doing anything else,
compare `video_duration_sec` (in the meta file) to the log span. They should
agree within ~1 second. If video is way shorter than the log, the recording
was broken — discard and redo.

### What scenarios to record

For BC training, you want the data to span what the deployed policy will see.
Priority order:

1. **Slow forward + backward pursuit** — person walks slowly forward then back,
   joystick tracks them with steady pressure. Most-valuable case.
2. **Standing-still at multiple distances** — park at 0.8 m, 1.0 m, 1.2 m,
   1.5 m for ~15 s each, joystick at zero. Teaches the policy the most
   important behavior: "do nothing when I'm at the right distance."
3. **Sudden stop in the middle of a pursuit** — person walks away, abruptly
   stops and holds for ~3 s, then resumes. Trains the damping behavior.
4. **Different starting distances** — start sessions at 0.5 m, 1.5 m, 2.0 m
   so the policy sees how to settle from each.

10–15 minutes of clean data across these scenarios is enough for a competent
policy. Day 1 recorded 173 seconds of mostly forward/backward and got val
MSE 0.017.

---

## 2. Organize downloaded files

```
RL data/
  <YYYY-MM-DD>/
    <session_name>/             # one folder per recording
      rl_manual_*.webm
      rl_manual_*.jsonl
      rl_manual_*.meta.json
```

Naming convention: prefix the session folder with `table_` or `floor_` if you
later want different deployments. Configs use globs by prefix.

---

## 3. Process recordings

Re-runs ArUco + YOLO on every video frame, time-aligns with the JSONL action
log, writes a tidy npz with the full per-frame state.

```
PY=/Users/tliu/Desktop/Workspace/Autonomous-Shopping-Cart/.venv/bin/python

$PY training/process_session.py "RL data/<date>/<session>"
# Multiple at once:
$PY training/process_session.py \
    "RL data/<date>/session_a" \
    "RL data/<date>/session_b" \
    "RL data/<date>/session_c"
```

Output goes to `training/data/processed/<stem>.npz`. Default behavior drops
frames where ArUco lost the marker; pass `--keep-lost` to keep them with a
visibility flag (useful for later "target lost" recovery training).

Each npz contains:

```
obs (N, 4)              [distance_m, heading_rad, lin_v, ang_v]   ← BC input
action (N, 2)           [left, right] joystick                     ← BC label
t_sec (N,)              video time at each row
visible (N,)            ArUco lock?
aruco_distance_m (N,)   raw distance (NaN when lost)
aruco_heading_rad (N,)
aruco_corners (N, 4, 2) marker corners in image
yolo_detected (N,)
yolo_person_conf (N,)
yolo_bbox (N, 4)        x1, y1, x2, y2
yolo_kxy (N, 17, 2)     17 COCO keypoints
yolo_kc (N, 17)         keypoint confidences
```

The npz path defaults to `training/data/processed/`. Use `--out-dir` to put
day-specific data in subfolders, e.g.:

```
$PY training/process_session.py --out-dir training/data/processed/4-23 \
    --keep-lost \
    "RL data/4:23/250pm" "RL data/4:23/252pm" ...
```

**Verify processing succeeded.** Look at the printout's `ArUco hit:` line.
Anything below ~70 % is suspicious — check that the marker was actually in
frame and that `--marker-cm` matches your printed marker (default 10 cm).

---

## 4. Train BC

### 4.1 Configs

Two YAML configs per training run — one for 1D (forward only), one for 2D
(left/right). Existing examples for the 4/23 data:

```
training/configs/4_23_1d.yaml      action_dim: 1
training/configs/4_23_2d.yaml      action_dim: 2
```

Both point at `training/data/processed/4-23/*.npz`. To train on a different
day's data, copy and edit `data_glob` and `out_dir`.

Key fields:

```yaml
action_dim: 1            # 1 = forward only (symmetric drive); 2 = [L, R]
obs_dim: 4
hidden_sizes: [64, 64]
data_glob: "training/data/processed/<day>/*.npz"
out_dir: "training/runs/<run_name>"
seed: 0
batch_size: 128
lr: 0.001
epochs: 300
val_frac: 0.15
device: auto              # auto picks MPS on Mac, CUDA if available, else CPU
save_every: 30            # epoch interval for checkpoint snapshots
```

### 4.2 Run training

Always train both 1D and 2D from the same data — the pair gives you both
deployment options without extra recording.

```
$PY training/train_bc.py --config training/configs/4_23_1d.yaml
$PY training/train_bc.py --config training/configs/4_23_2d.yaml
```

Each run produces:

```
training/runs/<run_name>/
  bc.pt                      best-val checkpoint (use this)
  history.json               per-epoch train/val MSE
  loss.png                   train+val loss curves
  checkpoints/
    epoch_0030.pt            snapshot every save_every epochs
    epoch_0060.pt
    ...
```

300 epochs on 2,738 rows takes <1 min on M-series Mac (MPS).

**Override hyperparameters without editing the YAML:**

```
$PY training/train_bc.py --config training/configs/4_23_1d.yaml \
    --override epochs=500 lr=5e-4 batch_size=64
```

### 4.3 What "good" looks like

For 4/23 data both 1D and 2D converge to **val MSE ≈ 0.017** (RMS action error
~0.13 in the [-1, 1] action space). Indicators of trouble:

- Val MSE plateaus above 0.04 → not enough variation in demos, or perception
  noise is dominating. Record more data.
- Train MSE keeps dropping while val MSE rises → overfitting. Reduce epochs
  or hidden sizes.
- 1D and 2D land at very different val MSE → likely a label-extraction bug;
  for typical 1D-pursuit data they should be within ~0.001 of each other.

### 4.4 Sanity-check the response curve

Verify the trained policy produces a sensible distance→action mapping before
deploying:

```python
import sys, torch, numpy as np
sys.path.insert(0, "training")
from models.policy import PolicyConfig, PolicyMLP

ck = torch.load("training/runs/4_23_1d/bc.pt", map_location="cpu", weights_only=False)
p = PolicyMLP(PolicyConfig(**ck["policy_cfg"]))
p.load_state_dict(ck["policy_state"]); p.eval()

for d in [0.7, 0.9, 1.0, 1.1, 1.3, 1.5]:
    obs = torch.tensor([d, 0.0, 0.0, 0.0], dtype=torch.float32)
    with torch.no_grad():
        out = p(obs.unsqueeze(0)).item()
    print(f"d={d:.2f}m  action={out:+.3f}")
```

Expect: positive (forward) when too close, ≈ 0 at the implicit setpoint,
negative (reverse) when too far. The day-1 1D model crosses zero near 1.0 m,
which matches Tyler's median driving distance.

---

## 5. Export for Pi deployment

The Pi runs numpy-only inference (no torch). Convert each torch checkpoint:

```
$PY training/export_policy.py training/runs/4_23_1d/bc.pt \
    -o training/runs/4_23_1d/bc_policy.npz

$PY training/export_policy.py training/runs/4_23_2d/bc.pt \
    -o training/runs/4_23_2d/bc_policy.npz
```

The npz embeds: weight/bias matrices per layer, activation names, obs mean/std,
`action_dim`, `obs_dim`. The Pi's `NumpyMLPPolicy` reproduces the torch output
to ~1e-8 max abs error.

---

## 6. Deploy via the dashboard

1. Connect to Pi.
2. **RL Training** tab → **Upload Policy (.npz)** → pick a `bc_policy.npz` file.
   The server inspects the npz's `action_dim` and routes:
   - `action_dim=1` → `bc_1d` slot
   - `action_dim=2` → `bc_2d` slot
3. The two **slot status mini-cards** show what's currently installed.
   You can have one of each at the same time.
4. Pick a **policy** from the dropdown:

   | Policy | Slot used | What it does |
   |---|---|---|
   | `bc_1d` | bc_1d | Trained 1D forward speed, applied symmetrically as `[v, v]` |
   | `bc_2d` | bc_2d | Trained 2D `[L, R]` directly |
   | `bc_2d_pivot` (Tier 1) | bc_2d | bc_2d output + heading P-controller. Continuous closed-loop heading correction; cart stays facing the marker. |
   | `bc_2d_translate` (Tier 2) | bc_2d | bc_2d normally, with a deterministic pivot-translate-pivot maneuver triggered by `|heading|` > 17° while the person is stationary. |

5. Set parameters:
   - `step_hz = 15` (matches your training data rate)
   - `action_scale = 0.25` for the **first** test of any new policy. Increase
     to 0.35 (training value) once you trust it not to bolt off the table.
   - `max_steps = 300` (~20 s)
6. Confirm vision tracking is on and the marker is visible. ArUco needs to
   produce a valid distance/heading for the policy to consume.
7. Click **Start Exploration + Record**. Both video and action log are
   captured (no extra setup).
8. Click **Stop** or let `max_steps` end it. Click **Download Session Bundle**
   to grab the new clip.

### Recommended deploy order

Lowest risk first:

1. `bc_1d` (5 min) — baseline, person walks forward and backward, cart should
   chase and retreat. Confirms BC works on hardware.
2. `bc_2d` (5 min) — same scenario; should look identical to bc_1d (training
   data had `L ≈ R` everywhere). If it diverges, something's miscalibrated.
3. `bc_2d_pivot` (5–10 min) — same as above, plus step ~30 cm to the side
   mid-test. Cart should rotate to face you continuously.
4. `bc_2d_translate` (10 min) — stand at ~1.0 m, then **stay still** and
   step laterally 0.5 m. The cart freezes momentarily, then spin-drives-spins
   to follow. The "stay still" condition matters: if you keep walking, the
   FSM debounce won't fire.

---

## 7. Tier 1 and Tier 2 tunable knobs

All in `cart_stack/agent/rl_session.py` near the top, redeploy required if
you change them:

```python
# Tier 1
_TIER1_K_HEADING = 0.6                  # turn gain. ↑ = snappier rotation
                                        #            ↓ = if it oscillates

# Tier 2
_TIER2_HEADING_TRIGGER_RAD = 0.30       # ~17°. ↑ to fire less often
_TIER2_DEBOUNCE_FRAMES = 3              # consecutive frames over threshold
_TIER2_PERSON_STATIONARY_M_PER_S = 0.10 # only fire when |lin_v| < this
_TIER2_PIVOT_OMEGA = 0.6                # rad/s during pivot phases
_TIER2_DRIVE_LINEAR = 0.30              # m/s during translate phase
_TIER2_MAX_TOTAL_S = 7.0                # safety cutoff for whole maneuver
_CART_MAX_SPEED_MPS = 0.9               # if your real cart is faster/slower
_CART_TRACK_WIDTH_M = 0.58              # measure between wheel centerlines
```

**If Tier 2 is consistently off-target by some fixed angle**, the open-loop
pivot duration is wrong — usually because `_CART_MAX_SPEED_MPS` doesn't match
your hardware. Re-measure with a stopwatch (forward command 1.0 for 2 s,
measure travelled distance, divide).

**If Tier 2 over-rotates**, lower `_TIER2_PIVOT_OMEGA`. Slower pivot = less
PWM-ratio error per second.

---

## 8. Online RL fine-tuning (optional, after BC works)

Once BC is deployed and behaving reasonably, you can refine with PPO:

```
# Sim first (warm up against env_sim_1d, free)
$PY training/train_ppo_1d.py --backend sim --total-steps 100000 \
    --init-bc training/runs/4_23_1d/bc.pt \
    --out-dir training/runs/4_23_1d_ppo

# On real hardware (slow, hardware-bound)
$PY training/train_ppo_1d.py --backend real \
    --pi-url http://<pi-ip>:8001 \
    --init-bc training/runs/4_23_1d/bc.pt \
    --total-steps 5000 --rollout-steps 64 \
    --out-dir training/runs/4_23_1d_ppo_real
```

Outputs an `ppo.pt` (final), `ppo_best.pt` (best mean episode reward),
periodic snapshots, and a 4-panel diagnostic plot (`ppo.png`) of reward,
log_std, policy loss, value loss.

Export to npz the same way as BC: `export_policy.py runs/.../ppo.pt -o ...`.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ArUco hit: 0/N` after process_session | marker not in frame, or wrong size | check `--marker-cm`, verify marker visible in raw video |
| `bc_1d policy requested but no checkpoint` | npz not uploaded yet | dashboard → Upload Policy |
| Policy uploads but slot says "not installed" | upload routed to wrong slot | check `action_dim` in your .pt config matches what you intended |
| Cart bolts forward at start of bc_*  | obs_norm mismatch (likely no marker) | confirm vision tracking is on, marker visible, distance reads non-zero |
| Tier 2 never triggers | `|heading|` never crosses threshold, or person isn't "stationary" | step laterally then *stop* before the FSM debounce times out |
| Cart oscillates with bc_2d_pivot | K_heading too high | lower `_TIER1_K_HEADING` |
| Loss plateaus at val MSE > 0.04 | data variation insufficient | record more / different scenarios |
| `cv2 cannot open *.webm` | missing VP8 codec | `ffmpeg -i in.webm out.mp4` and process the mp4 |
| Numpy ↔ torch output diverges | activation-name decode bug | re-run `save_initial_checkpoints.py` as a sanity test, then re-export |

---

## 10. Files reference

| File | Purpose |
|---|---|
| `process_session.py` | video + log → npz with ArUco/YOLO/action |
| `train_bc.py` | config-driven BC training |
| `train_ppo_1d.py` | PPO with sim/real backends, warm-starts from BC |
| `env_sim_1d.py` | deadband-reward 1D simulator for PPO sim mode |
| `export_policy.py` | torch .pt → numpy .npz for Pi runtime |
| `eval_yolo_distance.py` | compares YOLO torso → distance vs ArUco GT |
| `save_initial_checkpoints.py` | random-init placeholder checkpoints |
| `models/policy.py` | `PolicyMLP`, `StochasticPolicyMLP`, `ValueMLP` |
| `configs/4_23_*.yaml` | day-1 hyperparams + data globs |
| `runs/<setup>/` | trained checkpoints + plots |
| `data/sessions/<name>/` | raw recorded sessions you dropped in |
| `data/processed/<day>/*.npz` | machine-ready (obs, action, perception) |

For deeper details on the architecture, training results, and what Tyler's
distance distribution looked like, see [`../report/2026-04-23_bc_training.md`](../report/2026-04-23_bc_training.md).
