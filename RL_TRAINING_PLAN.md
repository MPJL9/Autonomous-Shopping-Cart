# Autonomous Shopping Cart — RL Training Plan

## Goal
Train a policy that follows a person at a target distance (~0.65m) using two motor commands (left, right) given perception inputs (distance, heading, velocities).

---

## Architecture Overview

```
[Camera] → [Perception] → [RL Policy] → [Motor Commands]
                |                |
         ArUco (Stage 1)    PPO (Stable Baselines3)
         YOLO  (Stage 2)    or Imitation Learning
```

**Observation space** (4D): `[distance_m, heading_rad, linear_vel, angular_vel]`
**Action space** (2D continuous): `[left_motor, right_motor]` in [-1, 1]

---

## Stage 1: ArUco Marker + RL (Current Focus)

Person wears a printed ArUco marker. Camera detects it and computes exact distance and angle via `cv2.aruco.estimatePoseSingleMarkers()`. This gives sub-centimeter ground truth — no learned perception model needed.

### Why ArUco first?
- Decouples RL development from CV model development
- Instant, accurate observations — RL training can start immediately
- ~1.3ms/frame on laptop, fast enough on Pi 3
- Team can validate the RL controller on real hardware before investing in markerless perception

### Training Approaches (pick one or combine)

#### Approach A: Simulation-first PPO
1. Train in the built-in physics simulator (`CartFollowEnv` with `backend="sim"`)
2. Simulated person moves with random walk + noise
3. Transfer to real robot via sim-to-real (observation normalization helps bridge the gap)
4. Fine-tune on real hardware if needed

**Existing code:** `cart_stack/rl/train.py`
```bash
cd cart_stack/rl
python train.py --mode sim --total-timesteps 500000
```

#### Approach B: Imitation Learning → RL Fine-tune
1. Human drives the cart via web dashboard, collecting (observation, action) pairs
2. Train a behavior cloning policy as a warm start
3. Fine-tune with PPO on the real robot

**Why IL first?** The reward function may not capture all nuances of good following behavior. Human demos give a strong initialization, reducing real-world training time and avoiding early random exploration that could damage hardware.

**Data collection:** Use the Veda web dashboard to teleoperate the cart. Record timestamped (obs, action) pairs at each control step.

#### Approach C: Real-world PPO only
1. Deploy random policy on real robot
2. Train PPO directly on hardware
3. Requires safety bounds (e-stop, distance limits) to prevent collisions

**Risk:** Random exploration on real hardware can be dangerous/slow. Only recommended after sim pre-training.

### Recommended Path: A → B → C
1. **Sim PPO** to get a reasonable baseline policy (~500k steps, ~30 min)
2. **Imitation learning** to collect 10-20 min of human driving data via dashboard
3. **Real-world fine-tuning** with PPO using the IL policy as initialization

---

## Stage 2: Markerless Perception (Future)

Replace ArUco with YOLO Pose + trained distance head so the person doesn't need a marker.

- **Distance:** MLP regressor using torso length + shoulder width features (best tested: MAE ~0.055m)
- **Angle:** Geometric computation from bbox center + calibrated focal length (no training needed)
- **RL policy transfers directly** — only the perception input source changes, observation format stays the same

---

## Reward Function (existing)

```python
reward = (
    -2.0 * dist_error          # penalize distance from 0.65m target
    -1.0 * heading_error        # penalize angle deviation
    -0.3 * jerk                 # penalize jerky motor commands
    +0.5 if dist_error < 0.1m   # bonus for being close
    +1.0 if dist_error < 0.05m AND heading_error < 0.1rad  # precision bonus
    -5.0 if terminal            # penalty for too close/far (episode ends)
)
```

Terminal conditions: distance < 0.2m or distance > 2.0m.

---

## PPO Hyperparameters (existing defaults)

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (MlpPolicy) |
| Learning rate | 3e-4 |
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| Normalization | VecNormalize (obs + reward) |
| Checkpoints | Every 10k steps |
| Eval | Every 5k steps |

---

## Key Files

| File | Purpose |
|------|---------|
| `cart_stack/rl/env.py` | Gymnasium env with sim and real backends |
| `cart_stack/rl/train.py` | PPO training script |
| `cart_stack/rl/deploy.py` | Inference wrapper for deployment |
| `cart_stack/agent/vision.py` | ArUco + person trackers |
| `cart_stack/agent/app.py` | Pi agent (motors, camera, API) |
| `cart_stack/dashboard/` | Web dashboard for teleoperation |

---

## Hardware Setup

- **Robot:** Raspberry Pi 3 + camera module (IMX708 Wide NoIR) + dual DC motors
- **Perception:** ArUco marker (15cm, DICT_4X4_50) worn on person's chest
- **Communication:** Pi runs FastAPI agent on port 8001; dashboard on laptop proxies to it
- **Safety:** E-stop via dashboard, terminal distance bounds in env

---

## TODO / Open Items

- [ ] Calibrate Pi camera intrinsics (needed for accurate ArUco pose estimation)
- [ ] Print ArUco marker (15cm side, ID from DICT_4X4_50)
- [ ] Test ArUco detection FPS on Pi 3
- [ ] Run sim training baseline and evaluate in simulator
- [ ] Build imitation learning data collection pipeline (record obs/action from dashboard)
- [ ] Build behavior cloning training script
- [ ] Test sim-to-real transfer
- [ ] Fine-tune on real hardware
- [ ] Tune reward function based on real-world behavior

---

## Quick Start

```bash
# 1. Install dependencies
pip install stable-baselines3 gymnasium

# 2. Train in simulation
cd cart_stack/rl
python train.py --mode sim --total-timesteps 500000

# 3. Evaluate
python train.py --mode sim --eval --model-path checkpoints/best_model.zip

# 4. Deploy to real robot
# (ensure Pi agent is running and ArUco marker is visible)
python train.py --mode real --model-path checkpoints/best_model.zip
```
