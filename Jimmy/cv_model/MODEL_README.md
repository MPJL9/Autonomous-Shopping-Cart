# Model Architecture

## Overview

The model extracts three quantities from a single rear-facing camera stream on every frame:

1. **Distance** (meters) — how far the person is from the cart.
2. **Angle** (radians) — horizontal angle of the person relative to the camera's optical axis.
3. **Turn intent** (classification) — whether the person intends to turn left, right, or go straight.

Distance and angle form the polar-coordinate observation `(Z, α)` fed to the RL agent. Turn intent is forwarded to the deterministic lane-change controller. The YOLO Pose backbone is frozen; only the downstream heads are trained.

## Pipeline

```
                        ┌─────────────────────┐
  Camera frame ───────► │  YOLOv8 Pose (Nano) │  ◄── frozen, not trained
                        │  ultralytics        │
                        └──────┬──────────────┘
                               │
                    bbox (x1,y1,x2,y2)
                    17 COCO keypoints (x,y,conf)
                               │
             ┌─────────────────┼──────────────────┐
             │                 │                  │
             ▼                 ▼                  ▼
  extract_distance_    estimate_person_   extract_turn_
  features()           angle()            features()
  → 5 bbox/keypoint   → geometric calc   → [shoulder_angle,
    features           from bbox center     hip_angle, twist,
             │         + focal_length       angular_vel,
             │                 │            bbox_center_vel]
             ▼                 │                  │
  ┌──────────────────┐         │        ┌─────────────────────┐
  │  DistanceMLP     │         │        │  TurnClassifier      │
  │  5→16→1          │         │        │  GRU / 1D-CNN        │
  │  (113 params)    │         │        │  over T-frame window │
  │  → meters        │         │        │  → P(L/R/straight)   │
  └──────────────────┘         │        └─────────────────────┘
             │                 │                  │
             └────────┬────────┘                  │
                      ▼                           ▼
              (Z, α) → RL agent         turn signal → lane controller
```

## Distance: MLP Regressor

A small MLP trained with MSE loss against ArUco ground-truth distances.

```
Input:  5 features
        → Linear(5, 16) → ReLU → Linear(16, 1) → Softplus
Output: distance in meters
```

### Distance features (5 per frame)

| Index | Name              | Description                          |
|-------|-------------------|--------------------------------------|
| 0     | `bbox_h_norm`     | bbox height / image height           |
| 1     | `bbox_w_norm`     | bbox width / image width             |
| 2     | `bbox_area_norm`  | bbox area / image area               |
| 3     | `torso_len_norm`  | torso pixel length / image height    |
| 4     | `shoulder_w_norm` | shoulder width / image width         |

- **Parameters:** 113
- **Trained with:** MSE loss against ArUco-marker ground truth.
- **Fallback (V1):** if data is scarce, `CalibratedDistance` uses `d = a / bbox_h_norm + b` (2 params, least-squares fit) as a simpler starting point.

## Angle Estimation: Pure Geometry

No training required. The horizontal angle is computed from the bounding-box center and the camera's calibrated focal length:

```
α = atan2(bbox_center_x − img_width / 2,  focal_length_x)
```

- Positive = person is to the right; negative = left.
- Falls back to an assumed 60° horizontal FOV if the camera is uncalibrated.
- Combined with distance, gives the RL agent a polar-coordinate observation `(Z, α)`.

## Turn Intent: Temporal Classifier

A small GRU (or 1D-CNN) operating over a sliding window of `T` frames.

```
Input:  window of T frames × 5 features  →  GRU / 1D-CNN
Output: P(left), P(right), P(straight)
```

### Turn features (5 per frame)

| Index | Name               | Description                                         |
|-------|--------------------|-----------------------------------------------------|
| 0     | `shoulder_angle`   | Angle of shoulder line (radians)                    |
| 1     | `hip_angle`        | Angle of hip line (radians)                         |
| 2     | `twist`            | `shoulder_angle − hip_angle`                        |
| 3     | `angular_vel`      | Change in shoulder angle from previous frame        |
| 4     | `bbox_center_vel`  | Change in bbox center x from previous frame (px)   |

**Key insight:** when a person is about to turn, their shoulders rotate before their hips, producing a non-zero twist. Angular velocity and bbox center velocity give the classifier additional early-warning signal across the window.

- **Window size T** defaults to 15 frames (~0.5s at 30 fps).
- **Trained with:** cross-entropy loss. Weak labels from shoulder-angle velocity thresholds, or light manual annotation.
- **Can be disabled** for the 1D (straight-line) MVP.

> **Note:** the exact classifier architecture (GRU vs 1D-CNN, hidden size) will be decided after initial experiments. `model.py` includes a linear baseline (`TurnClassifier`) for rapid prototyping; `model_v2.py` includes a 2-layer MLP variant (`TurnClassifierV2`).

## Parameter Count

| Component                 | Parameters         |
|---------------------------|--------------------|
| `DistanceMLP`             | 113                |
| `TurnClassifier` (linear, T=15) | 228 (5×3×3+3) |
| **Total trainable**       | **~341**           |

(YOLO Pose Nano backbone: ~3.3M params, frozen — not counted.)

## Training

### Distance only (1-D MVP)

```bash
python train.py --gt gt_data.npz --video walk.mp4
```

Runs YOLO on each labelled frame, extracts 5 distance features, trains the MLP with MSE against ArUco ground truth.

### Distance + turn intent

```bash
python train.py --gt gt_data.npz --video walk.mp4 \
    --train_turn --turn_labels turn_labels.npy
```

Trains distance MLP first, then trains the turn classifier on the 5-feature temporal windows.

## Data Flow

```
1. Record video of person wearing ArUco marker
2. collect_ground_truth.py  → gt_data.npz (frame_idx, distance_m, angle_rad, ...)
3. train.py loads gt_data.npz + video
       │
       ├── Runs YOLO Pose on each labelled frame
       ├── Extracts 5 features → trains DistanceMLP (MSE)
       └── (optional) Extracts 5-dim turn features → trains TurnClassifier
               │
               ▼
       distance_model.npz + turn_model.npz
4. inference.py loads models → runs real-time on webcam/video
```

## Files

| File              | Purpose                                                               |
|-------------------|-----------------------------------------------------------------------|
| `model.py`        | Feature extraction, `CalibratedDistance` (V1 fallback), `TurnClassifier` (linear baseline), `estimate_person_angle` |
| `model_v2.py`     | `DistanceMLP`, `TurnClassifierV2` (2-layer MLP), `estimate_person_angle` |
| `train.py`        | Training loop for V1 models                                           |
| `train_v2.py`     | Training loop for V2 MLP models                                       |
| `inference.py`    | Real-time inference (V1)                                              |
| `inference_v2.py` | Real-time inference (V2)                                              |
