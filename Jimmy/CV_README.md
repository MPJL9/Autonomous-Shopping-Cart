# Computer Vision — Follow-Me Cart

## Overview

The CV subsystem gives the follow-me cart two capabilities:

1. **Distance estimation** — predict the real-time distance (meters) between the cart's rear-facing camera and the person it is following.
2. **Angle estimation** — predict the horizontal angle (radians) of the person relative to the camera's optical axis, so the RL controller knows both *how far* and *in which direction* the person is.
3. **Turn-intent classification** — predict whether the person intends to **turn left**, **turn right**, or **go straight**, so the cart can anticipate direction changes.

Distance, angle, and turn-intent run on every frame and are fed to the RL / control module.

## Architecture

```
Camera frame
    │
    ▼
┌──────────────┐
│  YOLOv8 Pose │  (frozen backbone)
│  Nano        │
└──┬───────────┘
   │  bounding box + 17 COCO keypoints
   ▼
┌──────────────────────────────┐
│  Feature extraction          │
│  • bbox height, width, area  │
│  • torso length (px)         │
│  • shoulder / hip angles     │
│  • shoulder–hip twist        │
│  • bbox center x-offset      │
└──┬──────────────┬────────────┘
   │              │
   ▼              ▼
┌──────────┐  ┌──────────┐  ┌────────────────────┐
│ Distance │  │ Angle    │  │ Turn-intent        │
│ Head     │  │ Estim.   │  │ Classifier         │
│ (MLP)    │  │ (geom.)  │  │ (temporal, GRU)    │
│ → meters │  │ → rads   │  │ → L / R / straight │
└──────────┘  └──────────┘  └────────────────────┘
```

### Backbone — YOLOv8 Pose (Nano)

- Single forward pass produces person bounding box + 17 COCO keypoints (nose, shoulders, hips, etc.)
- Kept frozen; we only train the downstream heads.

### Distance Head

- Input features derived from the bounding box and keypoints (bbox height, torso pixel length, etc.)
- Simple MLP regressor that outputs distance in meters.
- Trained with MSE loss against ArUco-marker ground truth.

### Angle Estimation

- Computes the horizontal angle of the person relative to the camera's optical axis using the bounding-box center x-position.
- Uses the camera focal length from calibration (falls back to an assumed 60° horizontal FOV if uncalibrated).
- Pure geometry — no training required: `angle = atan2(bbox_center_x − img_center_x, focal_length_x)`.
- Positive angle means the person is to the right; negative means left.
- Combined with distance, this gives the RL controller a polar-coordinate observation of the person (distance, angle), which is a natural input representation for a following policy.

### Turn-Intent Classifier

- Input: a sliding window of pose features (shoulder angle, hip angle, shoulder–hip twist, angular velocity, bbox center velocity).
- Small GRU (or 1D-CNN) that outputs `P(left)`, `P(right)`, `P(straight)`.
- Trained with cross-entropy loss.
- For the 1-D (straight-line) MVP, this head can be disabled.

## Ground-Truth Data Collection

| Signal | How we get ground truth |
|--------|------------------------|
| Distance | Person wears a printed ArUco marker; calibrated camera gives metric distance via `cv2.aruco.estimatePoseSingleMarkers`. |
| Angle | No training needed — derived geometrically from bbox center and camera focal length. Accuracy can be validated against ArUco marker lateral position. |
| Turn intent | Weak labels from shoulder-angle velocity thresholds, or light manual annotation (~50 clips). |

**Important:** during training, the ArUco marker is randomly masked / blurred / erased so the model does **not** learn to depend on it.

## Folder Structure

```
ML_Project/
├── CV_README.md              ← this file
├── aruco/
│   ├── marker_generation/    ← generate & print ChArUco boards + single markers
│   ├── calibration/          ← camera calibration from ChArUco video
│   └── ground_truth/         ← extract ArUco ground-truth distances from video
├── model/
│   ├── model.py              ← DistanceTurnModel (YOLO backbone + heads)
│   ├── dataset.py            ← dataset & feature extraction
│   ├── train.py              ← training loop
│   └── inference.py          ← real-time inference script
└── ML_project_class/         ← original notes & demos
```

## Quick Start

### 1. Generate & print calibration board

```bash
cd aruco/marker_generation
python generate_charuco_board.py          # outputs PDF in board_output/
python generate_aruco_marker.py           # outputs single-marker PDFs in marker_output/
```

### 2. Calibrate the camera

```bash
cd aruco/calibration
python calibrate_camera.py --video <charuco_video.mp4> --out calib.npz
```

### 3. Collect ground-truth distance data

```bash
cd aruco/ground_truth
python collect_ground_truth.py --video <video.mp4> --calib ../calibration/calib.npz --out gt_data.npz
```

### 4. Train the model

```bash
cd model
python train.py --data_dir ../aruco/ground_truth --calib ../aruco/calibration/calib.npz
```

### 5. Run real-time inference

```bash
cd model
python inference.py --calib ../aruco/calibration/calib.npz
```

## Dependencies

```
ultralytics
opencv-python (opencv-contrib-python for ArUco)
numpy
torch
reportlab          # PDF generation for markers
```
