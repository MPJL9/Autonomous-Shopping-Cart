"""
Training script for the calibrated distance model (and optionally turn classifier).

Usage (distance only — 1-D MVP):
    python train.py \
        --gt ../aruco/ground_truth/gt_data.npz \
        --video walk_video.mp4

Usage (distance + turn):
    python train.py \
        --gt ../aruco/ground_truth/gt_data.npz \
        --video walk_video.mp4 \
        --train_turn --turn_labels turn_labels.npy
"""

import argparse

import cv2
import numpy as np
from ultralytics import YOLO

from model import (
    extract_distance_features,
    extract_turn_features,
    CalibratedDistance,
    TurnClassifier,
)

_PERSON_CLS = 0


def extract_all_features(video_path, gt_npz_path, yolo_model_path, need_turn=False):
    """Run YOLO on each labelled frame, extract distance & turn features."""
    gt = np.load(gt_npz_path)
    frame_indices = gt["frame_idx"]
    gt_distances = gt["distance_m"]

    model = YOLO(yolo_model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    needed = set(frame_indices.tolist())
    frame_to_rows = {}
    for i, fi in enumerate(frame_indices):
        frame_to_rows.setdefault(int(fi), []).append(i)

    bbox_h_norms = np.full(len(gt_distances), np.nan)
    turn_feats = np.zeros((len(gt_distances), 3), dtype=np.float32) if need_turn else None

    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx not in needed:
            fidx += 1
            continue

        results = model(frame, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            fidx += 1
            continue

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy().astype(int)
        kpts = results.keypoints
        if kpts is None:
            fidx += 1
            continue

        kpts_xy = kpts.xy.cpu().numpy()
        kpts_conf = kpts.conf.cpu().numpy()

        # Best person
        best_i, best_c = None, -1.0
        for i, (c, cls_id) in enumerate(zip(confs, clss)):
            if cls_id == _PERSON_CLS and c > best_c:
                best_c, best_i = float(c), i

        if best_i is not None:
            d_feats = extract_distance_features(
                boxes[best_i], kpts_xy[best_i], kpts_conf[best_i], H
            )
            for row in frame_to_rows[fidx]:
                bbox_h_norms[row] = d_feats["bbox_h_norm"]

            if need_turn:
                t_feats = extract_turn_features(kpts_xy[best_i], kpts_conf[best_i])
                for row in frame_to_rows[fidx]:
                    turn_feats[row] = t_feats

        fidx += 1

    cap.release()

    # Drop rows where YOLO didn't detect a person
    valid = ~np.isnan(bbox_h_norms)
    bbox_h_norms = bbox_h_norms[valid]
    gt_distances = gt_distances[valid]
    if need_turn:
        turn_feats = turn_feats[valid]

    print(f"Extracted features for {len(gt_distances)} / {valid.size} ground-truth entries")
    return bbox_h_norms, gt_distances, turn_feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="Ground-truth .npz")
    ap.add_argument("--video", required=True, help="Source video")
    ap.add_argument("--yolo_model", default="yolov8n-pose.pt")
    ap.add_argument("--save_dist", default="distance_model.npz")
    # Turn options
    ap.add_argument("--train_turn", action="store_true")
    ap.add_argument("--turn_labels", default=None, help=".npy with per-frame labels (0=L,1=R,2=S)")
    ap.add_argument("--window_size", type=int, default=15)
    ap.add_argument("--save_turn", default="turn_model.npz")
    args = ap.parse_args()

    # Extract features
    bbox_h_norms, gt_distances, turn_feats = extract_all_features(
        args.video, args.gt, args.yolo_model, need_turn=args.train_turn
    )

    # ── Fit distance model ──
    dist_model = CalibratedDistance()
    dist_model.fit(bbox_h_norms, gt_distances)

    preds = np.array([dist_model.predict(h) for h in bbox_h_norms])
    mse = np.mean((preds - gt_distances) ** 2)
    mae = np.mean(np.abs(preds - gt_distances))
    print(f"\nDistance model:  a={dist_model.a:.4f}  b={dist_model.b:.4f}")
    print(f"  MSE = {mse:.6f} m²")
    print(f"  MAE = {mae:.4f} m")

    dist_model.save(args.save_dist)
    print(f"Saved distance model to {args.save_dist}")

    # ── Fit turn classifier ──
    if args.train_turn:
        labels = np.load(args.turn_labels) if args.turn_labels else None
        if labels is None:
            print("\nSkipping turn training: no --turn_labels provided.")
            return

        # Build sliding windows
        T = args.window_size
        n = len(turn_feats)
        if n < T:
            print(f"\nNot enough samples ({n}) for window size {T}.")
            return

        windows = np.stack([turn_feats[i:i+T] for i in range(n - T + 1)])
        win_labels = labels[T-1:]  # label for last frame in each window

        print(f"\nTraining turn classifier on {len(windows)} windows...")
        turn_model = TurnClassifier(window_size=T)
        turn_model.fit(windows, win_labels)
        turn_model.save(args.save_turn)
        print(f"Saved turn model to {args.save_turn}")


if __name__ == "__main__":
    main()
