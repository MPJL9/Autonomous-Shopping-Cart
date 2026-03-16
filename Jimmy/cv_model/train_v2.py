"""
Training script for V2 model (MLP distance + 2-layer turn classifier).

Usage (distance only):
    python train_v2.py --gt ../aruco/ground_truth/gt_data.npz --video walk.mp4

Usage (distance + turn):
    python train_v2.py --gt ../aruco/ground_truth/gt_data.npz --video walk.mp4 \
        --train_turn --turn_labels turn_labels.npy
"""

import argparse

import cv2
import numpy as np
from ultralytics import YOLO

from model_v2 import (
    extract_distance_features_v2,
    extract_turn_features_v2,
    NUM_DIST_FEATURES_V2,
    DistanceMLP,
    TurnClassifierV2,
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

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    needed = set(frame_indices.tolist())
    frame_to_rows = {}
    for i, fi in enumerate(frame_indices):
        frame_to_rows.setdefault(int(fi), []).append(i)

    dist_feats = np.full((len(gt_distances), NUM_DIST_FEATURES_V2), np.nan)
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

        best_i, best_c = None, -1.0
        for i, (c, cls_id) in enumerate(zip(confs, clss)):
            if cls_id == _PERSON_CLS and c > best_c:
                best_c, best_i = float(c), i

        if best_i is not None:
            d_feat = extract_distance_features_v2(
                boxes[best_i], kpts_xy[best_i], kpts_conf[best_i], W, H
            )
            for row in frame_to_rows[fidx]:
                dist_feats[row] = d_feat

            if need_turn:
                t_feat = extract_turn_features_v2(kpts_xy[best_i], kpts_conf[best_i])
                for row in frame_to_rows[fidx]:
                    turn_feats[row] = t_feat

        fidx += 1

    cap.release()

    valid = ~np.isnan(dist_feats[:, 0])
    dist_feats = dist_feats[valid]
    gt_distances = gt_distances[valid]
    if need_turn:
        turn_feats = turn_feats[valid]

    print(f"Extracted features for {len(gt_distances)} / {valid.size} ground-truth entries")
    return dist_feats, gt_distances, turn_feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--yolo_model", default="yolov8n-pose.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save_dist", default="distance_model_v2.npz")
    ap.add_argument("--train_turn", action="store_true")
    ap.add_argument("--turn_labels", default=None)
    ap.add_argument("--window_size", type=int, default=15)
    ap.add_argument("--save_turn", default="turn_model_v2.npz")
    args = ap.parse_args()

    dist_feats, gt_distances, turn_feats = extract_all_features(
        args.video, args.gt, args.yolo_model, need_turn=args.train_turn
    )

    # ── Distance MLP ──
    print("\nTraining distance MLP...")
    dist_model = DistanceMLP()
    dist_model.fit(dist_feats, gt_distances, lr=args.lr, epochs=args.epochs)

    preds = np.array([dist_model.predict(x) for x in dist_feats])
    mse = np.mean((preds - gt_distances) ** 2)
    mae = np.mean(np.abs(preds - gt_distances))
    print(f"\n  MSE = {mse:.6f} m²")
    print(f"  MAE = {mae:.4f} m")

    dist_model.save(args.save_dist)
    print(f"Saved to {args.save_dist}")

    # ── Turn classifier ──
    if args.train_turn:
        labels = np.load(args.turn_labels) if args.turn_labels else None
        if labels is None:
            print("\nSkipping turn training: no --turn_labels provided.")
            return

        T = args.window_size
        n = len(turn_feats)
        if n < T:
            print(f"\nNot enough samples ({n}) for window size {T}.")
            return

        windows = np.stack([turn_feats[i:i+T] for i in range(n - T + 1)])
        win_labels = labels[T-1:]

        print(f"\nTraining turn classifier V2 on {len(windows)} windows...")
        turn_model = TurnClassifierV2(window_size=T)
        turn_model.fit(windows, win_labels, lr=args.lr, epochs=args.epochs)
        turn_model.save(args.save_turn)
        print(f"Saved to {args.save_turn}")


if __name__ == "__main__":
    main()
