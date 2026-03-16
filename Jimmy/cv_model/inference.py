"""
Real-time inference: YOLO Pose → distance + turn prediction.

Usage:
    python inference.py --dist_model distance_model.npz --source 0
    python inference.py --dist_model distance_model.npz --turn_model turn_model.npz --source video.mp4

Controls: q to quit
"""

import argparse
import collections

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
_TURN_LABELS = ["LEFT", "RIGHT", "STRAIGHT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist_model", required=True, help="distance_model.npz")
    ap.add_argument("--turn_model", default=None, help="turn_model.npz (optional)")
    ap.add_argument("--yolo_model", default="yolov8n-pose.pt")
    ap.add_argument("--source", default="0", help="Video path or camera index")
    ap.add_argument("--conf_min", type=float, default=0.25)
    args = ap.parse_args()

    yolo = YOLO(args.yolo_model)
    dist_model = CalibratedDistance.load(args.dist_model)

    turn_model = None
    feat_buffer = None
    if args.turn_model:
        turn_model = TurnClassifier.load(args.turn_model)
        feat_buffer = collections.deque(maxlen=turn_model.T)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = yolo(frame, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            cv2.imshow("Follow-Me Cart", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy().astype(int)
        kpts = results.keypoints
        if kpts is None:
            cv2.imshow("Follow-Me Cart", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        kpts_xy = kpts.xy.cpu().numpy()
        kpts_conf = kpts.conf.cpu().numpy()

        # Best person
        best_i, best_c = None, -1.0
        for i, (c, cls_id) in enumerate(zip(confs, clss)):
            if cls_id == _PERSON_CLS and c > best_c:
                best_c, best_i = float(c), i

        if best_i is None:
            cv2.imshow("Follow-Me Cart", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Distance
        d_feats = extract_distance_features(
            boxes[best_i], kpts_xy[best_i], kpts_conf[best_i], H
        )
        dist_m = dist_model.predict(d_feats["bbox_h_norm"])

        # Turn intent
        turn_text = ""
        if turn_model is not None:
            t_feats = extract_turn_features(kpts_xy[best_i], kpts_conf[best_i])
            feat_buffer.append(t_feats)
            if len(feat_buffer) == turn_model.T:
                window = np.stack(list(feat_buffer))
                probs = turn_model.predict_probs(window)
                cls_idx = int(probs.argmax())
                turn_text = (f"{_TURN_LABELS[cls_idx]} "
                             f"(L:{probs[0]:.2f} R:{probs[1]:.2f} S:{probs[2]:.2f})")

        # Draw
        x1, y1, x2, y2 = map(int, boxes[best_i])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"dist={dist_m:.2f}m", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if turn_text:
            cv2.putText(frame, turn_text, (x1, max(20, y1 - 35)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        for (x, y), c in zip(kpts_xy[best_i], kpts_conf[best_i]):
            if c >= args.conf_min:
                cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)

        cv2.imshow("Follow-Me Cart", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
