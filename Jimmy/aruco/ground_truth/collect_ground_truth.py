"""
Collect ground-truth distance data by detecting ArUco markers in a video.

The person wears a printed ArUco marker. This script detects the marker in
each frame, computes its 6-DoF pose using calibrated camera intrinsics, and
saves per-frame distance measurements to an .npz file.

Usage:
    python collect_ground_truth.py \
        --video walk_video.mp4 \
        --calib ../calibration/calib.npz \
        --out gt_data.npz \
        --marker_length_cm 15

Output .npz contains:
    frame_idx   — frame number (int32)
    time_sec    — timestamp in seconds (float64)
    marker_id   — detected ArUco marker ID (int32)
    center_px   — marker center in pixels, (N, 2)
    rvec        — rotation vector, (N, 3)
    tvec        — translation vector in meters, (N, 3)
    distance_m  — Euclidean distance in meters (float64)
    K, dist_coeffs — camera intrinsics (copied from calib file)
"""

import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(
        description="Extract ArUco ground-truth distances from video"
    )
    ap.add_argument("--video", required=True, help="Input video with visible ArUco marker")
    ap.add_argument("--calib", required=True, help="Camera calibration .npz file")
    ap.add_argument("--out", default="gt_data.npz", help="Output .npz path")
    ap.add_argument("--marker_length_cm", type=float, default=15.0,
                    help="Printed marker side length in cm (must match!)")
    ap.add_argument("--dict_name", default="DICT_4X4_50")
    ap.add_argument("--show", action="store_true", help="Show detections live")
    args = ap.parse_args()

    # Load calibration
    calib = np.load(args.calib)
    K = calib["K"]
    dist_coeffs = calib["dist"]

    marker_length_m = args.marker_length_cm / 100.0

    # ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, args.dict_name)
    )
    detector = cv2.aruco.ArucoDetector(
        aruco_dict, cv2.aruco.DetectorParameters()
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Accumulate detections
    frame_idx_list = []
    time_sec_list = []
    id_list = []
    center_px_list = []
    tvec_list = []
    rvec_list = []
    dist_list = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_length_m, K, dist_coeffs
            )

            for i, marker_id in enumerate(ids.flatten()):
                pts = corners[i][0]
                center_px = pts.mean(axis=0)
                tvec = tvecs[i][0].astype(np.float64)
                rvec = rvecs[i][0].astype(np.float64)
                distance_m = float(np.linalg.norm(tvec))

                frame_idx_list.append(frame_idx)
                time_sec_list.append(frame_idx / fps)
                id_list.append(int(marker_id))
                center_px_list.append(center_px.astype(np.float64))
                tvec_list.append(tvec)
                rvec_list.append(rvec)
                dist_list.append(distance_m)

            if args.show:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for i in range(len(ids)):
                    cv2.drawFrameAxes(frame, K, dist_coeffs,
                                      rvecs[i], tvecs[i],
                                      marker_length_m * 0.5)

        if args.show:
            cv2.imshow("Ground truth collection (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    # Pack into arrays
    n = len(id_list)
    np.savez(
        args.out,
        frame_idx=np.array(frame_idx_list, dtype=np.int32),
        time_sec=np.array(time_sec_list, dtype=np.float64),
        marker_id=np.array(id_list, dtype=np.int32),
        center_px=(np.stack(center_px_list) if n > 0
                   else np.zeros((0, 2), dtype=np.float64)),
        rvec=(np.stack(rvec_list) if n > 0
              else np.zeros((0, 3), dtype=np.float64)),
        tvec=(np.stack(tvec_list) if n > 0
              else np.zeros((0, 3), dtype=np.float64)),
        distance_m=np.array(dist_list, dtype=np.float64),
        K=K,
        dist_coeffs=dist_coeffs,
    )

    print(f"Saved {args.out}")
    print(f"Total detections: {n}")
    print(f"Frames processed: {frame_idx}")
    if n > 0:
        dists = np.array(dist_list)
        print(f"Distance range: {dists.min():.3f}m – {dists.max():.3f}m")


if __name__ == "__main__":
    main()
