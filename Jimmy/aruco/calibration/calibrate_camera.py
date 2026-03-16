"""
Camera calibration from a video of a ChArUco board.

Record a ~30-60s video slowly moving the printed ChArUco board through
different positions, angles, and distances. Then run:

    python calibrate_camera.py --video charuco_video.mp4 --out calib.npz

The output .npz contains:
    K          — 3x3 camera intrinsic matrix
    dist       — distortion coefficients
    image_size — (width, height)
    reproj_error — reprojection error (lower is better)
"""

import argparse

import cv2
import numpy as np


def sample_video_frames(video_path: str, every_n: int, max_frames: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    idx = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser(description="Calibrate camera from ChArUco video")
    ap.add_argument("--video", required=True, help="Path to ChArUco calibration video")
    ap.add_argument("--out", default="calib.npz", help="Output calibration file")
    ap.add_argument("--squares_x", type=int, default=7)
    ap.add_argument("--squares_y", type=int, default=5)
    ap.add_argument("--square_len_cm", type=float, default=3.0)
    ap.add_argument("--marker_len_cm", type=float, default=2.2)
    ap.add_argument("--dict_name", default="DICT_4X4_50")
    ap.add_argument("--every_n_frames", type=int, default=10,
                    help="Sample every N-th frame from video")
    ap.add_argument("--max_frames", type=int, default=250)
    ap.add_argument("--no_refine", action="store_true",
                    help="Skip sub-pixel corner refinement")
    ap.add_argument("--show_debug", action="store_true",
                    help="Show detected corners while processing")
    args = ap.parse_args()

    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, args.dict_name)
    )
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_len_cm / 100.0,
        args.marker_len_cm / 100.0,
        aruco_dict,
    )

    detector = cv2.aruco.ArucoDetector(
        aruco_dict, cv2.aruco.DetectorParameters()
    )

    frames = sample_video_frames(args.video, args.every_n_frames, args.max_frames)
    if not frames:
        raise RuntimeError("No frames sampled from video.")

    all_charuco_corners = []
    all_charuco_ids = []
    img_size = None
    used = 0

    for frame in frames:
        if img_size is None:
            img_size = (frame.shape[1], frame.shape[0])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None or len(ids) < 4:
            continue

        if not args.no_refine:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            for c in corners:
                cv2.cornerSubPix(gray, c, (3, 3), (-1, -1), crit)

        ok, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners, markerIds=ids, image=gray, board=board
        )

        if charuco_ids is None or len(charuco_ids) < 12:
            continue

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        used += 1

        if args.show_debug:
            vis = frame.copy()
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
            cv2.imshow("ChArUco detections (q to quit)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if args.show_debug:
        cv2.destroyAllWindows()

    if used < 10:
        raise RuntimeError(
            f"Only {used} usable frames. Need at least 10. "
            "Try: more angles, better lighting, closer board, longer video."
        )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
    ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=0,
        criteria=criteria,
    )

    np.savez(args.out, K=K, dist=dist,
             image_size=np.array(img_size),
             reproj_error=np.array([ret]))

    print(f"Saved calibration to: {args.out}")
    print(f"Used {used} frames")
    print(f"Reprojection error: {ret:.6f}")
    print(f"K =\n{K}")
    print(f"dist = {dist.ravel()}")


if __name__ == "__main__":
    main()
