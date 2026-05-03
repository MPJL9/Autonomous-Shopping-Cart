#!/usr/bin/env python3
"""
process_session.py — convert a recorded RL session (video + action log + meta)
into a training-ready npz with authoritative (obs, action) pairs.

For each video frame this script:
  1. Runs ArUco detection (using aruco/calibration/calib.npz) to get
     authoritative distance_m + heading_rad.
  2. Finds the nearest JSONL log step by wall-clock alignment.
  3. Builds obs = [distance_m, heading_rad, linear_vel, angular_vel]
     where the velocities are finite-difference estimates smoothed with EMA.
  4. Records the joystick action = [left, right] from that log step.

Output is written to training/data/processed/<session>.npz with arrays:
  obs       (N, 4)  float32
  action    (N, 2)  float32  (matches dashboard joystick, in [-1, 1])
  t_sec     (N,)    float32  (seconds from session start)
  visible   (N,)    bool     (ArUco detection flag)
  source    str              (session folder name, for bookkeeping)

Rows where ArUco never locked on and we can't form a valid distance are skipped
(visible=False are still written for the "lost target" signal later, but the
default --require-visible drops them).

Usage:
  python3 process_session.py <session-folder> [<session-folder> ...]
    [--out-dir training/data/processed]
    [--marker-cm 10]
    [--calib aruco/calibration/calib.npz]
    [--ema 0.6]
    [--keep-lost]      # write rows where ArUco lost the marker
    [--use-live-obs]   # use Pi-side obs from JSONL instead of offline ArUco
                       # (use when the recorded video has the live overlay
                       # burned in and the offline solve can't re-detect)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from datetime import datetime

import cv2
import numpy as np

# Repo layout anchors
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_CALIB = os.path.join(REPO, "aruco", "calibration", "calib.npz")
DEFAULT_OUT = os.path.join(HERE, "data", "processed")


def _parse_iso_ms(s: str) -> float:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).timestamp() * 1000.0


def _marker_obj(side_m: float) -> np.ndarray:
    h = side_m / 2.0
    return np.array(
        [[-h,  h, 0.0], [ h,  h, 0.0], [ h, -h, 0.0], [-h, -h, 0.0]],
        dtype=np.float64,
    )


def _detect_aruco(gray, detector, K, dist_coeffs, marker_obj, image_size):
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None
    best = 0
    if len(ids) > 1:
        W, H = image_size
        c = np.array([W / 2.0, H / 2.0])
        ds = [np.linalg.norm(x[0].mean(axis=0) - c) for x in corners]
        best = int(np.argmin(ds))
    pts = corners[best][0].astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        marker_obj, pts, K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        return None
    tvec = tvec.reshape(3)
    return {
        "distance_m": float(np.linalg.norm(tvec)),
        "heading_rad": float(math.atan2(tvec[0], tvec[2])),
        "corners": pts.astype(np.float32),  # (4, 2) image-space
        "marker_id": int(ids.flatten()[best]),
    }


def _detect_yolo(frame, yolo):
    res = yolo(frame, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
        return None
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    bi, bc = -1, -1.0
    for i, (c, cid) in enumerate(zip(confs, clss)):
        if cid == 0 and c > bc:  # 0 = person in COCO
            bi, bc = i, float(c)
    if bi < 0:
        return None
    return {
        "bbox": res.boxes.xyxy.cpu().numpy()[bi].astype(np.float32),  # (4,)
        "person_conf": bc,
        "kxy": res.keypoints.xy.cpu().numpy()[bi].astype(np.float32),  # (17, 2)
        "kc": res.keypoints.conf.cpu().numpy()[bi].astype(np.float32),  # (17,)
    }


def _load_session(folder: str) -> dict:
    meta_files = glob.glob(os.path.join(folder, "*.meta.json"))
    if not meta_files:
        raise FileNotFoundError(f"No .meta.json in {folder}")
    meta_path = meta_files[0]
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    stem = os.path.basename(meta_path)[: -len(".meta.json")]
    video_path = os.path.join(folder, f"{stem}.webm")
    log_path = os.path.join(folder, f"{stem}.jsonl")
    for p in (video_path, log_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {p}")

    header = None
    steps = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "header":
                header = rec
            elif rec.get("type") == "step":
                rec["_t_ms"] = _parse_iso_ms(rec["t"])
                steps.append(rec)
    if header is None or not steps:
        raise ValueError(f"Log missing header or steps: {log_path}")
    return {
        "folder": folder, "stem": stem,
        "meta": meta, "header": header, "steps": steps,
        "video_path": video_path, "log_path": log_path,
    }


def _find_step(steps, t_ms):
    """Return the step whose _t_ms is nearest t_ms (linear scan, sorted input)."""
    if t_ms <= steps[0]["_t_ms"]:
        return steps[0]
    if t_ms >= steps[-1]["_t_ms"]:
        return steps[-1]
    best = steps[0]
    best_err = abs(t_ms - best["_t_ms"])
    for s in steps[1:]:
        err = abs(t_ms - s["_t_ms"])
        if err < best_err:
            best = s
            best_err = err
        elif s["_t_ms"] - t_ms > best_err:
            break
    return best


def process(session, *, calib_path, marker_cm, ema_alpha,
            keep_lost, out_dir, with_yolo=True, yolo_weights=None,
            use_live_obs=False, verbose=True):
    meta = session["meta"]
    header = session["header"]
    steps = session["steps"]
    stem = session["stem"]
    video_path = session["video_path"]

    calib = np.load(calib_path)
    K, dist_coeffs = calib["K"], calib["dist"]
    image_size = tuple(int(x) for x in calib["image_size"])
    marker_side = marker_cm / 100.0
    marker_obj = _marker_obj(marker_side)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    yolo = None
    if with_yolo:
        if yolo_weights is None:
            yolo_weights = os.path.join(REPO, "yolov8n-pose.pt")
        if not os.path.exists(yolo_weights):
            raise FileNotFoundError(
                f"YOLO weights not found: {yolo_weights}. Pass --yolo-weights or --no-yolo.")
        from ultralytics import YOLO
        yolo = YOLO(yolo_weights)

    browser_t0_ms = float(meta["browser_started_at_ms"])
    pi_t0_ms = _parse_iso_ms(header["started_at"])
    offset_ms = pi_t0_ms - browser_t0_ms

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open {video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (W, H) != image_size:
        print(f"[warn] video size {W}x{H} != calib image_size {image_size}. "
              "Distances may be biased if the mjpeg stream was scaled.")

    fps_nominal = cap.get(cv2.CAP_PROP_FPS) or 15.0

    if verbose:
        print(f"=== {stem} ===")
        print(f"  video   : {W}x{H} @ ~{fps_nominal:.2f} fps")
        print(f"  log     : {len(steps)} steps @ {header['step_hz']:.1f} Hz  policy={header['policy']}")
        print(f"  offset  : pi-browser = {offset_ms:+.1f} ms")
        print(f"  marker  : {marker_cm:.1f} cm (dict 4x4_50)")
        print(f"  yolo    : {'enabled' if with_yolo else 'disabled'}")

    # Accumulators — one entry per video frame
    rows_obs = []; rows_action = []; rows_t = []; rows_visible = []
    rows_aruco_d = []; rows_aruco_h = []
    rows_aruco_corners = []       # (4, 2) per frame, NaN-filled when missing
    rows_yolo_detected = []; rows_yolo_conf = []
    rows_yolo_bbox = []; rows_yolo_kxy = []; rows_yolo_kc = []

    prev_dist = None; prev_head = None; prev_t = None
    ema_lin_v = 0.0; ema_ang_v = 0.0
    alpha = float(ema_alpha)

    aruco_hits = 0; yolo_hits = 0; both_hits = 0; neither_hits = 0
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_ms is None or pos_ms <= 0:
            pos_ms = (fi / max(fps_nominal, 1e-6)) * 1000.0
        t_sec = pos_ms / 1000.0
        pi_ms = browser_t0_ms + pos_ms + offset_ms
        step_rec = _find_step(steps, pi_ms)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aru = _detect_aruco(gray, detector, K, dist_coeffs, marker_obj, (W, H))
        yol = _detect_yolo(frame, yolo) if yolo is not None else None

        aruco_visible = aru is not None
        if aruco_visible:
            aruco_hits += 1
            d_a = aru["distance_m"]; h_a = aru["heading_rad"]
            rows_aruco_d.append(np.float32(d_a))
            rows_aruco_h.append(np.float32(h_a))
            rows_aruco_corners.append(aru["corners"])
        else:
            d_a = h_a = None
            rows_aruco_d.append(np.float32(np.nan))
            rows_aruco_h.append(np.float32(np.nan))
            rows_aruco_corners.append(np.full((4, 2), np.nan, dtype=np.float32))

        if use_live_obs:
            # Trust the Pi-side live tracker that produced the JSONL log.
            # Used when the recorded video has overlay burned in and the
            # offline solve can't re-detect the marker.
            live_obs = step_rec.get("obs", [0.0, 0.0, 0.0, 0.0])
            obs = np.array([
                float(live_obs[0]) if len(live_obs) > 0 else 0.0,
                float(live_obs[1]) if len(live_obs) > 1 else 0.0,
                float(live_obs[2]) if len(live_obs) > 2 else 0.0,
                float(live_obs[3]) if len(live_obs) > 3 else 0.0,
            ], dtype=np.float32)
            visible = True
        elif aruco_visible:
            if prev_dist is not None and prev_t is not None:
                dt = max(1e-3, t_sec - prev_t)
                inst_lin = (d_a - prev_dist) / dt
                dh = math.atan2(math.sin(h_a - prev_head), math.cos(h_a - prev_head))
                inst_ang = dh / dt
                ema_lin_v = alpha * inst_lin + (1 - alpha) * ema_lin_v
                ema_ang_v = alpha * inst_ang + (1 - alpha) * ema_ang_v
            prev_dist, prev_head, prev_t = d_a, h_a, t_sec
            obs = np.array([d_a, h_a, ema_lin_v, ema_ang_v], dtype=np.float32)
            visible = True
        else:
            ema_lin_v *= 0.9; ema_ang_v *= 0.9
            last_d = prev_dist if prev_dist is not None else 0.0
            last_h = prev_head if prev_head is not None else 0.0
            obs = np.array([last_d, last_h, ema_lin_v, ema_ang_v], dtype=np.float32)
            visible = False

        yolo_det = yol is not None
        if yolo_det:
            yolo_hits += 1
            rows_yolo_conf.append(np.float32(yol["person_conf"]))
            rows_yolo_bbox.append(yol["bbox"])
            rows_yolo_kxy.append(yol["kxy"])
            rows_yolo_kc.append(yol["kc"])
        else:
            rows_yolo_conf.append(np.float32(np.nan))
            rows_yolo_bbox.append(np.full(4, np.nan, dtype=np.float32))
            rows_yolo_kxy.append(np.full((17, 2), np.nan, dtype=np.float32))
            rows_yolo_kc.append(np.full(17, np.nan, dtype=np.float32))
        rows_yolo_detected.append(yolo_det)
        if yolo_det and visible:
            both_hits += 1
        if not yolo_det and not visible:
            neither_hits += 1

        rows_obs.append(obs)
        rows_action.append(np.array(step_rec["action"], dtype=np.float32))
        rows_t.append(np.float32(t_sec))
        rows_visible.append(bool(visible))
        fi += 1
    cap.release()

    obs_arr = np.stack(rows_obs)
    action_arr = np.stack(rows_action)
    t_arr = np.array(rows_t, dtype=np.float32)
    vis_arr = np.array(rows_visible, dtype=bool)
    aruco_d_arr = np.array(rows_aruco_d, dtype=np.float32)
    aruco_h_arr = np.array(rows_aruco_h, dtype=np.float32)
    aruco_corners_arr = np.stack(rows_aruco_corners) if rows_aruco_corners else np.zeros((0, 4, 2), dtype=np.float32)
    yolo_det_arr = np.array(rows_yolo_detected, dtype=bool)
    yolo_conf_arr = np.array(rows_yolo_conf, dtype=np.float32)
    yolo_bbox_arr = np.stack(rows_yolo_bbox) if rows_yolo_bbox else np.zeros((0, 4), dtype=np.float32)
    yolo_kxy_arr = np.stack(rows_yolo_kxy) if rows_yolo_kxy else np.zeros((0, 17, 2), dtype=np.float32)
    yolo_kc_arr = np.stack(rows_yolo_kc) if rows_yolo_kc else np.zeros((0, 17), dtype=np.float32)

    if not keep_lost:
        mask = vis_arr
        kept = int(mask.sum())
        if verbose:
            print(f"  dropped {fi - kept} rows where ArUco lost marker "
                  f"({fi} -> {kept})")
        obs_arr = obs_arr[mask]; action_arr = action_arr[mask]
        t_arr = t_arr[mask]; vis_arr = vis_arr[mask]
        aruco_d_arr = aruco_d_arr[mask]; aruco_h_arr = aruco_h_arr[mask]
        aruco_corners_arr = aruco_corners_arr[mask]
        yolo_det_arr = yolo_det_arr[mask]; yolo_conf_arr = yolo_conf_arr[mask]
        yolo_bbox_arr = yolo_bbox_arr[mask]; yolo_kxy_arr = yolo_kxy_arr[mask]
        yolo_kc_arr = yolo_kc_arr[mask]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stem}.npz")
    np.savez_compressed(
        out_path,
        obs=obs_arr, action=action_arr, t_sec=t_arr, visible=vis_arr,
        aruco_distance_m=aruco_d_arr,
        aruco_heading_rad=aruco_h_arr,
        aruco_corners=aruco_corners_arr,
        yolo_detected=yolo_det_arr,
        yolo_person_conf=yolo_conf_arr,
        yolo_bbox=yolo_bbox_arr,
        yolo_kxy=yolo_kxy_arr,
        yolo_kc=yolo_kc_arr,
        source=os.path.basename(session["folder"]),
        marker_cm=np.float32(marker_cm),
        pi_minus_browser_ms=np.float32(offset_ms),
        policy=header["policy"],
        step_hz=np.float32(header["step_hz"]),
    )

    if verbose:
        N = max(fi, 1)
        print(f"  ArUco hit:  {aruco_hits}/{fi} ({aruco_hits/N:.1%})")
        if yolo is not None:
            print(f"  YOLO  hit:  {yolo_hits}/{fi} ({yolo_hits/N:.1%})")
            print(f"  both  hit:  {both_hits}/{fi} ({both_hits/N:.1%})")
            print(f"  neither:    {neither_hits}/{fi} ({neither_hits/N:.1%})")
        print(f"  wrote    :  {out_path}   N_kept={len(obs_arr)}")
        if len(obs_arr) > 0:
            print(f"           dist range   [{obs_arr[:,0].min():.2f}, {obs_arr[:,0].max():.2f}] m")
            print(f"           action L     [{action_arr[:,0].min():+.2f}, {action_arr[:,0].max():+.2f}]")
            print(f"           action R     [{action_arr[:,1].min():+.2f}, {action_arr[:,1].max():+.2f}]")
    return {
        "out_path": out_path,
        "frames": fi,
        "aruco_hits": aruco_hits,
        "yolo_hits": yolo_hits,
        "both_hits": both_hits,
        "neither_hits": neither_hits,
        "kept": len(obs_arr),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="Session folder(s) to process.")
    ap.add_argument("--marker-cm", type=float, default=15.0,
                    help="Printed marker side length in cm (default 15; the "
                         "operator-worn marker in this project is ArUco id 2, "
                         "15 cm side, dict 4x4_50).")
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--ema", type=float, default=0.6,
                    help="EMA smoothing for velocity estimates (higher = more reactive).")
    ap.add_argument("--keep-lost", action="store_true",
                    help="Keep rows where ArUco lost the marker (for learning "
                         "the 'target lost' recovery behavior).")
    ap.add_argument("--use-live-obs", action="store_true",
                    help="Take obs from the JSONL Pi-side log instead of the "
                         "offline ArUco solve. Use this when the recorded "
                         "video has the live overlay burned in (offline ArUco "
                         "fails because the overlay covers the marker). Marks "
                         "all frames as visible since the live tracker already "
                         "succeeded on these.")
    ap.add_argument("--no-yolo", action="store_true",
                    help="Skip YOLO detection (ArUco-only npz, faster).")
    ap.add_argument("--yolo-weights", default=None,
                    help="Path to yolov8*-pose.pt. Default: <repo>/yolov8n-pose.pt")
    args = ap.parse_args()

    if not os.path.exists(args.calib):
        print(f"[fatal] calibration not found: {args.calib}", file=sys.stderr)
        sys.exit(1)

    for folder in args.folders:
        sess = _load_session(folder)
        process(
            sess,
            calib_path=args.calib,
            marker_cm=args.marker_cm,
            ema_alpha=args.ema,
            keep_lost=args.keep_lost,
            out_dir=args.out_dir,
            with_yolo=not args.no_yolo,
            yolo_weights=args.yolo_weights,
            use_live_obs=args.use_live_obs,
        )


if __name__ == "__main__":
    main()
