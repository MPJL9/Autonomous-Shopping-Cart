from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional on laptops
    cv2 = None

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional on laptops
    np = None


@dataclass
class VisionEstimate:
    frame_jpeg: bytes
    locked: bool
    distance_m: float | None = None
    heading_rad: float | None = None


# ---------------------------------------------------------------------------
# ArUco marker-based vision tracker
# ---------------------------------------------------------------------------

class ArUcoVisionTracker:
    """Precise distance/heading estimation using ArUco markers and calibrated camera."""

    def __init__(self) -> None:
        self.enabled = cv2 is not None and np is not None

        # Calibration file produced by aruco/calibration/calibrate_camera.py
        calib_path = os.getenv("CART_ARUCO_CALIB_FILE", "calib.npz")
        self._marker_length_m = float(os.getenv("CART_ARUCO_MARKER_SIZE_CM", "15")) / 100.0
        dict_name = os.getenv("CART_ARUCO_DICT", "DICT_4X4_50")
        self._target_marker_id: int | None = None
        _id_env = os.getenv("CART_ARUCO_MARKER_ID", "")
        if _id_env.strip():
            self._target_marker_id = int(_id_env)

        self._K: np.ndarray | None = None
        self._dist_coeffs: np.ndarray | None = None
        self._detector = None
        self._last_distance: float | None = None
        self._last_heading: float | None = None
        self._smooth_alpha = 0.4

        if not self.enabled:
            return

        # Load camera calibration
        calib_file = Path(calib_path)
        if calib_file.exists():
            calib = np.load(str(calib_file))
            self._K = calib["K"]
            self._dist_coeffs = calib["dist"]
        else:
            # Fall back to approximate intrinsics from FOV (less accurate)
            self._K = None
            self._dist_coeffs = None

        # ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dict_name)
        )
        self._detector = cv2.aruco.ArucoDetector(
            aruco_dict, cv2.aruco.DetectorParameters()
        )

    def _estimate_intrinsics(self, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
        """Approximate camera matrix from FOV when no calibration file is available."""
        hfov = math.radians(float(os.getenv("CART_CAMERA_HORIZONTAL_FOV_DEG", "62.2")))
        fx = (w / 2.0) / math.tan(hfov / 2.0)
        fy = fx  # assume square pixels
        K = np.array([[fx, 0, w / 2.0],
                       [0, fy, h / 2.0],
                       [0,  0,      1.0]], dtype=np.float64)
        return K, np.zeros(5, dtype=np.float64)

    def _choose_marker(self, ids, corners) -> int | None:
        """Pick the best marker. If a target ID is set, prefer it; otherwise pick largest."""
        if ids is None or len(ids) == 0:
            return None
        flat_ids = ids.flatten()
        if self._target_marker_id is not None and self._target_marker_id in flat_ids:
            return int(np.where(flat_ids == self._target_marker_id)[0][0])
        # Pick the marker with the largest bounding area
        best_idx = 0
        best_area = 0.0
        for i, pts in enumerate(corners):
            p = pts[0]
            w = np.linalg.norm(p[1] - p[0])
            h = np.linalg.norm(p[2] - p[1])
            area = w * h
            if area > best_area:
                best_area = area
                best_idx = i
        return best_idx

    def _draw_overlay(self, frame, corners, idx: int, distance_m: float, heading_rad: float) -> None:
        pts = corners[idx][0].astype(int)
        for j in range(4):
            cv2.line(frame, tuple(pts[j]), tuple(pts[(j + 1) % 4]), (0, 255, 0), 2)
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        label = f"ArUco {distance_m:.2f}m | {heading_rad:+.2f}rad"
        cv2.putText(frame, label, (cx - 80, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    def annotate(self, jpeg_bytes: bytes) -> VisionEstimate:
        if not self.enabled or cv2 is None or np is None or self._detector is None:
            return VisionEstimate(frame_jpeg=jpeg_bytes, locked=False)

        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return VisionEstimate(frame_jpeg=jpeg_bytes, locked=False)

        h, w = frame.shape[:2]
        K = self._K
        dist_coeffs = self._dist_coeffs
        if K is None:
            K, dist_coeffs = self._estimate_intrinsics(w, h)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        idx = self._choose_marker(ids, corners)
        if idx is None:
            self._last_distance = None
            self._last_heading = None
            return VisionEstimate(frame_jpeg=jpeg_bytes, locked=False)

        # Pose estimation
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners[idx]], self._marker_length_m, K, dist_coeffs
        )
        tvec = tvecs[0][0]  # [x, y, z] in camera frame (meters)
        distance_m = float(np.linalg.norm(tvec))
        heading_rad = float(math.atan2(tvec[0], tvec[2]))  # x/z = horizontal angle

        # Smooth readings
        if self._last_distance is not None:
            a = self._smooth_alpha
            distance_m = self._last_distance * (1 - a) + distance_m * a
            heading_rad = self._last_heading * (1 - a) + heading_rad * a
        self._last_distance = distance_m
        self._last_heading = heading_rad

        self._draw_overlay(frame, corners, idx, distance_m, heading_rad)
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            return VisionEstimate(frame_jpeg=jpeg_bytes, locked=False)

        return VisionEstimate(
            frame_jpeg=encoded.tobytes(),
            locked=True,
            distance_m=distance_m,
            heading_rad=heading_rad,
        )


def build_vision_tracker() -> ArUcoVisionTracker | PersonVisionTracker:
    """Factory: select tracker based on CART_VISION_MODE env var."""
    mode = os.getenv("CART_VISION_MODE", "hog").lower()
    if mode == "aruco":
        return ArUcoVisionTracker()
    return PersonVisionTracker()


class PersonVisionTracker:
    def __init__(self) -> None:
        self.enabled = cv2 is not None and np is not None
        self._detection_interval_s = float(os.getenv("CART_VISION_INTERVAL_SEC", "0.45"))
        self._person_height_m = float(os.getenv("CART_REFERENCE_PERSON_HEIGHT_M", "1.70"))
        self._camera_hfov_deg = float(os.getenv("CART_CAMERA_HORIZONTAL_FOV_DEG", "62.2"))
        self._camera_vfov_deg = float(os.getenv("CART_CAMERA_VERTICAL_FOV_DEG", "48.8"))
        self._scale_width_px = int(os.getenv("CART_VISION_DETECT_WIDTH_PX", "320"))
        self._last_detection_at = 0.0
        self._last_bbox: tuple[int, int, int, int] | None = None
        self._detector = None

        if self.enabled:
            detector = cv2.HOGDescriptor()
            detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self._detector = detector

    def _choose_detection(
        self,
        boxes: list[tuple[int, int, int, int]],
        frame_width: int,
    ) -> tuple[int, int, int, int] | None:
        if not boxes:
            return None

        center_x = frame_width / 2.0

        def score(box: tuple[int, int, int, int]) -> float:
            x, _, w, h = box
            area = w * h
            box_center = x + (w / 2.0)
            center_penalty = abs(box_center - center_x)
            return area - (center_penalty * 35.0)

        return max(boxes, key=score)

    def _smooth_bbox(
        self,
        bbox: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int] | None:
        if bbox is None:
            self._last_bbox = None
            return None
        if self._last_bbox is None:
            self._last_bbox = bbox
            return bbox

        alpha = 0.35
        smoothed = tuple(
            int((previous * (1.0 - alpha)) + (current * alpha))
            for previous, current in zip(self._last_bbox, bbox)
        )
        self._last_bbox = smoothed
        return smoothed

    def _detect_bbox(self, frame) -> tuple[int, int, int, int] | None:
        if not self.enabled or self._detector is None:
            return None

        frame_height, frame_width = frame.shape[:2]
        scaled_width = min(self._scale_width_px, frame_width)
        scale = scaled_width / float(frame_width)
        scaled_height = max(1, int(frame_height * scale))
        resized = cv2.resize(frame, (scaled_width, scaled_height))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        boxes, _ = self._detector.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)

        detected: list[tuple[int, int, int, int]] = []
        for x, y, w, h in boxes:
            detected.append(
                (
                    int(x / scale),
                    int(y / scale),
                    int(w / scale),
                    int(h / scale),
                )
            )

        return self._choose_detection(detected, frame_width)

    def _estimate_distance_and_heading(
        self,
        frame_width: int,
        frame_height: int,
        bbox: tuple[int, int, int, int],
    ) -> tuple[float, float]:
        x, _, w, h = bbox
        hfov = math.radians(self._camera_hfov_deg)
        vfov = math.radians(self._camera_vfov_deg)
        focal_x = (frame_width / 2.0) / math.tan(hfov / 2.0)
        focal_y = (frame_height / 2.0) / math.tan(vfov / 2.0)

        box_center_x = x + (w / 2.0)
        heading_rad = math.atan2(box_center_x - (frame_width / 2.0), focal_x)
        distance_m = max((focal_y * self._person_height_m) / max(h, 1), 0.15)
        return distance_m, heading_rad

    def _draw_overlay(
        self,
        frame,
        bbox: tuple[int, int, int, int],
        distance_m: float,
        heading_rad: float,
    ) -> None:
        x, y, w, h = bbox
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv2.rectangle(frame, top_left, bottom_right, (140, 242, 207), 2)
        cv2.rectangle(frame, (x, max(0, y - 34)), (min(frame.shape[1] - 1, x + 220), y), (17, 29, 25), -1)
        label = f"lock {distance_m:.2f}m | {heading_rad:+.2f}rad"
        cv2.putText(frame, label, (x + 10, max(16, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (222, 250, 241), 2)

    def annotate(self, jpeg_bytes: bytes) -> VisionEstimate:
        if not self.enabled or np is None or cv2 is None:
            return VisionEstimate(frame_jpeg=jpeg_bytes, locked=False)

        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return VisionEstimate(frame_jpeg=jpeg_bytes, locked=False)

        now = time.monotonic()
        bbox = self._last_bbox
        if bbox is None or now - self._last_detection_at >= self._detection_interval_s:
            bbox = self._detect_bbox(frame)
            self._last_detection_at = now

        bbox = self._smooth_bbox(bbox)
        if bbox is None:
            return VisionEstimate(frame_jpeg=jpeg_bytes, locked=False)

        distance_m, heading_rad = self._estimate_distance_and_heading(frame.shape[1], frame.shape[0], bbox)
        self._draw_overlay(frame, bbox, distance_m, heading_rad)
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            return VisionEstimate(frame_jpeg=jpeg_bytes, locked=False)

        return VisionEstimate(
            frame_jpeg=encoded.tobytes(),
            locked=True,
            distance_m=distance_m,
            heading_rad=heading_rad,
        )
