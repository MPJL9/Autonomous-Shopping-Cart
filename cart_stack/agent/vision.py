from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

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
