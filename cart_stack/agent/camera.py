from __future__ import annotations

import io
import os
import subprocess
import time
from collections.abc import Callable

from PIL import Image, ImageDraw

from cart_stack.shared.models import RobotSnapshot

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional on laptops
    cv2 = None

try:
    from picamera2 import Picamera2  # type: ignore
except Exception:  # pragma: no cover - optional on laptops
    Picamera2 = None


class CameraSource:
    def capture_jpeg(self, snapshot: RobotSnapshot) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        return None


class SyntheticCameraSource(CameraSource):
    def __init__(self, width: int = 960, height: int = 540) -> None:
        self.width = width
        self.height = height

    def capture_jpeg(self, snapshot: RobotSnapshot) -> bytes:
        image = Image.new("RGB", (self.width, self.height), color="#27422d")
        draw = ImageDraw.Draw(image)
        t = time.time()

        for index in range(18):
            x0 = int((index / 18) * self.width)
            band_height = int(self.height * 0.26 + (index % 3) * 8)
            draw.rectangle(
                [x0, self.height - band_height, x0 + int(self.width / 20), self.height],
                fill="#6b8859",
            )

        person_x = int(self.width / 2 + snapshot.heading_rad * 110)
        person_y = int(self.height * 0.48 + (t % 1.0 - 0.5) * 8)
        draw.ellipse([person_x - 18, person_y - 64, person_x + 18, person_y - 28], fill="#f1debe")
        draw.rectangle([person_x - 16, person_y - 26, person_x + 16, person_y + 56], fill="#e0c27e")

        cart_x = int(self.width / 2)
        cart_y = int(self.height * 0.82)
        draw.rounded_rectangle([cart_x - 70, cart_y - 30, cart_x + 70, cart_y + 20], radius=14, fill="#202d20")
        draw.rectangle([cart_x - 54, cart_y - 54, cart_x + 54, cart_y - 26], fill="#dac6a2")
        draw.ellipse([cart_x - 48, cart_y + 6, cart_x - 22, cart_y + 32], fill="#101810")
        draw.ellipse([cart_x + 22, cart_y + 6, cart_x + 48, cart_y + 32], fill="#101810")

        overlay = [
            "Pi agent demo feed",
            f"mode={snapshot.controller_mode} estop={snapshot.estop}",
            f"left={snapshot.left_command:.2f} right={snapshot.right_command:.2f}",
            f"gap={snapshot.distance_m:.2f}m heading={snapshot.heading_rad:.2f}rad",
        ]
        y = 24
        for line in overlay:
            draw.text((24, y), line, fill="#f6f3ea")
            y += 26

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()


class OpenCVCameraSource(CameraSource):
    def __init__(self, index: int = 0) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is not available.")
        self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            raise RuntimeError("Could not open the camera device.")
        self.fallback = SyntheticCameraSource()

    def capture_jpeg(self, snapshot: RobotSnapshot) -> bytes:
        ok, frame = self.capture.read()
        if not ok:
            return self.fallback.capture_jpeg(snapshot)

        cv2.putText(
            frame,
            f"gap={snapshot.distance_m:.2f}m heading={snapshot.heading_rad:.2f}rad",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 240, 240),
            2,
        )
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            return self.fallback.capture_jpeg(snapshot)
        return encoded.tobytes()

    def close(self) -> None:
        self.capture.release()


class Picamera2Source(CameraSource):
    def __init__(self) -> None:
        if Picamera2 is None or cv2 is None:
            raise RuntimeError("Picamera2 or OpenCV is unavailable.")
        self.camera = Picamera2()
        self.camera.configure(self.camera.create_preview_configuration(main={"size": (1280, 720)}))
        self.camera.start()
        # Let the CSI camera warm up so the first capture is less likely to fail during boot.
        time.sleep(0.2)
        self.fallback = SyntheticCameraSource()

    def capture_jpeg(self, snapshot: RobotSnapshot) -> bytes:
        try:
            frame = self.camera.capture_array()
            cv2.putText(
                frame,
                f"gap={snapshot.distance_m:.2f}m heading={snapshot.heading_rad:.2f}rad",
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            success, encoded = cv2.imencode(".jpg", frame)
            if success:
                return encoded.tobytes()
        except Exception:
            pass
        return self.fallback.capture_jpeg(snapshot)

    def close(self) -> None:
        self.camera.stop()


class RpicamJpegSource(CameraSource):
    def __init__(self) -> None:
        self.width = int(os.getenv("CART_CAMERA_WIDTH", "1280"))
        self.height = int(os.getenv("CART_CAMERA_HEIGHT", "720"))
        self.quality = int(os.getenv("CART_CAMERA_QUALITY", "85"))
        self.capture_timeout_s = float(os.getenv("CART_CAMERA_CAPTURE_TIMEOUT_SEC", "4.0"))
        self.min_interval_s = float(os.getenv("CART_CAMERA_MIN_INTERVAL_SEC", "0.25"))
        self._last_capture_at = 0.0
        self._last_frame: bytes | None = None
        self.fallback = SyntheticCameraSource(width=min(self.width, 960), height=min(self.height, 540))

    def _command(self) -> list[str]:
        return [
            "rpicam-jpeg",
            "-n",
            "-t",
            "1",
            "--immediate",
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--quality",
            str(self.quality),
            "-o",
            "-",
        ]

    def capture_jpeg(self, snapshot: RobotSnapshot) -> bytes:
        now = time.monotonic()
        if self._last_frame is not None and now - self._last_capture_at < self.min_interval_s:
            return self._last_frame

        try:
            result = subprocess.run(
                self._command(),
                capture_output=True,
                check=False,
                timeout=self.capture_timeout_s,
            )
            if result.returncode == 0 and result.stdout:
                self._last_capture_at = now
                self._last_frame = result.stdout
                return result.stdout
        except Exception:
            pass

        if self._last_frame is not None:
            return self._last_frame
        return self.fallback.capture_jpeg(snapshot)


class ResilientCameraSource(CameraSource):
    def __init__(
        self,
        factories: list[tuple[str, Callable[[], CameraSource]]],
        retry_interval_s: float = 5.0,
    ) -> None:
        self._factories = factories
        self._retry_interval_s = max(retry_interval_s, 0.0)
        self._fallback = SyntheticCameraSource()
        self._active: CameraSource = self._fallback
        self._active_name = "synthetic"
        self._next_retry_at = 0.0
        self._activate_best_available(initial=True)

    def _log(self, message: str) -> None:
        print(f"[camera] {message}", flush=True)

    def _swap_active(self, source: CameraSource, name: str) -> None:
        if self._active is not self._fallback:
            try:
                self._active.close()
            except Exception:
                pass
        self._active = source
        self._active_name = name

    def _activate_best_available(self, initial: bool = False) -> bool:
        for name, factory in self._factories:
            try:
                source = factory()
            except Exception as exc:
                self._log(f"{name} camera unavailable: {exc!r}")
                continue
            self._swap_active(source, name)
            self._log(f"Using {name} camera source.")
            return True

        self._next_retry_at = time.monotonic() + self._retry_interval_s
        if self._active is not self._fallback:
            self._swap_active(self._fallback, "synthetic")
        elif initial:
            self._active_name = "synthetic"
        self._log("Falling back to synthetic camera feed. Will retry real camera automatically.")
        return False

    def capture_jpeg(self, snapshot: RobotSnapshot) -> bytes:
        if self._active_name == "synthetic" and time.monotonic() >= self._next_retry_at:
            self._activate_best_available()

        try:
            return self._active.capture_jpeg(snapshot)
        except Exception as exc:
            self._log(f"{self._active_name} camera capture failed: {exc!r}")
            self._activate_best_available()
            return self._active.capture_jpeg(snapshot)

    def close(self) -> None:
        if self._active is not self._fallback:
            self._active.close()


def build_camera_source() -> CameraSource:
    mode = os.getenv("CART_CAMERA_MODE", "auto").strip().lower()
    retry_interval_s = float(os.getenv("CART_CAMERA_RETRY_SEC", "5.0"))

    if mode == "mock":
        return SyntheticCameraSource()

    factories: list[tuple[str, Callable[[], CameraSource]]] = []
    if mode in {"auto", "rpicam"}:
        factories.append(("rpicam", RpicamJpegSource))
    if mode in {"auto", "picamera2"} and Picamera2 is not None and cv2 is not None:
        factories.append(("picamera2", Picamera2Source))
    if mode in {"auto", "opencv"} and cv2 is not None:
        factories.append(("opencv", OpenCVCameraSource))

    if not factories:
        return SyntheticCameraSource()

    return ResilientCameraSource(factories, retry_interval_s=retry_interval_s)
