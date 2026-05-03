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
    """Persistent rpicam-vid process that streams MJPEG frames.

    Spawns rpicam-vid once and reads the continuous MJPEG output on a
    background thread, keeping the most recent complete frame ready for
    capture_jpeg. Avoids spawning a subprocess per frame, which was the
    main reason FPS tanked on prior iterations.
    """

    def __init__(self) -> None:
        import threading

        self.width = int(os.getenv("CART_CAMERA_WIDTH", "640"))
        self.height = int(os.getenv("CART_CAMERA_HEIGHT", "480"))
        self.quality = int(os.getenv("CART_CAMERA_QUALITY", "60"))
        self.framerate = int(os.getenv("CART_CAMERA_FRAMERATE", "15"))
        self._lock = threading.Lock()
        self._latest_frame: bytes | None = None
        self._process: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._stopping = False
        self.fallback = SyntheticCameraSource(width=min(self.width, 960), height=min(self.height, 540))
        self._start_process()

    def _start_process(self) -> None:
        import threading

        cmd = [
            "rpicam-vid",
            "-t", "0",
            "--codec", "mjpeg",
            "--width", str(self.width),
            "--height", str(self.height),
            "--framerate", str(self.framerate),
            "-q", str(self.quality),
            "-n",
            "-o", "-",
        ]
        self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._stopping = False
        self._reader_thread = threading.Thread(target=self._read_frames, daemon=True)
        self._reader_thread.start()

    def _read_frames(self) -> None:
        """Parse MJPEG frame boundaries from the persistent rpicam-vid stream."""
        SOI = b"\xff\xd8"
        EOI = b"\xff\xd9"
        buf = b""
        stdout = self._process.stdout
        while not self._stopping and stdout:
            chunk = stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                start = buf.find(SOI)
                if start == -1:
                    buf = b""
                    break
                end = buf.find(EOI, start + 2)
                if end == -1:
                    buf = buf[start:]
                    break
                frame = buf[start:end + 2]
                with self._lock:
                    self._latest_frame = frame
                buf = buf[end + 2:]

    def capture_jpeg(self, snapshot: RobotSnapshot) -> bytes:
        with self._lock:
            frame = self._latest_frame
        if frame is not None:
            return frame
        return self.fallback.capture_jpeg(snapshot)

    def close(self) -> None:
        self._stopping = True
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                pass
            self._process = None


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
