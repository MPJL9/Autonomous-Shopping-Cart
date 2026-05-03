from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx

from cart_stack.shared.commands import HELP_TEXT, execute_robot_command
from cart_stack.shared.models import ConnectionInfo, DashboardStatus, RobotSnapshot


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def make_idle_snapshot(last_command: str = "idle") -> RobotSnapshot:
    return RobotSnapshot(updated_at=_now_iso(), last_command=last_command)


class MockRobotBackend:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._left_command = 0.0
        self._right_command = 0.0
        self._controller_mode = "manual"
        self._estop = False
        self._linear_velocity_mps = 0.0
        self._angular_velocity_rad_s = 0.0
        self._distance_m = 0.82
        self._heading_rad = 0.08
        self._target_gap_m = 0.65
        self._fps = 18.0
        self._command_latency_ms = 24.0
        self._last_command = "idle"
        self._vision_enabled = False
        self._vision_locked = False
        self._vision_mode = "aruco"
        self._left1_motor_pin = 13
        self._left2_motor_pin = 16
        self._right1_motor_pin = 19
        self._right2_motor_pin = 12
        self._left1_invert = False
        self._left2_invert = False
        self._right1_invert = True
        self._right2_invert = True
        self._left1_trim = 0.0
        self._left2_trim = 0.0
        self._right1_trim = 0.0
        self._right2_trim = 0.0
        self._stop_deadband = 0.04
        self._logs: deque[str] = deque(maxlen=64)
        self._last_update = time.perf_counter()
        self._log("Mock cart ready. Browser camera mode is available for local testing.")

    async def close(self) -> None:
        return None

    def _log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self._logs.appendleft(f"[{stamp}] {message}")

    def _advance_unlocked(self) -> None:
        now = time.perf_counter()
        dt = min(now - self._last_update, 0.5)
        self._last_update = now
        wall_time = time.time()

        if self._controller_mode == "auto" and not self._estop:
            if self._vision_enabled and not self._vision_locked:
                self._left_command = 0.0
                self._right_command = 0.0
            else:
                gap_error = self._distance_m - self._target_gap_m
                forward = _clamp(gap_error * 1.35, -0.55, 0.55)
                turn = _clamp(-self._heading_rad * 0.75, -0.18, 0.18)
                self._left_command = _clamp(forward - turn, -1.0, 1.0)
                self._right_command = _clamp(forward + turn, -1.0, 1.0)

        if self._estop:
            self._left_command = 0.0
            self._right_command = 0.0

        target_linear = 0.92 * ((self._left_command + self._right_command) / 2.0)
        target_angular = 2.4 * ((self._right_command - self._left_command) / 2.0)

        linear_blend = min(dt * 2.6, 1.0)
        angular_blend = min(dt * 3.4, 1.0)
        self._linear_velocity_mps += (target_linear - self._linear_velocity_mps) * linear_blend
        self._angular_velocity_rad_s += (target_angular - self._angular_velocity_rad_s) * angular_blend

        if self._vision_enabled:
            self._vision_locked = True
            self._heading_rad = _wrap_angle(math.sin(wall_time * 0.55) * 0.22)
            self._distance_m = _clamp(
                self._target_gap_m + 0.16 + math.sin(wall_time * 0.72) * 0.14,
                0.25,
                3.5,
            )
        else:
            self._vision_locked = False
            self._heading_rad = _wrap_angle(self._heading_rad + self._angular_velocity_rad_s * dt)
            breathing = math.sin(wall_time * 0.8) * 0.01
            self._distance_m += (-self._linear_velocity_mps * 0.24 + breathing) * dt
            self._distance_m = _clamp(self._distance_m, 0.25, 3.5)

        self._fps = 17.0 + 2.4 * math.sin(wall_time * 0.4)
        self._command_latency_ms = 20.0 + abs(self._left_command - self._right_command) * 28.0

    async def set_drive(self, left: float, right: float) -> None:
        async with self._lock:
            self._advance_unlocked()
            self._controller_mode = "manual"
            self._left_command = _clamp(left, -1.0, 1.0)
            self._right_command = _clamp(right, -1.0, 1.0)
            self._last_command = f"drive {self._left_command:.2f} {self._right_command:.2f}"
            self._log(f"Manual drive set to left={self._left_command:.2f}, right={self._right_command:.2f}.")

    async def stop(self) -> None:
        async with self._lock:
            self._advance_unlocked()
            self._left_command = 0.0
            self._right_command = 0.0
            self._last_command = "stop"
            self._log("Stop command received.")

    async def set_mode(self, mode: str) -> None:
        async with self._lock:
            self._advance_unlocked()
            self._controller_mode = "auto" if mode == "auto" else "manual"
            self._last_command = mode
            self._log(f"Controller mode set to {self._controller_mode}.")

    async def set_estop(self, enabled: bool) -> None:
        async with self._lock:
            self._advance_unlocked()
            self._estop = enabled
            if enabled:
                self._left_command = 0.0
                self._right_command = 0.0
                self._log("Emergency stop engaged.")
                self._last_command = "estop"
            else:
                self._log("Emergency stop cleared.")
                self._last_command = "reset"

    async def set_target_gap(self, meters: float) -> None:
        async with self._lock:
            self._advance_unlocked()
            self._target_gap_m = _clamp(meters, 0.3, 2.5)
            self._last_command = f"gap {self._target_gap_m:.2f}"
            self._log(f"Target gap updated to {self._target_gap_m:.2f} m.")

    async def set_vision_tracking(self, enabled: bool) -> None:
        async with self._lock:
            self._advance_unlocked()
            self._vision_enabled = enabled
            self._vision_locked = enabled
            if not enabled:
                self._controller_mode = "manual"
            self._last_command = "calibrate_distance" if enabled else "clear_calibration"
            self._log("Vision tracking armed." if enabled else "Vision tracking cleared.")

    async def set_vision_mode(self, mode: str) -> None:
        # Mock backend just stores the mode for UI consistency.
        async with self._lock:
            self._vision_mode = mode
            self._last_command = f"vision_mode {mode}"
            self._log(f"Vision mode set to {mode}.")

    async def set_motor_pins(
        self, left1_pin: int, left2_pin: int, right1_pin: int, right2_pin: int
    ) -> None:
        async with self._lock:
            self._left1_motor_pin = left1_pin
            self._left2_motor_pin = left2_pin
            self._right1_motor_pin = right1_pin
            self._right2_motor_pin = right2_pin
            self._last_command = f"pins {left1_pin} {left2_pin} {right1_pin} {right2_pin}"
            self._log(
                f"Motor pins updated to L1={left1_pin} L2={left2_pin} R1={right1_pin} R2={right2_pin}."
            )

    async def set_motor_inverts(
        self, left1_invert: bool, left2_invert: bool, right1_invert: bool, right2_invert: bool
    ) -> None:
        async with self._lock:
            self._left1_invert = left1_invert
            self._left2_invert = left2_invert
            self._right1_invert = right1_invert
            self._right2_invert = right2_invert
            self._last_command = (
                f"invert {int(left1_invert)} {int(left2_invert)} {int(right1_invert)} {int(right2_invert)}"
            )
            self._log("Motor inverts updated.")

    async def set_motor_calibration(
        self,
        left1_trim: float,
        left2_trim: float,
        right1_trim: float,
        right2_trim: float,
        stop_deadband: float,
    ) -> None:
        async with self._lock:
            self._left1_trim = left1_trim
            self._left2_trim = left2_trim
            self._right1_trim = right1_trim
            self._right2_trim = right2_trim
            self._stop_deadband = stop_deadband
            self._last_command = (
                f"trim {left1_trim:.2f} {left2_trim:.2f} {right1_trim:.2f} {right2_trim:.2f} {stop_deadband:.3f}"
            )
            self._log("Motor calibration updated.")

    async def snapshot(self) -> RobotSnapshot:
        async with self._lock:
            self._advance_unlocked()
            return RobotSnapshot(
                controller_mode=self._controller_mode,
                estop=self._estop,
                left_command=round(self._left_command, 3),
                right_command=round(self._right_command, 3),
                linear_velocity_mps=round(self._linear_velocity_mps, 3),
                angular_velocity_rad_s=round(self._angular_velocity_rad_s, 3),
                distance_m=round(self._distance_m, 3),
                heading_rad=round(self._heading_rad, 3),
                target_gap_m=round(self._target_gap_m, 3),
                fps=round(self._fps, 2),
                command_latency_ms=round(self._command_latency_ms, 2),
                last_command=self._last_command,
                vision_enabled=self._vision_enabled,
                vision_locked=self._vision_locked,
                vision_mode=getattr(self, "_vision_mode", "aruco"),
                left1_motor_pin=self._left1_motor_pin,
                left2_motor_pin=self._left2_motor_pin,
                right1_motor_pin=self._right1_motor_pin,
                right2_motor_pin=self._right2_motor_pin,
                left1_invert=self._left1_invert,
                left2_invert=self._left2_invert,
                right1_invert=self._right1_invert,
                right2_invert=self._right2_invert,
                left1_trim=round(self._left1_trim, 3),
                left2_trim=round(self._left2_trim, 3),
                right1_trim=round(self._right1_trim, 3),
                right2_trim=round(self._right2_trim, 3),
                stop_deadband=round(self._stop_deadband, 3),
                logs=list(self._logs),
                updated_at=_now_iso(),
            )


class RemoteRobotBackend:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=5.0)
        self._last_snapshot = make_idle_snapshot("remote-idle")

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, payload: dict | None = None) -> RobotSnapshot:
        try:
            response = await self._client.request(method, path, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Remote Pi agent request failed: {exc}") from exc

        snapshot = RobotSnapshot.model_validate(response.json())
        self._last_snapshot = snapshot
        return snapshot

    async def set_drive(self, left: float, right: float) -> None:
        await self._request("POST", "/api/drive", {"left": left, "right": right})

    async def stop(self) -> None:
        await self._request("POST", "/api/stop")

    async def set_mode(self, mode: str) -> None:
        await self._request("POST", "/api/mode", {"mode": mode})

    async def set_estop(self, enabled: bool) -> None:
        await self._request("POST", "/api/estop", {"enabled": enabled})

    async def set_target_gap(self, meters: float) -> None:
        await self._request("POST", "/api/target-gap", {"target_gap_m": meters})

    async def set_vision_tracking(self, enabled: bool) -> None:
        await self._request("POST", "/api/vision/tracking", {"enabled": enabled})

    async def set_vision_mode(self, mode: str) -> None:
        await self._request("POST", "/api/vision/mode", {"mode": mode})

    async def set_motor_pins(
        self, left1_pin: int, left2_pin: int, right1_pin: int, right2_pin: int
    ) -> None:
        await self._request(
            "POST",
            "/api/motor-pins",
            {
                "left1_pin": left1_pin,
                "left2_pin": left2_pin,
                "right1_pin": right1_pin,
                "right2_pin": right2_pin,
            },
        )

    async def set_motor_inverts(
        self, left1_invert: bool, left2_invert: bool, right1_invert: bool, right2_invert: bool
    ) -> None:
        await self._request(
            "POST",
            "/api/motor-inverts",
            {
                "left1_invert": left1_invert,
                "left2_invert": left2_invert,
                "right1_invert": right1_invert,
                "right2_invert": right2_invert,
            },
        )

    async def set_motor_calibration(
        self,
        left1_trim: float,
        left2_trim: float,
        right1_trim: float,
        right2_trim: float,
        stop_deadband: float,
    ) -> None:
        await self._request(
            "POST",
            "/api/motor-calibration",
            {
                "left1_trim": left1_trim,
                "left2_trim": left2_trim,
                "right1_trim": right1_trim,
                "right2_trim": right2_trim,
                "stop_deadband": stop_deadband,
            },
        )

    async def rl_start(self, payload: dict) -> None:
        await self._request("POST", "/api/rl/start", payload)

    async def rl_stop(self) -> None:
        await self._request("POST", "/api/rl/stop")

    async def rl_log_list(self) -> dict:
        try:
            response = await self._client.get("/api/rl/log/list")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"RL log list failed: {exc}") from exc
        return response.json()

    async def rl_log_stream(self, name: str | None):
        """Stream the raw bytes of an RL log file (latest if name is None)."""
        path = "/api/rl/log"
        params = {"name": name} if name else {}
        try:
            async with self._client.stream("GET", path, params=params) as response:
                response.raise_for_status()
                filename = None
                disposition = response.headers.get("content-disposition", "")
                if "filename=" in disposition:
                    filename = disposition.split("filename=", 1)[1].strip('"; ')
                yield {"filename": filename}
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.HTTPError as exc:
            raise RuntimeError(f"RL log download failed: {exc}") from exc

    async def rl_policy_info(self) -> dict:
        try:
            response = await self._client.get("/api/rl/policy/info")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Policy info failed: {exc}") from exc
        return response.json()

    async def rl_policy_upload(self, body: bytes, slot: str | None = None) -> dict:
        params = {"slot": slot} if slot else None
        try:
            response = await self._client.post(
                "/api/rl/policy/upload",
                content=body,
                params=params,
                headers={"Content-Type": "application/octet-stream"},
                timeout=30.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Policy upload failed: {exc}") from exc
        return response.json()

    async def snapshot(self) -> RobotSnapshot:
        return await self._request("GET", "/api/status")

    async def camera_stream(self):
        try:
            async with self._client.stream("GET", "/stream.mjpg") as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Remote Pi camera stream failed: {exc}") from exc


def normalize_target(target: str) -> str:
    cleaned = target.strip()
    if not cleaned:
        return "mock"
    if cleaned.lower() in {"mock", "demo", "local"}:
        return "mock"

    ssh_style_user = None
    if "://" not in cleaned and "@" in cleaned and "/" not in cleaned:
        candidate_user, candidate_host = cleaned.split("@", maxsplit=1)
        if candidate_user and candidate_host:
            ssh_style_user = candidate_user
            cleaned = candidate_host

    if "://" not in cleaned:
        cleaned = f"http://{cleaned}"

    parsed = urlparse(cleaned)
    scheme = parsed.scheme or "http"
    host = parsed.netloc or parsed.path
    path = parsed.path if parsed.netloc else ""

    if ":" not in host:
        host = f"{host}:8001"

    normalized = f"{scheme}://{host}{path}".rstrip("/")

    if ssh_style_user:
        parsed_normalized = urlparse(normalized)
        host_label = parsed_normalized.netloc
        return f"{parsed_normalized.scheme}://{host_label}{parsed_normalized.path}".rstrip("/")

    return normalized


class DashboardController:
    def __init__(self) -> None:
        self._backend: MockRobotBackend | RemoteRobotBackend | None = None
        self._connection = ConnectionInfo()
        self._dashboard_logs: deque[str] = deque(maxlen=64)

    def _log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self._dashboard_logs.appendleft(f"[{stamp}] {message}")

    async def _set_backend(self, backend: MockRobotBackend | RemoteRobotBackend | None) -> None:
        old_backend = self._backend
        self._backend = backend
        if old_backend is not None:
            await old_backend.close()

    async def connect(self, target: str) -> tuple[str, DashboardStatus]:
        normalized = normalize_target(target)

        if normalized == "mock":
            await self._set_backend(MockRobotBackend())
            self._connection = ConnectionInfo(
                target="mock",
                mode="mock",
                connected=True,
                label="Mock cart",
                camera_mode="browser",
                camera_stream_url=None,
            )
            self._log("Connected to the local mock cart.")
            return "Connected to the mock cart.", await self.status()

        candidate = RemoteRobotBackend(normalized)
        try:
            await candidate.snapshot()
        except Exception:
            await candidate.close()
            raise

        await self._set_backend(candidate)
        host = urlparse(normalized).netloc
        self._connection = ConnectionInfo(
            target=normalized,
            mode="remote",
            connected=True,
            label=f"Pi agent @ {host}",
            camera_mode="remote",
            camera_stream_url="/api/camera/stream",
        )
        self._log(f"Connected to remote Pi agent at {normalized}.")
        return "Connected to the Pi agent.", await self.status()

    async def disconnect(self) -> tuple[str, DashboardStatus]:
        await self._set_backend(None)
        self._connection = ConnectionInfo()
        self._log("Disconnected from the robot.")
        return "Disconnected from the robot.", await self.status()

    async def status(self) -> DashboardStatus:
        if self._backend is None:
            return DashboardStatus(
                connection=self._connection,
                robot=make_idle_snapshot("awaiting-connection"),
                dashboard_logs=list(self._dashboard_logs),
            )

        try:
            snapshot = await self._backend.snapshot()
            self._connection.connected = True
        except Exception as exc:
            self._connection.connected = False
            self._log(str(exc))
            snapshot = make_idle_snapshot("connection-error")

        return DashboardStatus(
            connection=self._connection,
            robot=snapshot,
            dashboard_logs=list(self._dashboard_logs),
        )

    async def drive(self, left: float, right: float) -> DashboardStatus:
        if self._backend is None:
            raise RuntimeError("No robot connected. Use connect mock or connect http://<pi>:8001 first.")
        await self._backend.set_drive(left, right)
        return await self.status()

    async def stop(self) -> DashboardStatus:
        if self._backend is None:
            raise RuntimeError("No robot connected.")
        await self._backend.stop()
        return await self.status()

    async def execute_terminal_command(self, command: str) -> tuple[str, DashboardStatus]:
        stripped = command.strip()
        lowered = stripped.lower()

        if lowered == "help":
            help_message = f"{HELP_TEXT}, connect <mock|url>, disconnect"
            return help_message, await self.status()

        if lowered.startswith("connect"):
            parts = stripped.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError("Usage: connect mock OR connect http://10.117.30.17:8001")
            return await self.connect(parts[1])

        if lowered == "disconnect":
            return await self.disconnect()

        if self._backend is None:
            raise RuntimeError("No robot connected. Use connect mock or connect http://<pi>:8001 first.")

        message, _ = await execute_robot_command(stripped, self._backend)
        return message, await self.status()

    def can_stream_camera(self) -> bool:
        return isinstance(self._backend, RemoteRobotBackend)

    async def camera_stream(self):
        if not isinstance(self._backend, RemoteRobotBackend):
            raise RuntimeError("Remote camera streaming is only available when connected to a Pi agent.")
        async for chunk in self._backend.camera_stream():
            yield chunk

    async def close(self) -> None:
        await self._set_backend(None)
