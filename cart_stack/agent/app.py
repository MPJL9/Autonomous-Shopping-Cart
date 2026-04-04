from __future__ import annotations

import asyncio
import math
import os
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from cart_stack.agent.camera import build_camera_source
from cart_stack.agent.motors import build_motor_driver
from cart_stack.agent.vision import VisionEstimate, build_vision_tracker
from cart_stack.shared.commands import execute_robot_command
from cart_stack.shared.models import (
    DriveCommand,
    EstopChange,
    MotorCalibrationChange,
    MotorPinsChange,
    ModeChange,
    RobotSnapshot,
    TargetGapChange,
    TerminalCommand,
    VisionTrackingChange,
)


DEFAULT_PI_ENV = {
    "CART_AGENT_HOST": "0.0.0.0",
    "CART_AGENT_PORT": "8001",
    "CART_MOTOR_MODE": "servo",
    "CART_LEFT_PIN": "19",
    "CART_RIGHT_PIN": "13",
    "CART_LEFT_INVERT": "0",
    "CART_RIGHT_INVERT": "1",
    "CART_LEFT_TRIM": "0.0",
    "CART_RIGHT_TRIM": "0.0",
    "CART_STOP_DEADBAND": "0.04",
    "CART_IDLE_PWM_MODE": "off",
    "CART_CAMERA_MODE": "auto",
    "CART_CAMERA_RETRY_SEC": "5.0",
    "CART_TRACK_WIDTH_M": "0.58",
    "CART_MAX_SPEED_MPS": "0.9",
    "CART_VISION_INTERVAL_SEC": "0.45",
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


class RobotAgentRuntime:
    def __init__(self) -> None:
        self.driver = build_motor_driver()
        self.camera = build_camera_source()
        self.vision = build_vision_tracker()
        self._lock = asyncio.Lock()
        self._mode = "manual"
        self._estop = True
        self._heading_rad = 0.0
        self._distance_m = 0.78
        self._target_gap_m = 0.65
        self._linear_velocity_mps = 0.0
        self._angular_velocity_rad_s = 0.0
        self._fps = 16.0
        self._command_latency_ms = 18.0
        self._last_command = "agent-ready"
        self._vision_enabled = False
        self._vision_locked = False
        self._logs: deque[str] = deque(maxlen=64)
        self._last_update = time.perf_counter()
        self._max_speed_mps = float(os.getenv("CART_MAX_SPEED_MPS", "0.9"))
        self._track_width_m = float(os.getenv("CART_TRACK_WIDTH_M", "0.58"))
        self._env_file = Path(os.getenv("CART_ENV_FILE", str(Path(__file__).resolve().parents[2] / ".env.pi")))
        self._log("Pi agent started.")
        self._log(
            f"Motor driver {type(self.driver).__name__} on left GPIO {self.driver.left_pin}, right GPIO {self.driver.right_pin}."
        )
        self._log("Motors start disarmed. Use reset/arm before driving.")

    def _log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self._logs.appendleft(f"[{stamp}] {message}")

    def _persist_env_values(self, updates: dict[str, str]) -> None:
        lines: list[str] = []
        if self._env_file.exists():
            lines = self._env_file.read_text(encoding="utf-8").splitlines()

        indexes: dict[str, int] = {}
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key = stripped.split("=", maxsplit=1)[0]
            indexes[key] = index

        merged_updates = dict(DEFAULT_PI_ENV)
        merged_updates.update(updates)

        for key, value in merged_updates.items():
            rendered = f"{key}={value}"
            if key in indexes:
                lines[indexes[key]] = rendered
            else:
                lines.append(rendered)

        payload = "\n".join(lines).strip()
        self._env_file.write_text(f"{payload}\n" if payload else "", encoding="utf-8")

    def _advance_unlocked(self) -> None:
        now = time.perf_counter()
        dt = min(now - self._last_update, 0.5)
        self._last_update = now

        if self._mode == "auto" and not self._estop:
            if self._vision_enabled and not self._vision_locked:
                self.driver.state.left = 0.0
                self.driver.state.right = 0.0
            else:
                gap_error = self._distance_m - self._target_gap_m
                forward = _clamp(gap_error * 1.2, -0.55, 0.55)
                turn = _clamp(-self._heading_rad * 0.65, -0.2, 0.2)
                self.driver.state.left = _clamp(forward - turn, -1.0, 1.0)
                self.driver.state.right = _clamp(forward + turn, -1.0, 1.0)

        if self._estop:
            self.driver.state.left = 0.0
            self.driver.state.right = 0.0

        target_linear = self._max_speed_mps * ((self.driver.state.left + self.driver.state.right) / 2.0)
        target_angular = (
            self._max_speed_mps * (self.driver.state.right - self.driver.state.left) / max(self._track_width_m, 0.1)
        )

        blend_linear = min(dt * 2.4, 1.0)
        blend_angular = min(dt * 3.2, 1.0)
        self._linear_velocity_mps += (target_linear - self._linear_velocity_mps) * blend_linear
        self._angular_velocity_rad_s += (target_angular - self._angular_velocity_rad_s) * blend_angular

        self._heading_rad = _wrap_angle(self._heading_rad + self._angular_velocity_rad_s * dt)
        self._distance_m += -self._linear_velocity_mps * 0.22 * dt
        self._distance_m = _clamp(self._distance_m, 0.25, 3.5)
        self._fps = 15.0 + abs(self.driver.state.left - self.driver.state.right) * 1.5
        self._command_latency_ms = 14.0 + abs(self.driver.state.left - self.driver.state.right) * 18.0

    async def set_drive(self, left: float, right: float) -> None:
        async with self._lock:
            self._advance_unlocked()
            self._mode = "manual"
            if self._estop:
                await self.driver.stop()
                self._last_command = "estop"
                return
            await self.driver.set_drive(left, right)
            self._last_command = f"drive {self.driver.state.left:.2f} {self.driver.state.right:.2f}"
            self._log(f"Drive set to left={self.driver.state.left:.2f}, right={self.driver.state.right:.2f}.")

    async def stop(self) -> None:
        async with self._lock:
            self._advance_unlocked()
            await self.driver.stop()
            self._last_command = "stop"
            self._log("Stop command received.")

    async def set_mode(self, mode: str) -> None:
        async with self._lock:
            self._advance_unlocked()
            self._mode = "auto" if mode == "auto" else "manual"
            self._last_command = mode
            self._log(f"Controller mode set to {self._mode}.")

    async def set_estop(self, enabled: bool) -> None:
        async with self._lock:
            self._advance_unlocked()
            self._estop = enabled
            if enabled:
                await self.driver.stop()
                self._last_command = "estop"
                self._log("Motors disarmed.")
            else:
                self._last_command = "reset"
                self._log("Motors armed.")

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
            if not enabled:
                self._vision_locked = False
                self._mode = "manual"
            self._last_command = "calibrate_distance" if enabled else "clear_calibration"
            if enabled:
                self._log("Vision tracking armed. Waiting to lock onto a person.")
            else:
                self._log("Vision tracking cleared.")

    async def set_motor_pins(self, left_pin: int, right_pin: int) -> None:
        async with self._lock:
            self._advance_unlocked()
            await self.driver.set_motor_pins(left_pin, right_pin)
            self._persist_env_values(
                {
                    "CART_LEFT_PIN": str(self.driver.left_pin),
                    "CART_RIGHT_PIN": str(self.driver.right_pin),
                }
            )
            self._last_command = f"pins {self.driver.left_pin} {self.driver.right_pin}"
            self._log(
                f"Motor pins saved as left GPIO {self.driver.left_pin}, right GPIO {self.driver.right_pin}."
            )

    async def set_motor_calibration(self, left_trim: float, right_trim: float, stop_deadband: float) -> None:
        async with self._lock:
            self._advance_unlocked()
            await self.driver.set_motor_calibration(left_trim, right_trim, stop_deadband)
            self._persist_env_values(
                {
                    "CART_LEFT_TRIM": str(self.driver.left_trim),
                    "CART_RIGHT_TRIM": str(self.driver.right_trim),
                    "CART_STOP_DEADBAND": str(self.driver.deadband),
                }
            )
            self._last_command = (
                f"trim {self.driver.left_trim:.3f} {self.driver.right_trim:.3f} {self.driver.deadband:.3f}"
            )
            self._log(
                "Motor calibration saved "
                f"(left trim {self.driver.left_trim:.3f}, right trim {self.driver.right_trim:.3f}, deadband {self.driver.deadband:.3f})."
            )

    async def _apply_vision_estimate(self, estimate: VisionEstimate) -> None:
        async with self._lock:
            self._vision_locked = estimate.locked
            if estimate.locked:
                if estimate.distance_m is not None:
                    self._distance_m = _clamp(estimate.distance_m, 0.25, 3.5)
                if estimate.heading_rad is not None:
                    self._heading_rad = _wrap_angle(estimate.heading_rad)

    async def snapshot(self) -> RobotSnapshot:
        async with self._lock:
            self._advance_unlocked()
            return RobotSnapshot(
                controller_mode=self._mode,
                estop=self._estop,
                left_command=round(self.driver.state.left, 3),
                right_command=round(self.driver.state.right, 3),
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
                left_motor_pin=self.driver.left_pin,
                right_motor_pin=self.driver.right_pin,
                left_trim=round(self.driver.left_trim, 3),
                right_trim=round(self.driver.right_trim, 3),
                stop_deadband=round(self.driver.deadband, 3),
                logs=list(self._logs),
                updated_at=_now_iso(),
            )

    async def mjpeg_stream(self):
        while True:
            snapshot = await self.snapshot()
            frame = await asyncio.to_thread(self.camera.capture_jpeg, snapshot)
            if self._vision_enabled:
                estimate = await asyncio.to_thread(self.vision.annotate, frame)
                frame = estimate.frame_jpeg
                await self._apply_vision_estimate(estimate)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                + frame
                + b"\r\n"
            )
            await asyncio.sleep(0.12)

    async def close(self) -> None:
        await self.driver.close()
        self.camera.close()


app = FastAPI(title="Autonomous Shopping Cart Pi Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

runtime = RobotAgentRuntime()


@app.get("/api/status")
async def api_status() -> dict:
    return (await runtime.snapshot()).model_dump()


@app.post("/api/drive")
async def api_drive(command: DriveCommand) -> dict:
    await runtime.set_drive(command.left, command.right)
    return (await runtime.snapshot()).model_dump()


@app.post("/api/stop")
async def api_stop() -> dict:
    await runtime.stop()
    return (await runtime.snapshot()).model_dump()


@app.post("/api/mode")
async def api_mode(change: ModeChange) -> dict:
    await runtime.set_mode(change.mode)
    return (await runtime.snapshot()).model_dump()


@app.post("/api/estop")
async def api_estop(change: EstopChange) -> dict:
    await runtime.set_estop(change.enabled)
    return (await runtime.snapshot()).model_dump()


@app.post("/api/target-gap")
async def api_target_gap(change: TargetGapChange) -> dict:
    await runtime.set_target_gap(change.target_gap_m)
    return (await runtime.snapshot()).model_dump()


@app.post("/api/vision/tracking")
async def api_vision_tracking(change: VisionTrackingChange) -> dict:
    await runtime.set_vision_tracking(change.enabled)
    return (await runtime.snapshot()).model_dump()


@app.post("/api/motor-pins")
async def api_motor_pins(change: MotorPinsChange) -> dict:
    try:
        await runtime.set_motor_pins(change.left_pin, change.right_pin)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return (await runtime.snapshot()).model_dump()


@app.post("/api/motor-calibration")
async def api_motor_calibration(change: MotorCalibrationChange) -> dict:
    try:
        await runtime.set_motor_calibration(change.left_trim, change.right_trim, change.stop_deadband)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return (await runtime.snapshot()).model_dump()


@app.post("/api/command")
async def api_command(command: TerminalCommand) -> dict:
    try:
        _, snapshot = await execute_robot_command(command.command, runtime)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return snapshot.model_dump()


@app.get("/stream.mjpg")
async def stream_mjpg() -> StreamingResponse:
    return StreamingResponse(
        runtime.mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await runtime.close()
