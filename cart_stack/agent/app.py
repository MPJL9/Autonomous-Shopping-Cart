from __future__ import annotations

import asyncio
import math
import os
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from cart_stack.agent.camera import build_camera_source
from cart_stack.agent.motors import build_motor_driver
from cart_stack.agent.vision import VISION_MODES, VisionEstimate, build_vision_tracker
from cart_stack.shared.commands import execute_robot_command
from cart_stack.agent.rl_session import RlSession
from cart_stack.shared.models import (
    DriveCommand,
    EstopChange,
    MotorCalibrationChange,
    MotorInvertChange,
    MotorPinsChange,
    ModeChange,
    RlSessionStart,
    RlSessionStatus,
    RobotSnapshot,
    TargetGapChange,
    TerminalCommand,
    VisionTrackingChange,
    VisionModeChange,
)


DEFAULT_PI_ENV = {
    "CART_AGENT_HOST": "0.0.0.0",
    "CART_AGENT_PORT": "8001",
    "CART_MOTOR_MODE": "servo",
    "CART_LEFT1_PIN": "13",
    "CART_LEFT2_PIN": "16",
    "CART_RIGHT1_PIN": "19",
    "CART_RIGHT2_PIN": "12",
    "CART_LEFT1_INVERT": "0",
    "CART_LEFT2_INVERT": "0",
    "CART_RIGHT1_INVERT": "1",
    "CART_RIGHT2_INVERT": "1",
    "CART_LEFT1_TRIM": "0.0",
    "CART_LEFT2_TRIM": "0.0",
    "CART_RIGHT1_TRIM": "0.0",
    "CART_RIGHT2_TRIM": "0.0",
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
        self._vision_mode = getattr(self.vision, "mode", "aruco")
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
        self._rl = RlSession(self)
        self._log("Pi agent started.")
        pins = self.driver.pins()
        self._log(
            f"Motor driver {type(self.driver).__name__} on "
            f"L1={pins['left1']} L2={pins['left2']} R1={pins['right1']} R2={pins['right2']}."
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

    async def set_vision_mode(self, mode: str) -> None:
        """Switch the perception backend live (no agent restart needed).

        Reassigns `self.vision` to a fresh tracker for the requested mode.
        The new tracker lazily loads its model on first frame, so the
        first ~1 frame after switching may take a moment.
        """
        mode = (mode or "").lower()
        if mode not in VISION_MODES:
            raise ValueError(f"Unknown vision mode: {mode!r}. Valid: {VISION_MODES}")
        async with self._lock:
            self._advance_unlocked()
            try:
                self.vision = build_vision_tracker(mode)
                self._vision_mode = getattr(self.vision, "mode", mode)
                self._vision_locked = False  # cleared until the new tracker locks
                self._last_command = f"vision_mode {mode}"
                self._log(f"Vision mode set to {self._vision_mode}.")
            except Exception as exc:
                self._log(f"Failed to switch vision mode: {exc!r}")
                raise

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

    async def set_motor_pins(
        self, left1_pin: int, left2_pin: int, right1_pin: int, right2_pin: int
    ) -> None:
        async with self._lock:
            self._advance_unlocked()
            await self.driver.set_motor_pins(left1_pin, left2_pin, right1_pin, right2_pin)
            pins = self.driver.pins()
            self._persist_env_values(
                {
                    "CART_LEFT1_PIN": str(pins["left1"]),
                    "CART_LEFT2_PIN": str(pins["left2"]),
                    "CART_RIGHT1_PIN": str(pins["right1"]),
                    "CART_RIGHT2_PIN": str(pins["right2"]),
                }
            )
            self._last_command = (
                f"pins {pins['left1']} {pins['left2']} {pins['right1']} {pins['right2']}"
            )
            self._log(
                f"Motor pins saved as L1={pins['left1']} L2={pins['left2']} "
                f"R1={pins['right1']} R2={pins['right2']}."
            )

    async def set_motor_inverts(
        self, left1_invert: bool, left2_invert: bool, right1_invert: bool, right2_invert: bool
    ) -> None:
        async with self._lock:
            self._advance_unlocked()
            await self.driver.set_motor_inverts(left1_invert, left2_invert, right1_invert, right2_invert)
            inverts = self.driver.inverts()
            self._persist_env_values(
                {
                    "CART_LEFT1_INVERT": "1" if inverts["left1"] else "0",
                    "CART_LEFT2_INVERT": "1" if inverts["left2"] else "0",
                    "CART_RIGHT1_INVERT": "1" if inverts["right1"] else "0",
                    "CART_RIGHT2_INVERT": "1" if inverts["right2"] else "0",
                }
            )
            self._last_command = (
                f"invert {int(inverts['left1'])} {int(inverts['left2'])} "
                f"{int(inverts['right1'])} {int(inverts['right2'])}"
            )
            self._log(
                "Motor inverts saved "
                f"(L1={int(inverts['left1'])} L2={int(inverts['left2'])} "
                f"R1={int(inverts['right1'])} R2={int(inverts['right2'])})."
            )

    async def set_motor_calibration(
        self,
        left1_trim: float,
        left2_trim: float,
        right1_trim: float,
        right2_trim: float,
        stop_deadband: float,
    ) -> None:
        async with self._lock:
            self._advance_unlocked()
            await self.driver.set_motor_calibration(
                left1_trim, left2_trim, right1_trim, right2_trim, stop_deadband
            )
            trims = self.driver.trims()
            self._persist_env_values(
                {
                    "CART_LEFT1_TRIM": str(trims["left1"]),
                    "CART_LEFT2_TRIM": str(trims["left2"]),
                    "CART_RIGHT1_TRIM": str(trims["right1"]),
                    "CART_RIGHT2_TRIM": str(trims["right2"]),
                    "CART_STOP_DEADBAND": str(self.driver.deadband),
                }
            )
            self._last_command = (
                f"trim {trims['left1']:.2f} {trims['left2']:.2f} "
                f"{trims['right1']:.2f} {trims['right2']:.2f} {self.driver.deadband:.3f}"
            )
            self._log(
                "Motor calibration saved "
                f"(L1={trims['left1']:.3f} L2={trims['left2']:.3f} "
                f"R1={trims['right1']:.3f} R2={trims['right2']:.3f}, "
                f"deadband {self.driver.deadband:.3f})."
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
            pins = self.driver.pins()
            inverts = self.driver.inverts()
            trims = self.driver.trims()
            rl_status = self._rl.status()
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
                vision_mode=self._vision_mode,
                left1_motor_pin=pins["left1"],
                left2_motor_pin=pins["left2"],
                right1_motor_pin=pins["right1"],
                right2_motor_pin=pins["right2"],
                left1_invert=inverts["left1"],
                left2_invert=inverts["left2"],
                right1_invert=inverts["right1"],
                right2_invert=inverts["right2"],
                left1_trim=round(trims["left1"], 3),
                left2_trim=round(trims["left2"], 3),
                right1_trim=round(trims["right1"], 3),
                right2_trim=round(trims["right2"], 3),
                stop_deadband=round(self.driver.deadband, 3),
                rl_running=rl_status.running,
                rl_policy=rl_status.policy,
                rl_step=rl_status.step,
                rl_max_steps=rl_status.max_steps,
                rl_total_reward=round(rl_status.total_reward, 3),
                logs=list(self._logs),
                updated_at=_now_iso(),
            )

    async def mjpeg_stream(self):
        # Default 0.1s (10 fps). Was 0.033s (30 fps), which on the Pi 3B
        # created enough CPU contention from JPEG encoding + vision to
        # perturb servo PWM timing and cause visible motor jitter when
        # commanding through the dashboard. 10 fps is plenty for live
        # monitoring and leaves headroom for stable PWM.
        frame_interval_s = float(os.getenv("CART_STREAM_INTERVAL_SEC", "0.1"))
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
            if frame_interval_s > 0:
                await asyncio.sleep(frame_interval_s)

    async def observation(self) -> tuple[float, float, float, float]:
        """Expose the 4-D RL observation [distance, heading, linear_v, angular_v]."""
        async with self._lock:
            self._advance_unlocked()
            return (
                float(self._distance_m),
                float(self._heading_rad),
                float(self._linear_velocity_mps),
                float(self._angular_velocity_rad_s),
            )

    async def close(self) -> None:
        await self._rl.stop()
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


@app.post("/api/vision/mode")
async def api_vision_mode(change: VisionModeChange) -> dict:
    """Switch perception backend live: aruco | yolo | hog."""
    try:
        await runtime.set_vision_mode(change.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return (await runtime.snapshot()).model_dump()


@app.post("/api/motor-pins")
async def api_motor_pins(change: MotorPinsChange) -> dict:
    try:
        await runtime.set_motor_pins(
            change.left1_pin, change.left2_pin, change.right1_pin, change.right2_pin
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return (await runtime.snapshot()).model_dump()


@app.post("/api/motor-inverts")
async def api_motor_inverts(change: MotorInvertChange) -> dict:
    try:
        await runtime.set_motor_inverts(
            change.left1_invert, change.left2_invert, change.right1_invert, change.right2_invert
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return (await runtime.snapshot()).model_dump()


@app.post("/api/motor-calibration")
async def api_motor_calibration(change: MotorCalibrationChange) -> dict:
    try:
        await runtime.set_motor_calibration(
            change.left1_trim,
            change.left2_trim,
            change.right1_trim,
            change.right2_trim,
            change.stop_deadband,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return (await runtime.snapshot()).model_dump()


@app.post("/api/rl/start")
async def api_rl_start(change: RlSessionStart) -> dict:
    try:
        await runtime._rl.start(
            policy=change.policy,
            max_steps=change.max_steps,
            step_hz=change.step_hz,
            action_scale=change.action_scale,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return (await runtime.snapshot()).model_dump()


@app.post("/api/rl/stop")
async def api_rl_stop() -> dict:
    await runtime._rl.stop()
    return (await runtime.snapshot()).model_dump()


@app.get("/api/rl/status")
async def api_rl_status() -> dict:
    return runtime._rl.status().model_dump()


def _rl_log_dir() -> Path:
    return Path(os.getenv("CART_RL_LOG_DIR", str(Path.home() / "cart_runtime_logs" / "rl")))


@app.get("/api/rl/log/list")
async def api_rl_log_list() -> dict:
    log_dir = _rl_log_dir()
    if not log_dir.exists():
        return {"files": []}
    entries = []
    for path in sorted(log_dir.glob("rl_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = path.stat()
        entries.append(
            {
                "name": path.name,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            }
        )
    return {"files": entries}


@app.get("/api/rl/log")
async def api_rl_log(name: str | None = None, latest: int = 0) -> FileResponse:
    log_dir = _rl_log_dir()
    if not log_dir.exists():
        raise HTTPException(status_code=404, detail="No RL logs directory yet.")

    target: Path | None = None
    if name:
        # Prevent path traversal; only allow files inside log_dir.
        candidate = (log_dir / name).resolve()
        if not str(candidate).startswith(str(log_dir.resolve())):
            raise HTTPException(status_code=400, detail="Invalid log name.")
        if candidate.is_file():
            target = candidate
    else:
        logs = sorted(log_dir.glob("rl_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            target = logs[0]
        _ = latest  # arg accepted for clarity; default behavior already returns latest

    if target is None:
        raise HTTPException(status_code=404, detail="No matching RL log file.")

    return FileResponse(
        path=target,
        media_type="application/x-ndjson",
        filename=target.name,
    )


from cart_stack.agent.rl_session import policy_slot_path


def _slot_info(slot: str) -> dict:
    p = policy_slot_path(slot)
    if not p.exists():
        return {"slot": slot, "installed": False, "path": str(p)}
    stat = p.stat()
    return {
        "slot": slot,
        "installed": True,
        "path": str(p),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
    }


@app.get("/api/rl/policy/info")
async def api_rl_policy_info() -> dict:
    """Return the install state of all known policy slots.

    Physical slots (npz files; Tier-1 / Tier-2 wrappers ride on these):
      bc_2d_combined_v2       — corrected 4/29 + 5/1 combined (live-obs)
      bc_2d_5_1only_v2        — corrected 5/1 only (live-obs)
      bc_2d_combined_pre_fix  — OLD combined from before the live-obs fix
      bc_2d                   — 4/29 only, blockcap, no mirror
      bc_2d_mirror            — 4/29 only, blockcap, with mirror
    """
    slot_names = (
        "bc_2d_combined_v2",
        "bc_2d_5_1only_v2",
        "bc_2d_combined_pre_fix",
        "bc_2d",
        "bc_2d_mirror",
        "bc_2d_awr",
    )
    slots = [_slot_info(s) for s in slot_names]
    out = {"slots": slots}
    for name, info in zip(slot_names, slots, strict=True):
        out[name] = info
    return out


@app.post("/api/rl/policy/upload")
async def api_rl_policy_upload(request: Request, slot: str | None = None) -> dict:
    """Accept raw npz bytes and store in a slot.

    Slot resolution order:
      1. ?slot=<name> query param if provided (e.g. bc_2d_combined_v2).
      2. Otherwise: action_dim must be 2 (1D heads no longer supported)
         and the upload defaults to the bc_2d_combined_v2 slot.

    Valid slots: bc_2d_combined_v2, bc_2d_5_1only_v2,
                 bc_2d_combined_pre_fix, bc_2d, bc_2d_mirror.
    """
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(body) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Policy file too large (>20MB).")
    if not body.startswith(b"PK"):
        raise HTTPException(status_code=400, detail="File does not look like an npz (zip header missing).")

    import io
    import numpy as np
    try:
        with np.load(io.BytesIO(body), allow_pickle=False) as data:
            action_dim = int(data["action_dim"])
            obs_dim = int(data["obs_dim"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot read npz: {exc!r}") from exc

    if obs_dim != 4:
        raise HTTPException(status_code=400, detail=f"Unsupported obs_dim={obs_dim} (must be 4).")

    if slot is not None:
        if slot not in ("bc_2d_combined_v2", "bc_2d_5_1only_v2",
                        "bc_2d_combined_pre_fix", "bc_2d", "bc_2d_mirror",
                        "bc_2d_awr"):
            raise HTTPException(status_code=400, detail=f"Unknown slot {slot!r}.")
        if action_dim != 2:
            raise HTTPException(
                status_code=400,
                detail=f"slot={slot} expects action_dim=2, npz has action_dim={action_dim}.",
            )
    else:
        if action_dim != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported action_dim={action_dim} (only 2D heads are supported).",
            )
        slot = "bc_2d_combined_v2"

    p = policy_slot_path(slot)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    return {
        "installed": True, "slot": slot, "action_dim": action_dim,
        "path": str(p), "size_bytes": len(body),
    }


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
