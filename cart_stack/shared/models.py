from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ConnectionRequest(BaseModel):
    target: str = Field(
        default="mock",
        description="Use 'mock' for the demo backend or an http(s) base URL for the Pi agent.",
    )


class ConnectionInfo(BaseModel):
    target: str = "mock"
    mode: Literal["mock", "remote", "disconnected"] = "disconnected"
    connected: bool = False
    label: str = "No robot connected"
    camera_mode: Literal["browser", "remote", "placeholder"] = "placeholder"
    camera_stream_url: str | None = None


class DriveCommand(BaseModel):
    left: float = Field(ge=-1.0, le=1.0)
    right: float = Field(ge=-1.0, le=1.0)


class ModeChange(BaseModel):
    mode: Literal["manual", "auto"]


class EstopChange(BaseModel):
    enabled: bool


class TargetGapChange(BaseModel):
    target_gap_m: float = Field(gt=0.1, lt=5.0)


class VisionTrackingChange(BaseModel):
    enabled: bool


class VisionModeChange(BaseModel):
    mode: Literal["aruco", "yolo", "hog"]


class MotorPinsChange(BaseModel):
    left1_pin: int = Field(ge=0, le=27)
    left2_pin: int = Field(ge=0, le=27)
    right1_pin: int = Field(ge=0, le=27)
    right2_pin: int = Field(ge=0, le=27)


class MotorInvertChange(BaseModel):
    left1_invert: bool = False
    left2_invert: bool = False
    right1_invert: bool = False
    right2_invert: bool = False


class MotorCalibrationChange(BaseModel):
    left1_trim: float = Field(ge=-1.0, le=1.0)
    left2_trim: float = Field(ge=-1.0, le=1.0)
    right1_trim: float = Field(ge=-1.0, le=1.0)
    right2_trim: float = Field(ge=-1.0, le=1.0)
    stop_deadband: float = Field(ge=0.0, le=0.2)


class RlSessionStart(BaseModel):
    policy: Literal[
        "random", "zero", "forward", "manual",
        # Live-obs corrected (current generation)
        "bc_2d_combined_v2", "bc_2d_5_1only_v2",
        "bc_2d_combined_v2_pivot", "bc_2d_combined_v2_translate",
        # OLD combined restored for fallback testing
        "bc_2d_combined_pre_fix",
        "bc_2d_combined_pre_fix_pivot", "bc_2d_combined_pre_fix_translate",
        # 4/29-only variants (historical naming)
        "bc_2d", "bc_2d_mirror",
        "bc_2d_pivot", "bc_2d_translate",
        # AWR-refined bc_2d (offline RL on bc_2d_pivot rollouts)
        "bc_2d_awr",
        "bc_2d_awr_pivot", "bc_2d_awr_translate",
    ] = "random"
    max_steps: int = Field(default=300, ge=1, le=10000)
    step_hz: float = Field(default=5.0, gt=0.0, le=30.0)
    action_scale: float = Field(default=0.35, ge=0.0, le=1.0)


class RlSessionStatus(BaseModel):
    running: bool = False
    policy: str = "random"
    step: int = 0
    max_steps: int = 0
    last_reward: float = 0.0
    total_reward: float = 0.0
    log_path: str | None = None


class TerminalCommand(BaseModel):
    command: str = Field(min_length=1, max_length=120)


class RobotSnapshot(BaseModel):
    controller_mode: Literal["manual", "auto"] = "manual"
    estop: bool = False
    left_command: float = 0.0
    right_command: float = 0.0
    linear_velocity_mps: float = 0.0
    angular_velocity_rad_s: float = 0.0
    distance_m: float = 0.72
    heading_rad: float = 0.0
    target_gap_m: float = 0.65
    fps: float = 0.0
    command_latency_ms: float = 0.0
    last_command: str = "idle"
    vision_enabled: bool = False
    vision_locked: bool = False
    vision_mode: str = "aruco"
    left1_motor_pin: int = 13
    left2_motor_pin: int = 16
    right1_motor_pin: int = 19
    right2_motor_pin: int = 12
    left1_invert: bool = False
    left2_invert: bool = False
    right1_invert: bool = True
    right2_invert: bool = True
    left1_trim: float = 0.0
    left2_trim: float = 0.0
    right1_trim: float = 0.0
    right2_trim: float = 0.0
    stop_deadband: float = 0.04
    rl_running: bool = False
    rl_policy: str = "random"
    rl_step: int = 0
    rl_max_steps: int = 0
    rl_total_reward: float = 0.0
    logs: list[str] = Field(default_factory=list)
    updated_at: str


class DashboardStatus(BaseModel):
    connection: ConnectionInfo
    robot: RobotSnapshot
    dashboard_logs: list[str] = Field(default_factory=list)


class MotionModelInput(BaseModel):
    total_mass_kg: float = Field(default=24.0, gt=0)
    track_width_m: float = Field(default=0.58, gt=0.05)
    max_linear_speed_mps: float = Field(default=0.9, gt=0.05)
    zero_to_max_speed_s: float = Field(default=2.8, gt=0.05)
    cruise_speed_mph: float = Field(default=1.4, ge=0.0)
    turn_angle_deg: float = Field(default=90.0, gt=0.0, le=360.0)


class MotionModelResult(BaseModel):
    input_speed_mps: float
    weight_n: float
    weight_lbf: float
    average_acceleration_mps2: float
    estimated_acceleration_at_speed_mps2: float
    max_angular_speed_spin_rad_s: float
    max_angular_speed_at_speed_rad_s: float
    standstill_turn_time_s: float | None
    moving_turn_time_s: float | None
    turn_radius_m: float | None
    notes: list[str] = Field(default_factory=list)
    formulas: list[str] = Field(default_factory=list)
