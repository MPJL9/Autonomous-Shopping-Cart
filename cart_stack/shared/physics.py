from __future__ import annotations

import math

from .models import MotionModelInput, MotionModelResult


MPH_TO_MPS = 0.44704
KG_TO_LBF = 2.2046226218
G = 9.80665


def mph_to_mps(mph: float) -> float:
    return mph * MPH_TO_MPS


def _finite_or_none(value: float) -> float | None:
    return None if not math.isfinite(value) else round(value, 3)


def calculate_motion_model(inputs: MotionModelInput) -> MotionModelResult:
    target_speed_mps = mph_to_mps(inputs.cruise_speed_mph)
    clamped_speed_mps = min(target_speed_mps, inputs.max_linear_speed_mps * 0.99)

    weight_n = inputs.total_mass_kg * G
    weight_lbf = inputs.total_mass_kg * KG_TO_LBF
    avg_accel = inputs.max_linear_speed_mps / inputs.zero_to_max_speed_s

    speed_ratio = min(clamped_speed_mps / inputs.max_linear_speed_mps, 1.0)
    accel_at_speed = max(avg_accel * (1.0 - speed_ratio), 0.0)

    omega_spin = 2.0 * inputs.max_linear_speed_mps / inputs.track_width_m
    omega_at_speed = max(
        2.0 * (inputs.max_linear_speed_mps - clamped_speed_mps) / inputs.track_width_m,
        0.0,
    )

    angle_rad = math.radians(inputs.turn_angle_deg)
    standstill_turn_time = angle_rad / omega_spin if omega_spin > 0 else math.inf
    moving_turn_time = angle_rad / omega_at_speed if omega_at_speed > 0 else math.inf
    turn_radius = clamped_speed_mps / omega_at_speed if omega_at_speed > 0 else math.inf

    notes = [
        "Acceleration is dv/dt. It is not inherently constant, and most carts accelerate less as speed rises.",
        "The moving-turn estimate assumes a differential-drive cart with a fixed max wheel speed and symmetric left/right limits.",
        "Turn time still follows t = theta / omega, but omega usually shrinks as forward velocity consumes wheel-speed headroom.",
    ]
    formulas = [
        "v = (v_r + v_l) / 2",
        "omega = (v_r - v_l) / L",
        "omega_max(v) = 2 * (v_max - v) / L",
        "t_turn = theta / omega",
    ]

    return MotionModelResult(
        input_speed_mps=round(clamped_speed_mps, 3),
        weight_n=round(weight_n, 3),
        weight_lbf=round(weight_lbf, 3),
        average_acceleration_mps2=round(avg_accel, 3),
        estimated_acceleration_at_speed_mps2=round(accel_at_speed, 3),
        max_angular_speed_spin_rad_s=round(omega_spin, 3),
        max_angular_speed_at_speed_rad_s=round(omega_at_speed, 3),
        standstill_turn_time_s=_finite_or_none(standstill_turn_time),
        moving_turn_time_s=_finite_or_none(moving_turn_time),
        turn_radius_m=_finite_or_none(turn_radius),
        notes=notes,
        formulas=formulas,
    )

