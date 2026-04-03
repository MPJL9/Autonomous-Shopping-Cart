from __future__ import annotations

from typing import Protocol

from .models import RobotSnapshot


class RobotCommandTarget(Protocol):
    async def set_drive(self, left: float, right: float) -> None: ...

    async def stop(self) -> None: ...

    async def set_mode(self, mode: str) -> None: ...

    async def set_estop(self, enabled: bool) -> None: ...

    async def set_target_gap(self, meters: float) -> None: ...

    async def set_vision_tracking(self, enabled: bool) -> None: ...

    async def set_motor_pins(self, left_pin: int, right_pin: int) -> None: ...

    async def set_motor_calibration(self, left_trim: float, right_trim: float, stop_deadband: float) -> None: ...

    async def snapshot(self) -> RobotSnapshot: ...


HELP_TEXT = (
    "Commands: help, stop, estop, reset, manual, auto, forward <0..1>, "
    "reverse <0..1>, left <0..1>, right <0..1>, spin_left <0..1>, "
    "spin_right <0..1>, drive <left> <right>, gap <meters>, "
    "calibrate_distance, clear_calibration, follow_person, pins <left_gpio> <right_gpio>, "
    "trim <left_trim> <right_trim> [deadband], status"
)


def _parse_unit(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError("Expected a value between 0.0 and 1.0.")
    return parsed


async def execute_robot_command(command: str, target: RobotCommandTarget) -> tuple[str, RobotSnapshot]:
    tokens = command.strip().lower().split()
    if not tokens:
        snapshot = await target.snapshot()
        return "Enter a command.", snapshot

    action = tokens[0]

    if action == "help":
        snapshot = await target.snapshot()
        return HELP_TEXT, snapshot

    if action == "status":
        snapshot = await target.snapshot()
        return "Status refreshed.", snapshot

    if action == "stop":
        await target.stop()
        snapshot = await target.snapshot()
        return "Motors stopped.", snapshot

    if action == "estop":
        await target.set_estop(True)
        snapshot = await target.snapshot()
        return "Emergency stop engaged.", snapshot

    if action in {"reset", "clear_estop"}:
        await target.set_estop(False)
        snapshot = await target.snapshot()
        return "Emergency stop cleared.", snapshot

    if action in {"manual", "auto"}:
        await target.set_mode(action)
        snapshot = await target.snapshot()
        return f"Controller mode set to {action}.", snapshot

    if action == "gap":
        if len(tokens) != 2:
            raise ValueError("Usage: gap <meters>")
        await target.set_target_gap(float(tokens[1]))
        snapshot = await target.snapshot()
        return f"Target gap set to {tokens[1]} m.", snapshot

    if action == "calibrate_distance":
        await target.set_vision_tracking(True)
        snapshot = await target.snapshot()
        return "Distance calibration armed.", snapshot

    if action == "clear_calibration":
        await target.set_vision_tracking(False)
        snapshot = await target.snapshot()
        return "Distance calibration cleared.", snapshot

    if action == "follow_person":
        await target.set_vision_tracking(True)
        await target.set_mode("auto")
        snapshot = await target.snapshot()
        return "Live follow mode engaged.", snapshot

    if action == "pins":
        if len(tokens) != 3:
            raise ValueError("Usage: pins <left_gpio> <right_gpio>")
        left_pin = int(tokens[1])
        right_pin = int(tokens[2])
        await target.set_motor_pins(left_pin, right_pin)
        snapshot = await target.snapshot()
        return f"Motor pins updated to left GPIO {left_pin}, right GPIO {right_pin}.", snapshot

    if action == "trim":
        if len(tokens) not in {3, 4}:
            raise ValueError("Usage: trim <left_trim> <right_trim> [deadband]")
        left_trim = float(tokens[1])
        right_trim = float(tokens[2])
        stop_deadband = float(tokens[3]) if len(tokens) == 4 else 0.04
        await target.set_motor_calibration(left_trim, right_trim, stop_deadband)
        snapshot = await target.snapshot()
        return (
            f"Motor trims updated to left {left_trim:.3f}, right {right_trim:.3f}, deadband {stop_deadband:.3f}.",
            snapshot,
        )

    if action == "drive":
        if len(tokens) != 3:
            raise ValueError("Usage: drive <left> <right>")
        await target.set_drive(float(tokens[1]), float(tokens[2]))
        snapshot = await target.snapshot()
        return "Drive command applied.", snapshot

    if action in {"forward", "reverse", "left", "right", "spin_left", "spin_right"}:
        if len(tokens) != 2:
            raise ValueError(f"Usage: {action} <0..1>")
        magnitude = _parse_unit(tokens[1])

        if action == "forward":
            left, right = magnitude, magnitude
        elif action == "reverse":
            left, right = -magnitude, -magnitude
        elif action == "left":
            left, right = magnitude * 0.35, magnitude
        elif action == "right":
            left, right = magnitude, magnitude * 0.35
        elif action == "spin_left":
            left, right = -magnitude, magnitude
        else:
            left, right = magnitude, -magnitude

        await target.set_drive(left, right)
        snapshot = await target.snapshot()
        return f"{action.replace('_', ' ')} command applied.", snapshot

    raise ValueError(f"Unknown command '{action}'. Type 'help' to see the supported commands.")
