import asyncio

from cart_stack.dashboard.runtime import MockRobotBackend, normalize_target
from cart_stack.shared.commands import execute_robot_command


def test_normalize_target_adds_defaults() -> None:
    assert normalize_target("10.117.30.17") == "http://10.117.30.17:8001"
    assert normalize_target("mock") == "mock"
    assert normalize_target("veda@10.65.142.17") == "http://10.65.142.17:8001"


def test_mock_backend_accepts_drive_commands() -> None:
    async def run() -> None:
        backend = MockRobotBackend()
        await execute_robot_command("forward 0.4", backend)
        snapshot = await backend.snapshot()
        assert snapshot.left_command == 0.4
        assert snapshot.right_command == 0.4

        await execute_robot_command("gap 0.8", backend)
        snapshot = await backend.snapshot()
        assert snapshot.target_gap_m == 0.8

        await execute_robot_command("calibrate_distance", backend)
        snapshot = await backend.snapshot()
        assert snapshot.vision_enabled is True
        assert snapshot.vision_locked is True

        await execute_robot_command("pins 19 12", backend)
        snapshot = await backend.snapshot()
        assert snapshot.left_motor_pin == 19
        assert snapshot.right_motor_pin == 12

        await execute_robot_command("trim 0.03 -0.02 0.05", backend)
        snapshot = await backend.snapshot()
        assert snapshot.left_trim == 0.03
        assert snapshot.right_trim == -0.02
        assert snapshot.stop_deadband == 0.05

        await execute_robot_command("follow_person", backend)
        snapshot = await backend.snapshot()
        assert snapshot.controller_mode == "auto"
        assert snapshot.vision_enabled is True

        await execute_robot_command("clear_calibration", backend)
        snapshot = await backend.snapshot()
        assert snapshot.vision_enabled is False

    asyncio.run(run())
