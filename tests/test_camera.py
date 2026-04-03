from datetime import UTC, datetime

from cart_stack.agent.camera import ResilientCameraSource
from cart_stack.shared.models import RobotSnapshot


class _FailingSource:
    def capture_jpeg(self, snapshot: RobotSnapshot) -> bytes:
        raise RuntimeError("camera failed")

    def close(self) -> None:
        return None


class _WorkingSource:
    def __init__(self, payload: bytes = b"frame") -> None:
        self.payload = payload

    def capture_jpeg(self, snapshot: RobotSnapshot) -> bytes:
        return self.payload

    def close(self) -> None:
        return None


def _snapshot() -> RobotSnapshot:
    return RobotSnapshot(updated_at=datetime.now(UTC).isoformat())


def test_resilient_camera_source_uses_working_factory() -> None:
    def bad_factory():
        raise RuntimeError("camera unavailable")

    source = ResilientCameraSource(
        [
            ("bad", bad_factory),
            ("good", lambda: _WorkingSource(b"real")),
        ],
        retry_interval_s=0.0,
    )

    assert source.capture_jpeg(_snapshot()) == b"real"


def test_resilient_camera_source_recovers_after_capture_failure() -> None:
    calls = {"count": 0}

    def flaky_factory():
        calls["count"] += 1
        if calls["count"] == 1:
            return _FailingSource()
        return _WorkingSource(b"recovered")

    source = ResilientCameraSource([("flaky", flaky_factory)], retry_interval_s=0.0)

    assert source.capture_jpeg(_snapshot()) == b"recovered"
