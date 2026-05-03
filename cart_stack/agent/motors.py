from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    import lgpio  # type: ignore
except Exception:  # pragma: no cover - optional on laptops
    lgpio = None


MOTOR_KEYS = ("left1", "left2", "right1", "right2")


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _env_flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class DriveState:
    left: float = 0.0
    right: float = 0.0


@dataclass
class MotorConfig:
    pin: int
    invert: bool = False
    trim: float = 0.0


_TRIM_DEFAULTS = {
    # Per-wheel scalar trim from the 4/26 bench calibration
    # (see calibration table in report.tex). These are baked in so the
    # cart still has correct trims even if .env.pi is missing/reset; any
    # CART_<motor>_TRIM env var will override the default per-wheel.
    "left1":  -0.11,
    "left2":  -0.05,
    "right1": -0.15,
    "right2": -0.015,
}


def _load_motor_configs() -> dict[str, MotorConfig]:
    defaults = {
        "left1": MotorConfig(pin=int(os.getenv("CART_LEFT1_PIN", "13")), invert=_env_flag("CART_LEFT1_INVERT", False)),
        "left2": MotorConfig(pin=int(os.getenv("CART_LEFT2_PIN", "16")), invert=_env_flag("CART_LEFT2_INVERT", False)),
        "right1": MotorConfig(pin=int(os.getenv("CART_RIGHT1_PIN", "19")), invert=_env_flag("CART_RIGHT1_INVERT", True)),
        "right2": MotorConfig(pin=int(os.getenv("CART_RIGHT2_PIN", "12")), invert=_env_flag("CART_RIGHT2_INVERT", True)),
    }
    for key in MOTOR_KEYS:
        defaults[key].trim = float(
            os.getenv(f"CART_{key.upper()}_TRIM", str(_TRIM_DEFAULTS[key]))
        )
    return defaults


def _side_of(key: str) -> str:
    return "left" if key.startswith("left") else "right"


class BaseMotorDriver:
    def __init__(self) -> None:
        self.state = DriveState()
        self.motors: dict[str, MotorConfig] = _load_motor_configs()
        self.deadband = float(os.getenv("CART_STOP_DEADBAND", "0.04"))
        self.idle_mode = os.getenv("CART_IDLE_PWM_MODE", "off").strip().lower()

    # Convenience accessors so existing log lines keep working.
    @property
    def left_pin(self) -> int:
        return self.motors["left1"].pin

    @property
    def right_pin(self) -> int:
        return self.motors["right1"].pin

    def pins(self) -> dict[str, int]:
        return {k: m.pin for k, m in self.motors.items()}

    def trims(self) -> dict[str, float]:
        return {k: m.trim for k, m in self.motors.items()}

    def inverts(self) -> dict[str, bool]:
        return {k: m.invert for k, m in self.motors.items()}

    async def set_drive(self, left: float, right: float) -> None:
        self.state = DriveState(_clamp(left), _clamp(right))

    async def stop(self) -> None:
        await self.set_drive(0.0, 0.0)

    async def set_motor_pins(
        self, left1_pin: int, left2_pin: int, right1_pin: int, right2_pin: int
    ) -> None:
        new_pins = {"left1": int(left1_pin), "left2": int(left2_pin), "right1": int(right1_pin), "right2": int(right2_pin)}
        for key, pin in new_pins.items():
            self.motors[key].pin = pin

    async def set_motor_inverts(
        self, left1_invert: bool, left2_invert: bool, right1_invert: bool, right2_invert: bool
    ) -> None:
        inverts = {"left1": bool(left1_invert), "left2": bool(left2_invert), "right1": bool(right1_invert), "right2": bool(right2_invert)}
        for key, inv in inverts.items():
            self.motors[key].invert = inv

    async def set_motor_calibration(
        self,
        left1_trim: float,
        left2_trim: float,
        right1_trim: float,
        right2_trim: float,
        stop_deadband: float,
    ) -> None:
        trims = {"left1": float(left1_trim), "left2": float(left2_trim), "right1": float(right1_trim), "right2": float(right2_trim)}
        for key, trim in trims.items():
            self.motors[key].trim = trim
        self.deadband = float(stop_deadband)

    async def close(self) -> None:
        await self.stop()


class MockMotorDriver(BaseMotorDriver):
    """Safe default for laptops and non-Pi environments."""


class LgpioServoMotorDriver(BaseMotorDriver):
    """Drive four continuous-rotation servos through lgpio with jitter suppression.

    Jitter fix (adapted from the ExplainingComputers no-jitter approach):
      * Cache the last pulse width written to each pin and skip the write when the
        value hasn't changed — continuous re-writes of the same duty cycle are the
        main source of servo hunt/twitch on software PWM.
      * When a motor is in the stop deadband, drop the pulse entirely (tx_servo 0)
        instead of re-sending the neutral pulse. A CR servo without a pulse stops
        cleanly; a CR servo being fed its neutral pulse tends to creep and buzz.
    """

    def __init__(self) -> None:
        super().__init__()
        if lgpio is None:
            raise RuntimeError("lgpio is not available.")

        self.min_pulse_width = float(os.getenv("CART_MIN_PULSE_WIDTH", "0.0005"))
        self.max_pulse_width = float(os.getenv("CART_MAX_PULSE_WIDTH", "0.0025"))
        self.frame_width = float(os.getenv("CART_FRAME_WIDTH", "0.02"))
        self.frequency_hz = float(os.getenv("CART_SERVO_FREQ_HZ", "50.0"))

        self._chip = lgpio.gpiochip_open(0)
        self._last_pulse_us: dict[int, int] = {}
        for key in MOTOR_KEYS:
            pin = self.motors[key].pin
            self._claim_pin(pin)
            self._write_idle(pin)

    def _claim_pin(self, pin: int) -> None:
        try:
            lgpio.gpio_claim_output(self._chip, pin, 0)
        except Exception:
            pass

    def _release_pin(self, pin: int) -> None:
        try:
            lgpio.tx_servo(self._chip, pin, 0)
        except Exception:
            pass
        try:
            lgpio.gpio_write(self._chip, pin, 0)
        except Exception:
            pass
        try:
            lgpio.gpio_free(self._chip, pin)
        except Exception:
            pass
        self._last_pulse_us.pop(pin, None)

    def _pulse_width_for_value(self, value: float, motor: MotorConfig) -> float:
        command = _clamp((-value if motor.invert else value) + motor.trim)
        if abs(command) <= self.deadband:
            command = 0.0
        neutral = (self.min_pulse_width + self.max_pulse_width) / 2.0
        half_span = (self.max_pulse_width - self.min_pulse_width) / 2.0
        return neutral + (command * half_span)

    # Tolerance for reusing an existing pulse width without re-invoking tx_servo.
    # tx_servo restarts lgpio's PWM timer each call, which causes micro-glitches.
    # With joystick input arriving ~10 Hz and every tick producing a slightly
    # different floating-point command, exact-match dedup is ineffective — we
    # need a band around the last sent value.
    _TX_HYSTERESIS_US = 40

    def _tx(self, pin: int, pulse_us: int) -> None:
        last = self._last_pulse_us.get(pin)
        if last is not None and last >= 0 and abs(last - pulse_us) < self._TX_HYSTERESIS_US:
            return
        try:
            lgpio.tx_servo(self._chip, pin, pulse_us)
        except Exception:
            return
        self._last_pulse_us[pin] = pulse_us

    def _cut_pulses(self, pin: int) -> None:
        """Stop PWM on a pin. tx_servo(..., 0) rejects 0 as 'bad PWM micros'
        on some lgpio builds, so use tx_pwm(..., 0%) which is accepted everywhere.
        Falls back to driving the line low.
        """
        try:
            lgpio.tx_pwm(self._chip, pin, self.frequency_hz, 0.0)
        except Exception:
            pass
        try:
            lgpio.gpio_write(self._chip, pin, 0)
        except Exception:
            pass
        self._last_pulse_us[pin] = -1  # sentinel so next real pulse always resends

    def _write_pwm(self, pin: int, pulse_width_s: float, *, is_idle: bool) -> None:
        if is_idle:
            # Fully cut pulses for CR servos at rest: this is the key jitter fix.
            self._cut_pulses(pin)
            return
        pulse_us = max(500, min(2500, int(round(pulse_width_s * 1_000_000.0))))
        self._tx(pin, pulse_us)

    def _write_neutral(self, pin: int) -> None:
        neutral = (self.min_pulse_width + self.max_pulse_width) / 2.0
        pulse_us = max(500, min(2500, int(round(neutral * 1_000_000.0))))
        self._tx(pin, pulse_us)

    def _write_off(self, pin: int) -> None:
        self._cut_pulses(pin)

    def _write_idle(self, pin: int) -> None:
        if self.idle_mode == "neutral":
            self._write_neutral(pin)
        else:
            self._write_off(pin)

    def _apply_motor(self, key: str, side_value: float) -> None:
        motor = self.motors[key]
        # Idle on the user's intended value, not the trimmed one — otherwise a
        # nonzero trim leaks into the "stop" command and one wheel keeps
        # emitting pulses while another cuts off. That asymmetric stop is what
        # produced the at-rest jitter.
        is_idle = abs(side_value) <= self.deadband
        pulse = self._pulse_width_for_value(side_value, motor)
        self._write_pwm(motor.pin, pulse, is_idle=is_idle)

    async def set_drive(self, left: float, right: float) -> None:
        await super().set_drive(left, right)
        self._apply_motor("left1", self.state.left)
        self._apply_motor("left2", self.state.left)
        self._apply_motor("right1", self.state.right)
        self._apply_motor("right2", self.state.right)

    async def set_motor_pins(
        self, left1_pin: int, left2_pin: int, right1_pin: int, right2_pin: int
    ) -> None:
        new_pins = {"left1": int(left1_pin), "left2": int(left2_pin), "right1": int(right1_pin), "right2": int(right2_pin)}
        changed = any(self.motors[k].pin != new_pins[k] for k in MOTOR_KEYS)
        if not changed:
            return
        await self.stop()
        for key in MOTOR_KEYS:
            old_pin = self.motors[key].pin
            new_pin = new_pins[key]
            if old_pin != new_pin:
                self._release_pin(old_pin)
            self.motors[key].pin = new_pin
            self._claim_pin(new_pin)
            self._write_idle(new_pin)

    async def set_motor_calibration(
        self,
        left1_trim: float,
        left2_trim: float,
        right1_trim: float,
        right2_trim: float,
        stop_deadband: float,
    ) -> None:
        await super().set_motor_calibration(left1_trim, left2_trim, right1_trim, right2_trim, stop_deadband)
        await self.stop()

    async def stop(self) -> None:
        await super().stop()
        for key in MOTOR_KEYS:
            self._write_idle(self.motors[key].pin)

    async def close(self) -> None:
        try:
            await self.stop()
        finally:
            for key in MOTOR_KEYS:
                self._release_pin(self.motors[key].pin)
            try:
                lgpio.gpiochip_close(self._chip)
            except Exception:
                pass


class ServoMotorDriver(BaseMotorDriver):
    """gpiozero-based fallback driver for four continuous-rotation servos."""

    def __init__(self) -> None:
        super().__init__()
        from gpiozero import Servo

        factory = None
        if _env_flag("CART_USE_PIGPIO", default=True):
            try:
                from gpiozero.pins.pigpio import PiGPIOFactory

                factory = PiGPIOFactory()
            except Exception:
                factory = None

        self._servo_cls = Servo
        self._servo_kwargs = {
            "min_pulse_width": float(os.getenv("CART_MIN_PULSE_WIDTH", "0.0005")),
            "max_pulse_width": float(os.getenv("CART_MAX_PULSE_WIDTH", "0.0025")),
            "frame_width": float(os.getenv("CART_FRAME_WIDTH", "0.02")),
        }
        if factory is not None:
            self._servo_kwargs["pin_factory"] = factory

        self._servos: dict[str, object] = {}
        self._last_value: dict[str, float] = {}
        for key in MOTOR_KEYS:
            self._open_servo(key, self.motors[key].pin)

    def _open_servo(self, key: str, pin: int) -> None:
        old = self._servos.pop(key, None)
        servo = self._servo_cls(pin, **self._servo_kwargs)
        servo.mid()
        self._servos[key] = servo
        self._last_value.pop(key, None)
        if old is not None:
            try:
                old.close()
            except Exception:
                pass

    def _set_idle(self, key: str) -> None:
        servo = self._servos[key]
        if self.idle_mode == "neutral":
            servo.mid()
        else:
            try:
                servo.detach()
            except Exception:
                pass
        self._last_value[key] = 0.0

    def _apply_channel(self, key: str, side_value: float) -> None:
        motor = self.motors[key]
        # Idle on the user's intended value, not the trimmed one (see _apply_motor).
        if abs(side_value) <= self.deadband:
            self._set_idle(key)
            return
        command = _clamp((-side_value if motor.invert else side_value) + motor.trim)
        if self._last_value.get(key) == command:
            return
        self._servos[key].value = command
        self._last_value[key] = command

    async def set_drive(self, left: float, right: float) -> None:
        await super().set_drive(left, right)
        self._apply_channel("left1", self.state.left)
        self._apply_channel("left2", self.state.left)
        self._apply_channel("right1", self.state.right)
        self._apply_channel("right2", self.state.right)

    async def set_motor_pins(
        self, left1_pin: int, left2_pin: int, right1_pin: int, right2_pin: int
    ) -> None:
        new_pins = {"left1": int(left1_pin), "left2": int(left2_pin), "right1": int(right1_pin), "right2": int(right2_pin)}
        changed = any(self.motors[k].pin != new_pins[k] for k in MOTOR_KEYS)
        if not changed:
            return
        await self.stop()
        for key, pin in new_pins.items():
            self.motors[key].pin = pin
            self._open_servo(key, pin)

    async def set_motor_calibration(
        self,
        left1_trim: float,
        left2_trim: float,
        right1_trim: float,
        right2_trim: float,
        stop_deadband: float,
    ) -> None:
        await super().set_motor_calibration(left1_trim, left2_trim, right1_trim, right2_trim, stop_deadband)
        await self.stop()

    async def stop(self) -> None:
        await super().stop()
        for key in MOTOR_KEYS:
            self._set_idle(key)

    async def close(self) -> None:
        await self.stop()
        for key in MOTOR_KEYS:
            servo = self._servos.get(key)
            if servo is not None:
                try:
                    servo.close()
                except Exception:
                    pass


def build_motor_driver() -> BaseMotorDriver:
    if os.getenv("CART_MOTOR_MODE", "mock").strip().lower() == "mock":
        print("[motors] Using MockMotorDriver.", flush=True)
        return MockMotorDriver()

    try:
        driver = LgpioServoMotorDriver()
        pins = driver.pins()
        print(
            f"[motors] Using LgpioServoMotorDriver on L1={pins['left1']} L2={pins['left2']} R1={pins['right1']} R2={pins['right2']}.",
            flush=True,
        )
        return driver
    except Exception as exc:
        print(f"[motors] lgpio servo init failed, trying gpiozero Servo fallback: {exc!r}", flush=True)

    try:
        driver = ServoMotorDriver()
        pins = driver.pins()
        print(
            f"[motors] Using ServoMotorDriver on L1={pins['left1']} L2={pins['left2']} R1={pins['right1']} R2={pins['right2']}.",
            flush=True,
        )
        return driver
    except Exception as exc:
        print(f"[motors] Falling back to MockMotorDriver because servo init failed: {exc!r}", flush=True)
        return MockMotorDriver()
