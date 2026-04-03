from __future__ import annotations

import os
from dataclasses import dataclass

try:
    import lgpio  # type: ignore
except Exception:  # pragma: no cover - optional on laptops
    lgpio = None


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _env_flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class DriveState:
    left: float = 0.0
    right: float = 0.0


class BaseMotorDriver:
    def __init__(self) -> None:
        self.state = DriveState()
        self.left_pin = int(os.getenv("CART_LEFT_PIN", "18"))
        self.right_pin = int(os.getenv("CART_RIGHT_PIN", "13"))
        self.left_trim = float(os.getenv("CART_LEFT_TRIM", "0.0"))
        self.right_trim = float(os.getenv("CART_RIGHT_TRIM", "0.0"))
        self.deadband = float(os.getenv("CART_STOP_DEADBAND", "0.04"))
        self.idle_mode = os.getenv("CART_IDLE_PWM_MODE", "off").strip().lower()

    async def set_drive(self, left: float, right: float) -> None:
        self.state = DriveState(_clamp(left), _clamp(right))

    async def stop(self) -> None:
        await self.set_drive(0.0, 0.0)

    async def set_motor_pins(self, left_pin: int, right_pin: int) -> None:
        self.left_pin = int(left_pin)
        self.right_pin = int(right_pin)

    async def set_motor_calibration(self, left_trim: float, right_trim: float, stop_deadband: float) -> None:
        self.left_trim = float(left_trim)
        self.right_trim = float(right_trim)
        self.deadband = float(stop_deadband)

    async def close(self) -> None:
        await self.stop()


class MockMotorDriver(BaseMotorDriver):
    """Safe default for laptops and non-Pi environments."""


class LgpioServoMotorDriver(BaseMotorDriver):
    def __init__(self) -> None:
        super().__init__()
        if lgpio is None:
            raise RuntimeError("lgpio is not available.")

        self.left_invert = _env_flag("CART_LEFT_INVERT", default=False)
        self.right_invert = _env_flag("CART_RIGHT_INVERT", default=True)
        self.min_pulse_width = float(os.getenv("CART_MIN_PULSE_WIDTH", "0.0005"))
        self.max_pulse_width = float(os.getenv("CART_MAX_PULSE_WIDTH", "0.0025"))
        self.frame_width = float(os.getenv("CART_FRAME_WIDTH", "0.02"))
        self.frequency_hz = float(os.getenv("CART_SERVO_FREQ_HZ", "50.0"))
        self._chip = lgpio.gpiochip_open(0)
        self._claim_pin(self.left_pin)
        self._claim_pin(self.right_pin)
        self._write_idle(self.left_pin)
        self._write_idle(self.right_pin)

    def _claim_pin(self, pin: int) -> None:
        try:
            lgpio.gpio_claim_output(self._chip, pin, 0)
        except Exception:
            pass

    def _pulse_width_for_value(self, value: float, *, invert: bool, trim: float) -> float:
        command = _clamp((-value if invert else value) + trim)
        if abs(command) <= self.deadband:
            command = 0.0
        neutral = (self.min_pulse_width + self.max_pulse_width) / 2.0
        half_span = (self.max_pulse_width - self.min_pulse_width) / 2.0
        return neutral + (command * half_span)

    def _write_pwm(self, pin: int, pulse_width_s: float) -> None:
        pulse_width_us = max(500, min(2500, int(round(pulse_width_s * 1_000_000.0))))
        lgpio.tx_servo(self._chip, pin, pulse_width_us)

    def _write_neutral(self, pin: int) -> None:
        neutral = (self.min_pulse_width + self.max_pulse_width) / 2.0
        self._write_pwm(pin, neutral)

    def _write_off(self, pin: int) -> None:
        try:
            lgpio.tx_servo(self._chip, pin, 1500)
        except Exception:
            pass
        lgpio.gpio_write(self._chip, pin, 0)

    def _write_idle(self, pin: int) -> None:
        if self.idle_mode == "neutral":
            self._write_neutral(pin)
        else:
            self._write_off(pin)

    async def set_drive(self, left: float, right: float) -> None:
        await super().set_drive(left, right)
        left_pulse = self._pulse_width_for_value(self.state.left, invert=self.left_invert, trim=self.left_trim)
        right_pulse = self._pulse_width_for_value(self.state.right, invert=self.right_invert, trim=self.right_trim)
        self._write_pwm(self.left_pin, left_pulse)
        self._write_pwm(self.right_pin, right_pulse)

    async def set_motor_pins(self, left_pin: int, right_pin: int) -> None:
        left_pin = int(left_pin)
        right_pin = int(right_pin)
        if left_pin == self.left_pin and right_pin == self.right_pin:
            return
        await self.stop()
        self._claim_pin(left_pin)
        self._claim_pin(right_pin)
        self.left_pin = left_pin
        self.right_pin = right_pin
        self._write_idle(self.left_pin)
        self._write_idle(self.right_pin)

    async def set_motor_calibration(self, left_trim: float, right_trim: float, stop_deadband: float) -> None:
        await super().set_motor_calibration(left_trim, right_trim, stop_deadband)
        await self.stop()

    async def stop(self) -> None:
        await super().stop()
        self._write_idle(self.left_pin)
        self._write_idle(self.right_pin)

    async def close(self) -> None:
        try:
            await self.stop()
        finally:
            try:
                lgpio.gpio_write(self._chip, self.left_pin, 0)
            except Exception:
                pass
            try:
                lgpio.gpio_write(self._chip, self.right_pin, 0)
            except Exception:
                pass
            try:
                lgpio.gpio_free(self._chip, self.left_pin)
            except Exception:
                pass
            try:
                lgpio.gpio_free(self._chip, self.right_pin)
            except Exception:
                pass
            lgpio.gpiochip_close(self._chip)


class ServoMotorDriver(BaseMotorDriver):
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

        self.left_invert = _env_flag("CART_LEFT_INVERT", default=False)
        self.right_invert = _env_flag("CART_RIGHT_INVERT", default=True)
        self.left_servo = None
        self.right_servo = None
        self._open_servos(self.left_pin, self.right_pin)

    def _create_servo_pair(self, left_pin: int, right_pin: int):
        left_servo = self._servo_cls(left_pin, **self._servo_kwargs)
        right_servo = self._servo_cls(right_pin, **self._servo_kwargs)
        left_servo.mid()
        right_servo.mid()
        return left_servo, right_servo

    def _open_servos(self, left_pin: int, right_pin: int) -> None:
        left_servo, right_servo = self._create_servo_pair(left_pin, right_pin)
        old_left = self.left_servo
        old_right = self.right_servo
        self.left_servo = left_servo
        self.right_servo = right_servo
        self.left_pin = int(left_pin)
        self.right_pin = int(right_pin)
        if old_left is not None:
            old_left.close()
        if old_right is not None:
            old_right.close()

    def _set_idle(self, servo) -> None:
        if self.idle_mode == "neutral":
            servo.mid()
        else:
            servo.detach()

    def _apply_channel(self, servo, value: float, *, invert: bool, trim: float) -> None:
        command = _clamp((-value if invert else value) + trim)
        if abs(command) <= self.deadband:
            servo.mid()
        else:
            servo.value = command

    async def set_drive(self, left: float, right: float) -> None:
        await super().set_drive(left, right)
        self._apply_channel(self.left_servo, self.state.left, invert=self.left_invert, trim=self.left_trim)
        self._apply_channel(self.right_servo, self.state.right, invert=self.right_invert, trim=self.right_trim)

    async def set_motor_pins(self, left_pin: int, right_pin: int) -> None:
        left_pin = int(left_pin)
        right_pin = int(right_pin)
        if left_pin == self.left_pin and right_pin == self.right_pin:
            return
        await self.stop()
        self._open_servos(left_pin, right_pin)

    async def set_motor_calibration(self, left_trim: float, right_trim: float, stop_deadband: float) -> None:
        await super().set_motor_calibration(left_trim, right_trim, stop_deadband)
        await self.stop()

    async def stop(self) -> None:
        await super().stop()
        self._set_idle(self.left_servo)
        self._set_idle(self.right_servo)

    async def close(self) -> None:
        await self.stop()
        self.left_servo.close()
        self.right_servo.close()


def build_motor_driver() -> BaseMotorDriver:
    if os.getenv("CART_MOTOR_MODE", "mock").strip().lower() == "mock":
        print("[motors] Using MockMotorDriver.", flush=True)
        return MockMotorDriver()

    try:
        driver = LgpioServoMotorDriver()
        print(
            f"[motors] Using LgpioServoMotorDriver on left GPIO {driver.left_pin}, right GPIO {driver.right_pin}.",
            flush=True,
        )
        return driver
    except Exception as exc:
        print(f"[motors] lgpio servo init failed, trying gpiozero Servo fallback: {exc!r}", flush=True)

    try:
        driver = ServoMotorDriver()
        print(
            f"[motors] Using ServoMotorDriver on left GPIO {driver.left_pin}, right GPIO {driver.right_pin}.",
            flush=True,
        )
        return driver
    except Exception as exc:
        print(f"[motors] Falling back to MockMotorDriver because servo init failed: {exc!r}", flush=True)
        return MockMotorDriver()
