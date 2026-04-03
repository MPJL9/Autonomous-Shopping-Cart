from cart_stack.shared.models import MotionModelInput
from cart_stack.shared.physics import calculate_motion_model


def test_motion_model_turn_time_grows_with_speed() -> None:
    standstill = calculate_motion_model(MotionModelInput(cruise_speed_mph=0.0))
    moving = calculate_motion_model(MotionModelInput(cruise_speed_mph=1.6))

    assert standstill.standstill_turn_time_s is not None
    assert moving.moving_turn_time_s is not None
    assert moving.moving_turn_time_s > standstill.standstill_turn_time_s


def test_motion_model_reports_weight_and_acceleration() -> None:
    result = calculate_motion_model(MotionModelInput(total_mass_kg=30.0, zero_to_max_speed_s=3.0))

    assert result.weight_n > 250
    assert result.weight_lbf > 60
    assert result.average_acceleration_mps2 == 0.3

