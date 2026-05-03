"""Placeholder RL exploration session.

Scope
-----
This is a *scaffold* the dashboard can drive while the real PPO/IL pipeline in
``cart_stack/rl/`` matures. It runs a simple action policy on the live robot
runtime at a fixed step rate, logs ``(observation, action, reward, done)``
tuples to a JSONL file under ``~/cart_runtime_logs/rl/``, and can be started
and stopped via HTTP from the dashboard's RL Training tab.

Observation: ``[distance_m, heading_rad, linear_velocity_mps, angular_velocity_rad_s]``
(matches ``cart_stack/rl/env.py::CartFollowEnv``).

Action: ``[left, right]`` in ``[-1, 1]``.

Reward: ``-(distance - target_gap)**2`` (a toy proxy that matches the
distance-keeping spirit of the project; swap for the real reward when the
RL pipeline is wired in).

Policies
--------
- ``random``   : uniform random actions (useful for kicking off data logging)
- ``zero``     : always (0, 0) — for verifying the episode loop without moving
- ``forward``  : constant gentle forward command — for smoke-testing the cart
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from cart_stack.shared.models import RlSessionStatus

if TYPE_CHECKING:
    from cart_stack.agent.app import RobotAgentRuntime


_LOG_DIR = Path(os.getenv("CART_RL_LOG_DIR", str(Path.home() / "cart_runtime_logs" / "rl")))
_POLICY_DIR = Path(os.getenv("CART_RL_POLICY_DIR", str(Path.home() / "cart_runtime_logs" / "policy")))
_POLICY_PATH_COMBINED = _POLICY_DIR / "bc_2d_combined_v2.npz"   # 4/29 + 5/1 combined (live-obs corrected)
_POLICY_PATH_5_1_ONLY = _POLICY_DIR / "bc_2d_5_1only_v2.npz"    # 5/1 only (live-obs corrected)
_POLICY_PATH_PRE_FIX  = _POLICY_DIR / "bc_2d_combined_pre_fix.npz"  # OLD combined, broken-overlay era
_POLICY_PATH_4_29     = _POLICY_DIR / "bc_2d.npz"                # 4/29 only, blockcap, no mirror
_POLICY_PATH_4_29_MIR = _POLICY_DIR / "bc_2d_mirror.npz"         # 4/29 only, blockcap, with mirror
_POLICY_PATH_AWR      = _POLICY_DIR / "bc_2d_awr.npz"            # AWR-refined bc_2d on bc_2d_pivot rollouts
_POLICY_PATH = _POLICY_PATH_COMBINED  # legacy alias for older callers

# Physical slots = the npz files on disk that `policy_slot_path` resolves.
_VALID_SLOTS = {
    "bc_2d_combined_v2",
    "bc_2d_5_1only_v2",
    "bc_2d_combined_pre_fix",
    "bc_2d",
    "bc_2d_mirror",
    "bc_2d_awr",
}

# Policy names = what the dashboard sends. Each maps to (weights slot, wrapper).
# Raw policies use the BC output directly. Pivot adds a heading P-controller.
# Translate runs a deterministic pivot-translate-pivot FSM when off-axis.
_RAW_POLICIES = {
    "bc_2d_combined_v2", "bc_2d_5_1only_v2",
    "bc_2d_combined_pre_fix",
    "bc_2d", "bc_2d_mirror",
    "bc_2d_awr",
}
_PIVOT_POLICIES = {
    "bc_2d_combined_v2_pivot",
    "bc_2d_combined_pre_fix_pivot",
    "bc_2d_pivot",
    "bc_2d_awr_pivot",
}
_TRANSLATE_POLICIES = {
    "bc_2d_combined_v2_translate",
    "bc_2d_combined_pre_fix_translate",
    "bc_2d_translate",
    "bc_2d_awr_translate",
}
_BC_POLICIES = _RAW_POLICIES | _PIVOT_POLICIES | _TRANSLATE_POLICIES

# policy name -> weights slot (which npz to load)
_POLICY_TO_SLOT = {
    "bc_2d_combined_v2":                "bc_2d_combined_v2",
    "bc_2d_combined_v2_pivot":          "bc_2d_combined_v2",
    "bc_2d_combined_v2_translate":      "bc_2d_combined_v2",
    "bc_2d_5_1only_v2":                 "bc_2d_5_1only_v2",
    "bc_2d_combined_pre_fix":           "bc_2d_combined_pre_fix",
    "bc_2d_combined_pre_fix_pivot":     "bc_2d_combined_pre_fix",
    "bc_2d_combined_pre_fix_translate": "bc_2d_combined_pre_fix",
    "bc_2d":                            "bc_2d",
    "bc_2d_mirror":                     "bc_2d_mirror",
    "bc_2d_pivot":                      "bc_2d",
    "bc_2d_translate":                  "bc_2d",
    "bc_2d_awr":                        "bc_2d_awr",
    "bc_2d_awr_pivot":                  "bc_2d_awr",
    "bc_2d_awr_translate":              "bc_2d_awr",
}

# Distance deadband — when the cart is already inside the desired follow band,
# zero out the forward command so it doesn't creep. Hysteresis prevents bouncing
# in/out at the band edge. Applied to all bc_* policies; Tier 1's heading
# P-controller is preserved (the cart can still rotate to face a marker that
# drifted laterally), and Tier 2's open-loop maneuvers ignore the deadband
# entirely (the FSM has its own logic).
#
# Per-slot defaults match the demo equilibrium each slot was trained on:
#   - 4/29 era operator drove the cart toward d≈1.4m, runtime band [1.26, 1.56].
#   - 5/1 era operator drove toward d≈1.16m, runtime band [1.01, 1.32]
#     (= EVALUATION.md canonical band, combined Q25-Q75).
#   - pre_fix combined was trained on 4/29 + stale-5/1; its learned equilibrium
#     matches the 4/29 era so we use the 4/29 band.
# Env vars CART_BC_BAND_{MIN,MAX,HYST}_M override these as a global force-all.
_BAND_4_29 = (1.26, 1.56)
_BAND_5_1  = (1.01, 1.32)
_SLOT_BANDS: dict[str, tuple[float, float]] = {
    "bc_2d":                  _BAND_4_29,
    "bc_2d_mirror":           _BAND_4_29,
    "bc_2d_combined_pre_fix": _BAND_4_29,
    "bc_2d_combined_v2":      _BAND_5_1,
    "bc_2d_5_1only_v2":       _BAND_5_1,
    "bc_2d_awr":              _BAND_4_29,  # AWR-refined from 4/29 BC, ran on 4/29-band rollouts
}

# Legacy globals — kept so older callers/tests that imported them still work.
_BC_BAND_MIN_M = float(os.getenv("CART_BC_BAND_MIN_M", "1.26"))
_BC_BAND_MAX_M = float(os.getenv("CART_BC_BAND_MAX_M", "1.56"))
_BC_BAND_HYST_M = float(os.getenv("CART_BC_BAND_HYST_M", "0.05"))


def _resolve_band(slot: str) -> tuple[float, float, float]:
    """Return (band_min, band_max, hyst) for a weights slot.

    Env vars are a global override; if any are set, they apply to every slot.
    Otherwise we look up the slot in `_SLOT_BANDS` and fall back to the 4/29
    band for unknown slots (the legacy default).
    """
    if "CART_BC_BAND_MIN_M" in os.environ or "CART_BC_BAND_MAX_M" in os.environ:
        band_min = float(os.environ.get("CART_BC_BAND_MIN_M", "1.26"))
        band_max = float(os.environ.get("CART_BC_BAND_MAX_M", "1.56"))
    else:
        band_min, band_max = _SLOT_BANDS.get(slot, _BAND_4_29)
    hyst = float(os.getenv("CART_BC_BAND_HYST_M", "0.05"))
    return band_min, band_max, hyst


def _update_band_state(in_band: bool, distance: float,
                       band_min: float = _BC_BAND_MIN_M,
                       band_max: float = _BC_BAND_MAX_M,
                       hyst: float = _BC_BAND_HYST_M) -> bool:
    """Hysteresis. Returns updated in_band flag.

    - To enter the band: distance must fall inside [min, max].
    - To leave the band: distance must move past the boundary by HYST.
    """
    if in_band:
        if distance < band_min - hyst:
            return False
        if distance > band_max + hyst:
            return False
        return True
    if band_min <= distance <= band_max:
        return True
    return False


# Tier 1 — pivot-while-following gain. Maps |heading| (rad) to a turn command
# in normalized action units. K=0.6 means heading=0.5 rad gives turn=0.30.
_TIER1_K_HEADING = 0.6

# Tier 2 — open-loop pivot-translate-pivot maneuver. Triggered when the marker
# drifts beyond the threshold AND the person is not actively walking. All
# durations are open-loop: no encoders, so the calibrated max_speed and
# track_width below are best-effort.
_TIER2_HEADING_TRIGGER_RAD = 0.30      # ~17°
_TIER2_DEBOUNCE_FRAMES = 3              # consecutive frames over threshold
_TIER2_PERSON_STATIONARY_M_PER_S = 0.10 # |lin_vel| must be under this to trigger
_TIER2_PIVOT_OMEGA = 0.6                # rad/s during pivot phases (open-loop)
_TIER2_DRIVE_LINEAR = 0.30              # m/s during translate phase (open-loop)
_TIER2_MAX_TOTAL_S = 7.0                # safety cutoff for the whole maneuver
_CART_MAX_SPEED_MPS = 0.9               # matches CartFollowEnv default
_CART_TRACK_WIDTH_M = 0.58              # matches CartFollowEnv default


def _pivot_action(omega_radps: float) -> tuple[float, float]:
    """Differential-drive command for pure rotation at omega rad/s.
    Positive omega = counterclockwise (turn left, person at heading > 0)."""
    diff = omega_radps * _CART_TRACK_WIDTH_M / _CART_MAX_SPEED_MPS
    diff = max(-1.0, min(1.0, diff))
    return -diff / 2.0, +diff / 2.0


def _drive_action(v_mps: float) -> tuple[float, float]:
    """Differential-drive command for pure forward (or backward) motion."""
    s = v_mps / _CART_MAX_SPEED_MPS
    s = max(-1.0, min(1.0, s))
    return s, s


class _Tier2FSM:
    """Open-loop pivot → drive → pivot-back maneuver.

    Triggered when |heading| exceeds threshold consistently while the
    person is stationary. Geometry: at distance d, lateral offset of the
    person is d*sin(heading). The maneuver pivots toward the person by
    α=heading, drives forward by d*sin(heading) (the chord), then pivots
    -α to restore the original orientation. The cart ends up roughly in
    front of the person on their new lateral position, at the same distance.
    """

    IDLE = "idle"
    PIVOT_OUT = "pivot_out"
    DRIVE = "drive"
    PIVOT_BACK = "pivot_back"

    def __init__(self) -> None:
        self.state = self.IDLE
        self.over_count = 0
        self.alpha = 0.0
        self.s = 0.0
        self.t_phase_end = 0.0
        self.t_started = 0.0

    def _begin(self, *, heading: float, distance: float, t_now: float) -> None:
        # Sign convention: heading > 0 means person to the right of the cart's
        # forward axis (positive tvec_x). Our positive omega is CCW (turn left),
        # so to *face right* we need negative omega. alpha is the signed angle
        # to add to the cart's yaw — same sign as heading.
        self.alpha = float(heading)
        self.s = float(distance) * abs(math.sin(heading))
        self.state = self.PIVOT_OUT
        self.t_started = t_now
        # Phase 1 duration: how long to pivot at omega to cover |alpha|.
        t_pivot = abs(self.alpha) / max(_TIER2_PIVOT_OMEGA, 1e-3)
        self.t_phase_end = t_now + t_pivot

    def step(self, *, obs, t_now: float) -> tuple[float, float, str]:
        """Return (left, right, label). label is the FSM phase string."""
        d, heading, lin_v, _ = obs

        if self.state == self.IDLE:
            if (
                abs(heading) >= _TIER2_HEADING_TRIGGER_RAD
                and abs(lin_v) <= _TIER2_PERSON_STATIONARY_M_PER_S
            ):
                self.over_count += 1
                if self.over_count >= _TIER2_DEBOUNCE_FRAMES:
                    self._begin(heading=heading, distance=d, t_now=t_now)
                    self.over_count = 0
                    return *_pivot_action(_pivot_omega_for_alpha(self.alpha)), self.state
            else:
                self.over_count = 0
            return 0.0, 0.0, self.state

        # Safety cutoff
        if t_now - self.t_started > _TIER2_MAX_TOTAL_S:
            self.state = self.IDLE
            self.over_count = 0
            return 0.0, 0.0, "abort"

        if self.state == self.PIVOT_OUT:
            if t_now >= self.t_phase_end:
                self.state = self.DRIVE
                t_drive = abs(self.s) / max(_TIER2_DRIVE_LINEAR, 1e-3)
                self.t_phase_end = t_now + t_drive
                return *_drive_action(_TIER2_DRIVE_LINEAR), self.state
            return *_pivot_action(_pivot_omega_for_alpha(self.alpha)), self.state

        if self.state == self.DRIVE:
            if t_now >= self.t_phase_end:
                self.state = self.PIVOT_BACK
                t_pivot = abs(self.alpha) / max(_TIER2_PIVOT_OMEGA, 1e-3)
                self.t_phase_end = t_now + t_pivot
                return *_pivot_action(_pivot_omega_for_alpha(-self.alpha)), self.state
            return *_drive_action(_TIER2_DRIVE_LINEAR), self.state

        if self.state == self.PIVOT_BACK:
            if t_now >= self.t_phase_end:
                self.state = self.IDLE
                self.over_count = 0
                return 0.0, 0.0, "done"
            return *_pivot_action(_pivot_omega_for_alpha(-self.alpha)), self.state

        # Unknown state — reset.
        self.state = self.IDLE
        return 0.0, 0.0, self.state


def _pivot_omega_for_alpha(alpha: float) -> float:
    """Sign of omega to actually rotate the cart toward heading=0.

    Recall: positive omega is CCW (turn left). If the person is to the right
    (heading > 0, alpha > 0), we want to *face right* → negative omega.
    """
    sign = -1.0 if alpha > 0 else +1.0
    return sign * _TIER2_PIVOT_OMEGA


def policy_slot_path(slot: str) -> Path:
    """Resolve a slot name to the on-disk npz path.

    Physical slots (npz files):
      bc_2d_combined_v2       — corrected 4/29 + 5/1 combined (live-obs)
      bc_2d_5_1only_v2        — corrected 5/1 only (live-obs)
      bc_2d_combined_pre_fix  — OLD combined from before the live-obs fix;
                                paired with a heading P-controller (Tier-1)
                                this was the only working configuration on 5/1
      bc_2d                   — 4/29 only, blockcap, no mirror augment
      bc_2d_mirror            — 4/29 only, blockcap, with mirror augment

    Each is referenced by raw + Tier-1 (`_pivot`) + Tier-2 (`_translate`)
    policy names that the dashboard sends — see `_POLICY_TO_SLOT`.
    """
    return {
        "bc_2d_combined_v2":      _POLICY_PATH_COMBINED,
        "bc_2d_5_1only_v2":       _POLICY_PATH_5_1_ONLY,
        "bc_2d_combined_pre_fix": _POLICY_PATH_PRE_FIX,
        "bc_2d":                  _POLICY_PATH_4_29,
        "bc_2d_mirror":           _POLICY_PATH_4_29_MIR,
        "bc_2d_awr":              _POLICY_PATH_AWR,
    }.get(slot) or _raise_unknown_slot(slot)


def _raise_unknown_slot(slot: str):
    raise ValueError(f"Unknown policy slot: {slot}. Valid: {sorted(_VALID_SLOTS)}")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _select_action(policy: str, scale: float) -> tuple[float, float]:
    if policy == "zero":
        return 0.0, 0.0
    if policy == "forward":
        return scale, scale
    # "random" — uniform exploration
    return random.uniform(-scale, scale), random.uniform(-scale, scale)


class NumpyMLPPolicy:
    """Numpy-only MLP inference. Loaded from an npz exported by
    training/export_policy.py — lets the Pi run trained policies without torch.

    Expected npz keys: obs_mean, obs_std, W0, b0, W1, b1, ..., Wn, bn,
                       activations (e.g. b"tanh|tanh"), final (b"tanh" or b"none"),
                       action_dim, obs_dim, n_layers.
    """

    def __init__(self, path: Path):
        import numpy as np  # local import so rl_session stays light when unused

        self._np = np
        data = np.load(str(path), allow_pickle=True)
        self.obs_mean = data["obs_mean"].astype(np.float32)
        self.obs_std = data["obs_std"].astype(np.float32)
        self.action_dim = int(data["action_dim"])
        self.obs_dim = int(data["obs_dim"])
        self.n_layers = int(data["n_layers"])
        self.weights = [data[f"W{i}"].astype(np.float32) for i in range(self.n_layers)]
        self.biases = [data[f"b{i}"].astype(np.float32) for i in range(self.n_layers)]

        def _decode(field):
            # npz 0-d byte-string arrays: tobytes() gives the raw bytes; item()
            # returns bytes. bytes(field) sometimes yields a repr — avoid it.
            if hasattr(field, "tobytes"):
                return field.tobytes().decode("utf-8", errors="replace")
            return str(field)

        self.hidden_acts = [a for a in _decode(data["activations"]).split("|") if a]
        self.final_act = _decode(data["final"])
        self.path = str(path)

    @staticmethod
    def _act(x, name):
        if name in ("tanh", "Tanh"):
            import numpy as np
            return np.tanh(x)
        if name in ("relu", "ReLU"):
            import numpy as np
            return np.maximum(x, 0.0)
        if name in ("silu", "SiLU"):
            import numpy as np
            return x / (1.0 + np.exp(-x))
        return x  # identity fallback

    def act(self, obs: tuple[float, float, float, float]) -> "list[float]":
        """Run a single forward pass. Returns action as a python list."""
        np = self._np
        x = (np.asarray(obs, dtype=np.float32) - self.obs_mean) / (self.obs_std + 1e-6)
        for i in range(self.n_layers):
            x = x @ self.weights[i].T + self.biases[i]
            if i < self.n_layers - 1 and i < len(self.hidden_acts):
                x = self._act(x, self.hidden_acts[i])
        if self.final_act != "none":
            x = self._act(x, self.final_act)
        return [float(v) for v in np.atleast_1d(x)]


class RlSession:
    """Single-session manager. Only one episode can run at a time."""

    def __init__(self, runtime: "RobotAgentRuntime") -> None:
        self._runtime = runtime
        self._task: asyncio.Task | None = None
        self._running = False
        self._policy = "random"
        self._step = 0
        self._max_steps = 0
        self._last_reward = 0.0
        self._total_reward = 0.0
        self._log_path: Path | None = None
        self._numpy_policy: NumpyMLPPolicy | None = None
        self._in_band: bool = False
        self._band_min: float = _BC_BAND_MIN_M
        self._band_max: float = _BC_BAND_MAX_M
        self._band_hyst: float = _BC_BAND_HYST_M

    def status(self) -> RlSessionStatus:
        return RlSessionStatus(
            running=self._running,
            policy=self._policy,
            step=self._step,
            max_steps=self._max_steps,
            last_reward=round(self._last_reward, 4),
            total_reward=round(self._total_reward, 4),
            log_path=str(self._log_path) if self._log_path else None,
        )

    async def start(
        self,
        *,
        policy: str = "random",
        max_steps: int = 300,
        step_hz: float = 5.0,
        action_scale: float = 0.35,
    ) -> None:
        if self._running:
            raise RuntimeError("An RL session is already running. Stop it first.")

        self._numpy_policy = None
        self._tier2_fsm: _Tier2FSM | None = None
        self._in_band = False
        if policy in _BC_POLICIES:
            slot = _POLICY_TO_SLOT[policy]
            self._band_min, self._band_max, self._band_hyst = _resolve_band(slot)
            slot_path = policy_slot_path(slot)
            if not slot_path.exists():
                raise RuntimeError(
                    f"{policy} policy requested but no checkpoint at {slot_path}. "
                    "Upload one via the dashboard or place an npz there."
                )
            try:
                self._numpy_policy = NumpyMLPPolicy(slot_path)
            except Exception as exc:
                raise RuntimeError(f"Failed to load {policy} policy: {exc!r}") from exc
            if self._numpy_policy.action_dim != 2:
                raise RuntimeError(
                    f"{policy} requires action_dim=2 but the npz at "
                    f"{slot_path} has action_dim={self._numpy_policy.action_dim}."
                )
            if policy in _TRANSLATE_POLICIES:
                self._tier2_fsm = _Tier2FSM()
            # BC policies need real perception, not the runtime's initial constant.
            # Force vision tracking on so distance/heading reflect the marker.
            try:
                await self._runtime.set_vision_tracking(True)
            except Exception as exc:
                self._runtime._log(  # type: ignore[attr-defined]
                    f"Could not auto-enable vision tracking for {policy}: {exc!r}")

        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._log_path = _LOG_DIR / f"rl_{policy}_{stamp}.jsonl"

        self._policy = policy
        self._step = 0
        self._max_steps = int(max_steps)
        self._last_reward = 0.0
        self._total_reward = 0.0
        self._running = True
        self._task = asyncio.create_task(self._run(step_hz=step_hz, action_scale=action_scale))
        self._runtime._log(  # type: ignore[attr-defined]
            f"RL session started (policy={policy}, max_steps={max_steps}, "
            f"step_hz={step_hz:.1f}, log={self._log_path})."
        )

    async def stop(self) -> None:
        was_manual = self._policy == "manual"
        self._running = False
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        # Don't slam the motors after a manual session — the human may still
        # have the joystick engaged and expect to keep driving.
        if not was_manual:
            try:
                await self._runtime.driver.stop()
            except Exception:
                pass
        self._runtime._log("RL session stopped.")  # type: ignore[attr-defined]

    async def _run(self, *, step_hz: float, action_scale: float) -> None:
        period = 1.0 / max(step_hz, 0.5)
        is_manual = self._policy == "manual"
        try:
            with (self._log_path.open("a", encoding="utf-8") if self._log_path else _DummyCtx()) as fh:
                self._write_header(fh, policy=self._policy, step_hz=step_hz, action_scale=action_scale)
                while self._running and self._step < self._max_steps:
                    tick_start = time.perf_counter()
                    obs = await self._runtime.observation()
                    if is_manual:
                        # Record what the human is currently commanding via joystick /
                        # quick buttons. Do NOT call set_drive — that would overwrite
                        # the user's input.
                        state = self._runtime.driver.state
                        action = (float(state.left), float(state.right))
                    elif self._policy in _RAW_POLICIES and self._numpy_policy is not None:
                        # Raw 2D BC policy: emit independent L/R commands directly.
                        # Distance deadband still zeros forward when in-band so the
                        # cart doesn't creep at equilibrium.
                        a = self._numpy_policy.act(obs)
                        left = float(a[0]) * action_scale
                        right = float(a[1]) * action_scale
                        self._in_band = _update_band_state(
                            self._in_band, float(obs[0]),
                            self._band_min, self._band_max, self._band_hyst,
                        )
                        if self._in_band:
                            left = right = 0.0
                        action = (left, right)
                        try:
                            await self._runtime.set_drive(left, right)
                        except Exception:
                            pass
                    elif self._policy in _PIVOT_POLICIES and self._numpy_policy is not None:
                        # Tier 1: 2D BC for forward speed + P-controller on heading.
                        a = self._numpy_policy.act(obs)
                        base_l = float(a[0]) * action_scale
                        base_r = float(a[1]) * action_scale
                        # Distance deadband zeros the forward component but
                        # preserves the heading correction so the cart can
                        # still rotate to face a laterally-drifting marker.
                        self._in_band = _update_band_state(
                            self._in_band, float(obs[0]),
                            self._band_min, self._band_max, self._band_hyst,
                        )
                        if self._in_band:
                            base_l = base_r = 0.0
                        # heading > 0 = person to the right → turn right.
                        # right - left = -K * heading -> left += K*h/2, right -= K*h/2.
                        h = float(obs[1])
                        delta = _TIER1_K_HEADING * h * 0.5
                        left = max(-1.0, min(1.0, base_l + delta))
                        right = max(-1.0, min(1.0, base_r - delta))
                        action = (left, right)
                        try:
                            await self._runtime.set_drive(left, right)
                        except Exception:
                            pass
                    elif (self._policy in _TRANSLATE_POLICIES
                          and self._numpy_policy is not None
                          and self._tier2_fsm is not None):
                        # Tier 2: deterministic pivot-translate-pivot maneuver
                        # overrides BC when |heading| > threshold and person is
                        # stationary. Otherwise BC drives forward as normal.
                        t_now = time.time()
                        fsm_l, fsm_r, fsm_state = self._tier2_fsm.step(
                            obs=obs, t_now=t_now)
                        if fsm_state == _Tier2FSM.IDLE:
                            a = self._numpy_policy.act(obs)
                            left = float(a[0]) * action_scale
                            right = float(a[1]) * action_scale
                            # Apply the same distance deadband bc_2d uses, so
                            # in-band weak commands don't leak through and
                            # land in the motor stiction zone (the "shaky at
                            # stop" failure mode). The Tier-2 maneuver path
                            # below intentionally bypasses the deadband
                            # because the FSM has its own in-band logic.
                            self._in_band = _update_band_state(
                                self._in_band, float(obs[0]),
                                self._band_min, self._band_max, self._band_hyst,
                            )
                            if self._in_band:
                                left = right = 0.0
                        else:
                            # During maneuver phases, use FSM open-loop commands
                            # as-is (do not multiply by action_scale — the pivot
                            # / drive constants are already in physical units).
                            left, right = fsm_l, fsm_r
                        action = (left, right)
                        try:
                            await self._runtime.set_drive(left, right)
                        except Exception:
                            pass
                    else:
                        action = _select_action(self._policy, action_scale)
                        try:
                            await self._runtime.set_drive(action[0], action[1])
                        except Exception:
                            pass
                    reward = self._compute_reward(obs)
                    self._last_reward = reward
                    self._total_reward += reward
                    self._step += 1
                    done = self._step >= self._max_steps
                    self._write_step(fh, obs, action, reward, done)
                    elapsed = time.perf_counter() - tick_start
                    await asyncio.sleep(max(0.0, period - elapsed))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._runtime._log(f"RL session error: {exc!r}")  # type: ignore[attr-defined]
        finally:
            self._running = False
            try:
                await self._runtime.driver.stop()
            except Exception:
                pass

    def _compute_reward(self, obs: tuple[float, float, float, float]) -> float:
        distance, _heading, _linear, _angular = obs
        target_gap = float(getattr(self._runtime, "_target_gap_m", 0.65))
        error = distance - target_gap
        return -(error * error)

    def _write_header(self, fh, *, policy: str, step_hz: float, action_scale: float) -> None:
        if not hasattr(fh, "write"):
            return
        header = {
            "type": "header",
            "started_at": _now_iso(),
            "policy": policy,
            "step_hz": step_hz,
            "action_scale": action_scale,
            "obs_keys": ["distance_m", "heading_rad", "linear_velocity_mps", "angular_velocity_rad_s"],
            "action_keys": ["left", "right"],
            "reward": "-(distance - target_gap)^2",
        }
        fh.write(json.dumps(header) + "\n")
        fh.flush()

    def _write_step(
        self,
        fh,
        obs: tuple[float, float, float, float],
        action: tuple[float, float],
        reward: float,
        done: bool,
    ) -> None:
        if not hasattr(fh, "write"):
            return
        record = {
            "type": "step",
            "t": _now_iso(),
            "step": self._step,
            "obs": [round(o, 4) for o in obs],
            "action": [round(a, 4) for a in action],
            "reward": round(reward, 4),
            "done": done,
        }
        fh.write(json.dumps(record) + "\n")
        fh.flush()


class _DummyCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
