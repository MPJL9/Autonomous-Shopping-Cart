# Autonomous Shopping Cart

This repo now includes a local-first software stack for your shopping cart prototype:

- A tablet-style web dashboard with `Scan`, `Live`, and `Manual` workflows, built-in readme tabs, motion math, and Pi connection handling.
- A Raspberry Pi agent that exposes camera and motor endpoints over HTTP so the dashboard can connect by IP later.
- A small test suite plus GitHub Actions CI so your team can push code without breaking the basic control contract.

## What You Can Run Right Now

### 1. Start the dashboard on your laptop

```powershell
cd "C:\Users\Admin_4\Documents\New project\Autonomous-Shopping-Cart-main"
python -m pip install -e .[dev]
python -m cart_stack.dashboard.run
```

Or use the helper:

```powershell
.\scripts\start_dashboard.bat
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Use `mock` as the target first. That gives you:

- The dashboard UI
- A simulated robot backend
- Browser camera fallback for the live feed panel
- Manual joystick control
- A `Scan` mode for single-frame capture
- A `Live` mode for continuous camera plus auto-follow scaffolding
- A terminal for commands like `forward 0.4`, `spin_left 0.3`, `gap 0.7`, `auto`, `estop`
- A motion calculator for the milestone questions

### 2. Start the Pi agent later on the Raspberry Pi

```bash
cd ~/Autonomous-Shopping-Cart-main
python3 -m pip install -e .[pi]
export CART_MOTOR_MODE=servo
export CART_LEFT_PIN=18
export CART_RIGHT_PIN=13
python3 -m cart_stack.agent.run
```

Or use the deploy helper from Windows first:

```powershell
.\scripts\deploy_to_pi.ps1
```

That script will:

- archive the local repo
- copy it to `veda@10.65.142.17`
- unpack it on the Pi
- run [`pi_setup.sh`](C:/Users/Admin_4/Documents/New%20project/Autonomous-Shopping-Cart-main/scripts/pi_setup.sh)

After that, on the Pi:

```bash
bash ~/Autonomous-Shopping-Cart-main/scripts/pi_run_agent.sh
```

Then point the dashboard to:

```text
http://<pi-ip>:8001
```

You can also type the host as:

```text
veda@10.65.142.17
```

The dashboard will normalize that into the Pi agent URL automatically.

For hosted use, the dashboard now proxies the remote camera through `/api/camera/stream`, so teammates can stay on one dashboard URL. The important catch is that the dashboard server still needs a real network path to the Pi agent.

The dashboard will use:

- `GET /api/status`
- `POST /api/drive`
- `POST /api/stop`
- `POST /api/mode`
- `POST /api/estop`
- `POST /api/target-gap`
- `GET /stream.mjpg`

## Milestone Questions Covered in the Dashboard

The Motion Lab panel answers the exact planning questions you mentioned:

- acceleration
- weight
- turn time from rest
- turn time while moving at a chosen velocity

The formulas used are:

- `v = (v_r + v_l) / 2`
- `omega = (v_r - v_l) / L`
- `omega_max(v) = 2 * (v_max - v) / L`
- `t_turn = theta / omega`

Interpretation:

- Acceleration is `dv/dt`, so it is not automatically constant.
- For a real motorized cart, acceleration usually drops as speed rises because torque headroom shrinks.
- If both sides have the same max wheel speed, turn time gets worse as forward speed approaches max speed.

## Terminal Commands

The dashboard terminal supports:

- `help`
- `connect mock`
- `connect http://10.117.30.17:8001`
- `disconnect`
- `forward 0.4`
- `reverse 0.3`
- `left 0.4`
- `right 0.4`
- `spin_left 0.35`
- `spin_right 0.35`
- `drive 0.2 0.5`
- `stop`
- `manual`
- `auto`
- `estop`
- `reset`
- `gap 0.7`
- `status`

## Team Integration Path

### Hardware / Pi side

- `cart_stack/agent/motors.py`
  - Swap mock mode to servo mode with `CART_MOTOR_MODE=servo`
  - Calibrate trims with `CART_LEFT_TRIM` and `CART_RIGHT_TRIM`
  - If one side spins backward, flip `CART_LEFT_INVERT` or `CART_RIGHT_INVERT`

### Vision side

- Keep using the existing CV work under `Jimmy/cv_model/`
- Feed estimated distance, angle, and turn intent into the Pi agent or future controller layer
- The dashboard already has UI placeholders for those signals through status and overlay fields

### Control side

- Straight-line following can own `target_gap_m`
- Turn logic can own heading correction and later replace the placeholder auto-mode behavior
- The HTTP contract can stay stable while your teammates iterate internally

## Notes for the Current Hardware

- Your GPIO mapping from the prompt is already reflected in the Pi defaults:
  - left side: GPIO 18
  - right side: GPIO 13
- Because you are using continuous rotation servos, you will almost certainly need trim calibration.
- The current `auto` mode is a scaffold for the straight-line MVP. It is meant as an integration point, not the final learned controller.
- On a Raspberry Pi 3B or 4, the power connector is not also a laptop data link, so a single power cable will not replace Wi-Fi or Ethernet for coding.
- For teammates outside your local hotspot, use a secure tunnel or mesh VPN such as Tailscale instead of exposing the Pi directly to the public internet.

## Tests

```powershell
pytest
```

This currently checks:

- motion-model math
- target normalization for Pi URLs
- terminal command handling on the mock backend
