# Pi Setup — Autonomous Shopping Cart

Everything you need to talk to the Pi, push new code to it, and run the agent.

---

## Pi Credentials

| Field     | Value                              |
| --------- | ---------------------------------- |
| Hostname  | `veda`                             |
| Username  | `veda`                             |
| Password  | `veda123`                          |
| SSH port  | `22`                               |

### Known IP addresses (vary by network)

| Network                  | Pi IP           | SSH command                     | Agent URL                       |
| ------------------------ | --------------- | ------------------------------- | ------------------------------- |
| Jimmy's home WiFi        | `192.168.0.221` | `ssh veda@192.168.0.221`        | `http://192.168.0.221:8001`     |
| Jimmy's phone hotspot    | `172.20.10.5`   | `ssh veda@172.20.10.5`          | `http://172.20.10.5:8001`       |

> Pi IPs change per network (DHCP). After switching WiFi, look up the new IP with
> `hostname -I` on the Pi, or `arp -a` on the Mac once both are on the same LAN.
> Pass the current IP to `./start.sh <IP>` or set `CART_PI_IP=<IP>` in `cart.config`.

---

## How to Reach the Pi

### Primary method — use the IP directly

```bash
ssh veda@192.168.0.221
```

This is known to work. Use it.

### Why `ping cartpi.local` failed

The Pi's hostname is `veda`, not `cartpi` — it was set to `veda` in Pi Imager. The hostname-based address would be `veda.local`, not `cartpi.local`. Try:

```bash
ping veda.local
```

If that *also* fails, mDNS isn't working on your Mac or the Pi. Options:

- Stick with the IP (`192.168.0.221`) — simplest
- Enable the mDNS daemon on the Pi:
  ```bash
  # On the Pi
  sudo apt install -y avahi-daemon
  sudo systemctl enable --now avahi-daemon
  ```
  Then try `ping veda.local` again from the Mac.

### Finding the IP if it changes

When DHCP hands out a new address, re-find it from the Pi:

```bash
# On the Pi (via monitor/keyboard if SSH no longer works)
hostname -I
```

Or scan the LAN from the Mac:

```bash
arp -a | grep -i b8:27:eb   # Pi 3 MAC prefix (old Pis)
arp -a | grep -i dc:a6:32   # Pi 4 MAC prefix
```

---

## First-Time Setup on the Pi

Do this once after reflashing. All commands run **on the Pi** (SSH in first).

```bash
# 1. System packages
sudo apt update
sudo apt install -y \
  python3-venv python3-pip \
  python3-lgpio python3-picamera2 \
  git avahi-daemon

# 2. Make a project directory
mkdir -p ~/Autonomous-Shopping-Cart

# 3. Enable camera if using Pi Camera (not needed for USB cams)
# sudo raspi-config  →  Interface Options  →  Camera  →  Enable

# 4. (Optional) add veda to the gpio + video groups so sudo isn't needed
sudo usermod -aG gpio,video,dialout veda
```

Log out and back in after the `usermod` for the groups to stick.

---

## Deploying Code from Mac → Pi

Run this from your **Mac** whenever you want to push the latest source.

```bash
rsync -avz --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'runtime_logs' \
  --exclude '.DS_Store' \
  --exclude '*.egg-info' \
  /Users/tliu/Desktop/Workspace/Autonomous-Shopping-Cart/ \
  veda@192.168.0.221:~/Autonomous-Shopping-Cart/
```

Save it as a shell script (e.g. `~/push_to_pi.sh`) so you don't retype:

```bash
#!/bin/bash
PI_IP="${1:-192.168.0.221}"
rsync -avz --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude '*.pyc' --exclude 'runtime_logs' --exclude '.DS_Store' \
  --exclude '*.egg-info' \
  /Users/tliu/Desktop/Workspace/Autonomous-Shopping-Cart/ \
  veda@"$PI_IP":~/Autonomous-Shopping-Cart/
```

Then `chmod +x ~/push_to_pi.sh` and run `~/push_to_pi.sh 10.65.x.x`.

---

## Python Environment (one-time on the Pi)

```bash
# On the Pi
cd ~/Autonomous-Shopping-Cart

# Create a venv that can still see system packages (picamera2, lgpio)
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# Install the project
pip install -e .
```

From now on, every shell that runs the agent should `source .venv/bin/activate` first (or the run script handles it).

---

## Running the Agent

```bash
# On the Pi
cd ~/Autonomous-Shopping-Cart
bash scripts/pi_run_agent.sh
```

The agent listens on port `8001`. Leave it running — don't close the SSH session unless you use `tmux` / `screen` / `nohup` (see below).

### Keep it running after SSH logout

```bash
# Option A: tmux (recommended)
sudo apt install -y tmux
tmux new -s cart
bash scripts/pi_run_agent.sh
# Detach: Ctrl+B then D.   Reattach later: tmux attach -t cart

# Option B: nohup
nohup bash scripts/pi_run_agent.sh > ~/agent.log 2>&1 &
tail -f ~/agent.log
```

---

## Dashboard (runs on your Mac, connects to the Pi)

```bash
# On your Mac
cd /Users/tliu/Desktop/Workspace/Autonomous-Shopping-Cart
# (make sure the Mac side also has a venv with the project installed)
uvicorn cart_stack.dashboard.app:app --host 0.0.0.0 --port 8000
```

Open a browser to `http://localhost:8000`. In the sidebar, enter `192.168.0.221` in the "Pi Address" box and click **Connect**. The live stream should appear.

---

## Hardware — 4 Motors

The car has **four continuous-rotation servos** (one per wheel). Each motor connects to a single GPIO signal wire. Power comes from the battery — **do not power the servos from the Pi's 5V rail**; that causes brown-outs.

### Default GPIO assignments (editable from the dashboard sidebar)

| Wheel | GPIO (BCM) | Physical pin | Signal wire color | `.env.pi` variable |
| ----- | ---------- | ------------ | ----------------- | ------------------ |
| L1    | 13         | 33           | black             | `CART_LEFT1_PIN`   |
| L2    | 16         | 36           | purple            | `CART_LEFT2_PIN`   |
| R1    | 19         | 35           | white             | `CART_RIGHT1_PIN`  |
| R2    | 12         | 32           | gray              | `CART_RIGHT2_PIN`  |

Plus one **ground** wire (typically black on the servo/battery side) from battery
negative to any Pi GND pin (physical pins 6, 9, 14, 20, 25, 30, 34, 39). Without
this common ground, PWM signals have no reference and the servos ignore them.

### Calibration workflow

1. Connect to the Pi from the dashboard
2. Apply the four pin numbers you actually wired
3. Drive `forward 0.2` — watch each wheel
4. If a wheel spins **the wrong way**, check its box in **Wheel Direction** and click **Apply Direction**
5. If a stopped wheel **creeps**, nudge its trim (±0.02 at a time) in **Servo Trim**

---

## Troubleshooting

### "Low voltage" warning on the Pi

The Pi isn't getting enough power. Usually caused by:

- Servos sharing the Pi's 5V supply — **give servos their own battery**
- A weak USB-C / micro-USB cable feeding the Pi — use the official supply

### Agent won't start — "lgpio is not available"

```bash
sudo apt install -y python3-lgpio
# Then re-create the venv with --system-site-packages so it sees lgpio:
cd ~/Autonomous-Shopping-Cart
rm -rf .venv
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -e .
```

### Dashboard shows "Pi Offline"

1. `curl http://192.168.0.221:8001/api/status` from the Mac — does it respond?
2. If not, check the agent log on the Pi (`tail -f ~/agent.log`)
3. Check the Pi's firewall isn't blocking port 8001 (usually not enabled by default)

### Camera feed is blank

```bash
# On the Pi — test the camera independently
libcamera-hello --timeout 2000
# If that fails, re-enable the camera:
sudo raspi-config    # → Interface → Camera → Enable
```

### Wheels spin but cart doesn't move

- Physically: check all four wheels are contacting the ground (shim / re-tighten mounts if not)
- Logically: the two wheels on the same side may be mounted mirrored — flip the invert toggle for one of them so they spin in the mechanically-same direction

### Jittering / buzzing wheels at rest

Handled by the servo driver — at rest, pulses are cut entirely. If still buzzing, increase **Stop Deadband** slightly (e.g. 0.04 → 0.08).

---

## File Layout Reference

On the Pi after first setup:

```
/home/veda/
├── Autonomous-Shopping-Cart/         # synced from Mac via rsync
│   ├── .env.pi                       # persisted motor pins/trims live here
│   ├── .venv/                        # Python virtualenv (Pi-only, not synced)
│   ├── cart_stack/
│   └── scripts/pi_run_agent.sh
├── cart_runtime_logs/rl/             # RL session JSONL logs (auto-created)
├── agent.log                         # if running with nohup
└── push_to_pi.sh                     # optional: your local deploy script
```

---

## Quick Command Cheat Sheet

| What                     | Where | Command                                     |
| ------------------------ | ----- | ------------------------------------------- |
| SSH to Pi                | Mac   | `ssh veda@192.168.0.221`                          |
| Push code                | Mac   | `~/push_to_pi.sh 192.168.0.221` (once set up)     |
| Start agent              | Pi    | `bash scripts/pi_run_agent.sh`              |
| Start agent in tmux      | Pi    | `tmux new -s cart` then the above           |
| Reattach tmux            | Pi    | `tmux attach -t cart`                       |
| Stop agent (in its tmux) | Pi    | `Ctrl+C`                                    |
| Run dashboard            | Mac   | `uvicorn cart_stack.dashboard.app:app -p 8000` |
| Pi's IP                  | Pi    | `hostname -I`                               |
| Reboot Pi                | Pi    | `sudo reboot`                               |
| Shutdown Pi              | Pi    | `sudo shutdown now`                         |
