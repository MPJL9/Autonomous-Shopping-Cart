#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${1:-$HOME/Autonomous-Shopping-Cart-main}"
VENV_DIR="$APP_DIR/.venv"
ENV_FILE="$APP_DIR/.env.pi"
SERVICE_SRC="$APP_DIR/scripts/cart-agent.service"
SERVICE_DEST="/etc/systemd/system/cart-agent.service"
PI_SUDO_PASS="${PI_SUDO_PASS:-}"

as_root() {
  if sudo -n true >/dev/null 2>&1; then
    sudo "$@"
  elif [[ -n "$PI_SUDO_PASS" ]]; then
    printf '%s\n' "$PI_SUDO_PASS" | sudo -S "$@"
  else
    sudo "$@"
  fi
}

ensure_env_key() {
  local key="$1"
  local value="$2"
  if grep -q "^${key}=" "$ENV_FILE" 2>/dev/null; then
    return
  fi
  printf '%s=%s\n' "$key" "$value" >>"$ENV_FILE"
}

echo "[pi-setup] Updating apt metadata..."
as_root apt-get update

echo "[pi-setup] Installing base packages..."
as_root apt-get install -y \
  git \
  python3-gpiozero \
  python3-pip \
  python3-venv

for pkg in python3-pigpio pigpio-tools; do
  if apt-cache show "$pkg" >/dev/null 2>&1; then
    echo "[pi-setup] Installing optional package: $pkg"
    as_root apt-get install -y "$pkg"
  else
    echo "[pi-setup] Optional package not available in apt: $pkg"
  fi
done

for pkg in python3-picamera2 python3-opencv; do
  if apt-cache show "$pkg" >/dev/null 2>&1; then
    echo "[pi-setup] Installing optional package: $pkg"
    as_root apt-get install -y "$pkg"
  else
    echo "[pi-setup] Optional package not available in apt: $pkg"
  fi
done

if command -v pigpiod >/dev/null 2>&1 && systemctl list-unit-files pigpiod.service >/dev/null 2>&1; then
  echo "[pi-setup] Enabling pigpiod..."
  as_root systemctl enable pigpiod
  as_root systemctl restart pigpiod || true
else
  echo "[pi-setup] pigpiod not available on this image; continuing with gpiozero default pin factory."
fi

echo "[pi-setup] Creating virtual environment..."
python3 -m venv --system-site-packages "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -e "$APP_DIR"
chmod +x "$APP_DIR/scripts/pi_run_agent.sh" "$APP_DIR/scripts/pi_setup.sh"

if [[ ! -f "$ENV_FILE" ]]; then
  cat >"$ENV_FILE" <<'EOF'
CART_AGENT_HOST=0.0.0.0
CART_AGENT_PORT=8001
CART_MOTOR_MODE=servo
CART_LEFT_PIN=12
CART_RIGHT_PIN=13
CART_LEFT_INVERT=0
CART_RIGHT_INVERT=1
CART_LEFT_TRIM=0.0
CART_RIGHT_TRIM=0.0
CART_STOP_DEADBAND=0.04
CART_IDLE_PWM_MODE=off
CART_CAMERA_MODE=auto
CART_CAMERA_RETRY_SEC=5.0
CART_TRACK_WIDTH_M=0.58
CART_MAX_SPEED_MPS=0.9
CART_VISION_INTERVAL_SEC=0.45
EOF
  echo "[pi-setup] Wrote default Pi environment to $ENV_FILE"
else
  echo "[pi-setup] Reusing existing $ENV_FILE"
fi

ensure_env_key "CART_AGENT_HOST" "0.0.0.0"
ensure_env_key "CART_AGENT_PORT" "8001"
ensure_env_key "CART_MOTOR_MODE" "servo"
ensure_env_key "CART_LEFT_PIN" "12"
ensure_env_key "CART_RIGHT_PIN" "13"
ensure_env_key "CART_LEFT_INVERT" "0"
ensure_env_key "CART_RIGHT_INVERT" "1"
ensure_env_key "CART_LEFT_TRIM" "0.0"
ensure_env_key "CART_RIGHT_TRIM" "0.0"
ensure_env_key "CART_STOP_DEADBAND" "0.04"
ensure_env_key "CART_IDLE_PWM_MODE" "off"
ensure_env_key "CART_CAMERA_MODE" "auto"
ensure_env_key "CART_CAMERA_RETRY_SEC" "5.0"
ensure_env_key "CART_TRACK_WIDTH_M" "0.58"
ensure_env_key "CART_MAX_SPEED_MPS" "0.9"
ensure_env_key "CART_VISION_INTERVAL_SEC" "0.45"

if [[ -f "$SERVICE_SRC" ]]; then
  echo "[pi-setup] Installing systemd service..."
  SERVICE_TMP="$(mktemp)"
  sed "s|/home/veda/Autonomous-Shopping-Cart-main|$APP_DIR|g; s|User=veda|User=$(whoami)|g" "$SERVICE_SRC" >"$SERVICE_TMP"
  as_root cp "$SERVICE_TMP" "$SERVICE_DEST"
  rm -f "$SERVICE_TMP"
  as_root systemctl daemon-reload
  as_root systemctl enable cart-agent.service
  as_root systemctl restart cart-agent.service
fi

echo "[pi-setup] Setup complete."
echo
echo "Agent status:"
as_root systemctl --no-pager --full status cart-agent.service || true
echo
echo "Then connect the dashboard to:"
echo "  http://$(hostname -I | awk '{print $1}'):8001"
