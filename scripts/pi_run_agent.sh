#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$APP_DIR/.venv"
ENV_FILE="$APP_DIR/.env.pi"
export CART_ENV_FILE="$ENV_FILE"

# BC policies in training/runs were trained on ArUco-derived distance/heading.
# Default to ArUco at runtime so deploy-time perception matches training-time.
# Override in .env.pi if you want the HOG markerless tracker.
export CART_VISION_MODE="${CART_VISION_MODE:-aruco}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

exec "$VENV_DIR/bin/python" -m cart_stack.agent.run
