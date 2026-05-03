#!/bin/bash
# One-command launcher for the Veda Cart web interface.
#
# Usage:
#   ./start.sh                 # uses Pi IP from cart.config (or default 192.168.0.221)
#   ./start.sh 172.20.10.5     # override Pi IP for this run (use after switching networks)
#
# What this does:
#   1. Makes sure the Pi agent is running on the cart (starts it in a
#      persistent tmux session over SSH if it isn't already).
#   2. Sets up the Mac-side Python venv on first run.
#   3. Starts the dashboard on http://localhost:8000 — open that in your
#      browser, enter the Pi IP in the sidebar, click Connect.
#
# If the Pi isn't reachable, the dashboard still starts so you can connect
# to "mock" for local testing.
#
# Configure by copying cart.config.example to cart.config and editing it.
# Run from: your local machine (Mac/Linux). Do NOT run this on the Pi.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_lib.sh"

# Parse args: any non-flag positional arg overrides the configured Pi IP,
# and --restart-agent / --restart / -r forces a Pi-agent restart even if it's
# already running. Order between the IP and the flag doesn't matter.
PI_IP_OVERRIDE=""
FORCE_RESTART=0
for arg in "$@"; do
  case "${arg}" in
    --restart-agent|--restart|-r) FORCE_RESTART=1 ;;
    -*) echo "  warn: ignoring unknown flag '${arg}'" ;;
    *) [ -z "${PI_IP_OVERRIDE}" ] && PI_IP_OVERRIDE="${arg}" ;;
  esac
done
if [ -n "${PI_IP_OVERRIDE}" ]; then
  CART_PI_IP="${PI_IP_OVERRIDE}"
fi

LOCAL_DIR="$(resolve_local_project_dir)"
cd "${LOCAL_DIR}"

# ---- Step 1: ensure the Pi agent is running ----------------------------------
echo "Checking Pi at ${CART_PI_USER}@${CART_PI_IP}..."

if ssh -o ConnectTimeout=3 -o BatchMode=no -q "${CART_PI_USER}@${CART_PI_IP}" true 2>/dev/null; then
  echo "  Pi reachable. Making sure the agent is up..."
  ssh -T "${CART_PI_USER}@${CART_PI_IP}" bash <<EOF
    set -e
    if [ "${FORCE_RESTART}" = "1" ]; then
      echo "  Force-restart requested — killing existing tmux session."
      tmux kill-session -t cart 2>/dev/null || true
    elif curl -sf http://localhost:8001/api/status >/dev/null 2>&1; then
      echo "  Pi agent already running."
      exit 0
    fi
    tmux kill-session -t cart 2>/dev/null || true
    tmux new-session -d -s cart "cd ${CART_PI_PROJECT_DIR} && source .venv/bin/activate && bash scripts/pi_run_agent.sh"
    echo "  Pi agent launched in tmux session 'cart'."
EOF
else
  echo "  Pi not reachable. Dashboard will still start — connect to 'mock' in the sidebar for local testing."
fi

# ---- Step 2: ensure the Mac-side venv is ready -------------------------------
if [ ! -d ".venv" ]; then
  echo "First-time setup: creating .venv and installing the project..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e .
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python3 -c "import fastapi, httpx, pydantic, uvicorn" 2>/dev/null || pip install -e .
fi

# ---- Step 3: start the dashboard --------------------------------------------
# Kill any previous local dashboard listening on :8000 so uvicorn can bind cleanly.
DASHBOARD_PIDS="$(lsof -ti :8000 2>/dev/null || true)"
if [ -n "${DASHBOARD_PIDS}" ]; then
  echo "Killing previous dashboard on :8000 (pids: ${DASHBOARD_PIDS})"
  echo "${DASHBOARD_PIDS}" | xargs kill -9 2>/dev/null || true
  sleep 0.2
fi

echo
echo "Dashboard: http://localhost:8000"
echo "  Pi address to enter in sidebar: ${CART_PI_IP}"
echo "  (or 'mock' if the Pi is off)"
echo

exec python3 -m uvicorn cart_stack.dashboard.app:app --host 0.0.0.0 --port 8000
