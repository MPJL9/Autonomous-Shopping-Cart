#!/bin/bash
# Push the latest Autonomous-Shopping-Cart source from this machine to the Pi
# AND restart the Pi agent so it picks up the new code.
# Config:  copy cart.config.example to cart.config and edit if your setup differs.
# Usage:   ./deploy_to_pi.sh                # uses cart.config / defaults
#          ./deploy_to_pi.sh <pi-ip>        # override Pi IP for one run
#          ./deploy_to_pi.sh --no-restart   # rsync only, don't restart the agent
# Run from: your local machine (Mac/Linux).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_lib.sh"

RESTART_AGENT=1
POSITIONAL=()
for arg in "$@"; do
  case "${arg}" in
    --no-restart|--no-restart-agent) RESTART_AGENT=0 ;;
    *) POSITIONAL+=("${arg}") ;;
  esac
done

PI_IP_OVERRIDE="${POSITIONAL[0]:-}"
if [ -n "${PI_IP_OVERRIDE}" ]; then
  CART_PI_IP="${PI_IP_OVERRIDE}"
fi

LOCAL_SRC="$(resolve_local_project_dir)"
REMOTE_DEST="${CART_PI_USER}@${CART_PI_IP}:${CART_PI_PROJECT_DIR}/"

echo "Pushing ${LOCAL_SRC}/ → ${REMOTE_DEST}"

rsync -avz --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'runtime_logs' \
  --exclude '.DS_Store' \
  --exclude '*.egg-info' \
  --exclude '.env.pi' \
  "${LOCAL_SRC}/" \
  "${REMOTE_DEST}"

echo

# Copy bundled BC policies into the agent's runtime policy directory on the
# Pi, so the agent loads the latest weights on next restart. Bypasses the
# dashboard's install-builtin path entirely — no UI interaction needed.
LOCAL_POLICY_DIR="${LOCAL_SRC}/cart_stack/dashboard/policies"
NPZ_FILES=$(find "${LOCAL_POLICY_DIR}" -maxdepth 1 -name '*.npz' -type f 2>/dev/null | wc -l | tr -d ' ')
if [ "${NPZ_FILES}" -gt 0 ]; then
  echo "Pushing ${NPZ_FILES} bundled policies → ~/cart_runtime_logs/policy/ on Pi..."
  ssh -T "${CART_PI_USER}@${CART_PI_IP}" "mkdir -p ~/cart_runtime_logs/policy"
  rsync -avz "${LOCAL_POLICY_DIR}"/*.npz \
    "${CART_PI_USER}@${CART_PI_IP}:~/cart_runtime_logs/policy/"
  echo
else
  echo "  (no bundled .npz in ${LOCAL_POLICY_DIR}, skipping policy push)"
fi

if [ "${RESTART_AGENT}" = "1" ]; then
  echo "Restarting Pi agent so it picks up the new code..."
  ssh -T "${CART_PI_USER}@${CART_PI_IP}" bash <<EOF
    tmux kill-session -t cart 2>/dev/null || true
    tmux new-session -d -s cart "cd ${CART_PI_PROJECT_DIR} && source .venv/bin/activate && bash scripts/pi_run_agent.sh"
    echo "  Pi agent restarted in tmux session 'cart'."
EOF
  echo
  echo "Now hard-refresh the browser (Cmd+Shift+R) and reconnect."
else
  echo "Skipped agent restart (--no-restart). Restart manually when ready:"
  echo "  ssh ${CART_PI_USER}@${CART_PI_IP} 'tmux kill-session -t cart'"
  echo "  ./start.sh --restart-agent"
fi
