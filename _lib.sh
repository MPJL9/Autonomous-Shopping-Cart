# Shared helpers for start.sh and deploy_to_pi.sh. Sourced, not executed.

# Directory that contains this file (absolute).
CART_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The project root is the directory this file lives in, since these scripts
# ship at the root of the Autonomous-Shopping-Cart repo.
CART_LOCAL_PROJECT_DIR="${CART_SCRIPT_DIR}"

# Defaults — overridden by cart.config if present.
CART_PI_IP="${CART_PI_IP:-192.168.0.221}"
CART_PI_USER="${CART_PI_USER:-veda}"
CART_PI_PROJECT_DIR="${CART_PI_PROJECT_DIR:-~/Autonomous-Shopping-Cart}"

# Load user's config if present.
if [ -f "${CART_SCRIPT_DIR}/cart.config" ]; then
  # shellcheck disable=SC1090
  source "${CART_SCRIPT_DIR}/cart.config"
fi

# Sanity-check the project layout. Returns the absolute project dir or exits 1.
resolve_local_project_dir() {
  if [ ! -f "${CART_LOCAL_PROJECT_DIR}/pyproject.toml" ] || [ ! -d "${CART_LOCAL_PROJECT_DIR}/cart_stack" ]; then
    cat >&2 <<EOF
The project structure looks wrong. Expected pyproject.toml and cart_stack/
inside:
  ${CART_LOCAL_PROJECT_DIR}

Make sure start.sh is at the root of the Autonomous-Shopping-Cart repository.
EOF
    return 1
  fi
  echo "${CART_LOCAL_PROJECT_DIR}"
}
