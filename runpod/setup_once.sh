#!/bin/bash
# setup_once.sh — run ONCE per network volume (first pod ever)
# After this, only setup_pod.sh is needed on restarts.
set -e

OPENPI_DIR=/workspace/openpi

echo "=== [1/5] Installing system packages ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q ca-certificates cmake curl git libegl1-mesa libglu1-mesa rsync tmux vim xclip

echo "=== [1b/5] Configuring git identity ==="
git config --global user.name "Shubodh RunPod April"
git config --global user.email "p.saishubodh@gmail.com"

echo "=== [2/5] Cloning openpi ==="
cd /workspace
if [ -d "$OPENPI_DIR/.git" ]; then
  cd "$OPENPI_DIR"
elif [ -e "$OPENPI_DIR" ]; then
  echo "ERROR: $OPENPI_DIR exists but is not a git checkout." >&2
  exit 1
else
  git clone https://github.com/Shubodh/openpi.git
  cd "$OPENPI_DIR"
fi
git submodule update --init --recursive

# shellcheck source=/dev/null
source "$OPENPI_DIR/runpod/setup_common.sh"

echo "=== [2b/5] Setting up tmux config + plugins ==="
restore_tmux_config

echo "=== [3/5] Installing uv ==="
install_uv

echo "=== [4/5] Setting persistent cache/env paths on the network volume ==="
configure_persistent_env

echo "=== [5/5] Setting up LIBERO client venv and installing deps ==="
cd "$OPENPI_DIR"

echo "=== [5a/5] Configuring LIBERO paths non-interactively ==="
write_libero_config

echo "=== [5b/5] Installing persistent Python 3.8 and creating LIBERO venv ==="
ensure_libero_client_venv

echo "=== Installing LIBERO simulation deps into server venv (for main_patching_expt.py) ==="
ensure_server_patching_venv

echo ""
echo "=== setup_once.sh complete ==="
echo "Next: run 'bash /workspace/openpi/runpod/setup_agents.sh' to install Codex CLI."
echo "Then: 'bash /workspace/openpi/runpod/start_libero.sh' to launch server + client."
echo "(On future restarts, run setup_pod.sh first, then start_libero.sh)"
