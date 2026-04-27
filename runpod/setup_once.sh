#!/bin/bash
# setup_once.sh — run ONCE per network volume (first pod ever)
# After this, only setup_pod.sh is needed on restarts.
set -e

echo "=== [1/5] Installing system packages ==="
apt-get update -q && apt-get install -y -q tmux vim libegl1-mesa cmake  # libegl1-mesa for MUJOCO_GL=egl; cmake for egl-probe build

echo "=== [2/5] Cloning openpi ==="
cd /workspace
git clone https://github.com/Shubodh/openpi.git
cd openpi
git submodule update --init --recursive

echo "=== [3/5] Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

echo "=== [4/5] Setting OPENPI_DATA_HOME (checkpoint cache → network volume) ==="
export OPENPI_DATA_HOME=/workspace/openpi_assets
mkdir -p /workspace/openpi_assets
echo 'export OPENPI_DATA_HOME=/workspace/openpi_assets' >> ~/.bashrc

echo "=== [5/5] Setting up LIBERO client venv and installing deps ==="
cd /workspace/openpi
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip install -r examples/libero/requirements.txt -r third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero

echo ""
echo "=== setup_once.sh complete ==="
echo "Next: run 'bash /workspace/openpi/runpod/setup_agents.sh' to install Claude Code + Codex."
echo "Then: 'bash /workspace/openpi/runpod/start_libero.sh' to launch server + client."
echo "(On future restarts, run setup_pod.sh first, then start_libero.sh)"
