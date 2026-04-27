#!/bin/bash
# setup_pod.sh — run after EVERY pod restart (~3-5 min)
# Venv + checkpoint on /workspace survive pod stop — packages persist.
# uv and system packages are wiped on pod stop — reinstalled here.
# AI agents (Claude Code, Codex) are NOT reinstalled here — run setup_agents.sh separately if needed.
set -e

echo "=== [1/4] Installing system packages ==="
apt-get update -q && apt-get install -y -q tmux vim libegl1-mesa

echo "=== [2/4] Re-installing uv (wiped on pod stop) ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

echo "=== [3/4] Restoring OPENPI_DATA_HOME ==="
export OPENPI_DATA_HOME=/workspace/openpi_assets
echo 'export OPENPI_DATA_HOME=/workspace/openpi_assets' >> ~/.bashrc

echo "=== [4/4] Ensuring LIBERO venv deps are installed ==="
cd /workspace/openpi
source examples/libero/.venv/bin/activate
uv pip install -r examples/libero/requirements.txt -r third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero

echo ""
echo "=== setup_pod.sh complete ==="
echo "Next: run 'bash /workspace/openpi/runpod/start_libero.sh'"
echo "If you need AI agents: run 'bash /workspace/openpi/runpod/setup_agents.sh' first"
