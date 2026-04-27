#!/bin/bash
# setup_pod.sh — run after EVERY pod restart (fast, ~1-2 min)
# Venv + checkpoint on /workspace survive pod stop — no reinstall needed.
# uv and system packages are wiped on pod stop — reinstalled here.
# AI agents (Claude Code, Codex) are NOT reinstalled here — run setup_agents.sh separately if needed.
set -e

echo "=== [1/3] Installing system packages ==="
apt-get update -q && apt-get install -y -q tmux vim libegl1-mesa

echo "=== [2/3] Re-installing uv (wiped on pod stop) ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

echo "=== [3/3] Restoring OPENPI_DATA_HOME ==="
export OPENPI_DATA_HOME=/workspace/openpi_assets
echo 'export OPENPI_DATA_HOME=/workspace/openpi_assets' >> ~/.bashrc

echo ""
echo "=== setup_pod.sh complete ==="
echo "Next: run 'bash /workspace/openpi/runpod/start_libero.sh'"
echo "If you need AI agents: run 'bash /workspace/openpi/runpod/setup_agents.sh' first"
