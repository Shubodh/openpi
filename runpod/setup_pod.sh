#!/bin/bash
# setup_pod.sh — run after EVERY pod restart (fast, ~1-2 min)
# Venv + checkpoint on /workspace survive pod stop — no reinstall needed.
# uv, Claude Code, Codex, and system packages are wiped on pod stop — reinstalled here.
set -e

echo "=== [1/4] Installing system packages ==="
apt-get update -q && apt-get install -y -q tmux vim libegl1-mesa nodejs

echo "=== [2/4] Re-installing uv (wiped on pod stop) ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

echo "=== [3/4] Restoring OPENPI_DATA_HOME ==="
export OPENPI_DATA_HOME=/workspace/openpi_assets
echo 'export OPENPI_DATA_HOME=/workspace/openpi_assets' >> ~/.bashrc

echo "=== [4/4] Re-installing AI agents (wiped on pod stop) ==="
curl -fsSL https://claude.ai/install.sh | bash
source "$HOME/.local/bin/env" 2>/dev/null || true
npm install -g @openai/codex 2>/dev/null || echo "  (Codex install failed — check Node.js)"

echo ""
echo "=== setup_pod.sh complete ==="
echo "Next: run 'bash /workspace/openpi/runpod/start_libero.sh'"
