#!/bin/bash
# setup_pod.sh — run after EVERY pod restart (~2-3 min after persistent venv exists)
# Checkpoint on /workspace survives pod stop.
# uv and system packages are wiped on pod stop — reinstalled here.
# Python interpreters, uv cache, venvs, checkpoints, and repo live on /workspace.
# Codex CLI is NOT reinstalled here — run setup_agents.sh separately if needed.
set -e

OPENPI_DIR=/workspace/openpi
# shellcheck source=/dev/null
source "$OPENPI_DIR/runpod/setup_common.sh"

echo "=== [1/6] Installing system packages ==="
install_system_packages

echo "=== [1b/6] Verifying openpi checkout + submodules ==="
ensure_openpi_checkout

echo "=== [2/6] Restoring tmux config + plugins (wiped on pod stop) ==="
restore_tmux_config

echo "=== [3/6] Re-installing uv (wiped on pod stop) ==="
install_uv

echo "=== [4/6] Restoring persistent cache/env paths ==="
configure_persistent_env

echo "=== [5/6] Ensuring LIBERO venv deps are installed ==="
cd "$OPENPI_DIR"

echo "=== [5a/6] Configuring LIBERO paths non-interactively ==="
write_libero_config

echo "=== [5b/6] Configuring git identity ==="
configure_git_identity

echo "=== [5c/6] Ensuring persistent Python 3.8 and LIBERO venv ==="
ensure_libero_client_venv

echo "=== [6/6] Installing LIBERO simulation deps into server venv (for main_patching_expt.py) ==="
ensure_server_patching_venv

echo ""
echo "=== setup_pod.sh complete ==="
echo "Next: run 'bash /workspace/openpi/runpod/start_libero.sh'"
echo "If you need AI agents: run 'bash /workspace/openpi/runpod/setup_agents.sh' first"
