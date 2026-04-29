#!/bin/bash
# setup_pod.sh — run after EVERY pod restart (~15-20 min)
# Checkpoint on /workspace survives pod stop.
# uv, Python 3.8, and system packages are wiped on pod stop — reinstalled here.
# Venv is on /workspace but its Python symlink breaks (Python wiped) — recreated every restart.
# AI agents (Claude Code, Codex) are NOT reinstalled here — run setup_agents.sh separately if needed.
set -e

echo "=== [0/4] Configuring git identity ==="
git config --global user.name "Shubodh RunPod April"
git config --global user.email "p.saishubodh@gmail.com"

echo "=== [1/4] Installing system packages ==="
apt-get update -q && apt-get install -y -q tmux vim libegl1-mesa cmake rsync  # libegl1-mesa for MUJOCO_GL=egl; cmake for egl-probe build

echo "=== [2/4] Re-installing uv (wiped on pod stop) ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

echo "=== [3/4] Restoring OPENPI_DATA_HOME ==="
export OPENPI_DATA_HOME=/workspace/openpi_assets
echo 'export OPENPI_DATA_HOME=/workspace/openpi_assets' >> ~/.bashrc

echo "=== [4/4] Ensuring LIBERO venv deps are installed ==="
cd /workspace/openpi
# Python 3.8 binary is on container disk — wiped on pod stop — so venv symlink breaks every restart.
# Recreate the venv (packages in site-packages persist on /workspace, but uv venv resets them).
uv venv --python 3.8 --clear examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip install -r examples/libero/requirements.txt -r third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
# Extra openpi deps not pulled in by the libero requirements (needed for analysis scripts):
uv pip install sentencepiece "fsspec[gcs]" filelock tqdm-loggable

echo ""
echo "=== setup_pod.sh complete ==="
echo "Next: run 'bash /workspace/openpi/runpod/start_libero.sh'"
echo "If you need AI agents: run 'bash /workspace/openpi/runpod/setup_agents.sh' first"
