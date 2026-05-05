#!/bin/bash
# setup_pod.sh — run after EVERY pod restart (~15-20 min)
# Checkpoint on /workspace survives pod stop.
# uv, Python 3.8, and system packages are wiped on pod stop — reinstalled here.
# Venv is on /workspace but its Python symlink breaks (Python wiped) — recreated every restart.
# AI agents (Claude Code, Codex) are NOT reinstalled here — run setup_agents.sh separately if needed.
set -e


echo "=== [1/4] Installing system packages ==="
apt-get update -q && apt-get install -y -q tmux vim libegl1-mesa cmake rsync xclip git # libegl1-mesa for MUJOCO_GL=egl; cmake for egl-probe build; xclip for tmux clipboard

echo "=== [1b/4] Restoring tmux config + plugins (wiped on pod stop) ==="
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm 2>/dev/null || true
cp /workspace/openpi/runpod/tmux.conf ~/.tmux.conf
TMUX= ~/.tmux/plugins/tpm/bin/install_plugins || true

echo "=== [2/4] Re-installing uv (wiped on pod stop) ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

echo "=== [3/4] Restoring OPENPI_DATA_HOME ==="
export OPENPI_DATA_HOME=/workspace/openpi_assets
grep -qxF 'export OPENPI_DATA_HOME=/workspace/openpi_assets' ~/.bashrc || \
  echo 'export OPENPI_DATA_HOME=/workspace/openpi_assets' >> ~/.bashrc

echo "=== [4/4] Ensuring LIBERO venv deps are installed ==="
cd /workspace/openpi

echo "=== [4a/4] Configuring LIBERO paths non-interactively ==="
# Keep LIBERO config on the persistent workspace volume. Without this file, importing
# libero can prompt on stdin for a dataset path and crash non-interactive experiment runs.
export LIBERO_CONFIG_PATH=/workspace/openpi/.libero_config
mkdir -p "$LIBERO_CONFIG_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<'EOF'
assets: /workspace/openpi/third_party/libero/libero/libero/assets
bddl_files: /workspace/openpi/third_party/libero/libero/libero/bddl_files
benchmark_root: /workspace/openpi/third_party/libero/libero/libero
datasets: /workspace/openpi/third_party/libero/libero/datasets
init_states: /workspace/openpi/third_party/libero/libero/libero/init_files
EOF
grep -qxF 'export LIBERO_CONFIG_PATH=/workspace/openpi/.libero_config' ~/.bashrc || \
  echo 'export LIBERO_CONFIG_PATH=/workspace/openpi/.libero_config' >> ~/.bashrc


echo "=== [0/4] Configuring git identity ==="
git config --global user.name "Shubodh RunPod April"
git config --global user.email "p.saishubodh@gmail.com"

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

echo "=== [5/4] Installing LIBERO simulation deps into server venv (for main_patching_expt.py) ==="
# main_patching_expt.py loads JAX in-process (server venv, Python 3.11) but also steps
# LIBERO environments. Install the simulation deps here so both work in one process.
# uv sync repairs the server venv Python symlink (broken on pod stop/restart).
uv sync
# Use 'uv pip install' — uv-created venvs have no pip binary in bin/.
# Run from /workspace/openpi so uv targets the project venv (.venv).
# Install non-robosuite deps (robosuite is copied below — see note)
uv pip install \
  "mujoco>=3.2" imageio imageio-ffmpeg numpy "opencv-python>=4.6" scipy tqdm pyyaml \
  pyopengl etils tyro
uv pip install -e /workspace/openpi/packages/openpi-client
# Install LIBERO editable + its requirements.txt (setup.py alone misses bddl, easydict, gym, etc.)
# requirements.txt lines have leading spaces — use ^\s* not ^.
# Exclude: robosuite (copied below), training-only packages, and numpy (restored after).
uv pip install -e /workspace/openpi/third_party/libero
grep -viE "^\s*(robosuite|torch|wandb|transformers|thop|robomimic|numpy)" \
  /workspace/openpi/third_party/libero/requirements.txt | uv pip install -r /dev/stdin
# Restore numpy: LIBERO requirements downgrade to 1.22.4 which breaks JAX (needs np.dtypes, >=1.25).
uv pip install "numpy>=1.22.4,<2.0.0"
# Copy robosuite from LIBERO client venv — pip resolves to incompatible 1.5.x even with ==1.4.1 pin.
# Must rm -rf first: cp -r into existing dir nests instead of replacing.
SERVER_SITE=$(/workspace/openpi/.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
rm -rf "${SERVER_SITE}/robosuite"
cp -r /workspace/openpi/examples/libero/.venv/lib/python3.8/site-packages/robosuite \
      "${SERVER_SITE}/robosuite"

echo "=== [6/4] Verifying server venv patching imports ==="
PYTHONPATH="/workspace/openpi/third_party/libero:${PYTHONPATH:-}" \
  /workspace/openpi/.venv/bin/python - <<'PY'
import bddl
import jax
import libero
import libero.libero.envs
import numpy as np
import robosuite

major, minor = map(int, np.__version__.split(".")[:2])
assert (major, minor) >= (1, 25), f"NumPy {np.__version__} is too old for JAX"
print(f"server patching imports OK: jax={jax.__version__}, numpy={np.__version__}")
PY

echo ""
echo "=== setup_pod.sh complete ==="
echo "Next: run 'bash /workspace/openpi/runpod/start_libero.sh'"
echo "If you need AI agents: run 'bash /workspace/openpi/runpod/setup_agents.sh' first"
