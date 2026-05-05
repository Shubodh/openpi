#!/bin/bash
# setup_once.sh — run ONCE per network volume (first pod ever)
# After this, only setup_pod.sh is needed on restarts.
set -e

echo "=== [0/5] Configuring git identity ==="
git config --global user.name "Shubodh RunPod April"
git config --global user.email "p.saishubodh@gmail.com"

echo "=== [1/5] Installing system packages ==="
apt-get update -q && apt-get install -y -q tmux vim libegl1-mesa cmake rsync xclip git # libegl1-mesa for MUJOCO_GL=egl; cmake for egl-probe build; xclip for tmux clipboard

echo "=== [2/5] Cloning openpi ==="
cd /workspace
git clone https://github.com/Shubodh/openpi.git
cd openpi
git submodule update --init --recursive

echo "=== [2b/5] Setting up tmux config + plugins ==="
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm 2>/dev/null || true
cp /workspace/openpi/runpod/tmux.conf ~/.tmux.conf
TMUX= ~/.tmux/plugins/tpm/bin/install_plugins || true

echo "=== [3/5] Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

echo "=== [4/5] Setting persistent cache/env paths on the network volume ==="
export OPENPI_DATA_HOME=/workspace/openpi_assets
export UV_CACHE_DIR=/workspace/uv_cache
export UV_PYTHON_INSTALL_DIR=/workspace/python
mkdir -p "$OPENPI_DATA_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
grep -qxF 'export OPENPI_DATA_HOME=/workspace/openpi_assets' ~/.bashrc || \
  echo 'export OPENPI_DATA_HOME=/workspace/openpi_assets' >> ~/.bashrc
grep -qxF 'export UV_CACHE_DIR=/workspace/uv_cache' ~/.bashrc || \
  echo 'export UV_CACHE_DIR=/workspace/uv_cache' >> ~/.bashrc
grep -qxF 'export UV_PYTHON_INSTALL_DIR=/workspace/python' ~/.bashrc || \
  echo 'export UV_PYTHON_INSTALL_DIR=/workspace/python' >> ~/.bashrc

echo "=== [5/5] Setting up LIBERO client venv and installing deps ==="
cd /workspace/openpi
OPENPI_DIR=/workspace/openpi
LIBERO_VENV="$OPENPI_DIR/examples/libero/.venv"
LIBERO_PYTHON="$LIBERO_VENV/bin/python"
SERVER_VENV="$OPENPI_DIR/.venv"
SERVER_PYTHON="$SERVER_VENV/bin/python"

echo "=== [5a/5] Configuring LIBERO paths non-interactively ==="
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

echo "=== [5b/5] Installing persistent Python 3.8 and creating LIBERO venv ==="
uv python install 3.8 --install-dir "$UV_PYTHON_INSTALL_DIR"
PYTHON38=$(uv python find 3.8 --managed-python --no-project)
uv venv --python "$PYTHON38" "$LIBERO_VENV"
uv pip install --python "$LIBERO_PYTHON" -r examples/libero/requirements.txt -r third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install --python "$LIBERO_PYTHON" -e packages/openpi-client
uv pip install --python "$LIBERO_PYTHON" -e third_party/libero
# Extra openpi deps not pulled in by the libero requirements (needed for analysis scripts):
uv pip install --python "$LIBERO_PYTHON" sentencepiece "fsspec[gcs]" filelock tqdm-loggable

echo "=== Installing LIBERO simulation deps into server venv (for main_patching_expt.py) ==="
# uv sync creates/repairs the server venv (.venv) from pyproject.toml.
# LIBERO client venv must already be set up (step 5/5 above) before this runs,
# because we copy robosuite from it rather than pip-installing (see note below).
env -u VIRTUAL_ENV uv sync
# Use 'uv pip install' — uv-created venvs have no pip binary in bin/.
# Install with --python so an inherited active venv cannot capture these server deps.
# Install non-robosuite deps
uv pip install --python "$SERVER_PYTHON" \
  "mujoco>=3.2" imageio imageio-ffmpeg numpy "opencv-python>=4.6" scipy tqdm pyyaml \
  pyopengl etils tyro
uv pip install --python "$SERVER_PYTHON" -e /workspace/openpi/packages/openpi-client
# Install LIBERO editable + its requirements.txt (setup.py alone misses bddl, easydict, gym, etc.)
# requirements.txt lines have leading spaces — use ^\s* not ^.
# Exclude: robosuite (copied below), training-only packages, and numpy (restored after).
uv pip install --python "$SERVER_PYTHON" -e /workspace/openpi/third_party/libero
grep -viE "^\s*(robosuite|torch|wandb|transformers|thop|robomimic|numpy)" \
  /workspace/openpi/third_party/libero/requirements.txt | uv pip install --python "$SERVER_PYTHON" -r /dev/stdin
# Restore numpy: LIBERO requirements downgrade to 1.22.4 which breaks JAX (needs np.dtypes, >=1.25).
uv pip install --python "$SERVER_PYTHON" "numpy>=1.22.4,<2.0.0"
# Copy robosuite from LIBERO client venv — pip resolves to incompatible 1.5.x even with ==1.4.1 pin.
# Must rm -rf first: cp -r into existing dir nests instead of replacing.
SERVER_SITE=$("$SERVER_PYTHON" -c "import site; print(site.getsitepackages()[0])")
rm -rf "${SERVER_SITE}/robosuite"
cp -r "$LIBERO_VENV/lib/python3.8/site-packages/robosuite" \
      "${SERVER_SITE}/robosuite"

echo "=== Verifying server venv patching imports ==="
PYTHONPATH="/workspace/openpi/third_party/libero:${PYTHONPATH:-}" \
  "$SERVER_PYTHON" - <<'PY'
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
echo "=== setup_once.sh complete ==="
echo "Next: run 'bash /workspace/openpi/runpod/setup_agents.sh' to install Codex CLI."
echo "Then: 'bash /workspace/openpi/runpod/start_libero.sh' to launch server + client."
echo "(On future restarts, run setup_pod.sh first, then start_libero.sh)"
