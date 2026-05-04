#!/bin/bash
# setup_once.sh — run ONCE per network volume (first pod ever)
# After this, only setup_pod.sh is needed on restarts.
set -e

echo "=== [0/5] Configuring git identity ==="
git config --global user.name "Shubodh RunPod April"
git config --global user.email "p.saishubodh@gmail.com"

echo "=== [1/5] Installing system packages ==="
apt-get update -q && apt-get install -y -q tmux vim libegl1-mesa cmake rsync # libegl1-mesa for MUJOCO_GL=egl; cmake for egl-probe build

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
# Extra openpi deps not pulled in by the libero requirements (needed for analysis scripts):
uv pip install sentencepiece "fsspec[gcs]" filelock tqdm-loggable

echo "=== Installing LIBERO simulation deps into server venv (for main_patching_expt.py) ==="
# uv sync creates the server venv (.venv) from pyproject.toml; activate it explicitly.
# LIBERO client venv must already be set up (step 5/5 above) before this runs,
# because we copy robosuite from it rather than pip-installing (see note below).
deactivate 2>/dev/null || true
uv sync
# Use 'uv pip install' — uv-created venvs have no pip binary in bin/.
# cd to /workspace/openpi already done above; uv targets the project venv (.venv).
# Install non-robosuite deps
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

echo ""
echo "=== setup_once.sh complete ==="
echo "Next: run 'bash /workspace/openpi/runpod/setup_agents.sh' to install Claude Code + Codex."
echo "Then: 'bash /workspace/openpi/runpod/start_libero.sh' to launch server + client."
echo "(On future restarts, run setup_pod.sh first, then start_libero.sh)"
