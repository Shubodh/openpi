#!/bin/bash
# setup_pod.sh — run after EVERY pod restart (~15-20 min)
# Checkpoint on /workspace survives pod stop.
# uv, Python 3.8, and system packages are wiped on pod stop — reinstalled here.
# Venv is on /workspace but its Python symlink breaks (Python wiped) — recreated every restart.
# AI agents (Claude Code, Codex) are NOT reinstalled here — run setup_agents.sh separately if needed.
set -e


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

echo ""
echo "=== setup_pod.sh complete ==="
echo "Next: run 'bash /workspace/openpi/runpod/start_libero.sh'"
echo "If you need AI agents: run 'bash /workspace/openpi/runpod/setup_agents.sh' first"
