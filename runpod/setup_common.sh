#!/bin/bash
# Shared RunPod setup helpers. Source this file from setup_once.sh/setup_pod.sh.

OPENPI_DIR=${OPENPI_DIR:-/workspace/openpi}
OPENPI_REPO_URL=${OPENPI_REPO_URL:-https://github.com/Shubodh/openpi.git}
OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/workspace/openpi_assets}
UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/uv_cache}
UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-/workspace/python}
LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$OPENPI_DIR/.libero_config}
LIBERO_VENV=${LIBERO_VENV:-$OPENPI_DIR/examples/libero/.venv}
LIBERO_PYTHON=${LIBERO_PYTHON:-$LIBERO_VENV/bin/python}
SERVER_VENV=${SERVER_VENV:-$OPENPI_DIR/.venv}
SERVER_PYTHON=${SERVER_PYTHON:-$SERVER_VENV/bin/python}

export OPENPI_DATA_HOME UV_CACHE_DIR UV_PYTHON_INSTALL_DIR LIBERO_CONFIG_PATH

log_step() {
  echo ""
  echo "=== $* ==="
}

append_bashrc_once() {
  local line=$1
  grep -qxF "$line" ~/.bashrc 2>/dev/null || echo "$line" >> ~/.bashrc
}

install_system_packages() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -y -q \
    ca-certificates \
    cmake \
    curl \
    git \
    libegl1-mesa \
    libglu1-mesa \
    rsync \
    tmux \
    vim \
    xclip
}

ensure_openpi_checkout() {
  mkdir -p "$(dirname "$OPENPI_DIR")"
  if [ -d "$OPENPI_DIR/.git" ]; then
    cd "$OPENPI_DIR"
  elif [ -e "$OPENPI_DIR" ]; then
    echo "ERROR: $OPENPI_DIR exists but is not a git checkout." >&2
    return 1
  else
    cd "$(dirname "$OPENPI_DIR")"
    git clone "$OPENPI_REPO_URL" "$(basename "$OPENPI_DIR")"
    cd "$OPENPI_DIR"
  fi

  git submodule update --init --recursive
}

restore_tmux_config() {
  mkdir -p ~/.tmux/plugins
  git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm 2>/dev/null || true
  cp "$OPENPI_DIR/runpod/tmux.conf" ~/.tmux.conf
  TMUX= ~/.tmux/plugins/tpm/bin/install_plugins || true
}

install_uv() {
  curl -LsSf https://astral.sh/uv/install.sh | sh
  if [ -f "$HOME/.local/bin/env" ]; then
    # shellcheck source=/dev/null
    source "$HOME/.local/bin/env"
  fi
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null
}

configure_persistent_env() {
  mkdir -p "$OPENPI_DATA_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
  append_bashrc_once "export OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
  append_bashrc_once "export UV_CACHE_DIR=$UV_CACHE_DIR"
  append_bashrc_once "export UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR"
  append_bashrc_once "export LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH"
}

configure_git_identity() {
  git config --global user.name "Shubodh RunPod April"
  git config --global user.email "p.saishubodh@gmail.com"
}

write_libero_config() {
  mkdir -p "$LIBERO_CONFIG_PATH"
  cat > "$LIBERO_CONFIG_PATH/config.yaml" <<'EOF'
assets: /workspace/openpi/third_party/libero/libero/libero/assets
bddl_files: /workspace/openpi/third_party/libero/libero/libero/bddl_files
benchmark_root: /workspace/openpi/third_party/libero/libero/libero
datasets: /workspace/openpi/third_party/libero/libero/datasets
init_states: /workspace/openpi/third_party/libero/libero/libero/init_files
EOF
}

libero_venv_uses_persistent_python() {
  [ -x "$LIBERO_PYTHON" ] || return 1
  local base_python
  base_python=$("$LIBERO_PYTHON" -c 'import os, sys; print(os.path.realpath(getattr(sys, "_base_executable", sys.executable)))' 2>/dev/null || true)
  [[ "$base_python" == "$UV_PYTHON_INSTALL_DIR"/* ]]
}

libero_client_deps_ok() {
  [ -x "$LIBERO_PYTHON" ] || return 1
  PYTHONPATH="$OPENPI_DIR/third_party/libero:${PYTHONPATH:-}" \
  LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
  timeout 90 "$LIBERO_PYTHON" - <<'PY' >/dev/null
import importlib.metadata as metadata

import libero
import libero.libero.envs
import mujoco
import openpi_client
import robosuite
import sentencepiece

for package in ("torch", "fsspec", "filelock", "tqdm-loggable"):
    metadata.version(package)
PY
}

install_libero_client_deps() {
  uv pip install --python "$LIBERO_PYTHON" \
    -r examples/libero/requirements.txt \
    -r third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match
  uv pip install --python "$LIBERO_PYTHON" -e packages/openpi-client
  uv pip install --python "$LIBERO_PYTHON" -e third_party/libero
  uv pip install --python "$LIBERO_PYTHON" sentencepiece "fsspec[gcs]" filelock tqdm-loggable
}

ensure_libero_client_venv() {
  cd "$OPENPI_DIR"
  uv python install 3.8 --install-dir "$UV_PYTHON_INSTALL_DIR"
  local python38
  python38=$(uv python find 3.8 --managed-python --no-project)

  if libero_venv_uses_persistent_python; then
    echo "Reusing LIBERO venv backed by persistent Python."
  else
    echo "Creating LIBERO venv with persistent Python: $python38"
    uv venv --python "$python38" --clear "$LIBERO_VENV"
  fi

  if libero_client_deps_ok; then
    echo "LIBERO client deps already verified; skipping reinstall."
  else
    echo "Installing/repairing LIBERO client deps."
    install_libero_client_deps
    libero_client_deps_ok
  fi
}

server_patching_deps_ok() {
  [ -x "$SERVER_PYTHON" ] || return 1
  [ -d "$LIBERO_VENV/lib/python3.8/site-packages/robosuite" ] || return 1
  PYTHONPATH="$OPENPI_DIR/third_party/libero:${PYTHONPATH:-}" \
  LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
  timeout 90 "$SERVER_PYTHON" - <<'PY'
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
}

install_server_patching_deps() {
  uv pip install --python "$SERVER_PYTHON" \
    "mujoco>=3.2" imageio imageio-ffmpeg numpy "opencv-python>=4.6" scipy tqdm pyyaml \
    pyopengl etils tyro
  uv pip install --python "$SERVER_PYTHON" -e "$OPENPI_DIR/packages/openpi-client"
  uv pip install --python "$SERVER_PYTHON" -e "$OPENPI_DIR/third_party/libero"
  grep -viE "^\s*(robosuite|torch|wandb|transformers|thop|robomimic|numpy|matplotlib)" \
    "$OPENPI_DIR/third_party/libero/requirements.txt" | uv pip install --python "$SERVER_PYTHON" -r /dev/stdin
  uv pip install --python "$SERVER_PYTHON" "numpy>=1.25,<2.0.0"

  local server_site
  server_site=$("$SERVER_PYTHON" -c "import site; print(site.getsitepackages()[0])")
  rm -rf "${server_site}/robosuite"
  cp -r "$LIBERO_VENV/lib/python3.8/site-packages/robosuite" "${server_site}/robosuite"
}

ensure_server_patching_venv() {
  cd "$OPENPI_DIR"
  if server_patching_deps_ok; then
    return 0
  fi

  # Repairs the uv-managed server venv after pod restart. Clear VIRTUAL_ENV so an
  # active client shell cannot redirect packages into examples/libero/.venv.
  env -u VIRTUAL_ENV uv sync --inexact
  if server_patching_deps_ok; then
    return 0
  fi

  echo "Installing/repairing server venv LIBERO simulation deps."
  install_server_patching_deps
  server_patching_deps_ok
}
