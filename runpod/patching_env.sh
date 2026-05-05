#!/bin/bash
# patching_env.sh — activate SERVER venv + set env vars for in-process patching script
# Source this (don't run it) so exports stick in your current shell:
#   source /workspace/openpi/runpod/patching_env.sh
#
# Use this instead of libero_env.sh before running run_patching_phase1.sh.
# The patching script loads the JAX model in-process and must run in the server venv
# (Python 3.11 + JAX), not the LIBERO client venv (Python 3.8, no JAX).

OPENPI_DIR="/workspace/openpi"

cd "$OPENPI_DIR"
source "$OPENPI_DIR/.venv/bin/activate"
export PYTHONPATH=$PYTHONPATH:$OPENPI_DIR/third_party/libero
export MUJOCO_GL=egl
export OPENPI_DATA_HOME=/workspace/openpi_assets
export LIBERO_CONFIG_PATH=$OPENPI_DIR/.libero_config
export UV_CACHE_DIR=/workspace/uv_cache
export UV_PYTHON_INSTALL_DIR=/workspace/python

# Make sourcing this file robust after a pod restart or a fresh checkout. LIBERO's
# package import prompts for a dataset path if config.yaml is missing, which breaks
# non-interactive patching runs.
mkdir -p "$LIBERO_CONFIG_PATH" "$OPENPI_DATA_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
  cat > "$LIBERO_CONFIG_PATH/config.yaml" <<'EOF'
assets: /workspace/openpi/third_party/libero/libero/libero/assets
bddl_files: /workspace/openpi/third_party/libero/libero/libero/bddl_files
benchmark_root: /workspace/openpi/third_party/libero/libero/libero
datasets: /workspace/openpi/third_party/libero/libero/datasets
init_states: /workspace/openpi/third_party/libero/libero/libero/init_files
EOF
fi

echo ""
echo "Server venv active (Python $(python --version 2>&1 | cut -d' ' -f2), JAX $(python -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'NOT FOUND'))."
echo "Ready to run:"
echo "  bash /workspace/openpi/runpod/run_patching_phase1.sh"
