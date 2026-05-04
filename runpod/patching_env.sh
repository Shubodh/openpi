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

echo ""
echo "Server venv active (Python $(python --version 2>&1 | cut -d' ' -f2), JAX $(python -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'NOT FOUND'))."
echo "Ready to run:"
echo "  bash /workspace/openpi/runpod/run_patching_phase1.sh"
