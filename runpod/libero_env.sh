#!/bin/bash
# libero_env.sh — activate venv + set env vars for LIBERO client
# Source this (don't run it) so exports stick in your current shell:
#   source /workspace/openpi/runpod/libero_env.sh
#
# Run once server says "listening on :8000", then paste the python command below.

OPENPI_DIR="/workspace/openpi"

cd "$OPENPI_DIR"
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=egl
export OPENPI_DATA_HOME=/workspace/openpi_assets
export LIBERO_CONFIG_PATH=$OPENPI_DIR/.libero_config
export UV_CACHE_DIR=/workspace/uv_cache
export UV_PYTHON_INSTALL_DIR=/workspace/python

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
echo "Env ready. Run:"
echo "  python examples/libero/main_original.py --args.task-suite-name libero_object --args.video-out-path data/libero/videos/libero_object"
echo "  python examples/libero/main_original.py --args.task-suite-name libero_spatial --args.video-out-path data/libero/videos/libero_spatial"
