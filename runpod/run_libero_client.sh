#!/bin/bash
# run_libero_client.sh — run a LIBERO client against an already-running policy server
# Use this to add extra suite runs in parallel, or re-run after start_libero.sh is already up.
#
# Usage:
#   bash /workspace/openpi/runpod/run_libero_client.sh [suite] [num_trials] [seed] [video_out_path]
#
# Arguments (all optional, positional):
#   suite          Task suite name (default: libero_object)
#                  Options: libero_spatial | libero_object | libero_goal | libero_10 | libero_90
#   num_trials     Rollouts per task (default: 50 — matches published protocol)
#   seed           Random seed (default: 7)
#   video_out_path Where to save episode videos (default: data/libero/videos)
#
# Examples:
#   bash /workspace/openpi/runpod/run_libero_client.sh                            # libero_object, 50 trials
#   bash /workspace/openpi/runpod/run_libero_client.sh libero_spatial             # spatial, 50 trials
#   bash /workspace/openpi/runpod/run_libero_client.sh libero_object 10           # object, 10 trials (quick check)
#   bash /workspace/openpi/runpod/run_libero_client.sh libero_object 50 42        # object, 50 trials, seed 42

SUITE=${1:-libero_object}
NUM_TRIALS=${2:-50}
SEED=${3:-7}
VIDEO_OUT=${4:-data/libero/videos/$SUITE}
OPENPI_DIR="/workspace/openpi"

cd $OPENPI_DIR
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=egl
export OPENPI_DATA_HOME=/workspace/openpi_assets
export LIBERO_CONFIG_PATH=$OPENPI_DIR/.libero_config
export UV_CACHE_DIR=/workspace/uv_cache
export UV_PYTHON_INSTALL_DIR=/workspace/python
mkdir -p "$OPENPI_DATA_HOME" "$LIBERO_CONFIG_PATH" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
  cat > "$LIBERO_CONFIG_PATH/config.yaml" <<'EOF'
assets: /workspace/openpi/third_party/libero/libero/libero/assets
bddl_files: /workspace/openpi/third_party/libero/libero/libero/bddl_files
benchmark_root: /workspace/openpi/third_party/libero/libero/libero
datasets: /workspace/openpi/third_party/libero/libero/datasets
init_states: /workspace/openpi/third_party/libero/libero/libero/init_files
EOF
fi

echo "Suite: $SUITE | Trials/task: $NUM_TRIALS | Seed: $SEED | Videos: $OPENPI_DIR/$VIDEO_OUT"

python examples/libero/main_original.py \
  --args.task-suite-name "$SUITE" \
  --args.num-trials-per-task "$NUM_TRIALS" \
  --args.seed "$SEED" \
  --args.video-out-path "$VIDEO_OUT"
