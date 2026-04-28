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

echo "Suite: $SUITE | Trials/task: $NUM_TRIALS | Seed: $SEED | Videos: $OPENPI_DIR/$VIDEO_OUT"

python examples/libero/main_original.py \
  --args.task-suite-name "$SUITE" \
  --args.num-trials-per-task "$NUM_TRIALS" \
  --args.seed "$SEED" \
  --args.video-out-path "$VIDEO_OUT"
