#!/bin/bash
# libero_env.sh — activate venv + set env vars for LIBERO client
# Source this (don't run it) so exports stick in your current shell:
#   source /workspace/openpi/runpod/libero_env.sh
#
# Run once server says "listening on :8000", then paste the python command below.

SUITE=${1:-libero_object}
OPENPI_DIR="/workspace/openpi"

cd "$OPENPI_DIR"
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=egl

echo ""
echo "Env ready. Run:"
echo "  python examples/libero/main.py --args.task-suite-name $SUITE --args.video-out-path data/libero/videos/$SUITE"
echo ""
echo "Different suite: source libero_env.sh libero_spatial"
