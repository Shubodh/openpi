#!/bin/bash
# start_libero.sh — launch π₀.₅ server + LIBERO client in a tmux session
# Run setup_pod.sh first on a fresh pod restart.
#
# Usage:
#   bash /workspace/openpi/runpod/start_libero.sh
#   bash /workspace/openpi/runpod/start_libero.sh libero_spatial   # run a specific suite
#
# To attach later: tmux attach -t libero
# Pane 0 (left): policy server   Pane 1 (right): LIBERO client

SUITE=${1:-libero_object}
SESSION="libero"
OPENPI_DIR="/workspace/openpi"

# Kill any existing session
tmux kill-session -t $SESSION 2>/dev/null || true

# Pane 0: policy server
tmux new-session -d -s $SESSION
tmux send-keys -t $SESSION:0 \
  "cd $OPENPI_DIR && export OPENPI_DATA_HOME=/workspace/openpi_assets && uv run scripts/serve_policy.py --env LIBERO" \
  Enter

# Pane 1: waits for server, activates venv, sets env vars, then prints the run command for manual execution
tmux split-window -h -t $SESSION
tmux send-keys -t $SESSION:0.1 \
  "cd $OPENPI_DIR \
  && echo 'Waiting for policy server on :8000 (checkpoint download may take a while)...' \
  && until nc -z localhost 8000 2>/dev/null; do sleep 3; done \
  && echo 'Server ready.' \
  && source examples/libero/.venv/bin/activate \
  && export PYTHONPATH=\$PYTHONPATH:\$PWD/third_party/libero \
  && export MUJOCO_GL=egl \
  && echo '' \
  && echo 'Env ready. Run when ready:' \
  && echo '  python examples/libero/main.py --args.task-suite-name $SUITE --args.video-out-path data/libero/videos/$SUITE'" \
  Enter

tmux attach -t $SESSION
