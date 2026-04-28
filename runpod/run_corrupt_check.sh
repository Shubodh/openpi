#!/bin/bash
# run_corrupt_check.sh — run the prompt-ablation (corrupt-run) experiment
#
# Prerequisite: policy server must already be running (start_libero.sh),
# and venv must be active (source libero_env.sh).
#
# Runs two conditions on the milk task from libero_object (25 trials each):
#   Clean run  — correct prompt, baseline should be ~98%
#   Corrupt run — wrong object name, same scene/states
#
# Results are logged to stdout. Videos saved to data/libero/videos/corrupt_check_{clean,corrupt}/

set -e
OPENPI_DIR="/workspace/openpi"
cd "$OPENPI_DIR"

echo "=== [1/2] Clean run: correct prompt on milk task ==="
python examples/libero/main.py \
  --args.task-suite-name libero_object \
  --args.task-name-filter "milk" \
  --args.num-trials-per-task 25 \
  --args.video-out-path data/libero/videos/corrupt_check_clean

echo ""
echo "=== [2/2] Corrupt run: tomato sauce prompt on milk task ==="
python examples/libero/main.py \
  --args.task-suite-name libero_object \
  --args.task-name-filter "milk" \
  --args.corrupt-prompt "Pick up the tomato sauce and place it in the basket" \
  --args.num-trials-per-task 25 \
  --args.video-out-path data/libero/videos/corrupt_check_corrupt

echo ""
echo "=== Both conditions complete. Record results in status_cc/corrupt_run_experiment.md ==="
