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
# Output: tee'd to stdout AND scripts_outputs_txt/corrupt_check_YYYYMMDD_HHMMSS.txt
# Videos saved to data/libero/videos/corrupt_check_{clean,corrupt}/

set -eo pipefail
OPENPI_DIR="/workspace/openpi"
cd "$OPENPI_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="scripts_outputs_txt"
OUT_FILE="$OUT_DIR/corrupt_check_${TIMESTAMP}.txt"
mkdir -p "$OUT_DIR"

{
echo "=== corrupt_check run: $TIMESTAMP ==="
echo "=== [1/2] Clean run: correct prompt on milk task ==="
# "n" answers LIBERO's dataset path prompt on first run after pod restart
printf "n\n" | python examples/libero/main_corrupt_run_expt.py \
  --args.task-suite-name libero_object \
  --args.task-name-filter "milk" \
  --args.num-trials-per-task 10 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/corrupt_check_clean

echo ""
echo "=== [2/2] Corrupt run: tomato sauce prompt on milk task ==="
printf "n\n" | python examples/libero/main_corrupt_run_expt.py \
  --args.task-suite-name libero_object \
  --args.task-name-filter "milk" \
  --args.corrupt-prompt "Pick up the tomato sauce and place it in the basket" \
  --args.num-trials-per-task 10 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/corrupt_check_corrupt

echo ""
echo "=== Both conditions complete. Record results in status_cc/corrupt_run_experiment.md ==="
} 2>&1 | tee "$OUT_FILE"

echo ""
echo "=== Full log saved to: $OUT_FILE ==="
