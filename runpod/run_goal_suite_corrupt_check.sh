#!/bin/bash
# run_goal_suite_corrupt_check.sh — run the LIBERO-goal prompt-ablation experiment
#
# Prerequisite: policy server must already be running (start_libero.sh),
# and venv must be active (source libero_env.sh).
#
# Runs two conditions on the bowl task from libero_goal (25 trials each):
#   Clean run   — correct prompt on original task
#   Corrupt run — wine bottle prompt on the same original task/scene
#
# Output: tee'd to stdout AND scripts_outputs_txt/goal_suite_check_YYYYMMDD_HHMMSS.txt
# Videos saved to data/libero/videos/goal_suite_check_{clean,corrupt}/

set -eo pipefail
OPENPI_DIR="/workspace/openpi"
cd "$OPENPI_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="scripts_outputs_txt"
OUT_FILE="$OUT_DIR/goal_suite_check_${TIMESTAMP}.txt"
mkdir -p "$OUT_DIR"

{
echo "=== goal_suite_check run: $TIMESTAMP ==="
echo "=== [1/2] Clean run: bowl prompt on bowl task ==="
# "n" answers LIBERO's dataset path prompt on first run after pod restart
printf "n\n" | python examples/libero/main_corrupt_run_expt.py \
  --args.task-suite-name libero_goal \
  --args.task-name-filter "put the bowl on top of the cabinet" \
  --args.num-trials-per-task 25 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/goal_suite_check_clean

echo ""
echo "=== [2/2] Corrupt run: wine bottle prompt on bowl task ==="
printf "n\n" | python examples/libero/main_corrupt_run_expt.py \
  --args.task-suite-name libero_goal \
  --args.task-name-filter "put the bowl on top of the cabinet" \
  --args.corrupt-prompt "put the wine bottle on top of the cabinet" \
  --args.num-trials-per-task 25 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/goal_suite_check_corrupt

echo ""
echo "=== Both conditions complete. Record results in status_cc/corrupt_run_experiment.md ==="
} 2>&1 | tee "$OUT_FILE"

echo ""
echo "=== Full log saved to: $OUT_FILE ==="
