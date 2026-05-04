#!/bin/bash
# run_patching_phase1_baselines.sh — clean + corrupt baselines for Phase 1 KV-cache patching
#
# Prerequisite: policy server must already be running (start_libero.sh pane 0),
# and venv must be active (source libero_env.sh).
#
# Task pair (Phase 1): put_the_bowl_on_the_plate task, LIBERO-Goal suite.
#   Clean run   — correct prompt ("put the bowl on the plate"), N=25
#   Corrupt run — wrong destination ("put the bowl on the stove"),  N=25
#
# These are the D1 and D2 baselines from patching_implementation.md §5.
# Run these before the patched condition to establish the clean ceiling and corrupt floor.
#
# NOTE: these use the websocket server (same as other corrupt-check scripts).
#       The patched run (run_patching_phase1.sh) does NOT need the server.
#
# Output: tee'd to stdout AND scripts_outputs_txt/patching_phase1/baselines_YYYYMMDD_HHMMSS.txt
# Videos: data/libero/videos/patching_phase1_clean/ and patching_phase1_corrupt/

set -eo pipefail
OPENPI_DIR="/workspace/openpi"
cd "$OPENPI_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="scripts_outputs_txt/patching_phase1/baselines"
OUT_FILE="$OUT_DIR/baselines_${TIMESTAMP}.txt"
mkdir -p "$OUT_DIR"

{
echo "=== patching_phase1_baselines run: $TIMESTAMP ==="
echo "=== Task: put_the_bowl_on_the_plate (LIBERO-Goal) ==="
echo ""

echo "=== [D1] Clean run: 'put the bowl on the plate' prompt, N=25 ==="
printf "n\n" | python examples/libero/main_corrupt_run_expt.py \
  --args.task-suite-name libero_goal \
  --args.task-name-filter "put_the_bowl_on_the_plate" \
  --args.num-trials-per-task 25 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/patching_phase1/baselines/clean

echo ""
echo "=== [D2] Corrupt run: 'put the bowl on the stove' prompt, N=25 ==="
printf "n\n" | python examples/libero/main_corrupt_run_expt.py \
  --args.task-suite-name libero_goal \
  --args.task-name-filter "put_the_bowl_on_the_plate" \
  --args.corrupt-prompt "put the bowl on the stove" \
  --args.num-trials-per-task 25 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/patching_phase1/baselines/corrupt

echo ""
echo "=== Both baselines complete ==="
echo "=== Record clean and corrupt success rates in status_cc/patching_implementation.md §7.1 ==="
} 2>&1 | tee "$OUT_FILE"

echo ""
echo "=== Full log saved to: $OUT_FILE ==="
