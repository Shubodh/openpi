#!/bin/bash
# run_patching_phase1.sh — sanity check + patched run for Phase 1 KV-cache patching
#
# Prerequisite: source libero_env.sh (sets MUJOCO_GL=egl and PYTHONPATH; venv activation is
# overridden by uv run which uses the server venv with JAX + LIBERO simulation deps).
# *** NO POLICY SERVER NEEDED — model is loaded in-process. ***
# Do NOT start start_libero.sh before this script; the server is irrelevant here.
#
# What this does (in order):
#   [C3] Sanity check  — corrupt prompt + donor patch at ALL 788 positions, N=5.
#        Expected: success rate should approach the clean baseline (~70%+).
#        If sanity check fails (still ~0%), the patch mechanism is broken — stop here.
#
#   [D3] Patched run   — corrupt prompt + donor patch at position 594 only, N=25.
#        This is the main experimental condition.
#
# C3 and D3 map to checklist items in status_cc/patching_implementation.md §5.
# C1 (kv_cache shape) is auto-verified: look for "Donor KV cache harvested. K shape:"
#   in the log — expected: (18, 1, 788, 1, 256).
#
# Output: tee'd to stdout AND scripts_outputs_txt/patching_phase1/run_YYYYMMDD_HHMMSS.txt
# Videos: data/libero/videos/patched_posall_kv/  (sanity)
#         data/libero/videos/patched_pos594_kv/  (main run)

set -eo pipefail
OPENPI_DIR="/workspace/openpi"
cd "$OPENPI_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="scripts_outputs_txt/patching_phase1/patched"
OUT_FILE="$OUT_DIR/run_${TIMESTAMP}.txt"
mkdir -p "$OUT_DIR"

{
echo "=== patching_phase1 run: $TIMESTAMP ==="
echo "=== Task: put_the_bowl_on_the_plate (LIBERO-Goal) ==="
echo "=== Model loaded IN-PROCESS — no policy server needed ==="
echo ""

echo "=== [C3] Sanity check: patch ALL 788 positions, N=5 ==="
echo "=== Expected: success rate close to clean baseline (corrupt prompt, but full donor KV) ==="
printf "n\n" | uv run python examples/libero/main_patching_expt.py \
  --args.task-suite-name libero_goal \
  --args.task-name-filter "put the bowl on the plate" \
  --args.clean-prompt "put the bowl on the plate" \
  --args.corrupt-prompt "put the bowl on the stove" \
  --args.sanity-check \
  --args.num-trials-per-task 5 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/patching_phase1/patched

echo ""
echo "=== [C3] Sanity check complete. Check success rate above before continuing. ==="
echo "=== If near 0% — patch mechanism is broken. Stop and debug. ==="
echo "=== If near clean baseline — mechanism works. Proceeding to main run. ==="
echo ""

echo "=== [D3] Patched run: patch position 594 only (plate/stove token), N=25 ==="
printf "n\n" | uv run python examples/libero/main_patching_expt.py \
  --args.task-suite-name libero_goal \
  --args.task-name-filter "put the bowl on the plate" \
  --args.clean-prompt "put the bowl on the plate" \
  --args.corrupt-prompt "put the bowl on the stove" \
  --args.patch-positions "594" \
  --args.num-trials-per-task 25 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/patching_phase1/patched

echo ""
echo "=== All runs complete ==="
echo "=== Record results in status_cc/patching_implementation.md §7.1 ==="
echo "=== Key question: does patched success rate recover toward clean baseline? ==="
} 2>&1 | tee "$OUT_FILE"

echo ""
echo "=== Full log saved to: $OUT_FILE ==="
