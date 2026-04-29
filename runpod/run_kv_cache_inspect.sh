#!/bin/bash
# run_kv_cache_inspect.sh — run the KV-cache tokenizer inspection script on RunPod
#
# Prerequisite: source libero_env.sh (or any venv with openpi installed) before running.
# No model weights are loaded — tokenizer only (~800 KB download on first run).
#
# What this does:
#   Tokenizes the two contrastive LIBERO-Goal prompts with the PaliGemma tokenizer,
#   prints token IDs + piece strings with object-name tokens marked, and prints the
#   absolute prefix indices for 'bowl', 'wine', 'bottle' in the KV cache.
#
# Output: tee'd to stdout AND scripts_outputs_txt/kv_cache_inspect/inspect_YYYYMMDD_HHMMSS.txt

set -eo pipefail
OPENPI_DIR="/workspace/openpi"
cd "$OPENPI_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="scripts_outputs_txt/kv_cache_inspect"
OUT_FILE="$OUT_DIR/inspect_${TIMESTAMP}.txt"
mkdir -p "$OUT_DIR"

{
echo "=== kv_cache_inspect run: $TIMESTAMP ==="
echo "=== Script: tmp_kv_cache_sanity_check/inspect_kv_cache.py ==="
echo ""
python tmp_kv_cache_sanity_check/inspect_kv_cache.py
echo ""
echo "=== Done. Paste output into status_cc/kv_cache_findings.md Section 3. ==="
} 2>&1 | tee "$OUT_FILE"

echo ""
echo "=== Full log saved to: $OUT_FILE ==="
