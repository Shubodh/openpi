#!/bin/bash
# run_patching_phase1_verify.sh — fast pre-run verification for Phase 1 KV-cache patching
#
# Prerequisite: source libero_env.sh (venv active).
# NO model weights loaded, NO policy server needed. Tokenizer only (~800 KB on first run).
#
# What this does:
#   Tokenizes the Phase 1 contrastive prompt pair with the PaliGemma tokenizer and prints
#   the absolute KV-cache prefix index for each token. Use this to confirm that 'plate' and
#   'stove' both appear at absolute index 594 before trusting any patching results.
#
#   This satisfies checklist item C2 (token position verification).
#   Checklist item C1 (kv_cache shape) is verified automatically on the first run of
#   run_patching_phase1.sh — look for "Donor KV cache harvested. K shape:" in the log.
#
# Output: tee'd to stdout AND scripts_outputs_txt/patching_phase1_verify_YYYYMMDD_HHMMSS.txt

set -eo pipefail
OPENPI_DIR="/workspace/openpi"
cd "$OPENPI_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="scripts_outputs_txt/patching_phase1/verify"
OUT_FILE="$OUT_DIR/verify_${TIMESTAMP}.txt"
mkdir -p "$OUT_DIR"

{
echo "=== patching_phase1_verify run: $TIMESTAMP ==="
echo "=== Tokenizing Phase 1 contrastive pair for KV-cache index verification ==="
echo ""

python - <<'PYEOF'
import sys, os

# Tokenizer download (no openpi install needed)
CACHE = '/tmp/paligemma_tokenizer.model'
URL   = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
if not os.path.exists(CACHE):
    print(f"[tokenizer] Downloading from {URL} ...")
    import urllib.request
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    urllib.request.urlretrieve(URL, CACHE)
    print(f"[tokenizer] Downloaded ({os.path.getsize(CACHE)} bytes)")
else:
    print(f"[tokenizer] Using cached ({os.path.getsize(CACHE)} bytes)")

import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=CACHE)

PROMPTS = {
    "clean":   "put the bowl on the plate",
    "corrupt": "put the bowl on the stove",
}
LANG_OFFSET = 588   # 3 cameras × 196 tokens each = 588

print()
print("=" * 60)
print("PHASE 1 TOKEN POSITIONS")
print("=" * 60)
print(f"  Language region starts at absolute index: {LANG_OFFSET}")
print()

for key, prompt in PROMPTS.items():
    ids  = sp.encode(prompt, add_bos=True)
    pcs  = [sp.id_to_piece(i) for i in ids]
    print(f"  Prompt ({key}): {repr(prompt)}")
    print(f"  Token count (with BOS): {len(ids)}")
    for local_i, (tid, piece) in enumerate(zip(ids, pcs)):
        abs_i = LANG_OFFSET + local_i
        print(f"    [{local_i:2d}] id={tid:7d}  abs={abs_i}  {repr(piece)}")
    print()

print("=" * 60)
print("EXPECTED: 'plate' and 'stove' both at absolute index 594")
print("  If either differs, update patch_positions in run_patching_phase1.sh")
print("=" * 60)
PYEOF

echo ""
echo "=== Done. Record verified index in status_cc/patching_implementation.md §1.4 ==="
} 2>&1 | tee "$OUT_FILE"

echo ""
echo "=== Full log saved to: $OUT_FILE ==="
