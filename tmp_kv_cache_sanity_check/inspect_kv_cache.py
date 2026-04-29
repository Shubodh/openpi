"""
KV-cache sanity check for π₀.₅ on LIBERO-Goal.

Run on RunPod inside the openpi server venv (uv run) or the libero client venv:
    cd /workspace/openpi
    # Option A — server venv (has openpi installed):
    uv run python tmp_kv_cache_sanity_check/inspect_kv_cache.py
    # Option B — client venv:
    source examples/libero/.venv/bin/activate
    python tmp_kv_cache_sanity_check/inspect_kv_cache.py

What this script does:
  1. Tokenizes the two LIBERO-Goal contrastive prompts with the PaliGemma tokenizer.
  2. Prints the full token sequence (id + piece) for each prompt, marking the object-name tokens.
  3. Prints the statically-derived KV cache shape for π₀.₅ on LIBERO (no model load needed).
  4. Checks whether any of the three object-name words ('bowl', 'wine', 'bottle') are split
     into multiple subword tokens.

No model weights are loaded. Tokenizer is downloaded from GCS on first run (~800 KB).
"""

import sys
import os

# Allow running without openpi installed (reads tokenizer directly)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# --------------------------------------------------------------------------- #
# 1. Load the PaliGemma tokenizer
# --------------------------------------------------------------------------- #

try:
    from openpi.models.tokenizer import PaligemmaTokenizer
    tok = PaligemmaTokenizer()
    USE_OPENPI_TOKENIZER = True
    print("[tokenizer] Using openpi.models.tokenizer.PaligemmaTokenizer")
except Exception as e:
    print(f"[tokenizer] openpi tokenizer unavailable ({e}), falling back to raw sentencepiece")
    USE_OPENPI_TOKENIZER = False
    import sentencepiece
    from openpi.shared.download import maybe_download
    path = maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    sp = sentencepiece.SentencePieceProcessor(model_file=str(path))


def tokenize_raw(prompt: str) -> tuple[list[int], list[str]]:
    """Return (ids, pieces) for prompt with BOS, using the raw SP model."""
    if USE_OPENPI_TOKENIZER:
        # PaligemmaTokenizer.tokenize returns (tokens, mask) numpy arrays, padded to max_token_len.
        # We call the underlying SP directly via the private attr.
        sp = tok._tokenizer
        ids = sp.encode(prompt, add_bos=True)
    else:
        ids = sp.encode(prompt, add_bos=True)
    if USE_OPENPI_TOKENIZER:
        pieces = [tok._tokenizer.id_to_piece(i) for i in ids]
    else:
        pieces = [sp.id_to_piece(i) for i in ids]
    return ids, pieces


# --------------------------------------------------------------------------- #
# 2. Tokenize the contrastive prompt pair
# --------------------------------------------------------------------------- #

PROMPTS = {
    "clean":   "put the bowl on top of the cabinet",
    "corrupt": "put the wine bottle on top of the cabinet",
}

OBJECT_WORDS = {"bowl", "wine", "bottle"}

print()
print("=" * 60)
print("TOKENIZATION RESULTS")
print("=" * 60)

token_positions = {}  # prompt_key -> list of (abs_index_in_language_region, word)

for key, prompt in PROMPTS.items():
    ids, pieces = tokenize_raw(prompt)
    print(f"\nPrompt ({key}): {repr(prompt)}")
    print(f"  Tokens (with BOS): {len(ids)}")
    obj_positions = []
    for i, (tid, piece) in enumerate(zip(ids, pieces)):
        word = piece.lstrip('▁').lower()  # strip SentencePiece '▁' space prefix
        is_obj = word in OBJECT_WORDS
        marker = "  <-- OBJECT" if is_obj else ""
        print(f"  [{i:3d}] id={tid:7d}  piece={repr(piece)}{marker}")
        if is_obj:
            obj_positions.append((i, word))
    token_positions[key] = obj_positions
    if obj_positions:
        print(f"  Object-name token(s): {obj_positions}")
    else:
        print("  WARNING: no object-name tokens found!")


# --------------------------------------------------------------------------- #
# 3. Compute absolute prefix indices for the object-name tokens
# --------------------------------------------------------------------------- #

# Prefix sequence layout for π₀.₅ + LIBERO (3 cameras, 16×16 patches, 224×224 image):
# [0..195]   : base_0_rgb       (196 tokens, mask=True)
# [196..391] : left_wrist_0_rgb (196 tokens, mask=True)
# [392..587] : right_wrist_0_rgb(196 tokens, mask=False — zero image, padding)
# [588..787] : language tokens  (200 slots, actual tokens then padding)

IMAGE_TOKENS_PER_VIEW = (224 // 16) ** 2  # = 196
N_CAMERAS = 3
LANGUAGE_OFFSET = N_CAMERAS * IMAGE_TOKENS_PER_VIEW  # = 588
MAX_TOKEN_LEN = 200  # pi0.5 default

print()
print("=" * 60)
print("ABSOLUTE PREFIX POSITIONS (object-name tokens)")
print("=" * 60)
print(f"  Image token count per view  : {IMAGE_TOKENS_PER_VIEW}")
print(f"  Number of cameras           : {N_CAMERAS}")
print(f"  Language region starts at   : {LANGUAGE_OFFSET}")
print(f"  Max language token length   : {MAX_TOKEN_LEN}")
print(f"  Total prefix length         : {LANGUAGE_OFFSET + MAX_TOKEN_LEN}")
print()

for key, positions in token_positions.items():
    print(f"  {key}:")
    for (local_idx, word) in positions:
        abs_idx = LANGUAGE_OFFSET + local_idx
        print(f"    '{word}' → local index {local_idx} → absolute prefix index {abs_idx}")

# --------------------------------------------------------------------------- #
# 4. Statically-derived KV cache shape
# --------------------------------------------------------------------------- #

# From gemma.py get_config("gemma_2b"):
#   num_kv_heads = 1  (Multi-Query Attention)
#   head_dim     = 256
#   depth        = 18  (layers)
# KVCache type alias (gemma.py:336):
#   tuple[Float["l b t k h"], Float["l b t k h"]]
# With prefix_len = LANGUAGE_OFFSET + MAX_TOKEN_LEN = 788, batch=1:

DEPTH = 18
NUM_KV_HEADS = 1
HEAD_DIM = 256
BATCH = 1
PREFIX_LEN = LANGUAGE_OFFSET + MAX_TOKEN_LEN  # 788

print("=" * 60)
print("KV CACHE SHAPE (statically derived from gemma_2b config)")
print("=" * 60)
print(f"  kv_cache[0] (K): ({DEPTH}, {BATCH}, {PREFIX_LEN}, {NUM_KV_HEADS}, {HEAD_DIM})")
print(f"  kv_cache[1] (V): ({DEPTH}, {BATCH}, {PREFIX_LEN}, {NUM_KV_HEADS}, {HEAD_DIM})")
print(f"  = (layers, batch, prefix_seq_len, num_kv_heads, head_dim)")
print()
k_bytes = DEPTH * BATCH * PREFIX_LEN * NUM_KV_HEADS * HEAD_DIM * 2  # bfloat16
print(f"  Memory (bfloat16): {k_bytes / 1024:.1f} KB for K, same for V")
print(f"  Total cache: {2 * k_bytes / 1024:.1f} KB")

# --------------------------------------------------------------------------- #
# 5. Cache reuse confirmation (static code analysis result)
# --------------------------------------------------------------------------- #

print()
print("=" * 60)
print("CACHE REUSE (from static code analysis of pi0.py)")
print("=" * 60)
print("  sample_actions() flow:")
print("  1. embed_prefix(obs)  → prefix_tokens [B, 788, emb]")
print("  2. llm([prefix_tokens, None], ...)  → kv_cache  (computed ONCE)")
print("  3. for each diffusion step:")
print("       embed_suffix(obs, x_t, time)  → suffix_tokens")
print("       llm([None, suffix_tokens], kv_cache=kv_cache, ...)  → suffix_out")
print("       x_t += dt * action_out_proj(suffix_out)")
print("  Prefix KV cache is NEVER recomputed — reused across all diffusion steps.")
print("  Patching point: between step 2 and step 3 (overwrite kv_cache slots).")
print()
print("  Hook location: pi0.py:sample_actions, after line 237, before the step() loop.")
print("    `_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], ...)`")
print("    ← insert patch here →")
print("    `while cond(carry): carry = step(carry)`")
