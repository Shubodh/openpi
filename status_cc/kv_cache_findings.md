# KV-Cache Sanity Check — Findings

**Status:** 🔶 Partially complete — static analysis done; tokenizer verification pending (needs RunPod).

**Read this alongside:** `status_cc/misc/kv_cache_primer.md` for conceptual background.

**Verification script:** `tmp_kv_cache_sanity_check/inspect_kv_cache.py` — run on RunPod to confirm Section 2.

---

## 1. Cache Shape — ✅ Statically confirmed

**Source:** `src/openpi/models/gemma.py` — `get_config("gemma_2b")` (line 79) + `KVCache` type alias (line 336).

The KV cache for a single inference call is a tuple `(K, V)` where each has shape:

```
(layers, batch, prefix_seq_len, num_kv_heads, head_dim)
=  (18,     1,       788,           1,         256  )
```

Broken down:
| Dimension | Value | Source |
|-----------|-------|--------|
| layers | 18 | `gemma_2b.depth = 18` |
| batch | 1 | single episode rollout |
| prefix_seq_len | 788 | see prefix layout table below |
| num_kv_heads | 1 | `gemma_2b.num_kv_heads = 1` (Multi-Query Attention) |
| head_dim | 256 | `gemma_2b.head_dim = 256` |

Memory: `18 × 1 × 788 × 1 × 256 × 2 bytes (bfloat16) ≈ 7.2 MB` for K, same for V. Tiny — no practical constraint on patching.

**Note:** `num_kv_heads = 1` means π₀.₅ uses Multi-Query Attention (MQA) — all 8 query heads share a single K/V head. This simplifies patching: there is only one K and one V tensor per layer per position, not 8.

---

## 2. Prefix Sequence Layout — ✅ Statically confirmed

**Source:** `src/openpi/policies/libero_policy.py` (image mapping) + `src/openpi/models/pi0_config.py` (image specs) + `src/openpi/models/pi0.py:embed_prefix`.

For π₀.₅ running on LIBERO, the prefix has **788 tokens** in this fixed order:

| Slice | Camera / Region | Tokens | Valid? | Notes |
|-------|----------------|--------|--------|-------|
| [0..195] | `base_0_rgb` | 196 | ✅ Yes | agentview camera |
| [196..391] | `left_wrist_0_rgb` | 196 | ✅ Yes | wrist camera |
| [392..587] | `right_wrist_0_rgb` | 196 | ❌ No (mask=False) | zero image; tokens present in cache but masked in attention |
| [588..787] | language (prompt) | 200 | ✅ partial | padded to `max_token_len=200`; actual prompt tokens valid, rest padding |

**Image token count:** `(224 ÷ 16)² = 196` tokens per camera view (SigLIP with 16×16 patches, 224×224 input).

**Language region starts at absolute index 588.**

The right-wrist tokens occupy indices 392–587 in the cache tensor but have `input_mask=False`, so they cannot be attended to by any valid token (and cannot attend to any other token). Patching those positions would have no effect.

---

## 3. Tokenizer — 🔶 Needs RunPod verification

**What we know statically:** PaliGemma uses a SentencePiece tokenizer with vocab size 256,000 (same as Gemma). SentencePiece adds a `▁` prefix to represent the space before a word. In a 256K vocabulary, common English words like `bowl`, `wine`, `bottle` are almost certainly single tokens.

**What needs to be confirmed on RunPod:**

Run `tmp_kv_cache_sanity_check/inspect_kv_cache.py` from the openpi venv on RunPod. It will print:

1. The full token sequence for each prompt (ids + piece strings).
2. The local index of each object-name token within the tokenized prompt.
3. The absolute prefix index: `588 + local_index`.

**Expected output (to be filled in after RunPod run):**

For `"put the bowl on top of the cabinet"` (with BOS):
```
[0] BOS
[1] ▁put
[2] ▁the
[3] ▁bowl    ← object-name token, local index 3 → absolute prefix index 591
[4] ▁on
[5] ▁top
[6] ▁of
[7] ▁the
[8] ▁cabinet (may be 1 or 2 tokens — verify)
```

For `"put the wine bottle on top of the cabinet"` (with BOS):
```
[0] BOS
[1] ▁put
[2] ▁the
[3] ▁wine    ← object-name token 1, local index 3 → absolute prefix index 591
[4] ▁bottle  ← object-name token 2, local index 4 → absolute prefix index 592
[5] ▁on
...
```

**Key uncertainty:** whether `cabinet` is one token or two. Also whether `wine` and `bottle` are each one token (expected) or split further.

**Placeholder — update after RunPod run:**
```
clean:   bowl   → local [?], absolute prefix [?]
corrupt: wine   → local [?], absolute prefix [?]
         bottle → local [?], absolute prefix [?]
```

---

## 4. Cache Reuse — ✅ Statically confirmed

**Source:** `src/openpi/models/pi0.py:sample_actions` (lines 233–276).

The prefix KV cache is computed **once per episode** and reused across all diffusion steps:

```python
# Step 1: compute prefix cache once (line 237)
_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

# Step 2: reuse cache in every diffusion step (line 265, inside step())
(_, suffix_out), _ = self.PaliGemma.llm(
    [None, suffix_tokens], kv_cache=kv_cache, ...
)
```

Inside `Attention.__call__` (gemma.py:211–214), each suffix step reads from `kv_cache` and concatenates fresh suffix K/V on top:
```python
if kv_cache is not None:
    cache_k, cache_v = kv_cache
    k = jnp.concatenate([cache_k, k], axis=1)
    v = jnp.concatenate([cache_v, v], axis=1)
```

The cache itself is **never mutated** — it is a pure functional read. This is ideal for patching: we modify the cache tensor after step 1 and the modified values propagate into every diffusion step automatically.

---

## 5. Hook Point for Patching — ✅ Identified

**File:** `src/openpi/models/pi0.py`

**Function:** `Pi0.sample_actions`

**Location:** After line 237 (prefix cache built), before the `while cond(carry): carry = step(carry)` loop.

```python
# ← INSERT PATCH HERE ←
_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], ...)  # line 237
# patch: kv_cache = patch_kv_cache(kv_cache, donor_cache, abs_start, abs_end)
x_t, _ = jax.lax.while_loop(cond, step, (x_t, 0.0))         # suffix loop
```

For a clean patching interface without modifying the class, the simplest approach is a wrapper function that:
1. Calls `sample_actions` up to the cache-build step (via a custom method or monkey-patch).
2. Overwrites the target slice of `kv_cache`.
3. Continues the suffix loop with the modified cache.

Alternatively: subclass `Pi0` and override `sample_actions`, calling `super()` up to the cache build.

The exact patching architecture is a decision for the meta-repo discussion — this doc just confirms the hook exists and is clean.

---

## 6. Patching Design Implications

Given the above, the key design choice is how to handle the **token count mismatch** between `bowl` (1 token) and `wine bottle` (2 tokens):

**Option A — Patch only the first object-name token.** Overwrite the cache at absolute index 591 from the clean run onto the corrupt run. This transplants the `bowl` K/V into the `wine` slot. The `bottle` slot (index 592) keeps its corrupt-run values. This is the minimal intervention.

**Option B — Patch the full object-name span.** Determine the maximum span (2 tokens for `wine bottle`) and overwrite both positions. In the clean run, index 592 is `on`, not an object-name token — so we'd be transplanting `on` K/V into the `bottle` position. This is more aggressive and may have unexpected side effects.

**Option C — Use length-matched donors.** Craft a donor prompt that has `wine bottle` but means `bowl` (e.g., substitute `bowl` → `wine bowl` to match lengths). Harvest from that. Cleaner but requires a second clean-run variant.

**Recommendation going into meta-repo discussion:** Start with Option A (single-token patch). The `bowl` K/V at position 591 is the strongest signal; the `bottle` residual at 592 in the corrupt run is minimal contamination. If the result is ambiguous, try Option B or C.

---

## 7. Summary for Patching Code Author

| Question | Answer | Confidence |
|----------|--------|------------|
| KV cache shape | `(18, 1, 788, 1, 256)` each for K and V | ✅ Static |
| Language region start (absolute) | index 588 | ✅ Static |
| Object-name absolute index (bowl) | 591 (= 588 + 3) | 🔶 Expected; verify on RunPod |
| Object-name absolute indices (wine bottle) | 591, 592 (= 588 + 3, 588 + 4) | 🔶 Expected; verify on RunPod |
| Cache reused across diffusion steps | Yes | ✅ Static |
| Hook point | `pi0.py:sample_actions` after line 237 | ✅ Static |
| MQA (single K/V head per layer) | Yes, `num_kv_heads=1` | ✅ Static |
| Right-wrist tokens patchable | No — masked, no effect | ✅ Static |

**One RunPod run of `inspect_kv_cache.py` will fill in the 🔶 entries** and complete this document. No full model load needed — just the tokenizer.


# Verification by Codex

One useful nuance: pi05_libero sets discrete_state_input=False in src/openpi/training/config.py:743, so for this LIBERO config the tokenizer should be prompt-only, not Task: ..., State: ...; Action:. That makes the current raw-prompt tokenizer script direction plausible.
