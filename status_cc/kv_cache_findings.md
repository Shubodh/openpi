# KV-Cache Sanity Check — Findings

**Status:** ✅ Complete — static analysis + RunPod tokenizer verification done (2026-04-29).

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

## 3. Tokenizer — ✅ Verified on RunPod (2026-04-29)

**Source:** `scripts_outputs_txt/kv_cache_inspect/inspect_20260429_182504.txt`

**Background:** PaliGemma uses a SentencePiece tokenizer with vocab size 256,000 (same as Gemma). SentencePiece adds a `▁` prefix to represent the space before a word. In a 256K vocabulary, common English words like `bowl`, `wine`, `bottle` are almost certainly single tokens — this is confirmed below. BOS token is always prepended (id=2), so local token indices are 1-based relative to the prompt words.

All object-name words are single tokens. `cabinet` is also a single token. No surprises.

For `"put the bowl on top of the cabinet"` (9 tokens with BOS):
```
[0] id=2      <bos>
[1] id=1065   put
[2] id=573    ▁the
[3] id=14581  ▁bowl     ← object-name token, local index 3 → absolute prefix index 591
[4] id=611    ▁on
[5] id=2267   ▁top
[6] id=576    ▁of
[7] id=573    ▁the
[8] id=22402  ▁cabinet
```

For `"put the wine bottle on top of the cabinet"` (10 tokens with BOS):
```
[0] id=2      <bos>
[1] id=1065   put
[2] id=573    ▁the
[3] id=10058  ▁wine     ← object-name token 1, local index 3 → absolute prefix index 591
[4] id=12989  ▁bottle   ← object-name token 2, local index 4 → absolute prefix index 592
[5] id=611    ▁on
[6] id=2267   ▁top
[7] id=576    ▁of
[8] id=573    ▁the
[9] id=22402  ▁cabinet
```

**Verified results:**
```
clean:   bowl   → local 3, absolute prefix 591  (id=14581)
corrupt: wine   → local 3, absolute prefix 591  (id=10058)
         bottle → local 4, absolute prefix 592  (id=12989)
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
| Object-name absolute index (bowl) | 591 (= 588 + 3), token id=14581 | ✅ RunPod verified |
| Object-name absolute indices (wine bottle) | 591 (id=10058), 592 (id=12989) | ✅ RunPod verified |
| Cache reused across diffusion steps | Yes (within one `sample_actions` call) | ✅ Static + Codex |
| Hook point | `pi0.py:sample_actions` after line 237 | ✅ Static |
| MQA (single K/V head per layer) | Yes, `num_kv_heads=1` | ✅ Static |
| Right-wrist tokens patchable | No — masked, no effect | ✅ Static |

**Phase 1 complete.** All entries verified — static analysis + RunPod tokenizer run (2026-04-29).


# Verification by Codex

Verified against the local code on 2026-04-29.

The static claims are mostly correct: `gemma_2b` has 18 layers, 1 KV head, and head dim 256 (`src/openpi/models/gemma.py:get_config`); `Pi0Config(pi05=True)` gives 200 language slots (`src/openpi/models/pi0_config.py:__post_init__`); LIBERO supplies three 224x224 image streams, with right wrist masked false for non-FAST π0 (`src/openpi/policies/libero_policy.py:LiberoInputs`); and `pi05_libero` sets `discrete_state_input=False` (`src/openpi/training/config.py`), so prompt-only tokenizer inspection is the right check for this config.

Important caveat: the cache-reuse claim is accurate **within one `Pi0.sample_actions` / policy inference call**, across the diffusion denoising steps (`src/openpi/models/pi0.py:sample_actions`). I do not see evidence that the prefix KV cache is reused across a whole LIBERO episode or across separate replanning calls: `Policy.infer()` calls `sample_actions()` each server request (`src/openpi/policies/policy.py:infer`), the websocket server calls `policy.infer()` per request (`src/openpi/serving/websocket_policy_server.py:_handler`), and the LIBERO client requests a new action chunk when its current plan is exhausted (`examples/libero/main_corrupt_run_expt.py`). Patching should therefore happen after cache construction on each inference call, not once globally per episode.

Still unverified here: exact tokenizer ids/pieces for `bowl`, `wine`, and `bottle`. The local environment lacks `uv`, `jax`, and `sentencepiece`, so I could not run `tmp_kv_cache_sanity_check/inspect_kv_cache.py`. The document correctly marks those token positions as expected/pending RunPod verification.
