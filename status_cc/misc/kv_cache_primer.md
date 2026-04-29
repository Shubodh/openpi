# KV-Cache Primer — π₀.₅ + PaliGemma + openpi

**Audience:** anyone reading `kv_cache_findings.md` (or designing the patching code) who wants conceptual + implementation grounding without re-deriving everything from scratch.

**Scope:** what the KV cache *is*, why it is the right intervention point for our experiment, how openpi's inference path constructs and consumes it, and how PaliGemma's tokenizer turns prompt strings into the token positions we'll patch at.

**This is not the deliverable.** Findings (cache shapes, exact token indices, file:line hook points) live in `kv_cache_findings.md`. This doc is the conceptual scaffolding.

---

## 1. KV cache, from basics

### 1.1 What attention computes

A transformer attention layer takes a sequence of token embeddings and, for each position, produces three projections per head:

- **Q** (query) — "what am I looking for?"
- **K** (key)   — "what do I represent, as a thing to be looked up?"
- **V** (value) — "what do I contribute if attended to?"

Attention output at position `i` is a weighted sum over all `V` values, where the weights are softmax of `Q_i · K_j` across positions `j` (subject to a mask that says which `j` are visible to `i`).

**Key observation:** for a given input token, its `K` and `V` only depend on that token's embedding (and the layer's weights). They do not depend on which other tokens query them later. So once you have computed `K_j` and `V_j` for token `j`, you can **store them** and reuse them whenever a future query `Q_i` (i > j) needs to attend to position `j`.

### 1.2 What the cache is

The "KV cache" is exactly that storage: a tensor (per layer, per head) of shape roughly `[batch, num_heads, seq_len_so_far, head_dim]` for `K` and the same for `V`. Concretely in openpi (`src/openpi/models/gemma.py:211-214`):

```python
if kv_cache is not None:
    cache_k, cache_v = kv_cache
    k = jnp.concatenate([cache_k, k], axis=1)
    v = jnp.concatenate([cache_v, v], axis=1)
```

When new tokens arrive, you compute their fresh `K`/`V`, **concatenate onto the cache**, and run attention against the union. You never recompute `K`/`V` for tokens already in the cache.

### 1.3 Why this matters for us

In a VLA like π₀.₅, the prompt ("put the bowl on top of the cabinet") + image tokens form a **prefix** that doesn't change across the action-decoding loop. Only the action/noise tokens (the "suffix") change. So the prefix's `K`/`V` are computed **once per episode** and reused for every diffusion / flow-matching denoising step.

That single tensor of cached `K`/`V` for the prefix is therefore *the* place where the prompt's information is stored, in a form the action expert directly consumes. If we want to test "does the model behave as if the prompt said `bowl` even though we passed `wine bottle`?", the cleanest intervention is:

1. Run a forward pass with the **clean** prompt → harvest the cached `K, V` at the object-name token position.
2. Run inference with the **corrupt** prompt → before the suffix decoding loop runs, **overwrite** the cache slot at the object-name position with the values harvested in step 1.
3. Roll out and measure success.

This is the π₀.₅ analog of the SmolVLA residual-stream patch in the AXMech repo. The mechanism differs (KV cache vs. residual stream), but the principle is the same: localize the language signal, transplant it across two runs, observe behavior.

---

## 2. Our patching goal

**Hypothesis:** the language signal that decides which object to pick is concentrated at the object-name token position in the prefix KV cache.

**Test:** clean run + corrupt run + patched run on a contrastive LIBERO-Goal task pair (`put the bowl on top of the cabinet` vs `put the wine bottle on top of the cabinet`). If patching the object-name slot from clean→corrupt restores clean behavior, the hypothesis holds.

**Sanity check (this primer's host task):** before writing intervention code, confirm:

- Cache shape per layer (so we know what we're overwriting).
- That `bowl` and `wine bottle` map to a clean, identifiable token range — ideally a single subword for `bowl`, possibly two for `wine bottle`.
- That openpi actually reuses the prefix cache across action steps (vs. recomputing the prefix each step, which would invalidate the "patch once" plan).

---

## 3. openpi's π₀.₅ inference path — conceptual map

The relevant file is `src/openpi/models/pi0.py`. The flow during `sample_actions` (line 217+):

1. **Build the prefix.** `embed_prefix` (line 106) concatenates:
   - **Image tokens** from `PaliGemma.img(...)` — produced by the SigLIP image encoder, one chunk per camera view.
   - **Language tokens** from `PaliGemma.llm(..., method="embed")` — embeddings of the tokenized prompt.

   Order: images first, then language. Order matters for indexing.

2. **One forward pass to populate the cache.** Line 233-237:
   ```python
   _, kv_cache = self.PaliGemma.llm(
       [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
   )
   ```
   This runs the full PaliGemma LLM stack over the prefix and returns a `KVCache` — a per-layer collection of `(K, V)` tensors.

3. **Iterative suffix decoding.** A `step(...)` function (line 239+) embeds the current noisy actions + state as a "suffix", builds an attention mask that lets the suffix attend to the prefix via the cache, and calls the LLM again — this time passing `kv_cache=kv_cache` (line 265 in the `sample_actions` loop). The cache is **read-only at this stage**: suffix `K`/`V` are computed fresh per step but the prefix cache is reused unchanged.

4. **Action readout.** The suffix's final hidden states project through `action_out_proj` to produce the velocity field for flow matching, integrated over `num_steps` to yield the final action chunk.

**Why this is friendly to patching:** because step 2 happens once and step 3 reuses its output, we can intervene cleanly between step 2 and step 3 — overwrite a slot in `kv_cache` and let the rest of the loop run untouched.

**Open questions for the sanity check (resolved in `kv_cache_findings.md`):**
- Exactly what is the per-layer shape returned at step 2?
- Where in the prefix sequence (which absolute index range) does the language portion start?
- Does the policy server expose a hook between step 2 and step 3, or do we need a wrapper?

---

## 4. PaliGemma tokenizer wiring in openpi

The relevant file is `src/openpi/models/tokenizer.py`. There are two tokenizer classes; for π₀.₅ (non-FAST) the relevant one is `PaligemmaTokenizer` (line 14+).

### 4.1 What it is

A SentencePiece tokenizer downloaded from `gs://big_vision/paligemma_tokenizer.model`. It is the same tokenizer Google ships with PaliGemma; openpi just wraps it. It handles BOS, but no EOS by default for the prompt. Vocabulary size is the standard PaliGemma vocab (the last 128 tokens are reserved as special, per the FAST tokenizer's `_fast_skip_tokens = 128` comment at line 62).

### 4.2 What it produces

`PaligemmaTokenizer.tokenize(prompt)` returns a numpy array of token ids, with `add_bos=True` so id 0 is the `<bos>` token, then the SentencePiece subwords for the prompt.

Because SentencePiece works on byte-pair / unigram subwords, **whether a word is a single token depends on the word**. Common short words (`bowl`, `the`, `cabinet`) are usually single tokens. Multi-word expressions (`wine bottle`) are at least two tokens (one per word), and individual words like `bottle` may themselves split into multiple subwords if uncommon.

This is why the sanity check explicitly inspects token strings — assumptions about "one word = one token" are wrong often enough that we want to see the actual ids before designing the patch.

### 4.3 Where it plugs into the inference path

The tokenizer runs **upstream of the policy server** during observation construction. By the time the prompt reaches `embed_prefix`, it is already a tensor of token ids inside `obs.tokenized_prompt`. `embed_prefix` does:

```python
tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
```

i.e., it runs the LLM's input embedding lookup table to turn ids into embeddings, then concatenates after the image tokens. So the **prefix sequence layout** is:

```
[ image_tokens (from SigLIP) ][ language_tokens (from tokenizer + embed) ]
↑                            ↑
0                            len(image_tokens)
```

The object-name token's absolute index in the prefix is therefore:

```
absolute_index = len(image_tokens) + offset_of_object_name_in_tokenized_prompt
```

Both numbers need to be confirmed empirically — `len(image_tokens)` depends on the SigLIP config (number of cameras × tokens per image), and the offset depends on the tokenizer's behavior on the specific prompt.

### 4.4 Practical consequence for patching

To patch the cache at the object-name position we need to identify a contiguous slice `[start, end)` in the prefix sequence dimension that corresponds to the object name in **both** prompts (clean and corrupt), and we need them to align well enough to support an overwrite. If `bowl` is 1 token at position `P` in the clean run and `wine bottle` is 2 tokens at positions `[P, P+1)` in the corrupt run, the alignment isn't trivial — the patching design has to commit to one of:

- Patch only the first token of the object-name region.
- Patch a fixed-width window and accept that contributions from non-object tokens may bleed in.
- Re-run the clean prompt with `wine` and `bottle` substituted into the same template to harvest a length-matched donor, etc.

Choosing among these is design work for *after* the sanity check. The sanity check just needs to surface the ground truth that drives this choice.

---

## 5. What this primer does **not** cover

- The actual numerical shapes (those go in `kv_cache_findings.md` after a forward pass — or, if statically determinable, in the offline phase of the sanity check).
- Patching code architecture (open question — see `kv_cache_sanity_check.md` "Agent plan" and the SmolVLA reference in the AXMech repo).
- Why we picked LIBERO-Goal over LIBERO-Object (see `corrupt_run_experiment.md` and the meta repo's `step3_exploration.md`).
- How flow matching / the action expert produces actions from suffix hidden states. Not relevant for the patching mechanism — the cache is upstream of all that.

---

## 6. References

- `src/openpi/models/pi0.py` — `embed_prefix` (line 106), `sample_actions` (line 217), KV cache build (line 237), suffix loop (line 239+).
- `src/openpi/models/gemma.py` — `Attention.__call__` (line 164), cache concatenation (line 211–214), per-layer call (line 293).
- `src/openpi/models/tokenizer.py` — `PaligemmaTokenizer` (line 14).
- `status_cc/kv_cache_sanity_check.md` — the host task spec and execution plan.
- `status_cc/corrupt_run_experiment.md` — prior LIBERO-Object work and motivation for the suite pivot.
- AXMech repo (`~/claude_code_workspace/2026_AXMech/AXMech/`) — SmolVLA residual-stream patching (conceptually parallel, mechanically different).
