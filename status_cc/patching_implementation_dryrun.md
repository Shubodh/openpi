# Patching Implementation Dry-Run Doc

**Status:** Draft — for human review before coding begins.

**Scope:** Complete implementation plan for KV-cache patching on π₀.₅ + LIBERO-Goal. Covers architecture, code locations, indexing arithmetic, and all open decisions that need resolution before a line of implementation code is written.

**Reference documents:**
- `status_cc/kv_cache_findings.md` — authoritative source for shapes, positions, hook point
- `status_cc/misc/kv_cache_primer.md` — conceptual background
- SmolVLA patching scripts: `AXMech/scripts/run_patching.py` and `cache_activations.py`

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
   - [What we're building](#what-were-building)
   - [One script or three invocations?](#one-script-or-three-invocations)
2. [Hook Point — Exact Code Location](#2-hook-point--exact-code-location)
3. [Implementation Strategy: JAX vs PyTorch, and How to Patch](#3-implementation-strategy-jax-vs-pytorch-and-how-to-patch)
   - [Why SmolVLA's hook approach doesn't apply directly to JAX](#why-smolvlas-pytorch-hook-approach-doesnt-apply-directly-to-jax)
   - [The JAX way: functional array update](#the-jax-way-functional-array-update)
   - [Where does the donor KV cache come from?](#where-does-the-donor-kv-cache-come-from)
   - [Proposed implementation approach](#proposed-implementation-approach)
4. [Token Positions and Patch Options](#4-token-positions-and-patch-options)
   - [Patch options](#patch-options)
5. [Cache Shape and Indexing Arithmetic](#5-cache-shape-and-indexing-arithmetic)
6. [Token-Count Mismatch Handling](#6-token-count-mismatch-handling)
7. [Diffusion Step Propagation](#7-diffusion-step-propagation)
8. [Recovery Score Definition](#8-recovery-score-definition)
9. [End-to-End Walk-Through](#9-end-to-end-walk-through-single-patched-trial)
10. [Script Architecture](#10-script-architecture-proposed)
11. [Open Decisions Summary](#11-open-decisions-summary)
12. [What Is NOT in Scope](#12-what-is-not-in-scope-of-this-doc)
13. [Big-Picture Questions Came Up by the Human](#13-big-picture-questions-came-up-by-the-human)
   - [Does KV-cache patching even make sense as opposed to residual stream?](#does-kv-cache-patching-even-make-sense-as-opposed-to-residual-stream)
14. [Supplementary](#14-supplementary)
   - [14.1 Runtime call trace: how `sample_actions()` is reached](#141-runtime-call-trace-how-sample_actions-is-reached)
   - [14.2 JAX primer: JIT, immutable arrays, and while_loop](#142-jax-primer-jit-immutable-arrays-and-while_loop)
   - [14.3 JAX vs PyTorch: full analysis (reference only)](#143-jax-vs-pytorch-full-analysis-reference-only)


---

## 1. High-Level Architecture

### What we're building

Three run types, same as SmolVLA:

| Run | Prompt | KV cache | Purpose |
|-----|--------|----------|---------|
| **Clean** | `"put the bowl on top of the cabinet"` | built normally | establishes success ceiling |
| **Corrupt** | `"put the wine bottle on top of the cabinet"` | built normally | establishes failure floor |
| **Patched** | `"put the wine bottle on top of the cabinet"` | positions 591–592 overwritten with clean-run values | tests whether language signal is causal |

#### Optional later summary metric — recovery score (per-task, aggregated over trials):

```
recovery = (patched_success_rate − corrupt_success_rate) / (clean_success_rate − corrupt_success_rate)
```
- 0 = patch had no effect
- 1 = patch fully restores clean behavior
- Note: unlike SmolVLA (continuous action score), here the metric is binary task success (LIBERO provides `done` flag per episode). Recovery score is computed from success rates aggregated over the N trials per task.

**Important status of this metric:** This recovery formula is a dummy placeholder, not something to spend time optimizing before the first patching result. The main simulator result can simply be clean rollout success, corrupt rollout success, and patched-corrupt rollout success for each intervention setting. A normalized recovery score is only useful later as a compact summary metric, and even then only if the clean/corrupt baselines define a meaningful behavioral gap. In many LIBERO settings both clean and corrupt prompts can be near 100% successful because the task is visually obvious or the corrupt language does not reliably force failure; in that case the denominator is tiny or zero and the recovery score is meaningless.

If a heatmap-style patching analysis is needed later, more thought should go into defining a meaningful recovery score. The simplest practical option may be to save the last frame from each rollout to disk and run an offline VLM judge to label success/failure for the intended task. That would still produce a binary score rather than a continuous one, but it would let recovery-style summaries be computed offline from saved rollouts instead of forcing the online simulator `done` flag to carry all of the interpretation.

### One script or three invocations?

SmolVLA's approach: one script that runs clean and corrupt baselines at startup, then runs the patching loop (all three "run types" in a single process).

π₀.₅ difference: LIBERO evaluation is an interactive sim — we can't do a "forward pass" in isolation the way SmolVLA could with a fixed frame. We run full episodes with environment steps. Also, `main_corrupt_run_expt.py` already covers the baseline side: it can run the clean prompt and the corrupt prompt using the existing LIBERO loop.

**Updated plan: one existing baseline script plus one new patching script:**
1. `main_corrupt_run_expt.py` (already exists) → clean and corrupt baseline runs
2. `main_patching_expt.py` (new) → patched corrupt-prompt runs

`main_patching_expt.py` should be built by copying/adapting `main_corrupt_run_expt.py`, because it needs the same LIBERO task setup, rollout loop, seeding, logging, and video/save behavior. The new script should add only patch-specific controls: patch layer(s), patch kind (`K`, `V`, or `K+V`), patch position(s), donor mode, and output directory naming for patch configs.

**Open decision O1 resolved:** Do not merge clean/corrupt/patched into one large mode-switching script. Keep `main_corrupt_run_expt.py` as the baseline runner and create a separate `main_patching_expt.py` for intervention runs.

---

## 2. Hook Point — Exact Code Location

**File:** `src/openpi/models/pi0.py`

**Function:** `Pi0.sample_actions` (line 217)

**Exact location:** After line 237 (prefix cache built), before `def step` at line 239 and the `while_loop` at line 278.

```python
# src/openpi/models/pi0.py:233–279 (relevant excerpt)

# first fill KV cache with a forward pass of the prefix
prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)     # 234
prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)                  # 235
positions = jnp.cumsum(prefix_mask, axis=1) - 1                                 # 236
_, kv_cache = self.PaliGemma.llm(                                               # 237
    [prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

# ← PATCH GOES HERE (before `def step` at line 239) ←

def step(carry):                                                                 # 239
    ...
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [None, suffix_tokens], kv_cache=kv_cache, ...)                          # 265

x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))                          # 278
return x_0
```

**Why the patch must go before `def step` (line 239):** The `step` function closes over the Python variable `kv_cache`. By reassigning `kv_cache` to the patched version BEFORE `step` is defined, the closure captures the already-patched value — unambiguous. (Python closures are late-binding, so inserting between lines 237 and 278 but after line 239 would technically still work — the closure looks up `kv_cache` at call time — but inserting before line 239 is cleaner and preferred.)

The prefix KV cache is computed once per `sample_actions` call and reused read-only across all diffusion denoising steps (`kv_cache=kv_cache` at line 265 inside `step()`). Patching before `step` is defined propagates to all denoising steps within that one inference call automatically.

**Status of the cache-reuse caveat (from `kv_cache_findings.md`):** confirmed and not a blocker. The prefix cache is reused **within one `Pi0.sample_actions` / `Policy.infer()` call** across diffusion denoising steps, but it is not reused globally across a whole LIBERO episode. `Policy.infer()` calls `sample_actions()` for each server request, and the LIBERO client sends a new request whenever it needs a new action chunk. Therefore the hook point is still correct: because it lives inside `sample_actions`, the patch is applied on every inference call/action chunk automatically. For the per-step donor plan, this also means the clean donor prefix pass should be rebuilt from the same current observation on each inference call.

---

## 3. Implementation Strategy: JAX vs PyTorch, and How to Patch

For the full runtime path from the LIBERO eval script to `Pi0.sample_actions()`, see [Section 14.1](#runtime-call-trace-how-sample_actions-is-reached). The short version: the eval script asks the policy server for a new action chunk, the server calls `Policy.infer()`, and `Policy.infer()` calls the jitted `model.sample_actions()`.

**Execution path: JAX (resolved 2026-05-04).** All experiments to date use the default JAX path — `runpod/start_libero.sh` → `serve_policy.py --env LIBERO` → JAX orbax checkpoint → `src/openpi/models/pi0.py`. A PyTorch implementation exists and a conversion script (`examples/convert_jax_model_to_pytorch.py`) is provided, but JAX was chosen to keep baseline consistency and avoid the PyTorch-specific setup friction. Full JAX vs PyTorch analysis is in §14.3 for reference.

The "scary JAX JIT" is a non-issue — see §14.2 for a plain-English explanation. The patching implementation is ~15 lines of ordinary Python.

**Open decision O2:** With execution path resolved, choose between P1 (modify `pi0.py`) vs P3 (monkey-patch) vs O2b (bypass server) — see §10.

---

### Why SmolVLA's PyTorch hook approach doesn't apply directly to JAX

SmolVLA used PyTorch `register_forward_pre_hook` to intercept the residual stream mid-forward-pass. JAX has no equivalent hook mechanism — functions are pure and traced/compiled. If we stay on JAX, patching must be done via explicit functional array update inserted directly into `sample_actions` (see below). If we convert to PyTorch, hooks become available and the SmolVLA pattern applies.

### The JAX way: functional array update

**Applies when:** using the default JAX checkpoint (`start_libero.sh` → `serve_policy.py --env LIBERO` → `src/openpi/models/pi0.py`). This is the current setup for all prior LIBERO experiments.

JAX arrays are immutable. The patch is a pure functional update using `.at[...].set(...)`, inserted directly into `Pi0.sample_actions` in `src/openpi/models/pi0.py` between lines 237 (cache built) and 239 (`def step` defined):

```python
# kv_cache = (K, V)
# K shape: (18, 1, 788, 1, 256) — (layers, batch, prefix_seq, kv_heads, head_dim)
# V shape: same

K_donor, V_donor = donor_kv_cache
K_corrupt, V_corrupt = kv_cache

# Option A: patch only the object-name position (absolute index 591)
K_patched = K_corrupt.at[:, :, 591, :, :].set(K_donor[:, :, 591, :, :])
V_patched = V_corrupt.at[:, :, 591, :, :].set(V_donor[:, :, 591, :, :])
kv_cache = (K_patched, V_patched)
```

This is pure JAX, JIT-compatible, and requires no model modification beyond making the donor cache available.

**KVCache structure — confirmed from source:** `gemma.py:336` defines:
```python
KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]
```
So `kv_cache = (K, V)` where K and V each have shape `(layers, batch, seq_len, kv_heads, head_dim)`. The `l` (layers) dimension comes from `nn.scan` in `Module.setup()` — each `Block` layer returns per-layer `(k, v)` of shape `(B, S, K, H)`, and `nn.scan` stacks them into `(L, B, S, K, H)`. The indexing `K[:, :, 591, :, :]` patches all 18 layers at position 591 simultaneously. ✅ No debug verification needed for the shape; `jax.tree_util.tree_map(...)` can be used to sanity-check at runtime if desired.

### Where does the donor KV cache come from?

**Key constraint:** π₀.₅ uses bidirectional (non-causal) attention in the prefix. Verified from source: `embed_prefix` sets `ar_mask = [False] * n` for all image and language tokens (`pi0.py:125–133`). `make_attn_mask` (`pi0.py:41–44`) implements: `cumsum = jnp.cumsum(mask_ar, axis=1); attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]`. With all-False `mask_ar`, cumsum is all zeros, so `0 <= 0` is always True — every valid token attends to every other valid token. Consequently, language token K/V values at positions 591–592 depend on the image tokens. We cannot harvest the donor cache from a reference frame with different images.

**Solution:** For each patched inference call, run an additional clean-prompt prefix forward pass using the *current* observation's images:

**New update (2026-05-03):** See `status_cc/kv_cache_findings.md` for the fuller discussion of pre-computed donor vs per-step donor. In short, a pre-computed donor is easier and faster because the patched rollout only runs the corrupt/patched path at runtime, loading a clean KV cache harvested earlier. But that clean KV cache saw old images / old scene state. A per-step donor is more involved because each patched inference call internally builds both a clean donor cache and a corrupt recipient cache from the same current images, then patches the recipient before action decoding. This is slower and likely adds more implementation code, but it is the cleaner causal intervention because donor and recipient differ only in language, not in visual context.

```python
# Pseudocode inside patching logic, executed at each sample_actions call:

# Build donor KV cache: same current images, clean prompt
obs_clean = replace_prompt(observation, clean_tokenized_prompt)
prefix_tokens_c, prefix_mask_c, prefix_ar_mask_c = self.embed_prefix(obs_clean)
prefix_attn_mask_c = make_attn_mask(prefix_mask_c, prefix_ar_mask_c)
positions_c = jnp.cumsum(prefix_mask_c, axis=1) - 1
_, donor_kv_cache = self.PaliGemma.llm(
    [prefix_tokens_c, None], mask=prefix_attn_mask_c, positions=positions_c
)

# Now patch corrupt kv_cache with donor values (as above)
```

**Cost:** This doubles the prefix forward pass cost per action chunk request. The prefix pass is cheap (no diffusion steps), so this should be acceptable. Confirm timing on RunPod.

### Proposed implementation approach

**Approach P1 (recommended): minimal modification to `pi0.py`**

Add two optional parameters to `Pi0.sample_actions`:

```python
def sample_actions(
    self,
    rng,
    observation,
    *,
    num_steps=10,
    noise=None,
    clean_observation=None,   # NEW: if set, triggers KV patching
    patch_positions=(591,),   # NEW: absolute KV cache positions to overwrite
):
    ...
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], ...)

    if clean_observation is not None:
        kv_cache = self._apply_kv_patch(kv_cache, clean_observation, patch_positions)

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0

def _apply_kv_patch(self, corrupt_kv_cache, clean_observation, patch_positions):
    """Build donor KV cache from clean_observation, overwrite patch_positions."""
    clean_prefix_tokens, clean_prefix_mask, clean_prefix_ar_mask = self.embed_prefix(clean_observation)
    clean_prefix_attn_mask = make_attn_mask(clean_prefix_mask, clean_prefix_ar_mask)
    clean_positions = jnp.cumsum(clean_prefix_mask, axis=1) - 1
    _, donor_kv_cache = self.PaliGemma.llm(
        [clean_prefix_tokens, None], mask=clean_prefix_attn_mask, positions=clean_positions
    )
    K_d, V_d = donor_kv_cache
    K, V = corrupt_kv_cache
    for pos in patch_positions:
        K = K.at[:, :, pos, :, :].set(K_d[:, :, pos, :, :])
        V = V.at[:, :, pos, :, :].set(V_d[:, :, pos, :, :])
    return (K, V)
```

`clean_observation` is passed via `sample_kwargs` when constructing the `Policy` object... except that `clean_observation` changes per inference call (different images each step). This means it can't be a static `sample_kwargs` value.

**Revised P1 — per-call injection:** Pass `clean_observation` dynamically by updating `sample_kwargs` before each `infer()` call. `Policy.infer()` at `policy.py:82` does `sample_kwargs = dict(self._sample_kwargs)` on every call, so mutating `self._sample_kwargs` before each call is the right hook:

```python
# In main_patching_expt.py, inside the action-chunk request loop:
clean_obs_dict = make_obs_dict(env_obs, clean_prompt)   # same image, clean prompt
# Pre-process to match what Policy.infer() does internally (add batch dim, jnp.asarray):
clean_obs_jax = {k: jnp.asarray(v)[np.newaxis] for k, v in clean_obs_dict.items()}
clean_observation = Observation.from_dict(clean_obs_jax)
# Inject for this call
policy._sample_kwargs["clean_observation"] = clean_observation
action = policy.infer(corrupt_obs_dict)
```

**JAX JIT note:** `module_jit` wraps `sample_actions` with `jax.jit`. Adding `clean_observation` as a new dynamic kwarg will trigger a one-time recompilation on the first patched call (JAX sees a different pytree structure than the baseline calls). Subsequent calls with the same shapes use the cached compilation. This recompile is a one-off cost, not a per-call overhead.

**Approach P3 (alternative): monkey-patch `sample_actions` in new script, no pi0.py changes**

In `main_patching_expt.py`, after loading the model, replace its `sample_actions` with a closure:

```python
from openpi.models.pi0 import make_attn_mask  # make_attn_mask is module-level in pi0.py, importable
import types

original_sample_actions = model.sample_actions.__func__  # bound method's function

def patched_sample_actions(self, rng, observation, *, num_steps=10, noise=None):
    # ... full body of sample_actions with patch logic inserted after cache build
    pass

model.sample_actions = types.MethodType(patched_sample_actions, model)
```

This avoids touching `pi0.py` but requires duplicating `sample_actions` — fragile if upstream changes.

**Open decision O2:** P1 (minimal pi0.py modification) vs P3 (standalone monkey-patch). Leaning toward P1 for maintainability, but it requires touching core code. Discuss with human before coding.

---

## 4. Token Positions and Patch Options

From `kv_cache_findings.md` §3 (RunPod-verified):

| Token | Absolute KV cache index | Token id |
|-------|------------------------|----------|
| `bowl` | 591 | 14581 |
| `wine` (in "wine bottle") | 591 | 10058 |
| `bottle` | 592 | 12989 |

Language region starts at absolute index 588. Object-name token is at local position 3 within the prompt (BOS at local 0), → absolute 588 + 3 = 591.

### Patch options

**Option A (recommended start): patch only position 591.**
- Overwrite `kv_cache[:, :, 591, :, :]` from donor (bowl) into recipient (wine bottle).
- The `bowl` K/V replaces the `wine` K/V. The `bottle` K/V at position 592 is untouched.
- Minimal intervention; easiest to interpret.

**Option B: patch positions 591 and 592.**
- Overwrite both positions.
- At position 592 in the donor (clean) cache, the token is `on` (not an object-name token).
- We'd be transplanting `on` K/V into the `bottle` slot — potentially confusing the model about syntax.
- More aggressive, harder to interpret.

**Option C: use a length-matched donor prompt.**
- Run a second clean reference with prompt `"put the wine bowl on top of the cabinet"` (using `wine` as adjective, `bowl` as the object) to produce a 2-token object-name span matching `wine bottle`.
- Harvest positions 591–592 from this reference.
- Cleanest semantically, but requires defining and testing the length-matched prompt.

**Open decision O3:** Which option to implement first. Recommendation: start with Option A. If results are ambiguous (recovery score neither near 0 nor near 1), escalate to B or C.

---

## 5. Cache Shape and Indexing Arithmetic

From `kv_cache_findings.md` §1 and §7, confirmed against `gemma.py`:

| Dimension | Value | Source |
|-----------|-------|--------|
| layers | 18 | `gemma_2b.depth = 18` (`gemma.py:73`) |
| batch | 1 | single episode rollout |
| prefix_seq_len | 788 | total prefix tokens (3×196 images + 200 language) |
| num_kv_heads | 1 | MQA (`gemma_2b.num_kv_heads = 1`, `gemma.py:76`) |
| head_dim | 256 | `gemma_2b.head_dim = 256` (`gemma.py:77`) |

Full KV cache shape per K or V: `(18, 1, 788, 1, 256)`. ✅ Confirmed from `KVCache` type alias (`gemma.py:336`) and `nn.scan` stacking behavior.

Prefix layout:

| Absolute index range | Content | Valid? |
|---------------------|---------|--------|
| 0–195 | `base_0_rgb` (agentview camera) | ✅ |
| 196–391 | `left_wrist_0_rgb` | ✅ |
| 392–587 | `right_wrist_0_rgb` | ❌ masked out |
| 588–787 | language prompt (padded to 200) | ✅ (valid tokens only) |

Object-name positions (absolute):
- `bowl`: index 591 (= 588 + 3)
- `wine`: index 591 (= 588 + 3)
- `bottle`: index 592 (= 588 + 4)

**MQA note:** `num_kv_heads = 1` means all 8 query heads share a single K and V tensor. The patch operation on the `(18, 1, 788, 1, 256)` tensor is directly the K (or V) for all query heads — no per-head indexing needed.

**JAX indexing for Option A patch:**
```python
# K, V each: (18, 1, 788, 1, 256)
# Patch all 18 layers, batch 0, position 591, all kv_heads (1), all head dims (256)
K_patched = K_corrupt.at[:, :, 591, :, :].set(K_donor[:, :, 591, :, :])
V_patched = V_corrupt.at[:, :, 591, :, :].set(V_donor[:, :, 591, :, :])
```

---

## 6. Token-Count Mismatch Handling

The clean prompt has 9 tokens (with BOS); the corrupt prompt has 10 tokens (with BOS). This means the full prefix lengths differ by 1 token — positions 592–787 shift by 1 in absolute terms between clean and corrupt prompts.

**Critical implication:** The absolute positions of ALL tokens *after* the object name differ between clean and corrupt runs:
- Clean: `on` at position 592, `top` at 593, etc.
- Corrupt: `bottle` at 592, `on` at 593, etc.

For Option A (patch only position 591): only the object-name position is involved, so the 1-token shift doesn't affect the patch. The donor K/V at position 591 contains `bowl` context; the recipient K/V at position 591 contains `wine` context. These positions correspond in both runs.

**RoPE position consistency for Option A (key correctness property):** Positions are computed as `jnp.cumsum(prefix_mask, axis=1) - 1`. With identical images (196 valid + 196 valid + 196 masked + language tokens), both runs have 392 valid image tokens before the language region. The right-wrist tokens (392–587) are `prefix_mask=False`, contributing 0 to cumsum. Language token at local index 3 (absolute 591) gets `cumsum = 392 + 4 = 396`, RoPE position `= 395` — identical in both clean and corrupt runs. So the donor and recipient K/V at index 591 share the same RoPE position; only the token identity (bowl vs wine) differs. This makes Option A a clean semantic patch. ✅

For Options B/C: the shift matters and requires careful re-derivation of which absolute indices to patch.

**Padding:** Both prompts are padded to `max_token_len=200`. Actual token counts are 9 and 10; positions 597–787 (clean) and 598–787 (corrupt) are padding tokens. Padding tokens are not attended to (masked out by `tokenized_prompt_mask`). Overwriting their K/V would have no effect.

---

## 7. Diffusion Step Propagation

From `kv_cache_findings.md` §4 and `pi0.py:261–266`:

```python
# Inside step() — runs num_steps times (default 10) via jax.lax.while_loop:
(prefix_out, suffix_out), _ = self.PaliGemma.llm(
    [None, suffix_tokens],
    kv_cache=kv_cache,       # ← reads the (patched) prefix KV cache
    ...
)
```

In the P1 implementation, the patch is inserted between lines 237 and 239 — BEFORE `def step(carry):` is reached. `step` closes over `kv_cache` (Python late-binding), so when `step` is defined at line 239, it captures the already-patched `kv_cache`. `jax.lax.while_loop` traces `step` when called at line 278, at which point `kv_cache` in the closure refers to the patched JAX array. Inside the JIT-compiled execution, the patched K/V is a constant in the XLA graph for the loop body.

**The patch is done once; it affects all 10 diffusion steps automatically.** Each step reads `kv_cache` at line 265 — this is always the patched cache, never the original corrupt cache.

---

## 8. Recovery Score Definition

```
recovery = (patched_success_rate − corrupt_success_rate) / (clean_success_rate − corrupt_success_rate)
```

Operationally, for a contrastive task pair (bowl task = clean, wine bottle task = corrupt):
- `clean_success_rate`: fraction of N trials where clean-prompt policy completes the bowl task
- `corrupt_success_rate`: fraction of N trials where corrupt-prompt policy (wine bottle prompt) completes the bowl task
- `patched_success_rate`: fraction of N trials where patched policy (corrupt prompt + KV patch) completes the bowl task

N = 50 trials per task (existing setting from `main_corrupt_run_expt.py`).

Unlike SmolVLA where the metric was a continuous action dimension (shoulder_pan mean), here LIBERO provides a binary success signal. Recovery score of 1.0 = patching fully restores clean behavior; 0.0 = patching did nothing.

**Denominator caveat:** If `clean_success_rate ≈ corrupt_success_rate` (model is insensitive to this prompt pair), the recovery score is undefined. The baseline results already verified that the bowl/wine-bottle pair is discriminative on LIBERO-Goal — clean should be substantially higher than corrupt.

---

## 9. End-to-End Walk-Through (Single Patched Trial)

```
1. LIBERO env resets to initial state for bowl task.

2. Control loop begins:
   a. Env returns observation (images + state).
   b. Client sends observation to policy server (or calls infer() directly).

3. Server receives infer() call with corrupt obs (wine bottle prompt).

4. Inside Policy.infer() → Pi0.sample_actions(rng, corrupt_observation, clean_observation=clean_obs):
   a. Preprocess corrupt_observation.
   b. embed_prefix(corrupt_observation) → prefix_tokens (wine bottle prompt).
   c. PaliGemma.llm([prefix_tokens, None], ...) → corrupt_kv_cache.
   d. _apply_kv_patch(corrupt_kv_cache, clean_observation, patch_positions=(591,)):
      - embed_prefix(clean_observation) → prefix_tokens (bowl prompt).
      - PaliGemma.llm([clean_prefix_tokens, None], ...) → donor_kv_cache.
      - K_patched = K_corrupt.at[:, :, 591, :, :].set(K_donor[:, :, 591, :, :])
      - V_patched = V_corrupt.at[:, :, 591, :, :].set(V_donor[:, :, 591, :, :])
      - return (K_patched, V_patched)
   e. while_loop(cond, step, (noise, 1.0)) — 10 diffusion steps, each reading patched kv_cache.
   f. Return action chunk (50 timesteps × 7 dims).

5. Client executes action chunk in LIBERO env (5 steps, then replan).

6. Repeat from step 2 until task done or max_steps reached.

7. Record success/failure.

8. Repeat for N=50 trials, compute patched_success_rate.
```

**Sanity check (analog to SmolVLA's sanity_b):** Before the main eval loop, run one inference call where we patch ALL 788 positions (full prefix overwrite with donor cache). The model should see the clean K/V everywhere and produce behavior indistinguishable from the clean run. If the sanity check fails (behavior is still corrupt-like), the patch mechanism is broken.

---

## 10. Script Architecture (Proposed)

```
examples/libero/
├── main_original.py              (existing — clean run server)
├── main_corrupt_run_expt.py      (existing — corrupt run server)
└── main_patching_expt.py         (NEW — patched run, mirrors main_corrupt_run_expt.py)

src/openpi/models/pi0.py          (minor addition: clean_observation + patch_positions params)
```

`main_patching_expt.py` is structurally identical to `main_corrupt_run_expt.py` with these differences:
- Sends the corrupt prompt to the server (same as corrupt run) — so the env interaction is identical
- Passes `clean_observation` to each `infer()` call (thin wrapper over policy client or direct model)
- Additional args: `--patch_positions 591` (or `591,592` for Option B), `--sanity_check` flag

**Open decision O2 revisited:** The server/client split in π₀.₅ makes the SmolVLA "standalone script" analogy harder. Three sub-options:
- **O2a: Modify `pi0.py` + pass `clean_observation` through server API.** Clean but changes the server request format.
- **O2b: Bypass websocket server in `main_patching_expt.py`** — instantiate model directly (no server), run LIBERO loop + model forward pass in the same process. Loses the client/server separation but is maximally self-contained (closest to SmolVLA's approach).
- **O2c: Run patched server that monkey-patches Pi0 on startup.** New server launch script wraps the model instance.

O2b is closest to SmolVLA. O2a is cleanest architecturally. **Confirm with human before implementing.**

---

## 11. Open Decisions Summary

| ID | Decision | Options | Current Leaning |
|----|----------|---------|-----------------|
| O1 | Baseline and patching script split | Use existing `main_corrupt_run_expt.py` for clean/corrupt baselines; create new `main_patching_expt.py` for patched runs | **Resolved:** one baseline script plus one new patching script |
| O0 | JAX vs PyTorch execution path | Stay on JAX vs convert to PyTorch | **Resolved: JAX (2026-05-04)** |
| O2 | Patching integration approach | P1 (modify `pi0.py`) vs P3 (monkey-patch) vs O2b (bypass server) | P1, but ask human |
| O3 | Which token patch option to start with | A (pos 591 only) vs B (591–592) vs C (length-matched) | Option A |
| O4 | Donor cache per-step or fixed reference | Per-step (same images, different prompt) vs fixed reference | Per-step (more correct, bidirectional attention) |
| O5 | Number of trials per task | 50 (existing default) vs fewer for speed | 50 (keep consistent with prior runs) |
| O6 | Eval scope | Only contrastive pair (bowl/wine-bottle tasks) or full LIBERO-Goal suite | Start with contrastive pair, then broader |

---

## 12. What Is NOT in Scope of This Doc

- Writing the actual `main_patching_expt.py` script
- Modifying `pi0.py`
- **Runtime KVCache shape verification (first thing to do before writing indexing code):** The type alias and `nn.scan` behavior strongly suggest `(18, 1, 788, 1, 256)`, but `nn.scan`'s exact output axis ordering and its handling of `kv_cache=None` on the prefix pass were inferred statically. Before writing `K.at[:, :, 591, :, :]`, add a one-line debug print in `sample_actions` and confirm the shape: `print(jax.tree_util.tree_map(lambda x: x.shape, kv_cache))`.
- Deciding the order of implementation steps (RunPod session management)
- Visualization / analysis of results (separate from the patching run itself)

---

## 13. Big-Picture Questions Came Up by the Human

### Does KV-cache patching even make sense as opposed to residual stream?

For π₀.₅, the current proposal is to patch **K**, **V**, or **K+V** in the prefix cache, instead of patching the residual stream like we did in SmolVLA.

Conceptually, this still makes sense. We are **not changing weights**. `W_K` and `W_V` remain fixed. We are changing an **activation produced during a forward pass**: the cached key/value vector at a specific layer/token. That is analogous in spirit to residual-stream activation patching: run A, save an internal activation; run B, replace B's activation with A's; observe whether behavior changes. The activation just happens to live in a different coordinate system: residual stream vs attention key/value cache.

K/V are "after applying W" to the hidden state. More precisely, at each layer:

```text
residual / hidden state h
  -> layer norm / attention input
  -> W_Q, W_K, W_V projections
  -> Q, K, V
  -> attention(Q, K, V)
  -> attention output
  -> residual update
```

So K/V are not weights, and they are not "before activation" in the usual causal-tracing sense. They are intermediate activations inside the attention computation. Patching them asks a more specific question than residual patching. Residual patching asks: "If this token's whole representation were clean, does behavior recover?" K/V patching asks: "If this token presented clean lookup/address/value information to later attention, does behavior recover?"

There is not one "fundamentally correct" patch target. The target depends on the architecture and the causal question. In a standard LLM where we can rerun the whole sequence and access residual streams easily, residual-stream patching is often the default because it captures the token's full representation at a layer. But in π₀.₅, the prefix is explicitly cached and then consumed by the suffix/action decoder. That makes the KV cache a natural intervention point: the action suffix literally reads the prompt/image prefix through cached K/V.

For ActPatch / activation patching in LLMs generally, people patch many kinds of internal activations: residual stream, attention outputs, MLP outputs, individual heads, sometimes Q/K/V or attention patterns. Residual stream is common because it is broad and interpretable as "the representation at this token/layer." K/V patching is narrower: it intervenes on what other tokens can retrieve from that position via attention. That narrowness can be good or bad. Good because it aligns with π₀.₅'s cached-prefix mechanism. Bad because if the relevant information flows through some other channel, K/V patching may understate the causal role of the token.

So we should not treat "KV patching" as automatically equivalent to SmolVLA residual patching. It is a principled activation patch, but it answers a slightly different question. For the paper, that distinction is important. The claim should be something like: **the object-word information available to the action decoder through the cached prefix attention interface is causally important**, not simply "the residual representation at the object token is causally important."

---

## 14. Supplementary

### 14.1 Runtime call trace: how `sample_actions()` is reached

In the normal LIBERO server setup, the evaluation script does not call `Pi0.sample_actions()` directly. It asks the policy server for a new action chunk whenever the local `action_plan` queue is empty.

Concrete path for `main_corrupt_run_expt.py`:

```text
examples/libero/main_corrupt_run_expt.py
  -> eval_libero(args)                                      lines 56-238
  -> client = WebsocketClientPolicy(args.host, args.port)   line 81
  -> env.reset(); obs = env.set_init_state(...)             lines 108-112
  -> while rollout is running                               line 119
  -> if not action_plan: build `element` dict                lines 142-156
       observation/image
       observation/wrist_image
       observation/state
       prompt
  -> action_chunk = client.infer(element)["actions"]        line 159
  -> action_plan.extend(action_chunk[:args.replan_steps])   line 163
  -> env.step(action.tolist())                              line 168
```

Client/server hop:

```text
packages/openpi-client/src/openpi_client/websocket_client_policy.py
  -> WebsocketClientPolicy.infer(obs)                       lines 47-54
  -> msgpack-encode obs, send over websocket, wait for reply

src/openpi/serving/websocket_policy_server.py
  -> WebsocketPolicyServer._handler(...)                    lines 48-83
  -> obs = msgpack_numpy.unpackb(await websocket.recv())    line 58
  -> action = self._policy.infer(obs)                       line 61
  -> send packed action dict back to client                 line 71
```

Policy/model hop:

```text
src/openpi/policies/policy.py
  -> Policy.__init__ stores self._sample_actions            lines 61-65
     JAX path: self._sample_actions = nnx_utils.module_jit(model.sample_actions)
  -> Policy.infer(obs)                                      lines 68-106
  -> apply input transforms                                 lines 69-74
  -> observation = Observation.from_dict(inputs)            line 90
  -> self._sample_actions(rng, observation, **kwargs)       line 94

src/openpi/models/pi0.py
  -> Pi0.sample_actions(...)                                lines 217-279
  -> preprocess observation                                 line 225
  -> build prefix KV cache                                  lines 233-237
  -> patch hook goes here                                   after line 237
  -> diffusion loop reads patched kv_cache                  lines 239-278
```

For `main_patching_expt.py`, the outer LIBERO loop should mirror `main_corrupt_run_expt.py`. The main implementation decision is whether the patched path still goes through the websocket server (`client.infer(...) -> server -> Policy.infer(...) -> sample_actions(...)`) or bypasses it and calls the policy/model directly in-process. Either way, the model-level intervention point remains the same: immediately after the prefix KV cache is constructed in `Pi0.sample_actions()`.

---

### 14.2 JAX primer: JIT, immutable arrays, and while_loop

This section explains the three JAX concepts that appear in the patching code in plain terms. You do not need to understand JAX internals to implement the patch — this is just context so the syntax isn't surprising.

**`jax.jit` and recompilation**

`jax.jit` (and `nnx_utils.module_jit` which wraps it) compiles a Python function to an optimized XLA kernel the first time it is called. "Compiling" here means JAX traces through the function once using abstract placeholder values — it records what operations happen in what order, then hands the resulting computation graph to XLA (Google's tensor compiler), which produces native GPU/TPU code. The compiled kernel is cached by input signature (shapes + dtypes + pytree structure).

The practical consequence: the first call to a jitted function is slow (compilation), every subsequent call with the same input signature is fast (runs the cached kernel, no Python overhead). If input shapes change, JAX recompiles — once for the new shape, then caches again.

For patching: when we add `clean_observation=None` to `sample_actions` and call it with an actual `Observation` object for the first time, JAX sees a new pytree structure (the kwarg is now non-None) and recompiles. This recompile takes ~30–60 seconds on first patched call. After that, all subsequent patched calls use the cached compilation. You don't do anything to trigger or manage this — it happens automatically and silently. The only observable effect is a one-time pause on the first patched rollout.

**Immutable arrays and `.at[...].set(...)`**

JAX arrays are immutable by design — you cannot do `K[0, 0, 591, 0, :] = new_values` in-place. This is intentional: immutability lets JAX's tracing and automatic differentiation work correctly without tracking side effects. Instead, JAX provides a functional update syntax:

```python
K_patched = K.at[:, :, 591, :, :].set(new_values)
```

This returns a **new array** that is identical to `K` except at the specified indices, which are replaced by `new_values`. The original `K` is unchanged. Inside a JIT-compiled function, XLA typically optimizes this to an in-place buffer update under the hood, so there is no performance penalty compared to PyTorch's in-place mutation.

The syntax `arr.at[idx].set(val)` is the standard JAX pattern for "copy this array but with these positions replaced." You'll also see `.at[idx].add(val)`, `.at[idx].mul(val)`, etc. for other operations. For the KV cache patch, `.set(...)` is all we need.

**`jax.lax.while_loop`**

JAX's JIT traces Python code statically. A Python `for i in range(10)` loop is fine — JAX unrolls it at trace time. But a `while` loop whose exit condition depends on a runtime tensor value (like `while time >= threshold`) cannot be unrolled statically. `jax.lax.while_loop(cond_fn, body_fn, init_val)` is the JAX primitive for this: it compiles to an XLA while loop that runs entirely in compiled code, not Python.

In `pi0.py:sample_actions`, the diffusion denoising loop uses `jax.lax.while_loop` because the number of steps could depend on a runtime value. The `step` function (the loop body) closes over `kv_cache` — because Python closures are late-binding (they look up the variable at call time, not at definition time), reassigning `kv_cache` to the patched version before `def step` is evaluated means the loop body will use the patched cache. This is the core of why the patch point works the way it does.

**What you don't need to know**

You don't need to understand XLA, automatic differentiation, `jax.vmap`, `jax.grad`, custom_vjp, or any of the other JAX machinery. For the patching implementation, you are writing ordinary Python that happens to use JAX arrays. The JIT is transparent. The `.at[...].set(...)` replaces in-place mutation. The `while_loop` is just a loop. Everything else in `sample_actions` is standard Python/numpy-style code.

---

### 14.3 JAX vs PyTorch: full analysis (reference only)

**Decision: JAX (resolved 2026-05-04). This section is reference only — no action needed.**

**Why a PyTorch path exists.** `src/openpi/models_pytorch/pi0_pytorch.py` implements the same π₀.₅ model in PyTorch. It uses the HuggingFace `past_key_values` convention (standard `DynamicCache`-style tuple of per-layer `(K, V)`) and supports PyTorch forward hooks — the same mechanism used in SmolVLA. The upstream GCS checkpoint is JAX-only; the openpi project provides `examples/convert_jax_model_to_pytorch.py` for users who want PyTorch. (Community HuggingFace conversions may exist but would need apples-to-apples OpenPI server compatibility verification.)

**How conversion works.** `convert_jax_model_to_pytorch.py` maps JAX orbax params to PyTorch state dict (handling pi05's adaptive normalization layers via string-matching heuristics), then calls `load_state_dict(strict=False)` and saves as `model.safetensors`. `strict=False` means unmatched keys are silently left at random-init values — conversion correctness cannot be assumed without a baseline re-run.

**Conversion command (for reference):**
```bash
# Note: maybe_download resolves gs:// under $OPENPI_DATA_HOME/openpi-assets/...
# Actual local path: /workspace/openpi_assets/openpi-assets/checkpoints/pi05_libero
uv run python examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /workspace/openpi_assets/openpi-assets/checkpoints/pi05_libero \
    --config_name pi05_libero \
    --output_path /workspace/openpi_assets/openpi-assets/checkpoints/pi05_libero_pytorch \
    --precision bfloat16
```

**PyTorch patching would look like this** (hook point: `pi0_pytorch.py:sample_actions` after line 400, before the `while` loop):
```python
# past_key_values is DynamicCache; verify type(past_key_values) on RunPod first
for layer_idx in range(18):
    past_key_values.key_cache[layer_idx][:, :, 591, :] = donor_kv.key_cache[layer_idx][:, :, 591, :]
    past_key_values.value_cache[layer_idx][:, :, 591, :] = donor_kv.value_cache[layer_idx][:, :, 591, :]
```

**Key PyTorch-specific caveats that tipped the decision to JAX:**

1. **`torch.compile = "max-autotune"` is the default.** `Pi0Config.pytorch_compile_mode` defaults to `"max-autotune"` and `PI0Pytorch.__init__` overwrites `self.sample_actions` with a compiled version at instantiation (`pi0_pytorch.py:112–113`). For patching, must explicitly set `pytorch_compile_mode=None` — otherwise first patched call triggers a 10–15 min recompile.

2. **`transformers_replace` must be manually installed per environment** (`pi0_pytorch.py:118–125`):
   ```bash
   cp -r ./src/openpi/models_pytorch/transformers_replace/* \
       .venv/lib/python3.11/site-packages/transformers/
   ```

3. **Baseline re-run required.** `strict=False` loading means conversion correctness is not guaranteed. Must re-run clean/corrupt LIBERO-Goal baselines with the PyTorch model before trusting patching results.

**Tradeoff summary:**

| | JAX (chosen) | PyTorch |
|--|--------------|---------|
| Setup | Zero | Conversion + baseline re-run + `transformers_replace` install |
| Patching | `.at[...].set(...)` — 2 lines | Direct tensor mutation — familiar from SmolVLA |
| Compile friction | One silent recompile on first patched call | Must disable `max-autotune` for dev |
| Baseline consistency | All prior baselines are JAX | Requires re-run |
