# Patching Implementation Dry-Run Doc

**Status:** All open decisions resolved (2026-05-04) — ready for implementation.

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
8. [Results Metric](#8-results-metric)
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
   - [14.4 Phase 2 task pair: bowl vs wine-bottle (deferred)](#144-phase-2-task-pair-bowl-vs-wine-bottle-deferred)
   - [14.5 Recovery score (deferred — for heatmap analysis)](#145-recovery-score-deferred--for-heatmap-analysis)
   - [14.6 Per-step donor cache (deferred — for image-dependent patching)](#146-per-step-donor-cache-deferred--for-image-dependent-patching)


---

## 1. High-Level Architecture

### What we're building

Three run types, same as SmolVLA:

**Initial task pair (Phase 1):** `put_the_bowl_on_the_plate` vs `put_the_bowl_on_the_stove`. Same object (`bowl`), different destination (`plate` vs `stove`). Same prompt length — no token-count mismatch. Patch target: destination token (`plate`/`stove`), expected absolute index 594. Start here to get a working implementation and first results.

**Phase 2 (revisit after Phase 1 works):** `put the bowl on top of the cabinet` vs `put the wine bottle on top of the cabinet`. Different object token (`bowl` vs `wine`), token-count mismatch (1 vs 2 tokens). More complex; token positions verified in `kv_cache_findings.md`. See §14.4 for the full Phase 2 run table and token analysis.

| Run | Prompt | KV cache | Purpose |
|-----|--------|----------|---------|
| **Clean** | `"put the bowl on the plate"` | built normally | establishes success ceiling |
| **Corrupt** | `"put the bowl on the stove"` | built normally | establishes failure floor |
| **Patched** | `"put the bowl on the stove"` | position 594 overwritten with clean-run value | tests whether destination-language signal is causal |

**Primary output:** clean rollout success rate, corrupt rollout success rate, patched rollout success rate — reported directly, no normalisation needed for Phase 1. A recovery score formula exists for later analysis; see §14.5.

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

**O2 resolved: P1** — modify `pi0.py` directly (add `clean_observation` param + `_apply_kv_patch` method). See §10 for architecture details.

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

**Why language K/V depends on image content:** The KV cache is rebuilt at every `sample_actions` call from the current observation. Each language token's K/V is shaped by the scene's visual content — at minimum because image tokens precede language tokens in the prefix and are attended to (this holds even with purely causal attention). π₀.₅ additionally uses fully bidirectional (non-causal) attention in the prefix: `embed_prefix` sets `ar_mask = [False] * n` for all tokens (`pi0.py:125–133`), so `make_attn_mask` produces an all-True mask and every token attends to every other. This reinforces the image dependency further (image tokens also attend to language tokens, making the interaction richer). See `status_cc/kv_cache_findings.md` §4 and the primer in §14.2 for the original discussion of these terms. The practical consequence: a donor cache harvested from a different timestep has language K/V that reflects the scene at *that* timestep, not the current one.

**O4 resolved: pre-computed donor (2026-05-04).** Harvest the donor KV cache once before the rollout starts (using the initial observation + clean prompt), then reuse the same tensor for every inference call throughout the episode. The scene objects (plate, stove) don't move — only the robot arm does — so the language K/V for the destination token is stable enough that a t=0 donor is a valid stand-in for all subsequent timesteps. This is simpler and requires no extra forward pass during rollout.

```python
# In main_patching_expt.py, before the rollout loop:
clean_obs = make_obs_dict(initial_env_obs, clean_prompt)
clean_obs_jax = {k: jnp.asarray(v)[np.newaxis] for k, v in clean_obs.items()}
clean_observation = Observation.from_dict(clean_obs_jax)
# Run one clean prefix pass to get the donor cache:
_, donor_kv_cache = model.PaliGemma.llm(
    [clean_prefix_tokens, None], mask=clean_prefix_attn_mask, positions=clean_positions
)
# donor_kv_cache is now a fixed (K, V) tuple — pass it into sample_actions for every call
policy._sample_kwargs["donor_kv_cache"] = donor_kv_cache
```

For the per-step donor approach (rebuilds donor each call with current images — needed if image-dependent patching matters), see §14.6.

### Proposed implementation approach

**Approach P1 (resolved 2026-05-04): minimal modification to `pi0.py`**

Add two optional parameters to `Pi0.sample_actions`:

```python
def sample_actions(
    self,
    rng,
    observation,
    *,
    num_steps=10,
    noise=None,
    donor_kv_cache=None,       # NEW: pre-computed clean KV cache; if set, triggers patching
    patch_positions=(594,),    # NEW: absolute KV cache positions to overwrite
):
    ...
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], ...)

    if donor_kv_cache is not None:
        kv_cache = self._apply_kv_patch(kv_cache, donor_kv_cache, patch_positions)

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0

def _apply_kv_patch(self, corrupt_kv_cache, donor_kv_cache, patch_positions):
    K_d, V_d = donor_kv_cache
    K, V = corrupt_kv_cache
    for pos in patch_positions:
        K = K.at[:, :, pos, :, :].set(K_d[:, :, pos, :, :])
        V = V.at[:, :, pos, :, :].set(V_d[:, :, pos, :, :])
    return (K, V)
```

`donor_kv_cache` is a fixed JAX array tuple harvested once before the rollout. Pass it as a static `sample_kwargs` entry — no per-call mutation needed:

```python
# In main_patching_expt.py, before the rollout loop:
# (build clean prefix tokens from initial_obs + clean_prompt, then:)
_, donor_kv_cache = model.PaliGemma.llm(
    [clean_prefix_tokens, None], mask=clean_prefix_attn_mask, positions=clean_positions
)
policy._sample_kwargs["donor_kv_cache"] = donor_kv_cache
# Now every policy.infer() call will use the same donor cache — no per-call injection needed
```

**JAX JIT note:** `module_jit` wraps `sample_actions` with `jax.jit`. Adding `donor_kv_cache` as a new kwarg triggers a one-time recompilation on the first patched call. Subsequent calls use the cached compilation. Since `donor_kv_cache` is a fixed array (same shape every call), there is no further recompile after the first.

For the per-step donor variant (rebuilds donor each call with current images), see §14.6.

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

**O2 resolved: P1 (2026-05-04).** P3 is kept above for reference — it avoids touching `pi0.py` but requires duplicating `sample_actions`, which is fragile if upstream changes.

---

## 4. Token Positions and Patch Options

**Phase 1 pair** (`put the bowl on the plate` / `put the bowl on the stove`):

Both prompts are the same length (6 words + BOS = 7 tokens). The only differing token is the destination word at local index 6. Language region starts at absolute index 588, so:

| Token | Local index | Expected absolute index | Notes |
|-------|-------------|------------------------|-------|
| BOS | 0 | 588 | always present |
| `put` | 1 | 589 | same in both |
| `the` | 2 | 590 | same in both |
| `bowl` | 3 | 591 | same in both — object unchanged |
| `on` | 4 | 592 | same in both |
| `the` | 5 | 593 | same in both |
| `plate` / `stove` | 6 | **594** | differing token — patch target |

**⚠ Token positions not yet RunPod-verified for this pair.** Verify with `inspect_kv_cache.py` before writing indexing code (same script used for bowl/wine-bottle in §14.4).

### Patch options

**O3 resolved: Option A — patch only position 594 (2026-05-04).**
- Overwrite `kv_cache[:, :, 594, :, :]` from donor (`plate` run) into recipient (`stove` run).
- No token-count mismatch — both prompts have identical length, so position 594 corresponds exactly in both runs.
- RoPE position at absolute index 594: `cumsum = 392 + 7 = 399`, RoPE position = 398 — identical in both runs since image token count is the same. Clean semantic patch. ✅

**Revisit O3 for Phase 2** (bowl vs wine-bottle pair): token-count mismatch re-introduces the Option A/B/C tradeoff. See §14.4 for the full analysis.

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

Patch target positions (absolute) — **Phase 1**:
- `plate` / `stove`: expected index **594** (= 588 + 6) — ⚠ verify on RunPod

For reference, Phase 2 positions (RunPod-verified, see §14.4):
- `bowl`: 591, `wine`: 591, `bottle`: 592

**MQA note:** `num_kv_heads = 1` means all 8 query heads share a single K and V tensor. The patch operation on the `(18, 1, 788, 1, 256)` tensor is directly the K (or V) for all query heads — no per-head indexing needed.

**JAX indexing for Phase 1 Option A patch:**
```python
# K, V each: (18, 1, 788, 1, 256)
# Patch all 18 layers, batch 0, position 594, all kv_heads (1), all head dims (256)
K_patched = K_corrupt.at[:, :, 594, :, :].set(K_donor[:, :, 594, :, :])
V_patched = V_corrupt.at[:, :, 594, :, :].set(V_donor[:, :, 594, :, :])
```

---

## 6. Token-Count Mismatch Handling

**Phase 1 pair: no mismatch.** Both `"put the bowl on the plate"` and `"put the bowl on the stove"` are 7 tokens (with BOS). Every absolute position is identical between clean and corrupt runs. The patch at position 594 is a direct like-for-like swap with no alignment concerns.

**RoPE position consistency:** Language token at local index 6 (absolute 594) gets `cumsum = 392 + 7 = 399`, RoPE position = 398 — identical in both runs. Only the token identity (`plate` vs `stove`) differs. ✅

**Padding:** Both prompts are padded to `max_token_len=200`. Positions 595–787 are padding, masked out, and unaffected.

**Phase 2 mismatch (bowl vs wine-bottle) — deferred to §14.4.** The clean prompt has 9 tokens and corrupt has 10; all positions after the object name shift by 1. The full mismatch analysis and RoPE consistency proof for that pair are in §14.4.

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

## 8. Results Metric

**For Phase 1, report the three raw success rates directly:**
- `clean_success_rate`: fraction of N=25 trials where clean-prompt policy places bowl on the plate
- `corrupt_success_rate`: fraction of N=25 trials where corrupt-prompt policy (stove prompt) places bowl on the plate
- `patched_success_rate`: fraction of N=25 trials where patched policy (stove prompt + KV patch at 594) places bowl on the plate

No normalisation needed at this stage. A recovery score formula for later heatmap-style analysis is in §14.5.

---

## 9. End-to-End Walk-Through (Single Patched Trial)

```
1. LIBERO env resets to initial state for the `put_the_bowl_on_the_plate` task.

2. Control loop begins:
   a. Env returns observation (images + state).
   b. Client sends observation to policy server (or calls infer() directly).

3. Server receives infer() call with corrupt obs (`"put the bowl on the stove"` prompt).

4. Inside Policy.infer() → Pi0.sample_actions(rng, corrupt_observation, clean_observation=clean_obs):
   a. Preprocess corrupt_observation.
   b. embed_prefix(corrupt_observation) → prefix_tokens (stove prompt).
   c. PaliGemma.llm([prefix_tokens, None], ...) → corrupt_kv_cache.
   d. _apply_kv_patch(corrupt_kv_cache, clean_observation, patch_positions=(594,)):
      - embed_prefix(clean_observation) → prefix_tokens (plate prompt).
      - PaliGemma.llm([clean_prefix_tokens, None], ...) → donor_kv_cache.
      - K_patched = K_corrupt.at[:, :, 594, :, :].set(K_donor[:, :, 594, :, :])
      - V_patched = V_corrupt.at[:, :, 594, :, :].set(V_donor[:, :, 594, :, :])
      - return (K_patched, V_patched)
   e. while_loop(cond, step, (noise, 1.0)) — 10 diffusion steps, each reading patched kv_cache.
   f. Return action chunk (50 timesteps × 7 dims).

5. Client executes action chunk in LIBERO env (5 steps, then replan).

6. Repeat from step 2 until task done or max_steps reached.

7. Record success/failure.

8. Repeat for N=25 trials, compute patched_success_rate.
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
| O2 | Patching integration approach | P1 (modify `pi0.py`) vs P3 (monkey-patch) vs O2b (bypass websocket server entirely) | **Resolved: P1 (2026-05-04)** |
| O3 | Which token patch option to start with | A (single differing position) vs B (multi-position) vs C (length-matched donor) | **Resolved: Option A, pos 594 for Phase 1 (2026-05-04). Revisit for Phase 2 (bowl/wine-bottle) — see §14.4.** |
| O4 | Donor cache per-step or fixed reference | Per-step (rebuild donor each call) vs pre-computed (harvest once before rollout) | **Resolved: pre-computed (2026-05-04) — scene objects are static; see §3 "Where does the donor KV cache come from?" and §14.6 for per-step reference** |
| O5 | Number of trials per task | 50 (default in `main_corrupt_run_expt.py`) vs 25 | **Resolved: 25 (2026-05-04) — consistent with all prior bash scripts** |
| O6 | Eval scope | Only contrastive pair or full LIBERO-Goal suite | **Resolved: contrastive pair only for Phase 1 (plate/stove), broaden after (2026-05-04)** |

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

---

### 14.4 Phase 2 task pair: bowl vs wine-bottle (deferred)

**Status: deferred — implement after Phase 1 (plate/stove) is working.**

**Task pair:**
| Run | Prompt | Purpose |
|-----|--------|---------|
| **Clean** | `"put the bowl on top of the cabinet"` | establishes success ceiling |
| **Corrupt** | `"put the wine bottle on top of the cabinet"` | establishes failure floor |
| **Patched** | `"put the wine bottle on top of the cabinet"` + KV patch | tests whether object-name signal is causal |

**Token positions (RunPod-verified, 2026-04-29):**

| Token | Absolute KV cache index | Token id |
|-------|------------------------|----------|
| `bowl` | 591 | 14581 |
| `wine` (in "wine bottle") | 591 | 10058 |
| `bottle` | 592 | 12989 |

Language region starts at absolute index 588. Object-name token at local 3 → absolute 591.

**Token-count mismatch:** clean prompt = 9 tokens (with BOS), corrupt = 10 tokens. Positions 592–787 shift by 1 between runs. Full analysis below.

**Option A (recommended start for Phase 2): patch only position 591.**
Overwrite `kv_cache[:, :, 591, :, :]` — `bowl` K/V replaces `wine` K/V. The `bottle` K/V at 592 is untouched. Minimal intervention; easiest to interpret.

**RoPE position consistency for Option A:** Language token at local 3 (absolute 591) gets `cumsum = 392 + 4 = 396`, RoPE position = 395 — identical in both clean and corrupt runs (same image token count). Only the token identity differs. Clean semantic patch. ✅

**Option B: patch positions 591 and 592.**
At position 592 in the donor (clean) cache, the token is `on` — transplanting `on` K/V into the `bottle` slot. More aggressive, harder to interpret.

**Option C: use a length-matched donor prompt.**
Use `"put the wine bowl on top of the cabinet"` as the clean reference (2-token object span) to produce a matched donor for positions 591–592. Cleanest semantically, requires an additional reference run.

**Revisit O3 for Phase 2:** start with Option A; escalate to B or C if results are ambiguous.

**Results metric for Phase 2:** same three raw success rates as Phase 1 (clean / corrupt / patched). See §14.5 for recovery score if normalised summary is needed.

**JAX indexing for Phase 2 Option A (patch target: position 591):**
```python
K_patched = K_corrupt.at[:, :, 591, :, :].set(K_donor[:, :, 591, :, :])
V_patched = V_corrupt.at[:, :, 591, :, :].set(V_donor[:, :, 591, :, :])
```

---

### 14.5 Recovery score (deferred — for heatmap analysis)

**Not needed for Phase 1 or 2 initial runs. Revisit when doing systematic patching across positions or layers.**

The recovery score normalises the patching result against the clean/corrupt gap:

```
recovery = (patched_success_rate − corrupt_success_rate) / (clean_success_rate − corrupt_success_rate)
```

- 0 = patch had no effect
- 1 = patch fully restores clean behavior
- Unlike SmolVLA (continuous shoulder_pan mean), here LIBERO gives a binary `done` flag, so all three rates are success fractions over N=25 trials.

**Why it's deferred:** The formula is only meaningful if `clean_success_rate − corrupt_success_rate` is substantially nonzero. If both are near 100% (task visually obvious) or both near 0%, the denominator is tiny and the score is noise. Confirm the pair is discriminative from raw rates first.

**For heatmap-style analysis** (patching individual layers, individual positions, K vs V separately): a normalised recovery score is useful as a compact per-cell summary. At that stage, also consider saving the final rollout frame to disk and using an offline VLM judge rather than relying solely on the simulator `done` flag — this allows offline reanalysis without re-running rollouts.

---

### 14.6 Per-step donor cache (deferred — for image-dependent patching)

**Not needed for Phase 1. Revisit if scene dynamics matter (moving objects, manipulation mid-reach, or when language K/V image-dependency is suspected to affect results).**

The per-step approach rebuilds the donor KV cache on every `sample_actions` call using the *current* observation's images and the clean prompt. This ensures donor and recipient differ only in language, not in visual context — the cleanest possible causal intervention.

**Original P1 pseudocode for per-step donor** (inside `_apply_kv_patch`, called each inference):

```python
def _apply_kv_patch(self, corrupt_kv_cache, clean_observation, patch_positions):
    """Per-step: build donor from clean_observation (current images + clean prompt)."""
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

**Injection mechanism for per-step:** `clean_observation` changes each call (current images + clean prompt), so it cannot be a static `sample_kwargs` entry. Instead, mutate `policy._sample_kwargs["clean_observation"]` before each `infer()` call — `Policy.infer()` rebuilds `sample_kwargs = dict(self._sample_kwargs)` on every call (`policy.py:82`), so the mutation is picked up:

```python
# In main_patching_expt.py, inside the action-chunk request loop:
clean_obs_dict = make_obs_dict(env_obs, clean_prompt)
clean_obs_jax = {k: jnp.asarray(v)[np.newaxis] for k, v in clean_obs_dict.items()}
policy._sample_kwargs["clean_observation"] = Observation.from_dict(clean_obs_jax)
action_chunk = client.infer(corrupt_obs_dict)["actions"]
```

**Cost:** doubles the prefix forward pass per action chunk request (one corrupt + one clean). The prefix pass is fast relative to diffusion steps, so overhead should be acceptable — confirm on RunPod.

**Why language K/V is image-dependent** (recap from §3): π₀.₅ uses bidirectional prefix attention — every token (including language) attends to every other valid token. Even under causal attention, language tokens at positions 588+ attend to all preceding image tokens (0–587). Either way, the language K/V at any position encodes information from the current visual context. See `status_cc/kv_cache_findings.md` §4 for the original discussion.
