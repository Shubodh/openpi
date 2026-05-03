# Patching Implementation Dry-Run Doc

**Status:** Draft — for human review before coding begins.

**Scope:** Complete implementation plan for KV-cache patching on π₀.₅ + LIBERO-Goal. Covers architecture, code locations, indexing arithmetic, and all open decisions that need resolution before a line of implementation code is written.

**Reference documents:**
- `status_cc/kv_cache_findings.md` — authoritative source for shapes, positions, hook point
- `status_cc/misc/kv_cache_primer.md` — conceptual background
- SmolVLA patching scripts: `AXMech/scripts/run_patching.py` and `cache_activations.py`

---

## 1. High-Level Architecture

### What we're building

Three run types, same as SmolVLA:

| Run | Prompt | KV cache | Purpose |
|-----|--------|----------|---------|
| **Clean** | `"put the bowl on top of the cabinet"` | built normally | establishes success ceiling |
| **Corrupt** | `"put the wine bottle on top of the cabinet"` | built normally | establishes failure floor |
| **Patched** | `"put the wine bottle on top of the cabinet"` | positions 591–592 overwritten with clean-run values | tests whether language signal is causal |

Recovery score (per-task, aggregated over trials):
```
recovery = (patched_success_rate − corrupt_success_rate) / (clean_success_rate − corrupt_success_rate)
```
- 0 = patch had no effect
- 1 = patch fully restores clean behavior
- Note: unlike SmolVLA (continuous action score), here the metric is binary task success (LIBERO provides `done` flag per episode). Recovery score is computed from success rates aggregated over the N trials per task.

### One script or three invocations?

SmolVLA's approach: one script that runs clean and corrupt baselines at startup, then runs the patching loop (all three "run types" in a single process).

π₀.₅ difference: LIBERO evaluation is an interactive sim — we can't do a "forward pass" in isolation the way SmolVLA could with a fixed frame. We run full episodes with environment steps. Therefore:

**Proposed: three separate invocations (three scripts or one with a `--mode` flag):**
1. `main_original.py` (already exists) → clean run
2. `main_corrupt_run_expt.py` (already exists) → corrupt run
3. `main_patching_expt.py` (new) → patched run

This mirrors the existing pattern in `examples/libero/` and avoids making any one script too complex. The patching logic is self-contained in script 3.

**Open decision O1:** Single script with `--mode clean|corrupt|patched` vs. separate files. Current leaning: separate file for patching, since clean and corrupt already exist. Confirm with human.

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

The prefix KV cache is computed once per `sample_actions` call and reused read-only across all diffusion denoising steps (`kv_cache=kv_cache` at line 265 inside `step()`). Patching before `step` is defined propagates to all denoising steps automatically — no per-step intervention needed.

**Important caveat (from Codex verification):** The cache is rebuilt on every `Policy.infer()` call (`policy.py:94`), which is called per action chunk request from the LIBERO client. Patching must therefore happen on every `sample_actions` call, not once globally per episode.

---

## 3. Implementation Strategy: How to Patch in JAX

### Why SmolVLA's hook approach doesn't apply

SmolVLA used PyTorch `register_forward_pre_hook` to intercept the residual stream mid-forward-pass. JAX has no equivalent hook mechanism — functions are pure and traced/compiled.

### The JAX way: functional array update

JAX KV caches are immutable arrays. The patch is a pure functional update using `.at[...].set(...)`:

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
| O1 | One script with `--mode` vs separate scripts | Separate (clean/corrupt exist) vs merged | Separate (new `main_patching_expt.py`) |
| O2 | Patching integration approach | P1 (modify pi0.py) vs P3 (monkey-patch) vs O2b (bypass server) | P1, but ask human |
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
