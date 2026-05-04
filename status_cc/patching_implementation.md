# Patching Implementation

**Status:** In progress — begun 2026-05-04.

**Parent plan:** `status_cc/patching_implementation_dryrun.md` — all O0–O6 decisions resolved, full design rationale.

This document is the working reference for the actual implementation. It is more granular than the dry-run doc — it contains verified source-code facts, exact code to write, and a per-step checklist to fill in as work proceeds. Read this doc, not the dry-run, while writing code.

---

## Table of Contents

1. [Verified Source Facts](#1-verified-source-facts)
2. [Architecture Decisions (Final)](#2-architecture-decisions-final)
3. [Changes to `pi0.py`](#3-changes-to-pi0py)
4. [`main_patching_expt.py` Structure](#4-main_patching_exptpy-structure)
5. [Implementation Checklist](#5-implementation-checklist)
6. [RunPod Verification Steps](#6-runpod-verification-steps)
7. [Expected Outcomes and Interpretation Guide](#7-expected-outcomes-and-interpretation-guide)
8. [Results](#8-results)

---

## 1. Verified Source Facts

All claims below verified against source on 2026-05-04.

### 1.1 Hook point — `pi0.py`

File: `src/openpi/models/pi0.py`

```
line 217:  def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
line 225:  observation = _model.preprocess_observation(None, observation, train=False)
line 234:  prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
line 235:  prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
line 236:  positions = jnp.cumsum(prefix_mask, axis=1) - 1
line 237:  _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
             ← PATCH GOES HERE (between 237 and 239)
line 239:  def step(carry):
line 261:      (prefix_out, suffix_out), _ = self.PaliGemma.llm(
line 265:          kv_cache=kv_cache,       ← reads (patched) prefix cache
line 278:  x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
line 279:  return x_0
```

`step` is a Python closure over `kv_cache`. Python closures are late-binding: they look up `kv_cache` at call time (line 278), not at definition time (line 239). Reassigning `kv_cache` between lines 237 and 278 is sufficient — the patch does not need to precede `def step`. Inserting before `def step` (line 239) is cleaner and preferred.

### 1.2 KVCache shape — `gemma.py`

```python
# gemma.py:336
KVCache: TypeAlias = tuple[
    at.Float[at.Array, "l b _t _k _h"],   # K
    at.Float[at.Array, "l b _t _v _h"],   # V
]
```

Dimensions for `pi05_libero` (gemma_2b, `pi05=True`):

| dim | name | value | source |
|-----|------|-------|--------|
| `l` | layers | 18 | `gemma_2b.depth = 18` (gemma.py:81) |
| `b` | batch | 1 | single-episode rollout |
| `_t` | prefix seq len | 788 | 3×196 images + 200 language tokens |
| `_k` / `_v` | num_kv_heads | 1 | `gemma_2b.num_kv_heads = 1` (gemma.py:83) |
| `_h` | head_dim | 256 | `gemma_2b.head_dim = 256` (gemma.py:84) |

Full shape per K or V: `(18, 1, 788, 1, 256)`.

**How `nn.scan` produces this shape:** `Module.setup()` defines `self.layers = nn.scan(block_cls, in_axes=(0, ...), length=depth)` (gemma.py:365–381). `Block.__call__` returns `(xs, kv_cache)` where `kv_cache = (k, v)` from `Attention`. `nn.scan` stacks the per-layer `(k, v)` tuples into `(L, B, S, K, H)`. For the prefix pass (`kv_cache=None` input), each layer's output is `(1, 788, 1, 256)` → stacked to `(18, 1, 788, 1, 256)`. ✅

**How suffix pass uses the cache:** `nn.scan` with `in_axes=0` slices the full `(18,1,788,1,256)` tensor at axis 0 per layer, giving each `Block` a `(1,788,1,256)` slice. Inside `Attention.__call__` (gemma.py:211–214):
```python
if kv_cache is not None:
    cache_k, cache_v = kv_cache            # per-layer: (1, 788, 1, 256)
    k = jnp.concatenate([cache_k, k], axis=1)  # (1, 788+suffix_len, 1, 256)
    v = jnp.concatenate([cache_v, v], axis=1)
```
Patching `K[:, :, 594, :, :]` on the full `(18,1,788,1,256)` tensor propagates to all 18 layers. ✅

### 1.3 Prefix layout (verified from `pi0.py` + `pi0_config.py`)

`embed_prefix` iterates `obs.images` in insertion order. `pi0_config.py:69–77` defines:
```python
images={
    "base_0_rgb":       image_spec,   # agentview → tokens 0–195
    "left_wrist_0_rgb": image_spec,   # wrist     → tokens 196–391
    "right_wrist_0_rgb": image_spec,  # masked    → tokens 392–587
}
```
`ar_mask` is `[False]*196 + [False]*196 + [False]*196 + [False]*200` — fully bidirectional prefix attention (`pi0.py:125–133`). Language tokens start at absolute index 588.

### 1.4 Token positions for Phase 1 pair

**⚠ Not yet RunPod-verified for this pair.** Expected values from dry-run arithmetic:

| Token | Local index | Expected absolute index |
|-------|-------------|------------------------|
| BOS | 0 | 588 |
| `put` | 1 | 589 |
| `the` | 2 | 590 |
| `bowl` | 3 | 591 |
| `on` | 4 | 592 |
| `the` | 5 | 593 |
| `plate` / `stove` | 6 | **594** |

**Must verify with `inspect_kv_cache.py` on RunPod before trusting any indexing code.**

### 1.5 Policy and sample_kwargs wiring — `policy.py`

```python
# policy.py:53  — initial storage
self._sample_kwargs = sample_kwargs or {}

# policy.py:82  — per-call shallow copy
sample_kwargs = dict(self._sample_kwargs)

# policy.py:94  — passed to sample_actions
self._sample_actions(rng, observation, **sample_kwargs)
```

Setting `policy._sample_kwargs["donor_kv_cache"] = donor_kv_cache` before the rollout loop causes every `policy.infer()` call to pass `donor_kv_cache` to `sample_actions`. The shallow copy at line 82 copies the dict (not the JAX arrays, which are immutable). This mechanism works correctly. ✅

### 1.6 Model loading — `policy_config.py`

```python
# policy_config.py:57
model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
# ...
return _policy.Policy(model, transforms=[...], sample_kwargs=sample_kwargs, ...)
```

The patching script calls `create_trained_policy` (same as the server) to get a `Policy` object with the model already loaded and transforms wired up.

### 1.7 Input transform pipeline

`Policy.infer()` applies `self._input_transform` before creating `Observation.from_dict(inputs)`. For the donor obs, the same transforms must be applied. The transforms handle:
- Tokenization of the prompt string
- Image normalization / quantile normalization
- Key remapping (`observation/image` → `base_0_rgb`, etc.)

To build donor obs correctly in the patching script:
```python
clean_inputs = policy._input_transform(jax.tree.map(lambda x: x, clean_obs_dict))
clean_inputs_jax = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], clean_inputs)
clean_observation = Observation.from_dict(clean_inputs_jax)
# Then pass to model.build_donor_kv_cache(clean_observation)
```

### 1.8 `preprocess_observation` scope

`_model.preprocess_observation(None, observation, train=False)` is called at `pi0.py:225` INSIDE `sample_actions`. It handles image resize (if needed) and default mask filling. It is NOT called by `Policy.infer()` before constructing the Observation. Therefore, `build_donor_kv_cache` must call `preprocess_observation` internally (same pattern as `sample_actions`).

### 1.9 Server bypass — necessary consequence

The websocket server holds the `Policy` object. `main_corrupt_run_expt.py` communicates via `WebsocketClientPolicy.infer()` (msgpack over websocket). There is no mechanism for the client to inject a JAX array into the server's `policy._sample_kwargs`. Therefore, the patching script **must bypass the websocket server** and call `policy.infer()` in-process. This is not a contradiction of O2/P1 — P1 governs how `pi0.py` is modified; server bypass is a consequence of needing direct model access for pre-rollout donor harvesting.

The patching script: loads the policy in-process (no server launch), calls `policy.infer(element)` directly instead of `client.infer(element)`.

### 1.10 `module_jit` — `nnx_utils.py`

```python
# nnx_utils.py:15
def module_jit(meth, *jit_args, **jit_kwargs):
    # Freezes module state at call time; wraps method in jax.jit.
```

`Policy.__init__` sets `self._sample_actions = nnx_utils.module_jit(model.sample_actions)` (policy.py:64). The new `build_donor_kv_cache` method can be called either:
- Directly (no jit) for the one-off pre-rollout donor harvest, or
- Via `nnx_utils.module_jit(model.build_donor_kv_cache)` if jit is desired

For a one-time call, calling without jit is fine and avoids a compilation cost.

---

## 2. Architecture Decisions (Final)

| Decision | Resolution |
|----------|-----------|
| Execution path | JAX (default path, no conversion) |
| Script split | `main_corrupt_run_expt.py` = baselines; `main_patching_expt.py` = patched runs |
| `pi0.py` modification | P1: add `donor_kv_cache=None`, `patch_positions=(594,)` to `sample_actions`; add `_apply_kv_patch`; add `build_donor_kv_cache` |
| Server bypass | Patching script loads policy in-process; no websocket server needed |
| Patch target | Option A: position 594 only (Phase 1); revisit for Phase 2 |
| Donor cache | Pre-computed once from initial env obs + clean prompt before rollout |
| Trials | N=25 per run type |
| Eval scope | Phase 1: contrastive pair only (`plate`/`stove`) |

---

## 3. Changes to `pi0.py`

### 3.1 Modified `sample_actions` signature

```python
@override
def sample_actions(
    self,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    *,
    num_steps: int | at.Int[at.Array, ""] = 10,
    noise: at.Float[at.Array, "b ah ad"] | None = None,
    donor_kv_cache: _gemma.KVCache | None = None,   # NEW
    patch_positions: tuple[int, ...] = (594,),       # NEW
) -> _model.Actions:
```

### 3.2 Patch insertion (between lines 237 and 239)

```python
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    # KV-cache patching: overwrite specified positions with donor cache values
    if donor_kv_cache is not None:
        kv_cache = self._apply_kv_patch(kv_cache, donor_kv_cache, patch_positions)

    def step(carry):
```

### 3.3 New method: `_apply_kv_patch`

Add after `sample_actions` (before `compute_loss` or at end of class):

```python
def _apply_kv_patch(
    self,
    corrupt_kv_cache: _gemma.KVCache,
    donor_kv_cache: _gemma.KVCache,
    patch_positions: tuple[int, ...],
) -> _gemma.KVCache:
    """Replace specified prefix positions in corrupt cache with donor cache values."""
    K_d, V_d = donor_kv_cache
    K, V = corrupt_kv_cache
    for pos in patch_positions:
        K = K.at[:, :, pos, :, :].set(K_d[:, :, pos, :, :])
        V = V.at[:, :, pos, :, :].set(V_d[:, :, pos, :, :])
    return (K, V)
```

K and V each have shape `(18, 1, 788, 1, 256)`. The index `[:, :, pos, :, :]` patches all 18 layers at position `pos` simultaneously.

### 3.4 New method: `build_donor_kv_cache`

Add alongside `_apply_kv_patch`:

```python
def build_donor_kv_cache(self, observation: _model.Observation) -> _gemma.KVCache:
    """Run a prefix-only forward pass and return the resulting KV cache."""
    observation = _model.preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    return kv_cache
```

This mirrors the prefix-build block in `sample_actions` exactly. Takes a pre-transformed `Observation` (after `policy._input_transform` and `jnp.asarray` + batch-dim addition, but before `preprocess_observation`).

**JAX JIT note:** Calling `model.build_donor_kv_cache(clean_observation)` once without jit is fine for a pre-rollout harvest. The method is pure and can be jitted if desired.

---

## 4. `main_patching_expt.py` Structure

The script is a copy of `main_corrupt_run_expt.py` with these changes:

### 4.1 New/modified `Args` fields

```python
@dataclasses.dataclass
class Args:
    # same as main_corrupt_run_expt.py, with these changes:
    num_trials_per_task: int = 25              # was 50
    task_suite_name: str = "libero_goal"       # same default
    task_name_filter: str = "put_the_bowl_on_the_plate"  # Phase 1 task
    corrupt_prompt: str = "put the bowl on the stove"    # Phase 1 corrupt

    # NEW fields for patching
    clean_prompt: str = "put the bowl on the plate"      # donor source
    patch_positions: str = "594"               # comma-separated ints; verify on RunPod
    patch_k: bool = True
    patch_v: bool = True
    checkpoint_dir: str = "/workspace/openpi_assets/openpi-assets/checkpoints/pi05_libero"
    sanity_check: bool = False  # if True, patch all 788 positions (full donor override)
```

### 4.2 Policy loading (in-process, no server)

```python
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.models import model as _model

def load_policy(args: Args) -> _policy.Policy:
    train_config = _config.get_config("pi05_libero")
    return _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
    )
```

### 4.3 Donor harvest (before rollout loop)

```python
def harvest_donor_kv_cache(policy, initial_env_obs, clean_prompt, args):
    """Build donor KV cache from initial env obs + clean prompt."""
    import jax
    import jax.numpy as jnp

    # Build the obs dict in the same format as the rollout loop
    clean_obs_dict = {
        "observation/image": preprocess_img(initial_env_obs["agentview_image"], args.resize_size),
        "observation/wrist_image": preprocess_img(initial_env_obs["robot0_eye_in_hand_image"], args.resize_size),
        "observation/state": make_state(initial_env_obs),
        "prompt": clean_prompt,
    }

    # Apply policy input transforms (tokenization, normalization, key remapping)
    clean_inputs = policy._input_transform(jax.tree.map(lambda x: x, clean_obs_dict))
    clean_inputs_jax = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], clean_inputs)
    clean_observation = _model.Observation.from_dict(clean_inputs_jax)

    # Run prefix-only forward pass
    donor_kv_cache = policy._model.build_donor_kv_cache(clean_observation)
    return donor_kv_cache
```

### 4.4 Rollout loop changes

```python
# Before the episode loop:
initial_env_obs = env.set_init_state(initial_states[0])  # any initial state works; objects don't move
donor_kv_cache = harvest_donor_kv_cache(policy, initial_env_obs, args.clean_prompt, args)

patch_positions = tuple(int(p) for p in args.patch_positions.split(","))
if args.sanity_check:
    patch_positions = tuple(range(788))  # full prefix override

policy._sample_kwargs["donor_kv_cache"] = donor_kv_cache
policy._sample_kwargs["patch_positions"] = patch_positions

# Inside the action-chunk request block (replaces client.infer):
action_chunk = policy.infer(element)["actions"]   # direct call, no websocket
```

### 4.5 Key structural differences from `main_corrupt_run_expt.py`

| Aspect | `main_corrupt_run_expt.py` | `main_patching_expt.py` |
|--------|---------------------------|------------------------|
| Policy access | `WebsocketClientPolicy` (client) | `create_trained_policy` (in-process) |
| Action request | `client.infer(element)` | `policy.infer(element)` |
| Prompt in element | `args.corrupt_prompt` or task default | `args.corrupt_prompt` (stove) always |
| Pre-rollout setup | none | `harvest_donor_kv_cache` + set `_sample_kwargs` |
| `num_trials_per_task` | 50 | 25 |
| `task_name_filter` default | None | `"put_the_bowl_on_the_plate"` |

### 4.6 Logging / output

Keep the same video-saving and success-rate logging as `main_corrupt_run_expt.py`. Add one line at the top of each episode log: the current patch config (positions, K/V flags). Output directory should encode the patch config:
```
data/libero/videos/patched_pos594_kv/
```

---

## 5. Implementation Checklist

### Phase A — `pi0.py` changes

- [x] **A1.** Add `donor_kv_cache=None` and `patch_positions=(594,)` to `sample_actions` signature (after `noise=None`)
- [x] **A2.** Add patch logic block between lines 237 and 239 (see §3.2)
- [x] **A3.** Add `_apply_kv_patch` method (see §3.3)
- [x] **A4.** Add `build_donor_kv_cache` method (see §3.4)
- [x] **A5.** `_gemma` already imported (`import openpi.models.gemma as _gemma`, pi0.py:12) ✅

### Phase B — `main_patching_expt.py`

- [x] **B1.** Script created at `examples/libero/main_patching_expt.py`
- [x] **B2.** `Args` has new fields; `num_trials_per_task` defaults to 25
- [x] **B3.** Uses `create_trained_policy` in-process (no websocket server)
- [x] **B4.** `harvest_donor_kv_cache()` implemented — applies `policy._input_transform`, builds `Observation`, calls `model.build_donor_kv_cache`
- [x] **B5.** Donor harvest + `_sample_kwargs` injection happens once per task before episode loop
- [x] **B6.** Rollout calls `policy.infer(element)` directly
- [x] **B7.** Output dir encodes patch positions: `patched_pos594_kv/`
- [x] **B8.** No `host`/`port` args; no websocket client imports

### Phase Review AB — code review before RunPod

Review the two implementation files against the plan and source facts before moving to RunPod. Items below are the findings from the 2026-05-04 review.

**`pi0.py` — verified correct:**
- [x] **RAB1.** `build_donor_kv_cache` body is a verbatim copy of the prefix-build block in `sample_actions` (`embed_prefix` → `make_attn_mask` → `cumsum` → `llm([prefix_tokens, None], ...)`). No drift. ✅
- [x] **RAB2.** `build_donor_kv_cache` calls `preprocess_observation(None, obs, train=False)` before `embed_prefix` — same order as `sample_actions`. ✅
- [x] **RAB3.** `_apply_kv_patch` indexing `[:, :, pos, :, :]` is correct for `(18, 1, 788, 1, 256)`: patches all 18 layers at position `pos` simultaneously. ✅
- [x] **RAB4.** Patch call sits between `kv_cache` assignment (line 262) and `def step` (line 267), before `while_loop` at line 306. Closure captures patched value. ✅
- [x] **RAB5.** `donor_kv_cache=None` default means zero change to the existing code path when not patching. ✅
- [x] **RAB6.** `for pos in patch_positions` loop in `_apply_kv_patch` uses a Python-level tuple — JAX unrolls it at trace time. JIT-compatible. ✅
- [x] **RAB7.** `patch_positions` as a kwarg to the jitted `sample_actions`: it is a Python tuple of ints (not a JAX array), captured as a concrete value at trace time. First patched call recompiles once; subsequent calls with identical tuple reuse the kernel. ✅

**`main_patching_expt.py` — verified correct:**
- [x] **RAB8.** `harvest_donor_kv_cache` applies `policy._input_transform` (tokenization, normalization, key remapping) before building `Observation` — same pipeline as `Policy.infer()`. ✅
- [x] **RAB9.** Donor is harvested after `num_steps_wait` DUMMY_ACTION steps so physics are settled and images reflect a realistic scene. ✅
- [x] **RAB10.** Donor harvest uses `env.reset()` + `env.set_init_state(initial_states[0])` in an isolated block; each episode re-resets the env, so donor harvest does not contaminate episode initial states. ✅
- [x] **RAB11.** `policy.infer(element)` called directly in-process — no websocket server needed. `donor_kv_cache` and `patch_positions` reach `sample_actions` via `policy._sample_kwargs`. ✅
- [x] **RAB12.** `done = False` initialised before the episode loop; used correctly for video suffix and success counting even on exception-caused breaks. ✅

**Known deviations from §4.1 spec — accepted for Phase 1:**
- [ ] **RAB13.** `patch_k` / `patch_v` flags (K-only or V-only patching) are NOT implemented — `_apply_kv_patch` always patches both K and V. Accepted: K+V is the right default for Phase 1. Add selective K/V flags when doing the heatmap sweep.
- [ ] **RAB14.** `build_donor_kv_cache` is called without `module_jit` — runs eagerly (slow once, ~seconds, not minutes). Accepted: it is a one-time pre-rollout call per task. If this proves too slow on RunPod, wrap with `nnx_utils.module_jit`.

**Pre-existing inherited issues (not blocking):**
- [ ] **RAB15.** Final `total_successes / total_episodes` log crashes with ZeroDivisionError if `task_name_filter` matches no task. Not a risk for Phase 1 (filter is set correctly), but add a guard if broadening to full suite.

### Phase C — RunPod verification (before trusting results)

- [ ] **C1.** Shape debug: confirm kv_cache is `(18, 1, 788, 1, 256)` (see §6.1)
- [ ] **C2.** Token position check: run `inspect_kv_cache.py` on Phase 1 pair (see §6.2)
- [ ] **C3.** Sanity check run: patch all 788 positions → behavior should match clean run (see §6.3)

### Phase D — Phase 1 runs

- [x] **D1.** Clean baseline: `put the bowl on the plate`, N=25 — **100%** (25/25) ✅
- [x] **D2.** Corrupt baseline: `put the bowl on the stove` (stove prompt, plate task), N=25 — **0%** (0/25) ✅
- [ ] **D3.** Patched run: stove prompt + KV patch at pos 594, N=25 — record success rate

---

## 6. RunPod Verification Steps

### 6.1 Shape debug

Add a one-shot debug print inside `sample_actions` on the first call:

```python
_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
# DEBUG — remove before production runs:
import jax
print("kv_cache shapes:", jax.tree_util.tree_map(lambda x: x.shape, kv_cache))
```

Expected output: `kv_cache shapes: ((18, 1, 788, 1, 256), (18, 1, 788, 1, 256))`.

If K and V shapes differ from this, stop and investigate before writing any indexing code.

### 6.2 Token position verification

Use the existing `inspect_kv_cache.py` script (same one used for Phase 2 pair during sanity check):

```bash
# On RunPod, in /workspace/openpi:
uv run python examples/libero/inspect_kv_cache.py \
    --prompt "put the bowl on the plate" \
    --checkpoint_dir /workspace/openpi_assets/openpi-assets/checkpoints/pi05_libero
```

Check that `plate` token appears at absolute index 594. Then repeat with `"put the bowl on the stove"` and confirm `stove` is also at 594.

If either prompt tokenizes differently than expected (different absolute index), update `patch_positions` accordingly.

### 6.3 Sanity check

Run the patching script with `--sanity_check` flag (patches all 788 positions). This replaces the entire corrupt prefix cache with the clean donor. The model should behave identically to the clean-prompt run (put bowl on plate). If not:

- Patch mechanism is broken (check `_apply_kv_patch` and `build_donor_kv_cache`)
- Donor obs was built incorrectly (transforms not applied, wrong prompt, wrong images)

Run N=5 sanity episodes — a few successes is enough to confirm the mechanism works. Full N=25 not needed here.

---

## 7. Expected Outcomes and Interpretation Guide

This section defines what a "good" result looks like for each run condition and how to interpret deviations. **Read this before running and before recording results.**

### 7.1 What the two runs test

`run_patching_phase1.sh` runs two conditions back to back:

| Code | Run | Prompt sent to model | KV cache | What it tests |
|------|-----|----------------------|----------|---------------|
| C3 | Sanity check | `"put the bowl on the stove"` (corrupt) | ALL 788 positions replaced by clean donor | Does the patching mechanism work at all? |
| D3 | Main patched run | `"put the bowl on the stove"` (corrupt) | Only position 594 replaced by clean donor | Is pos 594 the primary causal position for task specification? |

### 7.2 Expected success rates

| Code | Expected success rate | Rationale |
|------|-----------------------|-----------|
| D1 (clean baseline) | ~100% | Already confirmed: 25/25 |
| D2 (corrupt baseline) | ~0% | Already confirmed: 0/25 |
| C3 (sanity check, all 788 pos patched) | **~100%** (≥80% acceptable) | Entire prefix KV cache is the clean donor — model has no access to "stove" signal during suffix generation. Should match D1. May be slightly below 100% because donor images come from episode 0 and are reused across all 5 episodes (minor scene variation). If C3 < 50%, the patch mechanism is broken — stop and debug before trusting D3. |
| D3 (patched pos 594 only) | **>0%, ideally ≥70%** | Only the destination token position is patched. If pos 594 is the sole causal position for task specification, D3 should approach D1 (~100%). Partial recovery (e.g. 30–70%) means other positions also carry destination signal. 0% means pos 594 is not causal — expand to more positions or revisit hypothesis. |

### 7.3 Interpretation matrix

| C3 | D3 | Interpretation | Action |
|----|----|----------------|--------|
| ~100% | ~100% | Pos 594 is the sole causal position. Strong hypothesis confirmation. | Record results; proceed to Phase 2 (sweep over more tasks/pairs). |
| ~100% | 30–80% | Pos 594 partially causal; other positions share the signal. | Broaden patch to include a window around 594 (e.g. 593–595 or all language tokens 588–787). |
| ~100% | ~0% | Pos 594 is not causal. Patching mechanism works (proven by C3) but wrong position targeted. | Check tokenization; inspect attention patterns at other positions. |
| ~0% | any | Patch mechanism broken regardless of D3. | Debug `_apply_kv_patch` and `build_donor_kv_cache`; check donor obs pipeline. |

### 7.4 Why C3 may not reach exactly 100%

The donor KV cache is harvested once from `initial_states[0]` and reused across all C3 episodes (N=5). This means:
- Image token positions (0–587): from episode 0's scene images, not the current episode's scene
- Language token positions (588–787): from "put the bowl on the plate" — correct for all episodes

LIBERO-Goal initial states vary in object placement. For the sanity check, this means the model sees the "wrong" images for episodes 1–4. Language dominates task specification in π₀.₅, so degradation should be small (≥80% expected), but exact 100% is unlikely. For D3 only position 594 is patched — image tokens are untouched — so this image-reuse issue does not affect D3.

---

## 8. Results

*(Fill in as runs complete.)*

### 8.1 Phase 1 — `put_the_bowl_on_the_plate` task (LIBERO-Goal)

| Run type | Prompt | KV cache | N | Success rate | Notes |
|----------|--------|----------|---|-------------|-------|
| Clean baseline (D1) | `"put the bowl on the plate"` | normal | 25 | **100%** (25/25) | `baselines_20260504_072520.txt` |
| Corrupt baseline (D2) | `"put the bowl on the stove"` | normal | 25 | **0%** (0/25) | `baselines_20260504_072520.txt` |
| Step 0 debug | `"put the bowl on the stove"` | pos 594 from donor | 1 | **0%** (0/1) | `run_20260504_172532_step0_debug_clean.txt`; donor differs from corrupt at pos594 |
| Step 1 C3 sanity | `"put the bowl on the stove"` | language positions 588-787 from precomputed donor | 5 | **0%** (0/5) | `run_20260504_173138_step1_c3_clean.txt`; language-only precomputed donor did not recover behavior |
| Step 4 C3 sanity | `"put the bowl on the stove"` | language positions 588-787 from per-step donor | 5 | **0%** (0/5) | `run_20260504_175753_step4_perstep_c3_clean.txt`; donor rebuilt from current obs each inference, still no recovery |
| Step 4 debug C3 sanity | `"put the bowl on the stove"` | full prefix positions 0-787 from per-step donor | 5 | **100%** (5/5) | `run_20260504_181140_step4_debug_perstep_fullprefix_c3_clean.txt`; full-prefix current-observation patch exactly recovers clean behavior |
| Step 5 control | `"put the bowl on the stove"` | full prefix positions 0-787 from per-step donor | 10 | **100%** (10/10) | `run_20260504_182602_step5_fullprefix_n10_clean.txt`; full-prefix recovery stable at N=10 |
| Step 5 D3 image prefix | `"put the bowl on the stove"` | image prefix positions 0-587 from per-step donor | 10 | **100%** (10/10) | `run_20260504_183412_step5_imageprefix_n10_clean.txt`; image-token K/V is sufficient for recovery |
| Step 5 image half A | `"put the bowl on the stove"` | image prefix positions 0-293 from per-step donor | 10 | **0%** (0/10) | `run_20260504_184132_step5_img0-293_n10_clean.txt`; first half of image tokens is insufficient |
| Step 5 image half B | `"put the bowl on the stove"` | image prefix positions 294-587 from per-step donor | 10 | **80%** (8/10) | `run_20260504_185204_step5_img294-587_n10_clean.txt`; second half contains most of the sufficient signal |
| Step 5 image quarter B1 | `"put the bowl on the stove"` | image prefix positions 294-440 from per-step donor | 10 | **10%** (1/10) | `run_20260504_185938_step5_img294-440_n10_clean.txt`; lower half of 294-587 is below meaningful threshold |
| Patched (D3, pos 594, K+V) | `"put the bowl on the stove"` | pos 594 from donor | 25 | — | |

### 8.2 Implementation notes

*(Surprises, deviations from plan, debugging notes — fill as work proceeds.)*

| Date | Note |
|------|------|
| 2026-05-04 | D1/D2 baselines: perfect 100%/0% separation on plate/stove pair. Clean ceiling is 100% (not ~96% as in the Apr 29 cabinet/wine-bottle run). Any D3 success above 0% is meaningful signal. |
| 2026-05-04 | Step 0 debug (`run_20260504_172532_step0_debug_clean.txt`): donor vs corrupt KV differs at pos594 (`K=3.000000`, `V=4.812500` L-inf); control pos688 `K=1.156250`. Patching pos594 is not a no-op. |
| 2026-05-04 | Step 1 C3 (`run_20260504_173138_step1_c3_clean.txt`): patching only language positions 588-787 from the precomputed t=0 donor still failed 0/5. Since prefix attention is bidirectional, stale-donor/image-conditioned language K/V remains a likely issue; proceeding to per-step donor. |
| 2026-05-04 | Step 4 C3 (`run_20260504_175753_step4_perstep_c3_clean.txt`): per-step donor with current images and language positions 588-787 also failed 0/5 (`pos594 K=3.375000`, `V=5.312500`; pos688 `K=1.179688`). D3 was not run because the sanity mechanism did not pass. |
| 2026-05-04 | Step 4 debug C3 (`run_20260504_181140_step4_debug_perstep_fullprefix_c3_clean.txt`): per-step full-prefix patch positions 0-787 passed 5/5. This validates `_apply_kv_patch` and shows language-only failure was due to leaving corrupt-prompt effects in non-language prefix positions, not a broken patch path. |
| 2026-05-04 | Step 5 control (`run_20260504_182602_step5_fullprefix_n10_clean.txt`): per-step full-prefix patch positions 0-787 passed 10/10, confirming the positive control is stable enough to use as the upper bound for localization. |
| 2026-05-04 | Step 5 D3 image-prefix probe (`run_20260504_183412_step5_imageprefix_n10_clean.txt`): patching only image-prefix positions 0-587 from a per-step clean donor passed 10/10. This clears the >2/10 meaningful threshold and shows the clean-vs-corrupt destination signal is available through image-token K/V after bidirectional prefix mixing. |
| 2026-05-04 | Step 5 image half A (`run_20260504_184132_step5_img0-293_n10_clean.txt`): positions 0-293 failed 0/10, so the sufficient image-prefix signal is not in the first half alone; test 294-587 next. |
| 2026-05-04 | Step 5 image half B (`run_20260504_185204_step5_img294-587_n10_clean.txt`): positions 294-587 recovered 8/10. This is above the meaningful threshold and localizes most of the recoverable signal to the second half of image tokens; recurse within 294-587. |
| 2026-05-04 | Step 5 image quarter B1 (`run_20260504_185938_step5_img294-440_n10_clean.txt`): positions 294-440 recovered only 1/10, below the meaningful threshold. Next probe should test positions 441-587. |

### 8.3 Current conclusion and next steps

**What we achieved:** The patching mechanism has a working positive control. With a per-step donor built from the current observation and clean prompt, replacing the full prefix KV cache (`0-787`) in the corrupt run recovers clean behavior at **5/5 episodes**. This meets the C3 sanity threshold and is recorded in `scripts_outputs_txt/patching_phase1/patched/1_SANITY_CHECK_SUCCESS.txt`.

**What it means:** `_apply_kv_patch` is functionally correct, `policy._sample_kwargs["donor_kv_cache"]` is reaching `sample_actions()`, and the clean donor KV cache can drive the corrupt-prompt rollout to the clean target. The earlier 0/5 results are therefore not evidence of a broken patch implementation. They show that patching only language slots (`588-787`) is insufficient in π₀.₅ because prefix attention is bidirectional: corrupt language affects image-token K/V as well, and those non-language cache entries remain corrupt unless patched.

**What it does not mean:** Full-prefix patching is a positive control, not yet a localized causal result. It proves the intervention can recover behavior, but it does not identify the minimal positions carrying the plate-vs-stove signal. Do not write `2_PATCHING_MEANINGFUL_RESULT.txt` from this run alone.

**Language effect caveat:** It is not correct to say that language has no effect on output behavior. The clean vs. corrupt baselines differ 25/25 vs. 0/25, so language is behaviorally load-bearing. The current patching result instead shows that, after π₀.₅'s bidirectional prefix pass, the behaviorally relevant language information is not confined to language-token K/V slots; it is also mixed into image-token K/V, where patching can recover clean behavior.

**Language-token patching result:** In this setup, patching the language-token K/V region alone had no meaningful behavioral effect: precomputed donor `588-787` failed 0/5, and per-step donor `588-787` also failed 0/5. The correct interpretation is not "language does nothing"; it is that swapping only the language-token cache entries is too late and too narrow after bidirectional prefix mixing. The corrupt prompt has already altered other prefix cache entries, especially image-token K/V, and those unpatched entries are sufficient to keep the rollout on the corrupt behavior.

**What did not work before and why this works now:** The original sanity run patched all 788 positions from a donor harvested once at t=0. That overwrote current image-token K/V with stale initial-scene image K/V at every rollout step, so the robot lost current visual context and failed 0/5. The next language-only fixes avoided stale image patching, but they still failed because π₀.₅ prefix attention is bidirectional: the corrupt prompt influences image-token K/V during the prefix pass, not only language-token K/V. The successful run fixes both problems at once: it rebuilds the donor from the current observation at each inference, so image K/V is current rather than stale, and it patches the full prefix (`0-787`), so no corrupt-prompt prefix state is left behind.

**Next:** Localize the sufficient patch set using the per-step donor script, starting from full-prefix success rather than language-only failure. Recommended N=10 probes:

| Probe | Patch positions | Question |
|-------|-----------------|----------|
| Full prefix repeat | `0-787` | Confirm full-prefix recovery is stable at N=10. |
| Image prefix only | `0-587` | Test whether corrupt-language effects in image-token K/V are sufficient to drive recovery. |
| Language prefix only | `588-787` | Already failed 0/5; rerun only if comparing same N/control conditions. |
| Image + target language window | `0-587` plus a small window around `594` | Test whether clean image-token K/V plus local destination token K/V is enough. |
| Binary search over `0-787` | contiguous halves, then recurse | Find the smallest sufficient region once a nontrivial subset gives >2/10. |

Once a nontrivial subset recovers **>2/10** at N=10, rerun that subset at N=25 and write `scripts_outputs_txt/patching_phase1/patched/2_PATCHING_MEANINGFUL_RESULT.txt` only if it exceeds the meaningful threshold.
