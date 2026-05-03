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
7. [Results](#7-results)

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

- [ ] **A1.** Add `donor_kv_cache=None` and `patch_positions=(594,)` to `sample_actions` signature (after `noise=None`)
- [ ] **A2.** Add patch logic block between lines 237 and 239 (see §3.2)
- [ ] **A3.** Add `_apply_kv_patch` method (see §3.3)
- [ ] **A4.** Add `build_donor_kv_cache` method (see §3.4)
- [ ] **A5.** Verify: the type annotation for `donor_kv_cache` uses `_gemma.KVCache | None` — confirm `_gemma` is imported in `pi0.py` (it is: `import openpi.models.gemma as _gemma`, line 12)

### Phase B — `main_patching_expt.py`

- [ ] **B1.** Copy `main_corrupt_run_expt.py` to `main_patching_expt.py`
- [ ] **B2.** Add new `Args` fields (§4.1); change `num_trials_per_task` default to 25
- [ ] **B3.** Replace `WebsocketClientPolicy` with in-process `create_trained_policy` (§4.2)
- [ ] **B4.** Implement `harvest_donor_kv_cache` (§4.3)
- [ ] **B5.** Add pre-episode-loop donor harvest and `_sample_kwargs` injection (§4.4)
- [ ] **B6.** Replace `client.infer(element)` with `policy.infer(element)` in rollout loop
- [ ] **B7.** Update output directory naming to encode patch config
- [ ] **B8.** Remove server-related args (`host`, `port`) and imports

### Phase C — RunPod verification (before trusting results)

- [ ] **C1.** Shape debug: confirm kv_cache is `(18, 1, 788, 1, 256)` (see §6.1)
- [ ] **C2.** Token position check: run `inspect_kv_cache.py` on Phase 1 pair (see §6.2)
- [ ] **C3.** Sanity check run: patch all 788 positions → behavior should match clean run (see §6.3)

### Phase D — Phase 1 runs

- [ ] **D1.** Clean baseline: `put the bowl on the plate`, N=25 — record success rate
- [ ] **D2.** Corrupt baseline: `put the bowl on the stove` (stove prompt, plate task), N=25 — record success rate
- [ ] **D3.** Patched run: stove prompt + KV patch at pos 594, N=25 — record success rate

*(Note: D1 and D2 may already exist from `main_corrupt_run_expt.py` runs if the task was previously evaluated on LIBERO-Goal. Check existing results before re-running.)*

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

## 7. Results

*(Fill in as runs complete.)*

### 7.1 Phase 1 — `put_the_bowl_on_the_plate` task (LIBERO-Goal)

| Run type | Prompt | KV cache | N | Success rate | Notes |
|----------|--------|----------|---|-------------|-------|
| Clean baseline | `"put the bowl on the plate"` | normal | 25 | — | |
| Corrupt baseline | `"put the bowl on the stove"` | normal | 25 | — | |
| Sanity check | `"put the bowl on the stove"` | full donor (all 788 pos) | 5 | — | should ≈ clean |
| Patched (pos 594, K+V) | `"put the bowl on the stove"` | pos 594 from donor | 25 | — | |

### 7.2 Implementation notes

*(Surprises, deviations from plan, debugging notes — fill as work proceeds.)*

| Date | Note |
|------|------|
| | |
