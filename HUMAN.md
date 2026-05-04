# Review: Phase 1 KV-cache patching implementation

Date: 2026-05-04

Scope reviewed:
- `runpod/run_patching_phase1_verify.sh`
- `runpod/run_patching_phase1_baselines.sh`
- `runpod/run_patching_phase1.sh`
- `examples/libero/main_patching_expt.py`
- `examples/libero/main_corrupt_run_expt.py`
- `src/openpi/models/pi0.py`
- Relevant docs in `status_cc/`, especially `patching_implementation.md`, `patching_implementation_dryrun.md`, and `kv_cache_findings.md`

## Bottom line

The implementation is structurally correct for the intended Phase 1 experiment:

- The patch hook is in the right model location: after the corrupt prefix KV cache is built in `Pi0.sample_actions()` and before the suffix/action diffusion loop consumes it.
- `main_patching_expt.py` correctly bypasses the websocket server, loads the policy in-process, harvests a donor cache, injects it via `policy._sample_kwargs`, and calls `policy.infer()` directly.
- The bash scripts call the intended Python scripts with the right tyro-style `--args.*` flags for these entry points.
- The baseline script correctly reuses `main_corrupt_run_expt.py` and therefore requires the normal policy server.
- The patching script correctly does not require the policy server.
- `run_patching_phase1_verify.sh` omits OpenPI's appended newline token, but this does not affect the destination-token index: `plate`/`stove` remain at expected absolute index 594.
- The all-position sanity path may compile slowly because `tuple(range(788))` is passed into jitted `sample_actions()` and unrolled as many `.at[..., pos, ...].set(...)` updates. A direct all-prefix replacement path would be cleaner if this becomes painful.
- `main_patching_expt.py` and `pi0.py` parse cleanly with Python `ast.parse`; I did not run LIBERO/JAX execution locally because this environment lacks the RunPod GPU/runtime setup.

However, I would not treat the current sanity-check interpretation as airtight. The all-position sanity check patches a donor prefix harvested from `initial_states[0]` into episodes using `initial_states[0..4]`. If those initial states differ, the all-position donor contains stale image-token K/V for most sanity episodes. A failed all-position sanity check would therefore not necessarily prove the patching mechanism is broken. It may prove that fully replacing the current episode's visual prefix with episode-0 visual K/V is too destructive.

## Findings

### 1. All-position sanity check is confounded by the fixed episode-0 donor

`main_patching_expt.py` harvests the donor once per task from `initial_states[0]`:

```python
initial_obs = env.set_init_state(initial_states[0])
...
donor_kv_cache = harvest_donor_kv_cache(...)
```

Then the sanity run uses:

```python
patch_positions = tuple(range(788))
```

That overwrites the entire prefix cache: both language and image-token slots. For episode 0, this is at least internally consistent. For episodes 1-4 in the N=5 sanity run, the donor image K/V may come from a different initial state than the actual rollout observation.

Consequence: the bash-script expectation that all-position patching "should approach clean baseline" is too strong under the current fixed-donor design. If C3 fails, do not immediately conclude `_apply_kv_patch` is broken.

Suggested fix for a cleaner C3: harvest the donor from the same current observation used for each inference call, at least for sanity mode. That means a per-call clean-prefix donor, or a simpler episode-level donor harvested from each episode's own post-wait initial observation before the rollout begins.

### 2. Fixed donor cache is an accepted design choice, but it is the main validity caveat for D3

For the main pos-594 patch, only one language token slot is overwritten. That is much less destructive than all-position patching, so the implementation can still answer the intended Phase 1 question.

Still, prefix attention is bidirectional over image and language tokens. The donor K/V at position 594 is not just the token embedding for `plate`; it is the result of a prefix forward pass that also saw the donor images. Reusing one donor from episode-0/t=0 across every inference call means the transplanted `plate` K/V is stale with respect to later robot states and other initial states.

This matches the resolved O4 decision, so it is not a code bug. But it should be recorded as a threat to interpretation: if D3 does not recover, the result may reflect fixed-donor staleness rather than absence of causal language signal.

### 3. `pi0.py` patching hook is correctly placed and correctly indexed

The model changes in `src/openpi/models/pi0.py` are correct:

- `build_donor_kv_cache()` mirrors the prefix-cache construction in `sample_actions()`.
- It calls `_model.preprocess_observation(..., train=False)` before `embed_prefix()`, matching the normal path.
- `_apply_kv_patch()` indexes `K[:, :, pos, :, :]` and `V[:, :, pos, :, :]`, which is the right slice for KV cache shape `(layers, batch, prefix_seq, kv_heads, head_dim)`.
- The patch is applied before the diffusion `while_loop`, so all denoising steps consume the patched prefix cache.

I also parsed both modified Python files with `ast.parse`; no syntax errors.

### 4. Bash script wiring is mostly correct

`run_patching_phase1_verify.sh`:
- Correctly runs without model weights or server.
- Correctly checks the Phase 1 pair with a language offset of 588.
- Minor caveat: the OpenPI `PaligemmaTokenizer` for this config appends a newline token after the prompt, while this verification snippet does not. That does not change the destination-token index: `plate`/`stove` are still local index 6, absolute 594. It only means the printed total token count is one shorter than the actual model prompt sequence.

`run_patching_phase1_baselines.sh`:
- Correctly calls `main_corrupt_run_expt.py`.
- Correctly needs the websocket server.
- Clean run omits `--args.corrupt-prompt`, so it uses the LIBERO task language. For this task that should be `"put the bowl on the plate"`.
- Corrupt run sets `--args.corrupt-prompt "put the bowl on the stove"` correctly.

`run_patching_phase1.sh`:
- Correctly calls `main_patching_expt.py` in-process.
- Correctly passes `--args.sanity-check` for the all-position run and `--args.patch-positions "594"` for D3.
- The `printf "n\n" |` prefix appears harmless but unnecessary for `main_patching_expt.py`; I do not see an input prompt in that script.

### 5. JAX/JIT caveat for `patch_positions`

`patch_positions` is passed as a Python tuple into a jitted `sample_actions()`. For `(594,)`, this should be fine. For `tuple(range(788))`, JAX will likely trace a very large unrolled sequence of `.at[..., pos, ...].set(...)` operations, and the tuple itself becomes a large argument pytree.

This may work, but it could make the first sanity-check compile very slow or memory-heavy. If this becomes a problem, replace the Python loop with a vectorized indexed update or add a dedicated `patch_all` path that simply returns the donor cache for all-position sanity.

For all-position K+V patching, the simplest equivalent is:

```python
if patch_all:
    kv_cache = donor_kv_cache
```

That would also avoid the 788-update compile.

### 6. Edge cases not blocking Phase 1

- `main_patching_expt.py` will divide by zero at final logging if `task_name_filter` matches no task. This is inherited from the baseline style and is not a risk for the current filter, but it should be guarded before broader use.
- `episode_idx` indexes directly into `initial_states`; this is fine for N=25 if LIBERO provides at least 25 initial states, as prior scripts assume.
- The docstring usage in `main_patching_expt.py` shows underscore flags like `--task_suite_name`, but the actual bash scripts correctly use tyro nested dashed flags like `--args.task-suite-name`.

## Recommended next action

Before trusting D3:

1. Run `run_patching_phase1_verify.sh` and confirm `plate` and `stove` are both absolute index 594.
2. Consider changing the C3 sanity check so the donor is harvested from the same episode/current observation, or interpret C3 conservatively under the fixed-donor design.
3. If the all-position sanity compile is painful, add a fast `patch_all` path rather than looping over all 788 positions.
4. Keep the main D3 result, but report the fixed-donor caveat alongside the success rate.
