# Agent Guide: KV-Cache Patching Debug (Phase 1)

---

## YOUR MISSION (read this first)

You are an autonomous agent debugging a broken KV-cache patching experiment on RunPod. The experiment ran and produced 0% success on BOTH the sanity check and the main patched run. Your job is to diagnose and fix this, iterating through the steps below until you achieve:

1. **Sanity check success** (C3 ≥ 3/5 episodes) → write `scripts_outputs_txt/patching_phase1/patched/1_SANITY_CHECK_SUCCESS.txt`
2. **Meaningful patching result** (D3 > 2/10 episodes) → write `scripts_outputs_txt/patching_phase1/patched/2_PATCHING_MEANINGFUL_RESULT.txt`

Work through the steps in order. Do not skip ahead. Commit after each step. Update `status_cc/patching_implementation.md §8.1` after each completed run.

**If you get stuck or something is unclear:** record what you tried and why it failed in the relevant result file, then move to the next step. Never spin on one approach indefinitely.

**You are on branch `patching_phase1_agentic`.** All code changes go here. Do not touch `kv-cache-patching`.

### Success thresholds (reference these throughout)

| Condition | Threshold | Action |
|-----------|-----------|--------|
| C3 sanity check passes | **≥ 3/5 episodes** (60%) | Write `1_SANITY_CHECK_SUCCESS.txt`; proceed to Step 2 |
| C3 sanity check fails | < 3/5 | Debug before proceeding — do NOT run N=25 D3 on a broken mechanism |
| D3 patching meaningful | **> 2/10 episodes** (20%) with N=10, or **> 5/25** (20%) with N=25 | Write `2_PATCHING_MEANINGFUL_RESULT.txt` |
| D3 patching weak but non-zero | 1–2/10 | Report it; continue binary search to find stronger patch set |
| D3 patching absent | 0/10 | Widen patch set (Step 3) or move to per-step donor (Step 4) |
| All approaches fail | 0 everywhere including Step 4 | Write `0_FAILURE_REPORT.txt`; stop |

**Context for these numbers:**
- Clean baseline D1 = 25/25 (100%) — ceiling
- Corrupt baseline D2 = 0/25 (0%) — floor
- Any D3 > 0% is signal; > 20% is "meaningful" (clearly above noise)
- C3 sanity at 60% threshold (not 100%) because the pre-computed donor has minor image-context contamination across episodes

---

## Environment Setup

Run this once at the start of every session (or verify it's already active):

```bash
cd /workspace/openpi
source /workspace/openpi/runpod/patching_env.sh
# Should print: Server venv active (Python 3.11.x, JAX x.x.x)
```

All experiment runs use plain `python` (NOT `uv run python`) after sourcing `patching_env.sh`.

Key paths:
- Main script: `examples/libero/main_patching_expt.py`
- Per-step donor script (Step 4, you create it): `examples/libero/main_patching_expt_per_step_donor.py`
- Agentic logs: `scripts_outputs_txt/patching_phase1/patched/agentic/`
- Agentic videos: `data/libero/videos/patching_phase1/patched/agentic/`
- Signal files: `scripts_outputs_txt/patching_phase1/patched/`

---

## Background: What Failed and Why

The previous run (`clean_log/run_20260504_153739_clean.txt`) gave 0/5 on sanity check and 0/25 on D3.

**Primary bug identified:** The sanity check patches ALL 788 KV cache positions with values from a donor built at t=0. Positions 0–587 are image tokens. By overwriting them with stale initial-state images at every rollout step, the model can no longer see the current scene — the robot arm flails wildly.

**Secondary observation:** The D3 video shows the robot placing the bowl on the STOVE with correct, purposeful behavior — as if the patch at pos 594 had zero effect. This could be because (a) the patch is a no-op, or (b) single-position patching is insufficient given that 18 layers of bidirectional attention has smeared the "stove" signal across all 200 language positions.

**Key architecture fact:** In π₀.₅, the prefix (788 tokens) is processed with FULL bidirectional attention across all 18 layers. The KV cache is the OUTPUT of this complete pass. By the time we patch, every language token position (588–787) has already incorporated information from every other position — including the "stove" signal at 594. Patching pos 594 only changes what 594 emits to suffix queries; it does not un-infect the neighboring positions. This is why we may need to patch all 200 language positions, not just 594.

Language tokens span positions **588–787** (200 tokens). Image tokens span **0–587** (3 × 196 = 588 tokens). Never patch image tokens with a pre-computed donor.

---

## Step 0 — Debug: Verify donor ≠ corrupt KV at pos 594

**Before running any rollout**, confirm that the donor and corrupt KV caches actually differ at the position we're patching. If they don't differ, patching is a no-op regardless of implementation correctness.

Add this check to `eval_libero()` in `main_patching_expt.py`, right after `donor_kv_cache` is harvested (after the `num_steps_wait` loop). Build a corrupt KV cache from the same initial obs but with the corrupt prompt, then compare:

```python
# --- Step 0 debug: verify donor != corrupt at pos 594 ---
corrupt_obs_dict = {
    "observation/image": _preprocess_img(initial_obs["agentview_image"], args.resize_size),
    "observation/wrist_image": _preprocess_img(initial_obs["robot0_eye_in_hand_image"], args.resize_size),
    "observation/state": _make_state(initial_obs),
    "prompt": args.corrupt_prompt,
}
corrupt_inputs = policy._input_transform(jax.tree.map(lambda x: x, corrupt_obs_dict))
corrupt_inputs_jax = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], corrupt_inputs)
corrupt_observation = _model.Observation.from_dict(corrupt_inputs_jax)
corrupt_kv = policy._model.build_donor_kv_cache(corrupt_observation)

K_d, V_d = donor_kv_cache
K_c, V_c = corrupt_kv
diff_K = float(jnp.max(jnp.abs(K_d[:, :, 594, :, :] - K_c[:, :, 594, :, :])))
diff_V = float(jnp.max(jnp.abs(V_d[:, :, 594, :, :] - V_c[:, :, 594, :, :])))
tqdm.tqdm.write(f"[DEBUG pos594] donor vs corrupt L-inf: K={diff_K:.6f}, V={diff_V:.6f}")

# Also check a middle language position as control
mid = 688
diff_K_mid = float(jnp.max(jnp.abs(K_d[:, :, mid, :, :] - K_c[:, :, mid, :, :])))
tqdm.tqdm.write(f"[DEBUG pos{mid}] donor vs corrupt L-inf: K={diff_K_mid:.6f}")
# --- end debug ---
```

**Run:** just trigger the script for 1 episode (add `--args.num-trials-per-task 1` and no `--sanity-check`) so it hits this code path and prints the norms. No full rollout needed.

**Interpret:**
- `diff_K` and `diff_V` >> 0 (e.g., > 0.01): donor and corrupt differ at pos 594 — patching will do something. Proceed to Step 1.
- `diff_K` ≈ 0 and `diff_V` ≈ 0: the two prompts produce IDENTICAL KV at pos 594. The tokenizer may put "plate"/"stove" at a different position. Check `inspect_kv_cache.py` output or re-run the Phase 1 verify script. Do not proceed until this is resolved.

**Log this:** record the norm values in `status_cc/patching_implementation.md §8.2` (implementation notes).

---

## Step 1 — Verify mechanism: patch all language tokens (C3, N=5)

**What changes:** Fix the sanity check to patch language tokens only (not image tokens).

In `eval_libero()` in `main_patching_expt.py`:
```python
# Before:
if args.sanity_check:
    patch_positions = tuple(range(788))

# After:
if args.sanity_check:
    patch_positions = tuple(range(588, 788))   # language tokens only; do NOT patch image tokens (0–587)
```

**Run C3 (sanity, N=5):**
```bash
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FULL="scripts_outputs_txt/patching_phase1/patched/agentic/run_${TIMESTAMP}_step1_c3_full.txt"
CLEAN="scripts_outputs_txt/patching_phase1/patched/agentic/run_${TIMESTAMP}_step1_c3_clean.txt"
mkdir -p scripts_outputs_txt/patching_phase1/patched/agentic
mkdir -p data/libero/videos/patching_phase1/patched/agentic/step1_c3

{
python examples/libero/main_patching_expt.py \
  --args.task-suite-name libero_goal \
  --args.task-name-filter "put the bowl on the plate" \
  --args.clean-prompt "put the bowl on the plate" \
  --args.corrupt-prompt "put the bowl on the stove" \
  --args.sanity-check \
  --args.num-trials-per-task 5 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/patching_phase1/patched/agentic/step1_c3
} 2>&1 \
  | tee "$FULL" \
  | tr '\r' '\n' | sed -u $'s/\x1b\\[[0-9;]*[A-Za-z]//g' \
  | grep -E --line-buffered '^(===|\[ep |\[DEBUG|INFO:root:|ERROR:root:)' \
  | tee "$CLEAN"
```

**Interpret:**
- ≥ 3/5 success: mechanism confirmed. Write `1_SANITY_CHECK_SUCCESS.txt`. Proceed to Step 2.
- < 3/5 success: mechanism still broken. Check full log for errors. Possible issues: (a) venv/import error, (b) patch not applying (check if `_apply_kv_patch` output is being used), (c) donor obs built incorrectly. Debug before proceeding.

**If C3 passes**, also do a quick D3 run (pos 594, N=5 for speed) to see if single-position patching already works:
```bash
# Same pattern but without --sanity-check, with --patch-positions "594", N=5
# Use label: step1_d3_pos594_quick
```
If D3 > 0/5 here, proceed directly to Step 2 with N=25.

**Commit** after Step 1 regardless of outcome.

---

## Step 2 — Test single-position causality: pos 594 only (N=25)

**What changes:** No code change. Use the default `--args.patch-positions "594"` from Step 1's codebase.

**Run D3 (N=25):**
```bash
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FULL="scripts_outputs_txt/patching_phase1/patched/agentic/run_${TIMESTAMP}_step2_d3_pos594_full.txt"
CLEAN="scripts_outputs_txt/patching_phase1/patched/agentic/run_${TIMESTAMP}_step2_d3_pos594_clean.txt"
mkdir -p data/libero/videos/patching_phase1/patched/agentic/step2_d3_pos594

{
python examples/libero/main_patching_expt.py \
  --args.task-suite-name libero_goal \
  --args.task-name-filter "put the bowl on the plate" \
  --args.clean-prompt "put the bowl on the plate" \
  --args.corrupt-prompt "put the bowl on the stove" \
  --args.patch-positions "594" \
  --args.num-trials-per-task 25 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/patching_phase1/patched/agentic/step2_d3_pos594
} 2>&1 \
  | tee "$FULL" \
  | tr '\r' '\n' | sed -u $'s/\x1b\\[[0-9;]*[A-Za-z]//g' \
  | grep -E --line-buffered '^(===|\[ep |\[DEBUG|INFO:root:|ERROR:root:)' \
  | tee "$CLEAN"
```

**Interpret:**
- > 5/25 (>20%): pos 594 is causal. Write `2_PATCHING_MEANINGFUL_RESULT.txt`. Done.
- 1–5/25: weak signal. Report it, proceed to Step 3 to see if widening helps.
- 0/25: pos 594 alone is insufficient. Proceed to Step 3.

**Commit** after Step 2.

---

## Step 3 — Find minimal sufficient patch set (binary search over language positions)

The goal is to find the smallest contiguous range of language positions that produces meaningful D3 recovery. This tells us how distributed the "stove" signal is.

Use N=10 for all binary search runs (fast iteration). Only run N=25 when you've found a promising range.

**3a. Upper bound: all 200 language positions (588–787)**

Generate the positions string:
```bash
LANG_POSITIONS=$(python -c "print(','.join(map(str, range(588, 788))))")
```

Run D3 with all language positions (N=10):
```bash
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# use label: step3a_d3_lang200
python examples/libero/main_patching_expt.py \
  ... \
  --args.patch-positions "$LANG_POSITIONS" \
  --args.num-trials-per-task 10 \
  ...
```

- If < 2/10: the pre-computed donor approach may be fundamentally broken. Go to Step 4.
- If ≥ 2/10: the distributed signal is there. Proceed to binary search (3b).

**3b. Binary search**

Split language positions into halves and test each:
- First half: 588–687 (100 positions): `python -c "print(','.join(map(str, range(588, 688))))"`
- Second half: 688–787 (100 positions): `python -c "print(','.join(map(str, range(688, 788))))"`

Run each with N=10. Whichever gives higher recovery, recurse into that half. Continue until you reach a range of ~10–20 positions or until further narrowing drops below 2/10.

Use label pattern: `step3b_d3_pos588-687`, `step3b_d3_pos688-787`, `step3c_d3_pos...`, etc.

**When you find a minimal sufficient set:** run it with N=25 and write `2_PATCHING_MEANINGFUL_RESULT.txt`.

**Commit** after each binary search iteration.

---

## Step 4 — Per-step donor (separate script)

Only reach here if Steps 1–3 all produce < 2/10 on D3 (or if Step 1 C3 itself fails to pass).

Create a new script `examples/libero/main_patching_expt_per_step_donor.py`. This is a SEPARATE file — do not add a flag to `main_patching_expt.py`. Keeping them separate makes it easy to diff the two approaches later.

**Key difference from `main_patching_expt.py`:** instead of harvesting donor once at the start, rebuild BOTH donor and recipient KV at each inference call using the CURRENT observation:

```python
# Inside the rollout loop, replace the policy.infer() call with:
def infer_with_per_step_patch(element, patch_positions):
    # Apply input transform (same as Policy.infer does internally)
    inputs = policy._input_transform(jax.tree.map(lambda x: x, element))
    inputs_jax = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = _model.Observation.from_dict(inputs_jax)

    # Build recipient KV (corrupt prompt, current images)
    recipient_kv = policy._model.build_donor_kv_cache(observation)

    # Build donor KV (clean prompt, same current images) 
    clean_element = dict(element)
    clean_element["prompt"] = args.clean_prompt
    clean_inputs = policy._input_transform(jax.tree.map(lambda x: x, clean_element))
    clean_inputs_jax = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], clean_inputs)
    clean_observation = _model.Observation.from_dict(clean_inputs_jax)
    donor_kv = policy._model.build_donor_kv_cache(clean_observation)

    # Patch and decode
    policy._sample_kwargs["donor_kv_cache"] = donor_kv
    policy._sample_kwargs["patch_positions"] = patch_positions
    # Inject patched KV... (see note below)
    return policy.infer(element)["actions"]
```

Note: per-step donor requires that `sample_actions` in `pi0.py` uses the donor_kv_cache from `_sample_kwargs` to patch AFTER building the recipient cache — which is already how it works. So the donor simply needs to be set in `_sample_kwargs` fresh at each step.

Start with all language positions (588–787), N=5 sanity + N=10 D3. Use label `step4_perstep`.

If per-step donor also fails at 0%: stop. Write `0_FAILURE_REPORT.txt` with a summary of all approaches tried and their results. The mechanism has a deeper bug requiring human review.

---

## Signal Files

Write these to `scripts_outputs_txt/patching_phase1/patched/` (NOT the agentic subdir):

**`1_SANITY_CHECK_SUCCESS.txt`** — write when C3 ≥ 3/5:
```
Sanity check passed on [DATE].
Step: [which step fixed it]
Success rate: X/5
Log: scripts_outputs_txt/patching_phase1/patched/agentic/[filename]
What fixed it: [one sentence — e.g., "patched only language positions 588-787 instead of all 788"]
See status_cc/patching_implementation.md §8.1 for full results table.
```

**`2_PATCHING_MEANINGFUL_RESULT.txt`** — write when D3 > 2/10:
```
Meaningful patching result on [DATE].
Step: [which step]
Patch positions: [what was patched — e.g., "pos 594 only" or "positions 620-650"]
Success rate: X/N (clean baseline: 25/25, corrupt baseline: 0/25)
Log: scripts_outputs_txt/patching_phase1/patched/agentic/[filename]
Interpretation: [one sentence — e.g., "pos 594 alone sufficient" or "positions 620-650 form minimal causal set"]
See status_cc/patching_implementation.md §8.1 for full results table.
```

**`0_FAILURE_REPORT.txt`** — write only if Step 4 also fails:
```
All approaches failed on [DATE].
Steps tried: [list]
Debug norms at pos 594: K=X, V=Y
Observations: [what the videos/logs showed at each step]
Hypothesis for next human investigation: [your best guess at the root cause]
```

---

## Result Recording

After every completed run (pass or fail), add a row to `status_cc/patching_implementation.md §8.1`:

| Run type | Prompt | KV cache | N | Success rate | Notes |
|----------|--------|----------|---|-------------|-------|
| [step label] | corrupt | [what was patched] | N | X/N | [log filename] |

---

## Commit Protocol

After each step (or each binary search iteration in Step 3):
```bash
git add -p   # stage only relevant changes
git commit -m "agentic step X: [one-line description of what was tried and result]"
```

Example messages:
- `agentic step0: debug norms show K=0.23, V=0.18 at pos594 — donor differs from corrupt`
- `agentic step1: fix sanity to language-only (588-787); C3=4/5 — mechanism confirmed`
- `agentic step2: D3 pos594 N=25 = 0/25 — single position insufficient`
- `agentic step3a: D3 all-language N=10 = 7/10 — signal distributed; proceeding to binary search`
