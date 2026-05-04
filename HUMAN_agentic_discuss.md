# Agentic Patching Debug — Planning Document

Human-facing planning doc for designing the autonomous agent loop that will debug and fix the KV-cache patching experiment. Iterate here before launching the agent.

---

## 1. Observed Failure (2026-05-04 run)

Log: `scripts_outputs_txt/patching_phase1/patched/clean_log/run_20260504_153739_clean.txt`

- **C3 (sanity, all 788 pos patched):** 0/5 — robot arm extends wildly, no coherent behavior
- **D3 (patched pos 594 only):** 0/25 — robot places bowl on STOVE (correct task structure, wrong target) — as if no patch was applied at all

**Video evidence:**
- Sanity: wild/incoherent arm → model acting blind (primary cause: image tokens patched with stale t=0 images)
- D3: sensible "stove" behavior → patch at pos 594 has zero observable effect on behavior

---

## 2. Root Cause Analysis

### Bug 1 (Primary): Sanity check patches image tokens with stale donor images

Donor KV cache is harvested once from `initial_states[0]` at t=0. It covers ALL 788 positions — including image token positions 0–587. During rollout, at each step `sample_actions` builds a fresh prefix KV cache from the current observation, then we overwrite ALL 788 positions with the stale donor. This erases the model's access to the current visual state for every step after t=0.

The robot arm flails wildly because it's "seeing" the initial frozen scene throughout a 300-step rollout while the actual scene changes around it.

**Fix:** Patch only language token positions (588–787). Image positions should always come from the model's own current-observation prefix pass.

### Bug 2 (Secondary — may compound Bug 1): Patch at pos 594 has no effect

D3 video shows the model faithfully executing the "stove" task — correct, purposeful behavior that ignores the patch entirely. Either:

- a) `_apply_kv_patch` is a silent no-op in JAX (e.g., the `.at[].set()` result is not being captured)
- b) Pos 594 alone carries insufficient signal — 18 layers of bidirectional prefix attention has distributed "stove" across ALL 200 language positions (588–787); patching one position doesn't shift the aggregate signal enough to change behavior
- c) Wrong index — verify pos 594 is actually where plate/stove differ in this prompt

---

## 3. Architectural Clarification: KV-cache vs Residual Stream Patching

**The key difference is WHEN the intervention happens, not bidirectional vs causal.**

In SmolVLA residual stream patching (layer 1, pos P):
- You intervene mid-forward-pass
- Layers 2–N recompute with the patched value
- Patch propagates through remaining layers, influencing all subsequent positions

In our KV-cache patching:
- The full 18-layer bidirectional prefix pass runs to COMPLETION first
- THEN we overwrite specific positions in the final KV cache output
- No recomputation, no propagation
- Neighboring positions (588–593, 595–787) already absorbed "stove" from 594 during those 18 layers — patching 594's final KV values cannot undo that

**Consequence:** Patching pos 594 only changes what 594 emits to suffix queries. It does not un-infect the neighboring language positions that already saw the corrupt 594. This is why patching all 200 language positions may be needed for a strong intervention.

---

## 4. Per-step Donor vs Pre-computed Donor

**Pre-computed donor** (current implementation, chosen in `kv_cache_findings.md`):
- Harvest donor KV once from `initial_states[0]`, reuse for all steps
- Language-only patch (Bug 1 fix) restores current visual state
- Subtle remaining issue: donor language token KVs were computed with episode 0's images as context (small contamination, worth measuring but not blocking)

**Per-step donor** (`kv_cache_findings.md` §end, deferred for PoC):
- At each step: current_obs + clean_prompt → donor KV; current_obs + corrupt_prompt → recipient KV; patch specific positions
- No staleness at all; only difference between donor and recipient is the language prompt
- Cleanest possible causal claim; costs one extra prefix forward pass per inference call

**Decision:** Pre-computed donor + language-only patching first. Per-step donor as a **separate script** if pre-computed approaches don't work.

---

## 5. Agent Approaches (Ordered — agent follows this sequence)

All agent logs → `scripts_outputs_txt/patching_phase1/patched/agentic/`

**Guiding principle:** Keep things clean and separate. Each approach is either a flag change or a separate script. This makes it easy to diff against the current branch later when updating `kv-cache-patching`.

The overall progression: verify mechanism → test single-position causality → find minimal sufficient patch set → per-step donor fallback.

### Step 0 — Debug: Verify patch is non-trivial (no rollout needed)

Before touching any rollout, confirm the donor and corrupt KV caches actually differ at pos 594. Add a one-shot check right after `harvest_donor_kv_cache`:

```python
# Build corrupt KV for comparison (discard after debug)
corrupt_kv = policy._model.build_donor_kv_cache(corrupt_observation)  # corrupt prompt, same obs
K_d, V_d = donor_kv_cache
K_c, V_c = corrupt_kv
diff_K = float(jnp.max(jnp.abs(K_d[:, :, 594, :, :] - K_c[:, :, 594, :, :])))
diff_V = float(jnp.max(jnp.abs(V_d[:, :, 594, :, :] - V_c[:, :, 594, :, :])))
tqdm.tqdm.write(f"[DEBUG] donor vs corrupt L-inf at pos 594: K={diff_K:.6f}, V={diff_V:.6f}")
```

Expected: both >> 0. If ~0, the two prompts produce identical KV at pos 594 — wrong position or tokenization issue, stop and investigate before running any rollout.

### Step 1 — Verify mechanism: patch ALL language positions (N=5)

This is the mechanism sanity check. Fix the sanity branch to patch only language tokens (not image tokens), and run N=5:

```python
# Before (current): patch_positions = tuple(range(788))    ← patches image tokens too (bug)
# After:            patch_positions = tuple(range(588, 788)) ← language only (200 positions)
```

Expected: ≥ 3/5 success. Robot should behave coherently. If < 3/5, the mechanism is broken — stop and debug before proceeding.

**This is the gate.** Do not proceed to Step 2 until Step 1 passes.

### Step 2 — Test single-position causality: pos 594 only (N=25)

With the mechanism confirmed, run D3 as originally designed — patch only pos 594:

```python
patch_positions = (594,)
```

Run N=25. This is the core scientific question: is pos 594 the causal position?

Expected outcomes:
- High recovery (>60%): pos 594 is the primary causal position → write `2_PATCHING_MEANINGFUL_RESULT.txt`, stop
- Partial recovery (20–60%): pos 594 is one of several causal positions → proceed to Step 3
- Near 0%: pos 594 alone is insufficient → proceed to Step 3

### Step 3 — Find minimal sufficient patch set (binary search over language positions)

If pos 594 alone is insufficient, the "stove" signal is distributed across language positions via 18 layers of bidirectional attention mixing. Systematically narrow down which positions are needed:

**3a.** Patch ALL 200 language positions (588–787), N=10. If this also fails, go to Step 4.

**3b.** If 3a succeeds: binary search to find the minimal subset.
- Try first half: positions 588–687 (N=10)
- Try second half: positions 688–787 (N=10)
- Whichever half gives recovery, recurse into it
- Continue until you find the smallest contiguous range that gives >20% recovery

Goal: identify the minimal patch set and report it in `2_PATCHING_MEANINGFUL_RESULT.txt`.

### Step 4 — Per-step donor (separate script)

If Steps 1–3 all fail: implement per-step donor as a NEW separate script `examples/libero/main_patching_expt_per_step_donor.py`.

Per-step donor structure (each inference step):
1. Build donor KV: current_obs + clean_prompt
2. Build recipient KV: current_obs + corrupt_prompt
3. Patch: `recipient_kv[patch_positions] = donor_kv[patch_positions]`
4. Decode action from patched recipient KV

Start with all language positions (588–787), N=5 sanity + N=10 D3. If this also gives 0%, stop and write `0_FAILURE_REPORT.txt` — the mechanism has a deeper bug requiring human review.

---

## 6. Signal Files (agent writes these)

All signal files → `scripts_outputs_txt/patching_phase1/patched/`

| File | Write when | Contents |
|------|-----------|----------|
| `1_SANITY_CHECK_SUCCESS.txt` | C3 ≥ 60% (3+/5) | Which step fixed it, success rate, pointer to agentic log file |
| `2_PATCHING_MEANINGFUL_RESULT.txt` | D3 > 20% (clearly above 0% corrupt floor) | Step used, positions patched, success rate, interpretation |
| `0_FAILURE_REPORT.txt` | Per-step donor also fails at Step 3 | What was tried, what debug norms showed, where to look next |

---

## 7. Output File Conventions (agent)

- Logs: `scripts_outputs_txt/patching_phase1/patched/agentic/run_YYYYMMDD_HHMMSS_<label>.txt`
- Label examples: `step0_debug`, `step1_c3_langonly`, `step1_d3_pos594`, `step2_d3_lang200`, `step3_perstep_c3`
- Videos: `data/libero/videos/patching_phase1/patched/agentic/<label>/`
- Each run: produce both full log (all output) and clean log (key result lines) using same split-logging pattern as `run_patching_phase1.sh`

---

## 8. Open Questions (resolve before launching agent)

- [x] Step 0 (debug norm check) goes FIRST before any rollout — confirmed
- [x] Per-step donor = separate script, not a flag — confirmed
- [x] Agentic outputs → `scripts_outputs_txt/patching_phase1/patched/agentic/` — confirmed
- [ ] Should the agent commit after each step? (Recommended: yes, small commits)
- [ ] Confirm: D3 > 2/10 (20%) is the threshold for "meaningful" — ok?
- [ ] Should agent update `status_cc/patching_implementation.md §8.1` after each run, or only at the end?

---

## 9. What to Do Next (before launching agent)

1. Create branch `patching_phase1_agentic`
2. Write the agent-facing guide (`status_cc/agent_patching_guide.md`) — more prescriptive than this doc; tells agent exactly what to run in what order
3. Create agentic output directory structure
4. Commit and push → pull on RunPod → launch agent there
