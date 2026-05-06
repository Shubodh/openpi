# Phase 2 — Implementation and Results

**Branch:** `axmech-agentic-phase-2`
**Status:** In progress — begun 2026-05-05.
**Agent guide:** `progress_cc/phase2/guide.md` — read this first before running anything.

---

## Table of Contents

1. [Context from Phase 1](#1-context-from-phase-1)
2. [Architecture Decisions for Phase 2](#2-architecture-decisions-for-phase-2)
3. [Phase 2a — Simple Pair 2 Battery](#3-phase-2a--simple-pair-2-battery)
4. [Phase 2b — Alpha Sweep](#4-phase-2b--alpha-sweep)
5. [Implementation Checklist](#5-implementation-checklist)
6. [Results — Phase 2a](#6-results--phase-2a)
7. [Results — Phase 2b](#7-results--phase-2b)
8. [Current Status and Next Steps](#8-current-status-and-next-steps)
9. [Results — Phase 2c](#9-results--phase-2c)

---

## 1. Context from Phase 1

**Phase 1 branch:** `patching_phase1_agentic`
**Phase 1 guide:** `status_cc/agent_patching_guide.md`
**Phase 1 implementation doc:** `status_cc/patching_implementation.md`
**Phase 1 signal files:** `scripts_outputs_txt/patching_phase1/patched/`
**Phase 1 scripts:** `examples/libero/main_patching_expt.py`, `examples/libero/main_patching_expt_per_step_donor.py`

### What Phase 1 established (on Simple Pair 1: `plate` ↔ `stove`)

| Finding | Result |
|---------|--------|
| Clean baseline D1 | 100% (25/25) |
| Corrupt baseline D2 | 0% (0/25) |
| Pre-computed donor (language-only, 588–787) | 0/5 — failed |
| Per-step donor (language-only, 588–787) | 0/5 — failed |
| Per-step donor (full prefix, 0–787) | 5/5 and 10/10 — **working positive control** |
| Per-step donor (image prefix, 0–587) | 10/10 — image-token K/V alone sufficient |
| Per-step donor (image half A, 0–293) | 0/10 — insufficient |
| Per-step donor (image half B, 294–587) | 8/10 — main signal region |
| Per-step donor (image quarter B1, 294–440) | 1/10 — insufficient alone |

**Phase 1 current state (as of 2026-05-05):** Binary search ongoing. Next probe is positions `441–587`. Layer localization and K/V separation are planned as 1b after localization completes.

**Key architectural finding:** π₀.₅'s prefix attention is fully bidirectional. By the time we patch the KV cache, the corrupt prompt has already mixed its signal into image-token K/V. Patching only language-token K/V is insufficient. The recoverable signal sits in image tokens, specifically in positions `294–587` (wrist camera and masked stream region).

**Pre-computed vs per-step:** Pre-computed full-prefix was tested as the original sanity check (`run_patching_phase1.sh` C3 condition) and failed because it overwrites current image-token K/V with stale t=0 donor images, blinding the model. **Per-step donor is the only working approach.** Phase 2 confirms this finding on a new pair and uses per-step donor throughout.

**Implementation notes:** Phase 2 reuses the existing baseline runner `examples/libero/main_corrupt_run_expt.py::eval_libero` for clean/corrupt no-patch controls; task filtering and prompt override happen at lines 95-100, websocket policy inference at lines 142-163, and video/result logging at lines 182-204. Patched runs use `examples/libero/main_patching_expt_per_step_donor.py::eval_libero`, which rebuilds a donor KV cache from the current observation before each action chunk at lines 301-310 and then calls `policy.infer()` with the corrupt prompt at line 311.

---

## 2. Architecture Decisions for Phase 2

| Decision | Resolution |
|----------|-----------|
| Task pair | Simple Pair 2: `put_the_bowl_on_top_of_the_cabinet` (clean) ↔ `put_the_wine_bottle_on_top_of_the_cabinet` (corrupt) — different object, same destination |
| Donor approach | Per-step donor only (`main_patching_expt_per_step_donor.py`) |
| Phase 2a goal | Replicate Phase 1's methodology on Simple Pair 2 → find minimal sufficient image-token patch set |
| Phase 2b goal | Alpha sweep: vary interpolation weight α ∈ {0, 0.25, 0.5, 0.75, 1.0} on Phase 2a's localized region |
| Alpha implementation | Add `patch_alpha: float = 1.0` to `_apply_kv_patch` and `sample_actions` in `pi0.py`; add `--args.patch-alpha` to per-step donor script |
| Alpha endpoint gate | α=0 must ≈ corrupt baseline (~0%), α=1 must reproduce per-step full-prefix result (≥60%) before running intermediate values |
| Output structure | Videos → `data/libero/videos/phase2/`; logs → `progress_cc/phase2/signal_files/logs/`; signal files → `progress_cc/phase2/signal_files/` |
| Script policy | Additive changes only to `pi0.py` — new default-preserving parameters. No modification of existing Phase 1 behavior. |

**Implementation notes:** The per-step donor path is implemented by `build_kv_cache_from_element()` in `examples/libero/main_patching_expt_per_step_donor.py` lines 124-133 and `make_element()` lines 136-142. Patch configuration is parsed and logged in `eval_libero()` lines 173-183, then wired into `policy._sample_kwargs` at lines 256-260 and again for each replanned action chunk at lines 303-310. Model-side patching is centralized in `src/openpi/models/pi0.py::_apply_kv_patch` lines 229-248 and invoked from `Pi0.sample_actions()` lines 279-282.

---

## 3. Phase 2a — Simple Pair 2 Battery

**Task pair (Simple Pair 2 — different object, same destination):**
- Environment task: `put_the_bowl_on_top_of_the_cabinet`
- Clean prompt: `"put the bowl on top of the cabinet"`
- Corrupt prompt: `"put the wine bottle on top of the cabinet"`
- Task name filter flag: `"put the bowl on top of the cabinet"`
- Scientific axis: Phase 1 tested same-object-different-destination (plate ↔ stove). This pair tests different-object-same-destination (bowl ↔ wine bottle, both on cabinet) — the orthogonal dimension.

**Experiment sequence (same methodology as Phase 1):**

| Code | Condition | N | Expected |
|------|-----------|---|----------|
| A-D1 | Clean baseline (correct prompt, normal KV) | 25 | ~100% |
| A-D2 | Corrupt baseline (wine bottle prompt, normal KV) | 25 | ~0% |
| A-C3 | Per-step full-prefix sanity (all 0–787, per-step donor) | 5 | ≥60% (3/5) |
| A-lang | Language-only sanity (588–787, per-step donor) | 5 | Unknown — Phase 1 found 0% on destination axis; object axis may differ |
| A-lang-binary | Binary search within 588–787 (only if A-lang passes) | N=10 per probe | Recurse until minimal |
| A-D3 | Image prefix (0–587, per-step donor) | 10 | >2/10 if Phase 1 finding holds |
| A-binary | Binary search within 0–587 to find minimal set | N=10 per probe | Recurse until minimal |
| A-final | Minimal set promoted to N=25 | 25 | >5/25 (20%) |

**Success gates:**
- A-C3 ≥ 3/5 before anything else. If A-C3 fails, stop and write `0_PHASE2A_FAILURE.txt`.
- A-lang is run after A-C3 regardless. If A-lang passes, run A-lang-binary before A-D3. If A-lang fails, proceed directly to A-D3.

**Implementation notes:** No new code was required for Phase 2a. Baselines used `main_corrupt_run_expt.py::eval_libero`, where `args.corrupt_prompt` can override task language at lines 99-100. Patching probes used `main_patching_expt_per_step_donor.py::eval_libero`; `--args.patch-positions` is parsed by `_parse_int_spec()` lines 155-166 and applied through `policy._sample_kwargs["patch_positions"]` at lines 256 and 306. The actual KV replacement for the requested absolute prefix positions happens in `pi0.py::_apply_kv_patch` lines 239-247.

---

## 4. Phase 2b — Alpha Sweep

**Prerequisite:** Phase 2a complete with a confirmed minimal patch set (A-final N=25 > 5/25).

**What changes in `pi0.py`:**

In `_apply_kv_patch`, add `alpha: float = 1.0` parameter and change the patch operation:
```python
# Before:
K = K.at[:, :, pos, :, :].set(K_d[:, :, pos, :, :])
V = V.at[:, :, pos, :, :].set(V_d[:, :, pos, :, :])

# After (with alpha interpolation):
K_corrupt_pos = K[:, :, pos, :, :]
V_corrupt_pos = V[:, :, pos, :, :]
K = K.at[:, :, pos, :, :].set(alpha * K_d[:, :, pos, :, :] + (1 - alpha) * K_corrupt_pos)
V = V.at[:, :, pos, :, :].set(alpha * V_d[:, :, pos, :, :] + (1 - alpha) * V_corrupt_pos)
```

In `sample_actions`, add `patch_alpha: float = 1.0` to the signature and pass it to `_apply_kv_patch`.

In `main_patching_expt_per_step_donor.py`, add `patch_alpha: float = 1.0` to `Args` and wire it into `policy._sample_kwargs["patch_alpha"]`.

**Endpoint verification (gate before intermediate values):**

| Condition | Alpha | Expected | Interpretation |
|-----------|-------|----------|----------------|
| B-alpha0 | 0.0 | ~0% (= corrupt baseline) | No patch applied; must match D2 |
| B-alpha1 | 1.0 | ≥60% (= per-step full-prefix) | Full donor patch; must match A-C3/A-final |

Run N=5 each. If either endpoint fails, the implementation is wrong — debug before proceeding.

**Intermediate sweep (only if both endpoints verified):**

| Condition | Alpha | N | Output |
|-----------|-------|---|--------|
| B-alpha025 | 0.25 | 10 | Success rate + videos |
| B-alpha05 | 0.50 | 10 | Success rate + videos |
| B-alpha075 | 0.75 | 10 | Success rate + videos |

**Output organization:**
- Videos: `data/libero/videos/phase2/alpha_sweep/alpha_{value}/` (e.g., `alpha_0.25/`, `alpha_0.50/`)
- Success rates: `progress_cc/phase2/signal_files/alpha_sweep_results.csv` with columns: `alpha,successes,trials,success_rate`
- A-D2 and A-C3 results from Phase 2a serve as the α=0 and α=1 reference values in the CSV

**Implementation notes:** Phase 2b added alpha interpolation in `src/openpi/models/pi0.py::_apply_kv_patch`: the `alpha` parameter is in the signature at line 233, corrupt and donor K/V slices are read at lines 240-243, and the interpolated write is performed at lines 244-245. `Pi0.sample_actions()` exposes `patch_alpha` at line 263 and passes it into `_apply_kv_patch()` at lines 279-282. The CLI surface is `examples/libero/main_patching_expt_per_step_donor.py::Args.patch_alpha` line 95; `eval_libero()` logs it at line 183 and writes it into `policy._sample_kwargs` at lines 260 and 310.

---

## 5. Implementation Checklist

### Phase 2a

- [ ] **A1.** Run D1 baseline (clean prompt, normal KV, N=25)
- [ ] **A2.** Run D2 baseline (corrupt prompt, normal KV, N=25)
- [ ] **A3.** Run C3 per-step full-prefix sanity (N=5) — gate; if fails stop
- [ ] **A4-lang.** Run language-only sanity (588–787, N=5) — if passes, run language binary search before A4-img
- [ ] **A4-img.** Run image prefix probe (0–587, N=10) — skip if A4-lang passed and localized
- [ ] **A5.** Binary search within whichever region passed (language 588–787 or image 0–587) → find minimal set
- [ ] **A6.** Promote minimal set to N=25 → write `progress_cc/phase2/signal_files/1_PHASE2A_MEANINGFUL_RESULT.txt`

### Phase 2b

- [ ] **B1.** Add `alpha` parameter to `_apply_kv_patch` in `src/openpi/models/pi0.py` (default=1.0, additive)
- [ ] **B2.** Add `patch_alpha` to `sample_actions` in `pi0.py` (default=1.0, additive)
- [ ] **B3.** Add `patch_alpha` to `Args` and `_sample_kwargs` in `main_patching_expt_per_step_donor.py`
- [ ] **B4.** Endpoint verification: run α=0 (N=5) and α=1 (N=5) — gate
- [ ] **B5.** Run α=0.25, 0.50, 0.75 (N=10 each), saving videos to per-alpha subdirectories
- [ ] **B6.** Write `progress_cc/phase2/signal_files/alpha_sweep_results.csv`
- [ ] **B7.** Write `progress_cc/phase2/signal_files/2_PHASE2B_ALPHA_SWEEP_COMPLETE.txt`

### Phase 2c (run after 2a+2b complete)

- [x] **C1-lang.** Run Pair A lang-only sanity (588–787, N=5) — failed 0/5; proceed to C1-img
- [x] **C1-img.** Run Pair A img sanity (294–587, N=5) — failed 0/5; proceed to C1-full
- [x] **C1-full.** Run Pair A full-prefix (0–787, N=5) — passed 5/5; binary search within 0–787
- [ ] **C2a-lang.** Run dest-only lang token (rack clean[595]→plate corrupt[594], N=10) — video-only
- [ ] **C2a-c1minimal.** Run dest-only with C1's minimal positions (N=10) — video-only
- [ ] **C2b-lang.** Run obj-only lang tokens (wine_bottle clean[591–592]→bowl corrupt[591–592], N=10) — video-only
- [ ] **C2b-c1minimal.** Run obj-only with C1's minimal positions (N=10) — video-only
- [ ] **C3-lang.** Run Pair D lang-only (588–787, N=5) — if passes, binary search + N=25; if fails, proceed to C3-img
- [ ] **C3-img.** Run Pair D img (294–587, N=5) — fallback
- [ ] **C3-full.** Run Pair D full-prefix (0–787, N=5) — last resort; if fails, stop
- [ ] **C4.** Write `3_PHASE2C_COMPLETE.txt` (or `0_PHASE2C_FAILURE.txt` if C1 fails all three regions)

**Implementation notes:** Checklist state is a manual tracking layer over the run logs and signal files; it does not drive code. Completed Phase 2a and 2b entries correspond to committed changes and logs from the scripts above. The only pending code change called out by the updated Phase 2c guide is `patch_source_positions` support, which is not present yet in `pi0.py::_apply_kv_patch` lines 229-248 or `main_patching_expt_per_step_donor.py::Args` lines 69-108.

---

## 6. Results — Phase 2a

*(Agent fills this in after each run.)*

### 6.1 Results table

**Implementation notes:** All A-D1/A-D2 rows were produced by `examples/libero/main_corrupt_run_expt.py::eval_libero`, using the prompt override path at lines 99-100 and websocket inference at lines 142-163. All patched rows were produced by `examples/libero/main_patching_expt_per_step_donor.py::eval_libero`; the donor cache is rebuilt from each live observation at lines 301-305 before the corrupt-prompt inference at line 311. Success/failure counts in the table come from the scripts' final log summaries: baseline logging at `main_corrupt_run_expt.py` lines 195-204, patched logging at `main_patching_expt_per_step_donor.py` lines 338-342.

| Run code | Prompt | Patch config | N | Success rate | Log file |
|----------|--------|-------------|---|-------------|----------|
| A-D1 | clean | normal | 25 | 24/25 (96%) | `progress_cc/phase2/signal_files/logs/run_20260505_162107_phase2a_baselines_clean.txt` |
| A-D2 | corrupt | normal | 25 | 0/25 (0%) | `progress_cc/phase2/signal_files/logs/run_20260505_162107_phase2a_baselines_clean.txt` |
| A-C3 | corrupt | per-step full-prefix 0–787 | 5 | 5/5 (100%) | `progress_cc/phase2/signal_files/logs/run_20260505_163750_phase2a_c3_fullprefix_clean.txt` |
| A-lang | corrupt | per-step language-only 588–787 | 5 | 0/5 (0%) | `progress_cc/phase2/signal_files/logs/run_20260505_164700_phase2a_lang588-787_clean.txt` |
| A-D3 | corrupt | per-step image prefix 0–587 | 10 | 10/10 (100%) | `progress_cc/phase2/signal_files/logs/run_20260505_165623_phase2a_img0-587_clean.txt` |
| A-bin-1a | corrupt | per-step image half 0–293 | 10 | 1/10 (10%) | `progress_cc/phase2/signal_files/logs/run_20260505_170647_phase2a_img0-293_clean.txt` |
| A-bin-1b | corrupt | per-step image half 294–587 | 10 | 7/10 (70%) | `progress_cc/phase2/signal_files/logs/run_20260505_171739_phase2a_img294-587_clean.txt` |
| A-bin-2a | corrupt | per-step image quarter 294–440 | 10 | 0/10 (0%) | `progress_cc/phase2/signal_files/logs/run_20260505_172654_phase2a_img294-440_clean.txt` |
| A-bin-2b | corrupt | per-step image quarter 441–587 | 10 | 0/10 (0%) | `progress_cc/phase2/signal_files/logs/run_20260505_173744_phase2a_img441-587_clean.txt` |
| A-final | corrupt | per-step image region 294–587 | 25 | 21/25 (84%) | `progress_cc/phase2/signal_files/logs/run_20260505_174905_phase2a_final_n25_clean.txt` |
| | | | | | |

### 6.2 Implementation notes

*(Surprises, deviations from plan, debugging notes.)*

Code-level summary: Phase 2a relied on existing per-step donor mechanics. `_parse_int_spec()` in `main_patching_expt_per_step_donor.py` lines 155-166 allowed both comma-separated and range-style patch specs, `_positions_tag()` lines 145-152 made stable output-directory tags, and `pi0.py::_apply_kv_patch` lines 239-247 overwrote the selected K/V cache positions across all layers by default.

| Date | Note |
|------|------|
| 2026-05-05 | A1/A2 baselines passed contrastive gate for Simple Pair 2: clean prompt recovered 24/25, corrupt wine-bottle prompt recovered 0/25 on the bowl/cabinet environment task. Proceeding to A3 full-prefix per-step sanity. |
| 2026-05-05 | A3 full-prefix per-step sanity passed 5/5, confirming the donor rebuild and KV patch path work for the bowl/cabinet vs wine_bottle/cabinet pair. Proceeding to A4 language-only sanity. |
| 2026-05-05 | A4 language-only sanity failed 0/5. This replicates Phase 1's language-slot insufficiency on the object-identity axis, so the search proceeds to image prefix positions 0–587. |
| 2026-05-05 | A4 image-prefix probe recovered 10/10, showing image-token K/V positions 0–587 are sufficient for Simple Pair 2 recovery. Proceeding to binary localization inside image tokens. |
| 2026-05-05 | Binary split 0–293 recovered only 1/10, below the meaningful threshold. Next probe is the complementary image half 294–587. |
| 2026-05-05 | Binary split 294–587 recovered 7/10, so this half is promoted for recursive localization. Next probes split it into 294–440 and 441–587. |
| 2026-05-05 | Recursive split 294–440 recovered 0/10, below threshold. Next probe is the complementary quarter 441–587. |
| 2026-05-05 | Recursive split 441–587 also recovered 0/10. Since the parent region 294–587 recovered 7/10 but both quarters failed, 294–587 is the smallest defensible contiguous region for promotion to N=25. |
| 2026-05-05 | A-final promoted positions 294–587 to N=25 and recovered 21/25, clearing the >5/25 meaningful-result threshold. Wrote `1_PHASE2A_MEANINGFUL_RESULT.txt`; proceeding to Phase 2b alpha interpolation. |
| 2026-05-05 | Phase 2b endpoint verification passed on positions 294–587: alpha=0.0 recovered 0/5 and alpha=1.0 recovered 5/5. Proceeding to intermediate alpha values without self-correction loops. |
| 2026-05-05 | Alpha sweep alpha=0.25 recovered 0/10. Recorded without rerun per guide. |
| 2026-05-05 | Alpha sweep alpha=0.50 recovered 0/10. Recorded without rerun per guide. |
| 2026-05-05 | Alpha sweep alpha=0.75 recovered 0/10. Wrote `alpha_sweep_results.csv` and `2_PHASE2B_ALPHA_SWEEP_COMPLETE.txt`; proceeding to Phase 2c. |

### 6.3 Minimal patch set found

*(Fill after binary search complete.)*

**Implementation notes:** The minimal-set claim is an experiment-selection result, not a separate code path. The same `main_patching_expt_per_step_donor.py::eval_libero` code path handled full prefix, image half, image quarter, and final promotion by changing only `--args.patch-positions`; model-side behavior stayed in `pi0.py::_apply_kv_patch` lines 239-247. Because both child quarters failed after parent `294-587` passed, `294-587` is recorded as the smallest defensible contiguous region for this phase.

- Minimal sufficient positions: 294–587
- Success rate at N=25: 21/25 (84%)
- Comparison to Phase 1 result (294–587 or similar): Phase 2a matches Phase 1's main image-token half. Phase 1 later narrowed to 294–514, while Phase 2a's direct quarters 294–440 and 441–587 both failed alone, so the defensible Phase 2a contiguous set is 294–587.

---

## 7. Results — Phase 2b

*(Agent fills this in after each run.)*

### 7.1 Endpoint verification

**Implementation notes:** Endpoint verification exercised the new alpha code without changing patch positions. At alpha 0, `pi0.py::_apply_kv_patch` lines 244-245 writes the corrupt K/V values back into the cache, so behavior should match the corrupt baseline. At alpha 1, the same lines write full donor K/V, matching the previous patch behavior. The CLI value came from `main_patching_expt_per_step_donor.py::Args.patch_alpha` line 95 and was passed to the model through `policy._sample_kwargs["patch_alpha"]` lines 260 and 310.

| Alpha | N | Success rate | Matches expected? |
|-------|---|-------------|------------------|
| 0.0 | 5 | 0/5 (0%) | yes |
| 1.0 | 5 | 5/5 (100%) | yes |

### 7.2 Alpha sweep results

**Implementation notes:** Intermediate alpha runs used the same per-step donor path as Phase 2a and changed only `--args.patch-alpha`. The interpolation is applied independently to K and V at every selected cache position in `pi0.py::_apply_kv_patch` lines 240-245; `patch_k` and `patch_v` gates are still respected at lines 246-247. The CSV was written as an experiment artifact in `progress_cc/phase2/signal_files/alpha_sweep_results.csv`.

| Alpha | Successes | Trials | Success rate | Video path |
|-------|-----------|--------|-------------|-----------|
| 0.00 | 0 | 25 | 0% | (= A-D2) |
| 0.25 | 0 | 10 | 0% | `data/libero/videos/phase2/alpha_sweep/alpha_0.25/` |
| 0.50 | 0 | 10 | 0% | `data/libero/videos/phase2/alpha_sweep/alpha_0.50/` |
| 0.75 | 0 | 10 | 0% | `data/libero/videos/phase2/alpha_sweep/alpha_0.75/` |
| 1.00 | 5 | 5 | 100% | (= alpha endpoint; A-final was 21/25) |

### 7.3 Interpretation

The sweep shows a sharp threshold over the sampled values: alpha 0.25, 0.50, and 0.75 all recovered 0/10, while alpha 1.0 recovered 5/5 at the endpoint and the full Phase 2a region recovered 21/25. No intermediate sampled alpha produced meaningful recovery.

**Implementation notes:** This interpretation depends on the additive alpha implementation only; no thresholding or alternate model path was added. The relevant code remains `pi0.py::_apply_kv_patch` lines 229-248, `Pi0.sample_actions()` lines 251-264 and 279-282, and `main_patching_expt_per_step_donor.py::eval_libero` lines 256-260 and 303-311.

---


## 9. Results — Phase 2c

*(Agent fills this in. Run only after Phase 2b complete and `2_PHASE2B_ALPHA_SWEEP_COMPLETE.txt` exists.)*

**Implementation notes:** Phase 2c has only completed C1-lang so far, using the same per-step donor patch script and model alpha-capable patch path as Phase 2a/2b. The old failure file from the previous guide is superseded by the updated guide: C1 now continues from lang to image positions `294-587`, then full prefix `0-787` before declaring failure. The required future `patch_source_positions` feature for C2a is not implemented yet; current `_apply_kv_patch` reads and writes the same absolute position (`pos`) at `pi0.py` lines 239-245.

### 9.1 Results table

**Implementation notes:** C1-lang used `main_patching_expt_per_step_donor.py::eval_libero` with `--args.patch-positions` covering `588-787`; `eval_libero()` parsed those positions at lines 173-179, rebuilt the clean-prompt donor cache at lines 301-305, and inferred under the corrupt prompt at line 311. Pending C2a destination-only remapping will require adding source-position support because "rack" is clean absolute position 595 while "plate" is corrupt absolute position 594.

| Run code | Pair | Patch region | N | Result | Metric |
|----------|------|-------------|---|--------|--------|
| C1-lang | A (BOTH, lang tokens 588–787) | lang 588–787 | 5 | 0/5 (0%) | automated success rate |
| C1-img | A (BOTH, img tokens 294–587) | img 294–587 | 5 | 0/5 (0%) | automated success rate |
| C1-full | A (BOTH, full prefix 0–787) | full 0–787 | 5 | 5/5 (100%) | automated success rate |
| C1-bin-1a | A (BOTH, full-prefix half 0–393) | prefix 0–393 | 10 | 0/10 (0%) | automated success rate |
| C1-bin-1b | A (BOTH, full-prefix half 394–787) | prefix 394–787 | 10 | | automated success rate |
| C2a-lang | A (dest-only, rack clean[595]→plate corrupt[594]) | lang single token | 10 | | video-only |
| C2a-c1minimal | A (dest-only, C1 minimal positions) | C1 minimal | 10 | | video-only |
| C2b-lang | A (obj-only, wine_bottle clean[591–592]→bowl corrupt[591–592]) | lang token span | 10 | | video-only |
| C2b-c1minimal | A (obj-only, C1 minimal positions) | C1 minimal | 10 | | video-only |
| C3-lang | D (motor-class flip, lang 588–787) | lang 588–787 | 5 | | automated success rate |
| C3-img | D (motor-class flip, img 294–587) | img 294–587 | 5 | | automated success rate |
| C3-full | D (motor-class flip, full prefix 0–787) | full 0–787 | 5 | | automated success rate |

### 9.2 Interpretation

C1-lang failed 0/5 — language-only patching insufficient for cross-pair flip (consistent with Phase 1 and Phase 2a findings). C1-img also failed 0/5, so the Phase 2a image region `294-587` is not sufficient for the harder both-object-and-destination Pair A flip. C1-full passed 5/5, so cross-pair patching is possible via broader prefix KV. The first binary half, 0–393, failed 0/10; 394–787 is next. All C2 and Pair D rows pending.

**Implementation notes:** No new interpretation code exists; results are read from LIBERO done flags emitted by the evaluation scripts. For C2a/C2b, the guide marks results as video-only because the effective compositional tasks are not LIBERO-Goal success conditions, so the implementation should save videos via `main_patching_expt_per_step_donor.py` lines 330-336 and avoid drawing automated success conclusions.

---

## 8. Current Status and Next Steps

*(Agent updates this section at the end of each work session.)*

**Last updated:** 2026-05-06 — C1 binary half 0–393 failed 0/10; running 394–787 next.

**Current state:** Phase 2a and 2b complete. Phase 2c Pair A C1-lang failed 0/5, C1-img failed 0/5, and C1-full passed 5/5. The previous `0_PHASE2C_FAILURE.txt` is superseded by this result and should be removed or overwritten when final Phase 2c signaling is written.

**Next action:** Run C1 binary half positions 394–787, N=10. Promote/recurse if meaningful; if it fails too, 0–787 remains the smallest defensible set because the parent full-prefix run passed.

**Implementation notes:** Before full Phase 2c continuation, code should be extended only if C1 passes and C2 is reached. The planned additive change is `patch_source_positions`: add a donor-source position tuple to `pi0.py::_apply_kv_patch` beside the current `patch_positions` argument at lines 229-234, pass it through `Pi0.sample_actions()` lines 251-264 and 279-282, expose it in `main_patching_expt_per_step_donor.py::Args` lines 69-108, parse it near lines 173-179, and write it into `policy._sample_kwargs` beside the existing patch settings at lines 256-260 and 303-310.
