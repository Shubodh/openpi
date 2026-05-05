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

- [ ] **C1.** Run Pair A BOTH case sanity (N=5) + main run (N=10): `wine_bottle/rack ↔ bowl/plate` — gate for all of Phase 2c
- [ ] **C2a.** Run Pair A destination-only (N=10): clean=`wine_bottle/rack`, corrupt=`bowl/plate`, patch only destination token (rack→plate pos) — **video-only, no automated metric**; human inspects videos later for bowl/rack behavior
- [ ] **C2b.** Run Pair A object-only (N=10): clean=`wine_bottle/rack`, corrupt=`bowl/plate`, patch only object tokens (wine_bottle→bowl pos 591-592) — **video-only, no automated metric**; human inspects videos later for wine_bottle/plate behavior
- [ ] **C3.** Run Pair D sanity (N=5) + main run (N=10): `bowl/stove ↔ turn_on_stove` — motor-class flip test
- [ ] **C4.** Write `3_PHASE2C_COMPLETE.txt` (or `0_PHASE2C_FAILURE.txt` if C1 fails)

---

## 6. Results — Phase 2a

*(Agent fills this in after each run.)*

### 6.1 Results table

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

### 6.3 Minimal patch set found

*(Fill after binary search complete.)*

- Minimal sufficient positions: 294–587
- Success rate at N=25: 21/25 (84%)
- Comparison to Phase 1 result (294–587 or similar): Phase 2a matches Phase 1's main image-token half. Phase 1 later narrowed to 294–514, while Phase 2a's direct quarters 294–440 and 441–587 both failed alone, so the defensible Phase 2a contiguous set is 294–587.

---

## 7. Results — Phase 2b

*(Agent fills this in after each run.)*

### 7.1 Endpoint verification

| Alpha | N | Success rate | Matches expected? |
|-------|---|-------------|------------------|
| 0.0 | 5 | 0/5 (0%) | yes |
| 1.0 | 5 | 5/5 (100%) | yes |

### 7.2 Alpha sweep results

| Alpha | Successes | Trials | Success rate | Video path |
|-------|-----------|--------|-------------|-----------|
| 0.00 | | | | (= A-D2) |
| 0.25 | | | | `data/libero/videos/phase2/alpha_sweep/alpha_0.25/` |
| 0.50 | | | | `data/libero/videos/phase2/alpha_sweep/alpha_0.50/` |
| 0.75 | | | | `data/libero/videos/phase2/alpha_sweep/alpha_0.75/` |
| 1.00 | | | | (= A-C3/A-final) |

### 7.3 Interpretation

*(Agent writes one paragraph after sweep is complete: is the transition graded or binary? Which alpha first shows meaningful recovery?)*

---


## 9. Results — Phase 2c

*(Agent fills this in. Run only after Phase 2b complete and `2_PHASE2B_ALPHA_SWEEP_COMPLETE.txt` exists.)*

### 9.1 Results table

| Run code | Pair | Corrupt prompt | N | Result | Metric |
|----------|------|---------------|---|--------|--------|
| C1-sanity | A (BOTH, all lang tokens 588-787) | put the bowl on the plate | 5 | | automated success rate |
| C1-main | A (BOTH, all lang tokens 588-787) | put the bowl on the plate | 10 | | automated success rate |
| C2a | A (dest-only patch, pos 595→594) | put the bowl on the plate | 10 | | video-only: bowl/rack behavior? |
| C2b | A (obj-only patch, pos 591-592) | put the bowl on the plate | 10 | | video-only: wine_bottle/plate behavior? |
| C3-sanity | D (motor-class flip) | turn on the stove | 5 | | automated success rate |
| C3-main | D (motor-class flip) | turn on the stove | 10 | | automated success rate |

### 9.2 Interpretation

*(Agent writes one paragraph after Phase 2c is complete: did cross-pair patching work? Was the encoding decomposable? Did motor-class flip occur?)*

---

## 8. Current Status and Next Steps

*(Agent updates this section at the end of each work session.)*

**Last updated:** 2026-05-05 — Phase 2b endpoint verification complete.

**Current state:** Alpha endpoints passed on positions 294–587: alpha=0.0 recovered 0/5 and alpha=1.0 recovered 5/5.

**Next action:** Run intermediate alpha sweep values 0.25, 0.50, and 0.75 at N=10 each.
