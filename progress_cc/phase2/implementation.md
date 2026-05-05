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
| Task pair | Simple Pair 2: `put_the_bowl_on_top_of_the_cabinet` (clean) ↔ `put_the_bowl_on_the_stove` (corrupt) |
| Donor approach | Per-step donor only (`main_patching_expt_per_step_donor.py`) |
| Phase 2a goal | Replicate Phase 1's methodology on Simple Pair 2 → find minimal sufficient image-token patch set |
| Phase 2b goal | Alpha sweep: vary interpolation weight α ∈ {0, 0.25, 0.5, 0.75, 1.0} on Phase 2a's localized region |
| Alpha implementation | Add `patch_alpha: float = 1.0` to `_apply_kv_patch` and `sample_actions` in `pi0.py`; add `--args.patch-alpha` to per-step donor script |
| Alpha endpoint gate | α=0 must ≈ corrupt baseline (~0%), α=1 must reproduce per-step full-prefix result (≥60%) before running intermediate values |
| Output structure | Videos → `data/libero/videos/phase2/`; logs → `progress_cc/phase2/signal_files/logs/`; signal files → `progress_cc/phase2/signal_files/` |
| Script policy | Additive changes only to `pi0.py` — new default-preserving parameters. No modification of existing Phase 1 behavior. |

---

## 3. Phase 2a — Simple Pair 2 Battery

**Task pair:**
- Environment task: `put_the_bowl_on_top_of_the_cabinet`
- Clean prompt: `"put the bowl on top of the cabinet"`
- Corrupt prompt: `"put the bowl on the stove"`
- Task name filter flag: `"put the bowl on top of the cabinet"`

**Experiment sequence (same methodology as Phase 1):**

| Code | Condition | N | Expected |
|------|-----------|---|----------|
| A-D1 | Clean baseline (correct prompt, normal KV) | 25 | ~100% |
| A-D2 | Corrupt baseline (stove prompt, normal KV) | 25 | ~0% |
| A-C3 | Per-step full-prefix sanity (all 0–787, per-step donor) | 5 | ≥60% (3/5) |
| A-D3 | Image prefix (0–587, per-step donor) | 10 | >2/10 if finding holds |
| A-binary | Binary search within 0–587 to find minimal set | N=10 per probe | Recurse until minimal |
| A-final | Minimal set promoted to N=25 | 25 | >5/25 (20%) |

**Success gate:** A-C3 ≥ 3/5 before running binary search. If A-C3 fails, stop and write `0_PHASE2A_FAILURE.txt`.

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
- [ ] **A3.** Run C3 per-step full-prefix sanity (N=5) — gate before binary search
- [ ] **A4.** Run image prefix probe (0–587, N=10)
- [ ] **A5.** Binary search within image positions → find minimal set
- [ ] **A6.** Promote minimal set to N=25 → write `progress_cc/phase2/signal_files/1_PHASE2A_MEANINGFUL_RESULT.txt`

### Phase 2b

- [ ] **B1.** Add `alpha` parameter to `_apply_kv_patch` in `src/openpi/models/pi0.py` (default=1.0, additive)
- [ ] **B2.** Add `patch_alpha` to `sample_actions` in `pi0.py` (default=1.0, additive)
- [ ] **B3.** Add `patch_alpha` to `Args` and `_sample_kwargs` in `main_patching_expt_per_step_donor.py`
- [ ] **B4.** Endpoint verification: run α=0 (N=5) and α=1 (N=5) — gate
- [ ] **B5.** Run α=0.25, 0.50, 0.75 (N=10 each), saving videos to per-alpha subdirectories
- [ ] **B6.** Write `progress_cc/phase2/signal_files/alpha_sweep_results.csv`
- [ ] **B7.** Write `progress_cc/phase2/signal_files/2_PHASE2B_ALPHA_SWEEP_COMPLETE.txt`

---

## 6. Results — Phase 2a

*(Agent fills this in after each run.)*

### 6.1 Results table

| Run code | Prompt | Patch config | N | Success rate | Log file |
|----------|--------|-------------|---|-------------|----------|
| A-D1 | clean | normal | 25 | | |
| A-D2 | corrupt | normal | 25 | | |
| A-C3 | corrupt | per-step full-prefix 0–787 | 5 | | |
| A-D3 | corrupt | per-step image prefix 0–587 | 10 | | |
| | | | | | |

### 6.2 Implementation notes

*(Surprises, deviations from plan, debugging notes.)*

| Date | Note |
|------|------|
| | |

### 6.3 Minimal patch set found

*(Fill after binary search complete.)*

- Minimal sufficient positions: TBD
- Success rate at N=25: TBD
- Comparison to Phase 1 result (294–587 or similar): TBD

---

## 7. Results — Phase 2b

*(Agent fills this in after each run.)*

### 7.1 Endpoint verification

| Alpha | N | Success rate | Matches expected? |
|-------|---|-------------|------------------|
| 0.0 | 5 | | (expect ~0%) |
| 1.0 | 5 | | (expect ≥60%) |

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

## 8. Current Status and Next Steps

*(Agent updates this section at the end of each work session.)*

**Last updated:** 2026-05-05 — document initialized.

**Current state:** Not started. Begin with Phase 2a (A1: D1 baseline).

**Next action:** Run D1 and D2 baselines on Simple Pair 2 using `main_corrupt_run_expt.py`.
