---
name: phase2-guide
purpose: Agent guide for Phase 2 — KV-cache patching on Simple Pair 2 (bowl/cabinet ↔ wine_bottle/cabinet) and alpha sweep for steerability, on π₀.₅ + LIBERO-Goal.
when-to-read: You are the autonomous agent running Phase 2. Read this before touching any code or running any commands.
tl;dr:
  - Phase 2a: replicate Phase 1's methodology on Simple Pair 2 (different object, same destination — cabinet) → find minimal image-token patch set.
  - Phase 2b: alpha sweep on Phase 2a's minimal set — verify endpoints first (α=0 ≡ corrupt, α=1 ≡ clean), then run intermediates.
  - Per-step donor only. Pre-computed donor is known to fail (stale images). Do not use main_patching_expt.py for patching runs — use main_patching_expt_per_step_donor.py.
  - No success signal for intermediate alpha values — just save videos and CSV. Do not attempt to interpret or iterate on intermediate results.
---

# Phase 2 Agent Guide

---

## YOUR MISSION

You are running Phase 2 of the AXMech activation-patching experiment on π₀.₅ + LIBERO-Goal. Phase 1 (on branch `patching_phase1_agentic`) is running in parallel and focuses on localizing the minimal patch set and sweeping layers/K-vs-V on Simple Pair 1 (`plate` ↔ `stove`). Your job is complementary: run the same initial battery on a **new task pair** (Simple Pair 2: `bowl` ↔ `wine bottle`, same destination — cabinet), then run an **alpha sweep** to test steerability.

**You are on branch `axmech-agentic-phase-2`.** Do not touch any other branch. Do not modify files that Phase 1 is using without adding new parameters with defaults that preserve existing behavior.

**Success conditions:**

1. **Phase 2a complete** → minimal sufficient image-token patch set found for Simple Pair 2, N=25 result recorded → write `progress_cc/phase2/signal_files/1_PHASE2A_MEANINGFUL_RESULT.txt`
2. **Phase 2b complete** → alpha sweep run with verified endpoints, videos saved, CSV written → write `progress_cc/phase2/signal_files/2_PHASE2B_ALPHA_SWEEP_COMPLETE.txt`

**If stuck:** record what you tried, why it failed, and move to the next step. Never spin on one approach indefinitely. If Phase 2a fails at the sanity check and you cannot debug within 2 iterations, write `progress_cc/phase2/signal_files/0_PHASE2A_FAILURE.txt` and stop.

---

## Success Thresholds

| Condition | Threshold | Action |
|-----------|-----------|--------|
| A-C3 sanity passes | ≥ 3/5 (60%) | Proceed to binary search |
| A-C3 sanity fails | < 3/5 | Debug (≤2 iterations); if still failing, write `0_PHASE2A_FAILURE.txt` and stop |
| Binary search meaningful | > 2/10 (20%) at N=10 | Promote to N=25 |
| Binary search absent | 0/10 | Widen (try combined regions); if all fail, document and stop Phase 2a |
| A-final N=25 meaningful | > 5/25 (20%) | Write `1_PHASE2A_MEANINGFUL_RESULT.txt`; proceed to Phase 2b |
| B endpoint α=0 | ≈ 0% (within 10pp of D2) | Endpoint verified |
| B endpoint α=1 | ≥ 60% | Endpoint verified |
| Either endpoint fails | — | Alpha implementation is wrong — debug before running intermediates |

**Context:** Phase 1 found clean baseline = 100%, corrupt = 0% on Simple Pair 1. Expect similar separation on Simple Pair 2. Any A-C3 > 60% confirms the mechanism works on this pair.

---

## Environment Setup

Run once at the start of every session:

```bash
cd /workspace/openpi
source /workspace/openpi/runpod/patching_env.sh
# Should print: Server venv active (Python 3.11.x, JAX x.x.x)
git checkout axmech-agentic-phase-2
git pull origin axmech-agentic-phase-2
```

**Key paths — use these exactly, do not improvise:**

| Path | Purpose |
|------|---------|
| `progress_cc/phase2/implementation.md` | Results doc — update after every run |
| `progress_cc/phase2/signal_files/` | Signal files (write here when conditions met) |
| `progress_cc/phase2/signal_files/logs/` | All run logs (full + clean) |
| `progress_cc/phase2/signal_files/alpha_sweep_results.csv` | Alpha sweep output |
| `data/libero/videos/phase2/` | All videos for this phase |
| `src/openpi/models/pi0.py` | Model file — Phase 2b adds alpha parameter here |
| `examples/libero/main_patching_expt_per_step_donor.py` | Primary patching script |
| `examples/libero/main_corrupt_run_expt.py` | Baselines script |

**Phase 1 reference (read-only — do not modify):**

| Path | What it contains |
|------|-----------------|
| `status_cc/patching_implementation.md` | Phase 1 full results, architectural findings, binary search progress |
| `status_cc/agent_patching_guide.md` | Phase 1 agent guide |
| `scripts_outputs_txt/patching_phase1/patched/1_SANITY_CHECK_SUCCESS.txt` | Phase 1 sanity result |
| `scripts_outputs_txt/patching_phase1/patched/2_PATCHING_MEANINGFUL_RESULT.txt` | Phase 1 main result (may not exist yet if still running) |

Read `status_cc/patching_implementation.md §8.3` before starting to understand what Phase 1 found and why per-step donor is required.

---

## Background: What Phase 1 Found and Why It Matters for Phase 2

Phase 1 established that π₀.₅'s prefix attention is **fully bidirectional**: by the time we patch the KV cache, the corrupt prompt has already mixed its signal into image-token K/V entries. Patching only language-token K/V (positions 588–787) does not recover clean behavior — the corrupt signal remains in image tokens. The working approach is per-step donor patching on image-token positions, specifically positions `294–587` show the strongest signal so far.

**What this means for Phase 2:**
- Skip pre-computed donor entirely — it fails because it overwrites current image K/V with stale initial-scene images
- Skip language-token-only patching — it fails for the same bidirectional-mixing reason Phase 1 found
- Start directly with per-step full-prefix sanity, then probe image tokens
- Expect the minimal sufficient set to also fall in the wrist/masked-camera image-token region (`196–587`), though it may differ from Phase 1's `294–587`

**Simple Pair 2 tests the orthogonal dimension — different object, same destination:**
- Phase 1: `plate` vs `stove` — same object (`bowl`), different destination. Established that destination encoding is causally active and localizable in image-token K/V.
- Phase 2: `bowl` vs `wine bottle` — same destination (`cabinet`), different object. Clean: "put the bowl on top of the cabinet"; corrupt: "put the wine bottle on top of the cabinet". Tests whether object-identity encoding is equally patchable.

The key question: does the image-token localization finding from Phase 1 generalize to a different-object-same-destination pair? If yes, the mechanism captures object selection, not just destination encoding.

---

## Task Pair Reference

| Parameter | Value |
|-----------|-------|
| Environment task | `put_the_bowl_on_top_of_the_cabinet` |
| `--args.task-name-filter` | `"put the bowl on top of the cabinet"` |
| `--args.clean-prompt` | `"put the bowl on top of the cabinet"` |
| `--args.corrupt-prompt` | `"put the wine bottle on top of the cabinet"` |
| `--args.task-suite-name` | `libero_goal` |
| `--args.checkpoint-dir` | `/workspace/openpi_assets/openpi-assets/checkpoints/pi05_libero` |

---

## Log Command Pattern

Use this pattern for every run. Substitute `LABEL` with a descriptive name (e.g., `phase2a_d1_clean`, `phase2a_c3_fullprefix`, `phase2a_img0-587`):

```bash
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LABEL="<descriptive_label>"
FULL="progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_${LABEL}_full.txt"
CLEAN="progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_${LABEL}_clean.txt"
mkdir -p progress_cc/phase2/signal_files/logs
```

Pipe pattern:
```bash
{ python examples/libero/<script>.py <args> } 2>&1 \
  | tee "$FULL" \
  | tr '\r' '\n' | sed -u $'s/\x1b\\[[0-9;]*[A-Za-z]//g' \
  | grep -E --line-buffered '^(===|\[ep |\[DEBUG|INFO:root:|ERROR:root:)' \
  | tee "$CLEAN"
```

---

## Phase 2a Steps

### Step A1 + A2 — Baselines (D1 clean + D2 corrupt)

Run both baselines back to back. Use `main_corrupt_run_expt.py` (same as Phase 1 baselines).

```bash
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FULL="progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_phase2a_baselines_full.txt"
CLEAN="progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_phase2a_baselines_clean.txt"
mkdir -p progress_cc/phase2/signal_files/logs
mkdir -p data/libero/videos/phase2/baselines

{
python examples/libero/main_corrupt_run_expt.py \
  --args.task-suite-name libero_goal \
  --args.task-name-filter "put the bowl on top of the cabinet" \
  --args.corrupt-prompt "put the wine bottle on top of the cabinet" \
  --args.num-trials-per-task 25 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/phase2/baselines
} 2>&1 \
  | tee "$FULL" \
  | tr '\r' '\n' | sed -u $'s/\x1b\\[[0-9;]*[A-Za-z]//g' \
  | grep -E --line-buffered '^(===|\[ep |\[DEBUG|INFO:root:|ERROR:root:)' \
  | tee "$CLEAN"
```

**Interpret:**
- D1 (clean prompt run): expect ~100%. If < 80%, the model does not reliably solve this task — stop and report.
- D2 (corrupt prompt run): expect ~0%. If > 20%, the two prompts produce similar behavior — the pair is not contrastive enough, stop and report.
- Record both in `progress_cc/phase2/implementation.md §6.1`.

**Commit:** `git add -p && git commit -m "phase2a A1+A2: baselines D1=X/25 D2=Y/25"`

---

### Step A3 — Per-step full-prefix sanity (C3, N=5)

This is the mechanism gate. Per-step donor, all 788 positions. If this fails, the patching infrastructure is broken on this pair.

```bash
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FULL="progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_phase2a_c3_fullprefix_full.txt"
CLEAN="progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_phase2a_c3_fullprefix_clean.txt"
mkdir -p data/libero/videos/phase2/c3_fullprefix

{
python examples/libero/main_patching_expt_per_step_donor.py \
  --args.task-suite-name libero_goal \
  --args.task-name-filter "put the bowl on top of the cabinet" \
  --args.clean-prompt "put the bowl on top of the cabinet" \
  --args.corrupt-prompt "put the wine bottle on top of the cabinet" \
  --args.patch-positions "$(python -c "print(','.join(map(str, range(788))))")" \
  --args.num-trials-per-task 5 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/phase2/c3_fullprefix
} 2>&1 \
  | tee "$FULL" \
  | tr '\r' '\n' | sed -u $'s/\x1b\\[[0-9;]*[A-Za-z]//g' \
  | grep -E --line-buffered '^(===|\[ep |\[DEBUG|INFO:root:|ERROR:root:)' \
  | tee "$CLEAN"
```

**Interpret:**
- ≥ 3/5: mechanism confirmed. Proceed to A4.
- < 3/5: debug. Check that the task name filter matches exactly, the prompt is correct, and the env initializes properly. Allow ≤ 2 debug iterations. If still failing after 2, write `progress_cc/phase2/signal_files/0_PHASE2A_FAILURE.txt` and stop.

**Commit after A3.**

---

### Step A4 — Image prefix probe (0–587, N=10)

```bash
POSITIONS=$(python -c "print(','.join(map(str, range(588))))")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FULL="progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_phase2a_img0-587_full.txt"
CLEAN="progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_phase2a_img0-587_clean.txt"
mkdir -p data/libero/videos/phase2/img0-587

{
python examples/libero/main_patching_expt_per_step_donor.py \
  --args.task-suite-name libero_goal \
  --args.task-name-filter "put the bowl on top of the cabinet" \
  --args.clean-prompt "put the bowl on top of the cabinet" \
  --args.corrupt-prompt "put the wine bottle on top of the cabinet" \
  --args.patch-positions "$POSITIONS" \
  --args.num-trials-per-task 10 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/phase2/img0-587
} 2>&1 \
  | tee "$FULL" \
  | tr '\r' '\n' | sed -u $'s/\x1b\\[[0-9;]*[A-Za-z]//g' \
  | grep -E --line-buffered '^(===|\[ep |\[DEBUG|INFO:root:|ERROR:root:)' \
  | tee "$CLEAN"
```

**Interpret:**
- > 2/10: image-token K/V carries the signal on this pair. Proceed to binary search (A5).
- 0–2/10: image prefix alone insufficient. Try full language prefix (588–787) at N=10 before concluding. If that also fails, write `0_PHASE2A_FAILURE.txt`.

---

### Step A5 — Binary search within image positions

Use N=10 for all binary search probes. Mirror Phase 1's approach: split into halves, recurse into whichever half clears > 2/10.

**Image token layout for reference:**

| Positions | Camera | Grid |
|-----------|--------|------|
| 0–195 | `base_0_rgb` (agentview) | 14×14 |
| 196–391 | `left_wrist_0_rgb` (wrist) | 14×14 |
| 392–587 | `right_wrist_0_rgb` (masked stream) | 14×14 |

**Halves:**
- Half A: `0–293` → `python -c "print(','.join(map(str, range(294))))"`
- Half B: `294–587` → `python -c "print(','.join(map(str, range(294, 588))))"`

Use label pattern: `phase2a_img0-293`, `phase2a_img294-587`, `phase2a_img294-440`, etc.

Continue until either:
- A region of ~20–50 positions clears > 2/10 consistently, or
- Further narrowing drops below threshold (the previous larger region is the minimal set)

**Commit after each binary search iteration.**

---

### Step A6 — Promote to N=25 and write signal file

Run the minimal sufficient set at N=25:

```bash
POSITIONS="<minimal set from binary search>"
# ... (same command pattern as A4, N=25, label: phase2a_final_n25)
```

**If > 5/25:** write `progress_cc/phase2/signal_files/1_PHASE2A_MEANINGFUL_RESULT.txt`:
```
Phase 2a meaningful result — [DATE]
Task pair: put_the_bowl_on_top_of_the_cabinet (clean) / put_the_wine_bottle_on_top_of_the_cabinet (corrupt)
Minimal patch positions: [list]
Success rate: X/25
Clean baseline: Y/25, Corrupt baseline: Z/25
Log: progress_cc/phase2/signal_files/logs/[filename]
Comparison to Phase 1: Phase 1 found 294-587 as main region on plate/stove (same obj, diff dest) pair.
See progress_cc/phase2/implementation.md §6.3 for interpretation.
```

**Commit and proceed to Phase 2b.**

---

## Phase 2b Steps

**Prerequisite:** `progress_cc/phase2/signal_files/1_PHASE2A_MEANINGFUL_RESULT.txt` exists.

### Step B1–B3 — Add alpha parameter (code changes)

**`src/openpi/models/pi0.py` — three changes:**

**Change 1:** `_apply_kv_patch` signature — add `alpha: float = 1.0`:
```python
def _apply_kv_patch(
    self,
    corrupt_kv_cache: _gemma.KVCache,
    donor_kv_cache: _gemma.KVCache,
    patch_positions: tuple[int, ...],
    alpha: float = 1.0,          # NEW — 1.0 preserves existing behavior exactly
) -> _gemma.KVCache:
```

**Change 2:** `_apply_kv_patch` body — replace the `.set()` calls:
```python
# Before:
K = K.at[:, :, pos, :, :].set(K_d[:, :, pos, :, :])
V = V.at[:, :, pos, :, :].set(V_d[:, :, pos, :, :])

# After:
K_corrupt_pos = K[:, :, pos, :, :]
V_corrupt_pos = V[:, :, pos, :, :]
K = K.at[:, :, pos, :, :].set(alpha * K_d[:, :, pos, :, :] + (1 - alpha) * K_corrupt_pos)
V = V.at[:, :, pos, :, :].set(alpha * V_d[:, :, pos, :, :] + (1 - alpha) * V_corrupt_pos)
```

**Change 3:** `sample_actions` — add `patch_alpha: float = 1.0` to signature and pass to `_apply_kv_patch`:
```python
def sample_actions(
    self,
    rng,
    observation,
    *,
    num_steps=10,
    noise=None,
    donor_kv_cache=None,
    patch_positions=(594,),
    patch_alpha: float = 1.0,    # NEW
):
    ...
    if donor_kv_cache is not None:
        kv_cache = self._apply_kv_patch(kv_cache, donor_kv_cache, patch_positions, alpha=patch_alpha)
```

**`examples/libero/main_patching_expt_per_step_donor.py` — two changes:**

Add to `Args`:
```python
patch_alpha: float = 1.0   # Interpolation weight: 1.0 = full donor, 0.0 = no patch
```

Wire into `_sample_kwargs` (alongside `donor_kv_cache` and `patch_positions`):
```python
policy._sample_kwargs["patch_alpha"] = args.patch_alpha
```

**Verify the changes are additive:** `alpha=1.0` default must reproduce identical results to before. Run the A-C3 full-prefix condition once with no `--args.patch-alpha` flag to confirm results match.

**Commit B1–B3:** `git commit -m "phase2b B1-B3: add alpha interpolation to pi0.py and per-step donor script"`

---

### Step B4 — Endpoint verification (gate)

Run α=0 and α=1 each at N=5 on the **minimal patch set from Phase 2a**:

```bash
# alpha=0 (expect ~0% — no effective patch)
POSITIONS="<minimal set from Phase 2a>"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p data/libero/videos/phase2/alpha_sweep/alpha_0.00

{
python examples/libero/main_patching_expt_per_step_donor.py \
  --args.task-suite-name libero_goal \
  --args.task-name-filter "put the bowl on top of the cabinet" \
  --args.clean-prompt "put the bowl on top of the cabinet" \
  --args.corrupt-prompt "put the wine bottle on top of the cabinet" \
  --args.patch-positions "$POSITIONS" \
  --args.patch-alpha 0.0 \
  --args.num-trials-per-task 5 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/phase2/alpha_sweep/alpha_0.00
} 2>&1 | tee "progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_phase2b_alpha0_full.txt" \
  | tr '\r' '\n' | sed -u $'s/\x1b\\[[0-9;]*[A-Za-z]//g' \
  | grep -E --line-buffered '^(===|\[ep |ERROR:root:)' \
  | tee "progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_phase2b_alpha0_clean.txt"
```

Repeat for α=1 (label: `phase2b_alpha1`, video dir: `alpha_1.00`).

**Endpoint pass criteria:**
- α=0: success rate ≤ 10% (within noise of corrupt baseline ~0%)
- α=1: success rate ≥ 60%

**If either fails:** the alpha implementation is wrong. Check that `patch_alpha` is reaching `_apply_kv_patch`. Add a debug print: `print(f"[DEBUG] alpha={alpha}, K_corrupt_pos norm={float(jnp.linalg.norm(K_corrupt_pos)):.4f}")` to confirm the interpolation is executing. Do not run intermediate values until endpoints are verified.

**Commit after B4.**

---

### Step B5 — Intermediate alpha sweep

Only run if both endpoints verified. Run α ∈ {0.25, 0.50, 0.75} at N=10 each.

For each alpha value, use this pattern (substitute `ALPHA_STR` with `0.25`, `0.50`, `0.75`):

```bash
ALPHA=0.25
ALPHA_STR="0.25"
POSITIONS="<minimal set from Phase 2a>"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "data/libero/videos/phase2/alpha_sweep/alpha_${ALPHA_STR}"

{
python examples/libero/main_patching_expt_per_step_donor.py \
  --args.task-suite-name libero_goal \
  --args.task-name-filter "put the bowl on top of the cabinet" \
  --args.clean-prompt "put the bowl on top of the cabinet" \
  --args.corrupt-prompt "put the wine bottle on top of the cabinet" \
  --args.patch-positions "$POSITIONS" \
  --args.patch-alpha $ALPHA \
  --args.num-trials-per-task 10 \
  --args.save-all-videos \
  --args.video-out-path "data/libero/videos/phase2/alpha_sweep/alpha_${ALPHA_STR}"
} 2>&1 \
  | tee "progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_phase2b_alpha${ALPHA_STR}_full.txt" \
  | tr '\r' '\n' | sed -u $'s/\x1b\\[[0-9;]*[A-Za-z]//g' \
  | grep -E --line-buffered '^(===|\[ep |ERROR:root:)' \
  | tee "progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_phase2b_alpha${ALPHA_STR}_clean.txt"
```

Record success rate in `progress_cc/phase2/implementation.md §7.2` after each run.

**No self-correction loop for intermediate values.** Each alpha is run once at N=10. Do not re-run or adjust based on intermediate results — just record and move to the next alpha.

**Commit after each alpha run.**

---

### Step B6 — Write CSV and signal file

After all five alpha values are recorded, write the CSV:

```bash
cat > progress_cc/phase2/signal_files/alpha_sweep_results.csv << 'EOF'
alpha,successes,trials,success_rate
0.00,<D2_successes>,25,<D2_rate>
0.25,<successes>,10,<rate>
0.50,<successes>,10,<rate>
0.75,<successes>,10,<rate>
1.00,<A_C3_successes>,5,<A_C3_rate>
EOF
```

Then write `progress_cc/phase2/signal_files/2_PHASE2B_ALPHA_SWEEP_COMPLETE.txt`:
```
Phase 2b alpha sweep complete — [DATE]
Patch positions: [minimal set from Phase 2a]
Endpoint verification: alpha=0 → X/5 (expected ~0%), alpha=1 → Y/5 (expected ≥60%)
Intermediate results: alpha=0.25 → A/10, alpha=0.50 → B/10, alpha=0.75 → C/10
CSV: progress_cc/phase2/signal_files/alpha_sweep_results.csv
Videos: data/libero/videos/phase2/alpha_sweep/
See progress_cc/phase2/implementation.md §7 for full results and interpretation.
```

**Final commit:**
```bash
git add progress_cc/phase2/ src/openpi/models/pi0.py examples/libero/main_patching_expt_per_step_donor.py
git commit -m "$(cat <<'EOF'
phase2b B6: alpha sweep complete

Endpoint verified: alpha=0=X/5, alpha=1=Y/5.
Intermediate: 0.25=A/10, 0.50=B/10, 0.75=C/10.
CSV and signal file written. See progress_cc/phase2/implementation.md §7.
EOF
)"
```

---

## Result Recording

Update `progress_cc/phase2/implementation.md` after **every run**:
- Add a row to §6.1 (Phase 2a) or §7.1/§7.2 (Phase 2b)
- Update §8 (current status and next steps) with one line saying what was just run and what comes next

Update §8 **before committing** so that if the pod restarts, the doc tells the next session exactly where to resume.

---

## Commit Protocol

```bash
git add -p   # stage only files you modified
git commit -m "$(cat <<'EOF'
phase2a/2b <step>: <one-line result>

[Optional: what you observed, why surprising, what next]
EOF
)"
```

Rules:
- Commit after every step.
- Stage only `progress_cc/phase2/`, `src/openpi/models/pi0.py`, and `examples/libero/main_patching_expt_per_step_donor.py` — nothing else.
- Never amend published commits.
- Never touch `patching_phase1_agentic` branch or any Phase 1 files under `status_cc/` or `scripts_outputs_txt/patching_phase1/`.

---

## Phase 2c Steps (Challenging Pairs — Cross-Object + Cross-Destination)

**Prerequisite:** `progress_cc/phase2/signal_files/2_PHASE2B_ALPHA_SWEEP_COMPLETE.txt` exists. Do not begin Phase 2c until Phase 2a and 2b are fully done.

**Disclaimer:** Phase 2c experiments all have automated success metrics (the done flag always measures the clean task outcome — see `status_cc/misc/libero_suite_choice_detailed.md §Automated success metric` for the full explanation). However, some corrupt prompts (e.g., "put the bowl on the rack") are compositionally novel — the model has not been trained on that exact combination. The corrupt unpatched baseline may show unusual behavior, which is expected and worth noting in videos. The patched-run success rate is still reliable. **Agent self-correction rule:** C1 is the correctness anchor. If C1 fails (< 2/5), do not run C2a, C2b, or Pair D — the mechanism does not work on cross-pair patching and further runs have no scientific value.

**Scientific questions:**
- Can KV-cache patching flip behavior when BOTH object AND destination differ (C1)?
- Is the encoding decomposable — can we flip destination alone (C2a) or object alone (C2b)?
- Can patching flip qualitatively different motor behaviors, not just object/destination selection (Pair D)?

---

### Phase 2c — Pair A: `put_the_wine_bottle_on_the_rack` (clean) ↔ `put_the_bowl_on_the_plate` (corrupt)

| Parameter | Value |
|-----------|-------|
| Environment task | `put_the_wine_bottle_on_the_rack` |
| `--args.task-name-filter` | `"put the wine bottle on the rack"` |
| `--args.clean-prompt` | `"put the wine bottle on the rack"` |
| `--args.corrupt-prompt` | `"put the bowl on the plate"` |
| Success metric | Automated (done flag checks wine bottle on rack) |

Scientific contrast with Phase 2a: Phase 2a changes only the object (bowl→wine bottle), same destination (cabinet). Pair A changes BOTH object (wine bottle vs bowl) AND destination (rack vs plate) simultaneously.

---

### Step C1 — Pair A BOTH case: sanity (N=5) + main run (N=10)

```bash
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FULL="progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_phase2c_pairA_both_sanity_full.txt"
CLEAN="progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_phase2c_pairA_both_sanity_clean.txt"
mkdir -p data/libero/videos/phase2c/pairA_both_sanity

{
python examples/libero/main_patching_expt_per_step_donor.py \
  --args.task-suite-name libero_goal \
  --args.task-name-filter "put the wine bottle on the rack" \
  --args.clean-prompt "put the wine bottle on the rack" \
  --args.corrupt-prompt "put the bowl on the plate" \
  --args.patch-positions "$(python -c "print(','.join(map(str, range(788))))")" \
  --args.num-trials-per-task 5 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/phase2c/pairA_both_sanity
} 2>&1 \
  | tee "$FULL" \
  | tr '\r' '\n' | sed -u $'s/\x1b\\[[0-9;]*[A-Za-z]//g' \
  | grep -E --line-buffered '^(===|\[ep |\[DEBUG|INFO:root:|ERROR:root:)' \
  | tee "$CLEAN"
```

**Interpret:**
- ≥ 3/5: cross-pair patching works. Run N=10 (label `phase2c_pairA_both_n10`, same command, `--args.num-trials-per-task 10`). Proceed to C2a.
- < 3/5: stop Phase 2c. Write `0_PHASE2C_FAILURE.txt` (see below). Do not run C2a, C2b, or Pair D.

**Commit after C1.**

---

### Step C2a — Pair A, destination-only sub-test (automated metric)

Same object (wine bottle), different destination (rack vs cabinet). Tests whether destination encoding alone drives the behavioral difference.

| Parameter | Value |
|-----------|-------|
| `--args.clean-prompt` | `"put the wine bottle on the rack"` |
| `--args.corrupt-prompt` | `"put the wine bottle on top of the cabinet"` |
| `--args.task-name-filter` | `"put the wine bottle on the rack"` |
| Success metric | Automated (both are LIBERO-Goal tasks) |

```bash
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FULL="progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_phase2c_pairA_destonly_full.txt"
CLEAN="progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_phase2c_pairA_destonly_clean.txt"
mkdir -p data/libero/videos/phase2c/pairA_destonly

{
python examples/libero/main_patching_expt_per_step_donor.py \
  --args.task-suite-name libero_goal \
  --args.task-name-filter "put the wine bottle on the rack" \
  --args.clean-prompt "put the wine bottle on the rack" \
  --args.corrupt-prompt "put the wine bottle on top of the cabinet" \
  --args.patch-positions "$(python -c "print(','.join(map(str, range(788))))")" \
  --args.num-trials-per-task 10 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/phase2c/pairA_destonly
} 2>&1 \
  | tee "$FULL" \
  | tr '\r' '\n' | sed -u $'s/\x1b\\[[0-9;]*[A-Za-z]//g' \
  | grep -E --line-buffered '^(===|\[ep |\[DEBUG|INFO:root:|ERROR:root:)' \
  | tee "$CLEAN"
```

**Interpret:** Record success rate in `progress_cc/phase2/implementation.md §9.1`. No minimum threshold — just record and proceed to C2b.

**Commit after C2a.**

---

### Step C2b — Pair A, object-only sub-test

Same destination (rack), different object (wine bottle vs bowl). Corrupt prompt "put the bowl on the rack" is compositionally novel (not a LIBERO-Goal task), but the done flag still measures the clean task (wine bottle on rack) — automated metric holds.

| Parameter | Value |
|-----------|-------|
| `--args.clean-prompt` | `"put the wine bottle on the rack"` |
| `--args.corrupt-prompt` | `"put the bowl on the rack"` (not a LIBERO-Goal task — model may show novel behavior on the unpatched corrupt run) |
| `--args.task-name-filter` | `"put the wine bottle on the rack"` |
| Success metric | Automated for patched run (done flag checks wine bottle on rack). Unpatched corrupt baseline may show unusual behavior — inspect videos. |

```bash
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FULL="progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_phase2c_pairA_objonly_full.txt"
CLEAN="progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_phase2c_pairA_objonly_clean.txt"
mkdir -p data/libero/videos/phase2c/pairA_objonly

{
python examples/libero/main_patching_expt_per_step_donor.py \
  --args.task-suite-name libero_goal \
  --args.task-name-filter "put the wine bottle on the rack" \
  --args.clean-prompt "put the wine bottle on the rack" \
  --args.corrupt-prompt "put the bowl on the rack" \
  --args.patch-positions "$(python -c "print(','.join(map(str, range(788))))")" \
  --args.num-trials-per-task 10 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/phase2c/pairA_objonly
} 2>&1 \
  | tee "$FULL" \
  | tr '\r' '\n' | sed -u $'s/\x1b\\[[0-9;]*[A-Za-z]//g' \
  | grep -E --line-buffered '^(===|\[ep |\[DEBUG|INFO:root:|ERROR:root:)' \
  | tee "$CLEAN"
```

**Interpret:** Record success rate in §9.1. Note that the done flag is valid for the patched run. Inspect videos to observe whether the model goes for wine bottle (expected if patching works on object encoding).

**Commit after C2b.**

---

### Phase 2c — Pair D: `put_the_bowl_on_the_stove` (clean) ↔ `turn_on_the_stove` (corrupt)

| Parameter | Value |
|-----------|-------|
| Environment task | `put_the_bowl_on_the_stove` |
| `--args.task-name-filter` | `"put the bowl on the stove"` |
| `--args.clean-prompt` | `"put the bowl on the stove"` |
| `--args.corrupt-prompt` | `"turn on the stove"` |
| Success metric | Automated (done flag checks bowl on stove) |

**Scientific question:** Can patching flip between qualitatively different motor behaviors — pick-place (put bowl on stove) vs. knob-turn (turn on stove)? This tests whether KV patching controls action class, not just object/destination selection.

### Step C3 — Pair D sanity (N=5) + main run (N=10)

```bash
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FULL="progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_phase2c_pairD_sanity_full.txt"
CLEAN="progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_phase2c_pairD_sanity_clean.txt"
mkdir -p data/libero/videos/phase2c/pairD_sanity

{
python examples/libero/main_patching_expt_per_step_donor.py \
  --args.task-suite-name libero_goal \
  --args.task-name-filter "put the bowl on the stove" \
  --args.clean-prompt "put the bowl on the stove" \
  --args.corrupt-prompt "turn on the stove" \
  --args.patch-positions "$(python -c "print(','.join(map(str, range(788))))")" \
  --args.num-trials-per-task 5 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/phase2c/pairD_sanity
} 2>&1 \
  | tee "$FULL" \
  | tr '\r' '\n' | sed -u $'s/\x1b\\[[0-9;]*[A-Za-z]//g' \
  | grep -E --line-buffered '^(===|\[ep |\[DEBUG|INFO:root:|ERROR:root:)' \
  | tee "$CLEAN"
```

**Interpret:**
- ≥ 3/5: proceed to N=10 (label `phase2c_pairD_n10`). Motor-class flip is possible.
- < 3/5: note failure. Motor-class patching may require different patch positions or may not be achievable via prefix KV alone. Record and stop.

**Commit after C3.**

---

### Step C4 — Write Phase 2c signal file

After all Phase 2c sub-experiments:

Write `progress_cc/phase2/signal_files/3_PHASE2C_COMPLETE.txt`:
```
Phase 2c complete — [DATE]

Pair A (BOTH, wine_bottle/rack ↔ bowl/plate):
  C1 sanity: X/5; N=10: Y/10
  C2a (dest-only, wine_bottle/rack ↔ wine_bottle/cabinet): Z/10
  C2b (obj-only, wine_bottle/rack ↔ bowl/rack): W/10

Pair D (bowl/stove ↔ turn_on_stove):
  C3 sanity: X/5; N=10: Y/10

Videos: data/libero/videos/phase2c/
See progress_cc/phase2/implementation.md §9 for interpretation.
```

If C1 failed, write `progress_cc/phase2/signal_files/0_PHASE2C_FAILURE.txt`:
```
Phase 2c failure — [DATE]
C1 sanity (wine_bottle/rack ↔ bowl/plate, full prefix, N=5): X/5 — below threshold.
Cross-pair patching does not appear to work. Phase 2c aborted.
See progress_cc/phase2/implementation.md §9 for details.
```

**Final commit for Phase 2c:**
```bash
git add progress_cc/phase2/
git commit -m "$(cat <<'EOF'
phase2c complete: Pair A C1=Y/10 C2a=Z/10 C2b=W/10, Pair D C3=Y/10

[Optional: key observation about cross-pair patching and motor-class flip]
EOF
)"
```

---

## Updated Signal Files Summary

| File | Write when | Location |
|------|-----------|----------|
| `0_PHASE2A_FAILURE.txt` | A-C3 fails after 2 debug iterations | `progress_cc/phase2/signal_files/` |
| `1_PHASE2A_MEANINGFUL_RESULT.txt` | A-final N=25 > 5/25 | `progress_cc/phase2/signal_files/` |
| `2_PHASE2B_ALPHA_SWEEP_COMPLETE.txt` | All alpha values run, CSV written | `progress_cc/phase2/signal_files/` |
| `3_PHASE2C_COMPLETE.txt` | All Phase 2c sub-experiments run | `progress_cc/phase2/signal_files/` |
| `0_PHASE2C_FAILURE.txt` | C1 fails (< 2/5) | `progress_cc/phase2/signal_files/` |
