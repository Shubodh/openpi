---
name: phase2-guide
purpose: Agent guide for Phase 2 — KV-cache patching on Simple Pair 2 (bowl/cabinet ↔ wine_bottle/cabinet) and alpha sweep for steerability, on π₀.₅ + LIBERO-Goal.
when-to-read: You are the autonomous agent running Phase 2. Read this before touching any code or running any commands.
tl;dr:
  - Phase 2a: replicate Phase 1's methodology on Simple Pair 2 (different object, same destination — cabinet) → test language-token patching (new vs Phase 1) then image-token, find minimal sufficient patch set.
  - Phase 2b: alpha sweep on Phase 2a's minimal set — verify endpoints first (α=0 ≡ corrupt, α=1 ≡ clean), then run intermediates.
  - Phase 2c: C1 patches ALL language tokens (wine_bottle/rack ↔ bowl/plate, automated metric); C2a patches ONLY destination tokens ("rack" from clean into corrupt's "plate" position → effective behavior bowl/rack, video-only); C2b patches ONLY object tokens ("wine bottle" from clean into corrupt's "bowl" position → effective behavior wine_bottle/plate, video-only). Pair D motor-class flip (automated metric). C1/C2a/C2b all use identical clean+corrupt prompts — only patch positions differ.
  - Per-step donor only. Pre-computed donor is known to fail (stale images). Do not use main_patching_expt.py for patching runs — use main_patching_expt_per_step_donor.py.
  - No success signal for intermediate alpha values — just save videos and CSV. Do not attempt to interpret or iterate on intermediate results.
---

# Phase 2 Agent Guide

---

## YOUR MISSION

You are running Phase 2 of the AXMech activation-patching experiment on π₀.₅ + LIBERO-Goal. Phase 1 (on branch `patching_phase1_agentic`) is complete. Your job: run the same initial battery on a **new task pair** (Simple Pair 2: `bowl` ↔ `wine bottle`, same destination — cabinet), run an **alpha sweep** to test steerability, then run **challenging cross-pair experiments** (Phase 2c) that flip both object and destination simultaneously and test motor-class flip.

**You are on branch `axmech-agentic-phase-2`.** Do not touch any other branch. Do not modify files that Phase 1 is using without adding new parameters with defaults that preserve existing behavior.

**Success conditions:**

1. **Phase 2a complete** → minimal sufficient patch set found for Simple Pair 2 (language or image tokens), N=25 result recorded → write `progress_cc/phase2/signal_files/1_PHASE2A_MEANINGFUL_RESULT.txt`
2. **Phase 2b complete** → alpha sweep run with verified endpoints, videos saved, CSV written → write `progress_cc/phase2/signal_files/2_PHASE2B_ALPHA_SWEEP_COMPLETE.txt`
3. **Phase 2c complete** → all Pair A (C1/C2a/C2b) and Pair D (C3) experiments run → write `progress_cc/phase2/signal_files/3_PHASE2C_COMPLETE.txt` (or `0_PHASE2C_FAILURE.txt` if C1 fails)

**If stuck:** record what you tried, why it failed, and move to the next step. Never spin on one approach indefinitely. If Phase 2a fails at the sanity check and you cannot debug within 2 iterations, write `progress_cc/phase2/signal_files/0_PHASE2A_FAILURE.txt` and stop.

---

## Success Thresholds

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Per-step full-prefix sanity (A-C3) passes | ≥ 3/5 (60%) | Proceed to A4-lang |
| Per-step full-prefix sanity (A-C3) fails | < 3/5 | Debug ≤2 iterations; if still failing, write `0_PHASE2A_FAILURE.txt` and stop |
| Language-only sanity (A4-lang) passes | ≥ 3/5 | Run binary search within 588–787 before testing image prefix |
| Language-only sanity (A4-lang) fails | < 3/5 | Note it (replicates Phase 1 on object axis); proceed to A4-img |
| Binary search probe (N=10) meaningful | > 2/10 (20%) | Promote region; recurse |
| Binary search probe (N=10) absent | 0/10 | Widen (try combined regions); if all fail, document and stop Phase 2a |
| Minimal set N=25 run (A-final) meaningful | > 5/25 (20%) | Write `1_PHASE2A_MEANINGFUL_RESULT.txt`; proceed to Phase 2b |
| Alpha endpoint α=0 (B-alpha0) | ≤ 10% | Endpoint verified |
| Alpha endpoint α=1 (B-alpha1) | ≥ 60% | Endpoint verified |
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
| `progress_cc/phase2/guide.md` | **Current guide** — you are reading this |
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
| `progress_cc/phase1/implementation.md` | Phase 1 full results, architectural findings, binary search progress |
| `progress_cc/phase1/guide.md` | Phase 1 agent guide |
| `progress_cc/phase1/signal_files/patched/1_SANITY_CHECK_SUCCESS.txt` | Phase 1 sanity result |
| `progress_cc/phase1/signal_files/patched/2_PATCHING_MEANINGFUL_RESULT.txt` | Phase 1 main result |

Read `progress_cc/phase1/implementation.md §8.3` before starting to understand what Phase 1 found and why per-step donor is required.

---

## Background: What Phase 1 Found and Why It Matters for Phase 2

Phase 1 established that π₀.₅'s prefix attention is **fully bidirectional**: by the time we patch the KV cache, the corrupt prompt has already mixed its signal into image-token K/V entries. Patching only language-token K/V (positions 588–787) does not recover clean behavior — the corrupt signal remains in image tokens. The working approach is per-step donor patching on image-token positions, specifically positions `294–587` show the strongest signal so far.

**What this means for Phase 2:**
- Skip pre-computed donor entirely — it fails because it overwrites current image K/V with stale initial-scene images
- Do NOT skip language-token-only patching — Phase 1 tested destination encoding (plate ↔ stove) and found it failed. Phase 2 tests object-identity encoding (bowl ↔ wine bottle), a different axis. Language-only is tested systematically in Step A4-lang before falling back to image tokens.
- Start with per-step full-prefix sanity (A3), then language-only sanity (A4-lang), then image prefix (A4-img)
- If language-only sanity passes (unlike Phase 1): run binary search within language positions before testing image
- If language-only sanity fails: note it as a cross-axis replication of Phase 1's finding, proceed to image prefix

**Simple Pair 2 tests the orthogonal dimension — different object, same destination:**
- Phase 1: `plate` vs `stove` — same object (`bowl`), different destination. Established that destination encoding, after bidirectional mixing, lives in image-token K/V not language-token K/V.
- Phase 2: `bowl` vs `wine bottle` — same destination (`cabinet`), different object. Tests whether object-identity encoding shows the same pattern or stays more localized to language tokens.

The key question: does the image-token localization finding from Phase 1 generalize to a different-object-same-destination pair? If language-only patching works here (unlike Phase 1), it would mean object-identity mixes less into image K/V than destination encoding does — a meaningful architectural distinction.

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

### Step A4-lang — Language-only sanity (588–787, N=5)

**Scientific purpose:** Phase 1 found language-only patching fails on a destination-axis pair. Phase 2 uses an object-axis pair — test whether the result differs before assuming image tokens are the only route.

```bash
POSITIONS=$(python -c "print(','.join(map(str, range(588, 788))))")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FULL="progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_phase2a_lang588-787_full.txt"
CLEAN="progress_cc/phase2/signal_files/logs/run_${TIMESTAMP}_phase2a_lang588-787_clean.txt"
mkdir -p data/libero/videos/phase2/lang588-787

{
python examples/libero/main_patching_expt_per_step_donor.py \
  --args.task-suite-name libero_goal \
  --args.task-name-filter "put the bowl on top of the cabinet" \
  --args.clean-prompt "put the bowl on top of the cabinet" \
  --args.corrupt-prompt "put the wine bottle on top of the cabinet" \
  --args.patch-positions "$POSITIONS" \
  --args.num-trials-per-task 5 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/phase2/lang588-787
} 2>&1 \
  | tee "$FULL" \
  | tr '\r' '\n' | sed -u $'s/\x1b\\[[0-9;]*[A-Za-z]//g' \
  | grep -E --line-buffered '^(===|\[ep |\[DEBUG|INFO:root:|ERROR:root:)' \
  | tee "$CLEAN"
```

**Interpret:**
- ≥ 3/5: language-token K/V carries the object-identity signal. Run binary search within 588–787 using N=10 probes (halves: 588–687, 688–787) before moving to image prefix. Update §6.1.
- < 3/5: replicates Phase 1 finding on the object axis — language-only is insufficient. Note it in §6.1, proceed to A4-img without further language search.

**Commit after A4-lang.**

---

### Step A4-img — Image prefix probe (0–587, N=10)

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
- 0–2/10: image prefix alone insufficient. If A4-lang also failed, write `0_PHASE2A_FAILURE.txt`. If A4-lang passed, the signal is in language tokens — skip image binary search.

---

### Step A5 — Binary search (language or image positions)

Use N=10 for all binary search probes. Mirror Phase 1's approach: split into halves, recurse into whichever half clears > 2/10.

**Which positions to search:**
- If A4-lang passed: search within **588–787** (language). Halves: 588–687, 688–787.
- If A4-lang failed and A4-img passed: search within **0–587** (image). Halves below.

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

**Token mismatch note:** Phase 2a's clean and corrupt prompts differ in token count ("bowl"=1 token vs "wine bottle"=2 tokens), so language positions are not perfectly aligned between donor and corrupt. In practice this is low risk: image binary search (0–587) is unaffected by text length; bulk language patching (A4-lang, entire 588–787) overwrites the whole region so misalignment is a wash; and the binary search stopping criterion (~20–50 positions) is coarse enough to never require single-token precision. **If the language path (A4-lang passed) narrows to individual token positions, apply the token alignment convention from Step C2-prep before targeting specific tokens.**

**Commit after each binary search iteration.**

---

### Step A6 — Promote to N=25 and write signal file

Run the minimal sufficient set at N=25:

```bash
POSITIONS="<minimal set from binary search>"
# ... (same command pattern as A4-lang or A4-img depending on which passed, N=25, label: phase2a_final_n25)
```

**If > 5/25:** write `progress_cc/phase2/signal_files/1_PHASE2A_MEANINGFUL_RESULT.txt`:
```
Phase 2a meaningful result — [DATE]
Task pair: put_the_bowl_on_top_of_the_cabinet (clean) / put_the_wine_bottle_on_top_of_the_cabinet (corrupt)
Minimal patch positions: [list]
Success rate: X/25
Clean baseline: Y/25, Corrupt baseline: Z/25
Log: progress_cc/phase2/signal_files/logs/[filename]
Language-only (A4-lang): [X/5 — passed/failed]
Comparison to Phase 1: Phase 1 found language-only failed and 294-587 (image) as main region on plate/stove (same obj, diff dest) pair.
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
- Add a row to §6.1 (Phase 2a), §7.1/§7.2 (Phase 2b), or §9.1 (Phase 2c)
- Update §8 (current status and next steps) with one line saying what was just run and what comes next

Update §8 **before committing** so that if the pod restarts, the doc tells the next session exactly where to resume.

---

## Commit Protocol

```bash
git add -p   # stage only files you modified
git commit -m "$(cat <<'EOF'
phase2a/2b/2c <step>: <one-line result>

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

**Disclaimer:** Only C1 and Pair D have automated success metrics (done flag measures the clean task). **C2a and C2b are video-only** — they patch only a subset of language tokens into the SAME bowl/plate corrupt run, producing compositionally novel effective tasks (bowl/rack for C2a, wine_bottle/plate for C2b) that are not in LIBERO-Goal 10, so the done flag gives no useful signal. Save videos and move on — video interpretation is the human's job, not the agent's. **Agent self-correction rule:** C1 is the correctness anchor. If C1 fails (< 2/5), do not run C2a, C2b, or Pair D — write `0_PHASE2C_FAILURE.txt` and stop.

**What C1, C2a, C2b, Pair D mean (defined in full below):**
- **C1, C2a, C2b all use the SAME prompt pair:** clean=`"put the wine bottle on the rack"`, corrupt=`"put the bowl on the plate"`. What differs is *which KV token positions* are patched from clean into corrupt.
- **C1** — Patch ALL language positions (588–787). Full clean signal → automated metric (done flag: wine bottle on rack?).
- **C2a** — Patch ONLY the destination token ("rack", ~position 595 in clean) into the corrupt run's destination position ("plate", ~position 594 in corrupt). Effective behavior: bowl/rack — compositionally novel. **Video-only. No automated metric.**
- **C2b** — Patch ONLY the object tokens ("wine bottle", positions 591–592 in clean) into the corrupt run's object position ("bowl", position 591 in corrupt). Effective behavior: wine_bottle/plate — compositionally novel. **Video-only. No automated metric.**
- **Pair D** — Separate pair. clean=`"put the bowl on the stove"`, corrupt=`"turn on the stove"`. Full prefix patch. Motor-class flip: pick-place vs. knob-turn. Automated metric.

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
  --args.patch-positions "$(python -c "print(','.join(map(str, range(588, 788))))")" \
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
- < 2/5: stop Phase 2c. Write `0_PHASE2C_FAILURE.txt` (see below). Do not run C2a, C2b, or Pair D.

**Commit after C1.**

---

### Step C2-prep — Identify object and destination token positions

Before running C2a or C2b, verify the exact KV positions for "wine bottle" and "rack" in the clean prompt, and "bowl" and "plate" in the corrupt prompt. The expected positions (RunPod-verified 2026-04-29):

| Token | Prompt | Local index | Absolute KV position |
|-------|--------|-------------|----------------------|
| "wine" | clean | 3 | 591 |
| "bottle" | clean | 4 | 592 |
| "rack" | clean | 7 | 595 |
| "bowl" | corrupt | 3 | 591 |
| "plate" | corrupt | 6 | 594 |

*Absolute KV position = 588 + local index. Language tokens start at absolute index 588 (image region occupies 0–587).*

Run this tokenizer check to confirm (from the `openpi/` directory):

```bash
python -c "
import glob, sentencepiece as spm

# Find the PaliGemma SentencePiece tokenizer
candidates = (glob.glob('**/*.model', recursive=True) +
              glob.glob(os.path.expanduser('~/.cache/**/*.model'), recursive=True))
tok_path = next((p for p in candidates if 'tokenizer' in p.lower()), None)
if tok_path is None:
    raise RuntimeError('No tokenizer.model found — try: find . -name tokenizer.model')
sp = spm.SentencePieceProcessor(); sp.Load(tok_path)

LANG_START = 588  # absolute KV index where language tokens begin
for prompt, label in [
    ('put the wine bottle on the rack', 'clean'),
    ('put the bowl on the plate',       'corrupt'),
]:
    tokens = sp.EncodeAsPieces(prompt)
    print(f'--- {label}: \"{prompt}\" ---')
    for i, tok in enumerate(tokens):
        print(f'  local[{i:2d}]  abs[{LANG_START + i}]  {tok!r}')
    print()
import os  # needed for expanduser above — move to top if running standalone
"
```

If verification confirms the positions above, proceed. **Record actual positions if they differ from the table** — they determine C2a and C2b patch-positions.

---

**Token alignment convention — self-contained, applies to any future pair**

When clean and corrupt prompts differ in token count, two distinct problems arise. Both are addressed here so the agent does not need to rediscover the logic for each new pair.

**Problem 1 — Span mismatch.** A word in one prompt has more tokens than the corresponding word in the other. For example, "wine bottle" (2 tokens) vs "bowl" (1 token).

Convention:
- **For object tokens (C2b):** patch the full span of the longer name. "wine bottle" (positions 591–592 in clean) → overwrite positions 591–592 in the corrupt run. Position 591 maps wine→bowl (same-index, direct). Position 592 maps "bottle" (clean) → "on" (corrupt) — this is an approximation, since "bottle" and "on" are different tokens, but the pair is read together as a unit. Run both 591+592 first. If behavior looks wrong (robot acts on neither object correctly), retry with 591 only.
- **For all other single-word differences:** patch only that one position. No span consideration needed.

**Problem 2 — Position shift.** Because the two prompts have different-length spans at word W, every token *after* W shifts by N positions between clean and corrupt, where N = (clean span length) − (corrupt span length). This offset is constant for all subsequent tokens.

Example here: "wine bottle" (N=2) vs "bowl" (N=1) → shift = +1 for clean relative to corrupt. So "rack" at clean[595] corresponds to corrupt[594] — you must write clean[595] → corrupt[594], not clean[594] → corrupt[594].

Convention:
- Whenever positions differ between clean and corrupt for a target token, use `--args.patch-positions` for the destination (corrupt) position and `--args.patch-source-positions` for the source (donor) position (e.g., `--args.patch-positions "594" --args.patch-source-positions "595"`).
- Never assume same-index for tokens that come after a span-length difference. Always compute the offset from the tokenizer output above.
- If the per-step donor script does not yet support `--args.patch-source-positions` / `--args.patch-dest-positions`, add these parameters first (additive change; when only `--args.patch-positions` is provided, behavior is unchanged — source and dest positions are identical).

**General procedure for any new pair:** tokenize both prompts, find the first word where token counts differ, compute the shift (clean_span_len − corrupt_span_len), apply that offset to all subsequent token positions. Record source and dest positions explicitly in the experiment table before running.

> Deeper rationale on span-mismatch options (A/B/C) is in `status_cc/kv_cache_findings.md §6`. Refer to it only if the conventions above produce ambiguous results.

---

### Step C2-code — Add source→dest position remapping (only if not already present)

Check whether `main_patching_expt_per_step_donor.py` already has `--args.patch-source-positions`. If yes, skip this step. If not, make the following **additive** changes (defaults preserve all existing behavior):

**`src/openpi/models/pi0.py` — `_apply_kv_patch`:**

Add `patch_source_positions` parameter (default `None` = same as patch_positions, i.e., no change for existing callers):

```python
def _apply_kv_patch(
    self,
    corrupt_kv_cache: _gemma.KVCache,
    donor_kv_cache: _gemma.KVCache,
    patch_positions: tuple[int, ...],          # destination positions in corrupt cache
    patch_layers: tuple[int, ...] | None = None,
    patch_k: bool = True,
    patch_v: bool = True,
    patch_source_positions: tuple[int, ...] | None = None,  # NEW — source positions in donor cache; None = same as patch_positions
) -> _gemma.KVCache:
    K_d, V_d = donor_kv_cache
    K, V = corrupt_kv_cache
    layer_indices = tuple(range(K.shape[0])) if patch_layers is None else patch_layers
    src_positions = patch_source_positions if patch_source_positions is not None else patch_positions  # NEW
    for src_pos, dst_pos in zip(src_positions, patch_positions):                                       # CHANGED (was: for pos in patch_positions)
        K_patched = K.at[layer_indices, :, dst_pos, :, :].set(K_d[layer_indices, :, src_pos, :, :])   # CHANGED (was: pos, pos)
        V_patched = V.at[layer_indices, :, dst_pos, :, :].set(V_d[layer_indices, :, src_pos, :, :])   # CHANGED (was: pos, pos)
        K = jax.lax.cond(patch_k, lambda _: K_patched, lambda _: K, operand=None)
        V = jax.lax.cond(patch_v, lambda _: V_patched, lambda _: V, operand=None)
    return (K, V)
```

**`src/openpi/models/pi0.py` — `sample_actions`:** add parameter and pass through:

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
    patch_layers=None,
    patch_k=True,
    patch_v=True,
    patch_source_positions=None,   # NEW
):
    ...
    if donor_kv_cache is not None:
        kv_cache = self._apply_kv_patch(
            kv_cache, donor_kv_cache, patch_positions,
            patch_layers, patch_k, patch_v,
            patch_source_positions=patch_source_positions,   # NEW
        )
```

**`examples/libero/main_patching_expt_per_step_donor.py` — `Args` and wiring:**

```python
# In Args dataclass — add after patch_positions:
patch_source_positions: str = ""  # NEW — if set, donor positions to read from (comma-separated); defaults to same as patch_positions

# In main(), after patch_positions is parsed — add:
patch_source_positions = _parse_int_spec(args.patch_source_positions) if args.patch_source_positions.strip() else None

# In _sample_kwargs wiring — add alongside patch_positions:
policy._sample_kwargs["patch_source_positions"] = patch_source_positions
```

**Verify:** run A-C3 full-prefix with no `--args.patch-source-positions` flag — result must match the earlier A-C3 run. This confirms the default path is unchanged.

**Commit:** `git commit -m "phase2c C2-code: add patch_source_positions for source→dest remapping (additive, default=None)"`

---

### Step C2a — Pair A, destination-only sub-test (video-only)

Same clean/corrupt prompts as C1. Patch ONLY the destination token ("rack", clean position 595) into the corrupt run's destination position ("plate", corrupt position 594). Effective behavior: bowl/rack — compositionally novel, not in LIBERO-Goal 10.

| Parameter | Value |
|-----------|-------|
| `--args.clean-prompt` | `"put the wine bottle on the rack"` |
| `--args.corrupt-prompt` | `"put the bowl on the plate"` (same as C1) |
| `--args.task-name-filter` | `"put the wine bottle on the rack"` |
| Patch positions | Source: 595 (clean "rack") → Destination: 594 (corrupt "plate") |
| Success metric | **Video-only. No automated metric.** Done flag checks wine_bottle/rack, which won't trigger for bowl/rack behavior. |

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
  --args.corrupt-prompt "put the bowl on the plate" \
  --args.patch-positions "594" \
  --args.patch-source-positions "595" \
  --args.num-trials-per-task 10 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/phase2c/pairA_destonly
} 2>&1 \
  | tee "$FULL" \
  | tr '\r' '\n' | sed -u $'s/\x1b\\[[0-9;]*[A-Za-z]//g' \
  | grep -E --line-buffered '^(===|\[ep |\[DEBUG|INFO:root:|ERROR:root:)' \
  | tee "$CLEAN"
```

**No automated metric.** Save videos to `data/libero/videos/phase2c/pairA_destonly` and record the run in `progress_cc/phase2/implementation.md §9.1`. Video interpretation is the human's job. Proceed to C2b.

**Commit after C2a.**

---

### Step C2b — Pair A, object-only sub-test (video-only)

Same clean/corrupt prompts as C1. Patch ONLY the object tokens ("wine bottle", clean positions 591–592) into the corrupt run's object position ("bowl", corrupt position 591). Effective behavior: wine_bottle/plate — compositionally novel, not in LIBERO-Goal 10.

Note: position 591 is same-index in both prompts (wine=591 in clean, bowl=591 in corrupt) so this is a direct substitution. Position 592 is "bottle" in clean but "on" in corrupt — patch both 591+592 as a first attempt; if behavior is odd, try 591 alone.

| Parameter | Value |
|-----------|-------|
| `--args.clean-prompt` | `"put the wine bottle on the rack"` |
| `--args.corrupt-prompt` | `"put the bowl on the plate"` (same as C1) |
| `--args.task-name-filter` | `"put the wine bottle on the rack"` |
| Patch positions | 591, 592 (clean "wine", "bottle" → corrupt "bowl", "on") |
| Success metric | **Video-only. No automated metric.** Done flag checks wine_bottle/rack, which won't trigger for wine_bottle/plate behavior. |

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
  --args.corrupt-prompt "put the bowl on the plate" \
  --args.patch-positions "591,592" \
  --args.num-trials-per-task 10 \
  --args.save-all-videos \
  --args.video-out-path data/libero/videos/phase2c/pairA_objonly
} 2>&1 \
  | tee "$FULL" \
  | tr '\r' '\n' | sed -u $'s/\x1b\\[[0-9;]*[A-Za-z]//g' \
  | grep -E --line-buffered '^(===|\[ep |\[DEBUG|INFO:root:|ERROR:root:)' \
  | tee "$CLEAN"
```

**No automated metric.** Save videos to `data/libero/videos/phase2c/pairA_objonly` and record the run in `progress_cc/phase2/implementation.md §9.1`. Video interpretation is the human's job.

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

Pair A (clean=wine_bottle/rack, corrupt=bowl/plate — same prompts for C1/C2a/C2b, patch positions differ):
  C1 sanity (all lang tokens 588-787): X/5; N=10: Y/10 — automated metric
  C2a (dest-only, patch rack→plate pos): videos saved to data/libero/videos/phase2c/pairA_destonly/
  C2b (obj-only, patch wine_bottle→bowl pos 591-592): videos saved to data/libero/videos/phase2c/pairA_objonly/

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

## Signal Files Summary

| File | Write when | Location |
|------|-----------|----------|
| `0_PHASE2A_FAILURE.txt` | A-C3 fails after 2 debug iterations | `progress_cc/phase2/signal_files/` |
| `1_PHASE2A_MEANINGFUL_RESULT.txt` | A-final N=25 > 5/25 | `progress_cc/phase2/signal_files/` |
| `2_PHASE2B_ALPHA_SWEEP_COMPLETE.txt` | All alpha values run, CSV written | `progress_cc/phase2/signal_files/` |
| `3_PHASE2C_COMPLETE.txt` | All Phase 2c sub-experiments run | `progress_cc/phase2/signal_files/` |
| `0_PHASE2C_FAILURE.txt` | C1 fails (< 2/5) | `progress_cc/phase2/signal_files/` |
