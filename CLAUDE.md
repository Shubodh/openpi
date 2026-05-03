# openpi — Agent Guide

This is the **Shubodh fork** of the Physical Intelligence [openpi](https://github.com/physical-intelligence/openpi) repo, extended with pod setup infrastructure for running π₀.₅ + LIBERO evaluations on RunPod.

---

## Added infrastructure (not in upstream)

| Directory | What's in it |
|-----------|-------------|
| `runpod/` | Bash scripts for pod lifecycle — setup, launch, evaluation |
| `docs/runpod_setup.md` | Full operational guide: agent-first setup, LIBERO eval, video download, GPU rationale |

## When you're asked to set up the pod or run LIBERO

1. **Scripts to run:** `runpod/` — start here. Each script has a comment header explaining when and how to use it.
2. **Full guide:** `docs/runpod_setup.md` — covers the complete workflow including agent permissions, parallel suite runs, and baseline verification numbers.
3. **Quick orientation:** `runpod/README.md` — one-line descriptions of each script.

---

## When you're asked about the research / experiments

This fork is used for the AXMech mechanistic interpretability project. Experiment design, suite choices, and high-level research status live in a separate repo (`AXMech_meta-discussion-FS`).

**Experiment-specific implementation work lives here**, in `status_cc/`. When a new experiment is ready to implement, a briefing document is added to `status_cc/` with everything an agent needs to implement and run it without reading the meta repo.

### status_cc document index

| File | What it's for |
|------|---------------|
| `status_cc/corrupt_run_experiment.md` | Prompt-ablation check on LIBERO-Object — full design, results, and pivot decision |
| `status_cc/kv_cache_sanity_check.md` | KV-cache inspection task brief — phased plan (primer → static analysis → RunPod verification) |
| `status_cc/kv_cache_findings.md` | KV-cache findings — cache shape, token positions, hook point, patching options (authoritative reference for patching implementation) |
| `status_cc/misc/kv_cache_primer.md` | Conceptual background on KV-cache patching — read before kv_cache_findings.md |
| `status_cc/misc/libero_suite_choice.md` | LIBERO suite decision rationale |
| `status_cc/misc/libero_suite_choice_detailed.md` | Full technical reference for all four LIBERO suites |
| `status_cc/misc/openpi_scripts_primer.md` | Primer on the runpod scripts and workflow |

---

## Experiment status

| Experiment | Status | Key output |
|------------|--------|------------|
| Prompt-ablation check (LIBERO-Object) | ✅ Complete | Language load-bearing (96% clean, 36% corrupt); suite pivot to LIBERO-Goal |
| KV-cache sanity check | ✅ Complete (2026-04-29) | `kv_cache_findings.md` — shape, token positions, hook point all confirmed |
| LIBERO-Goal baseline (clean + corrupt prompt) | ✅ Complete (2026-05-03) | Both behave as expected; model is sensitive to language on LIBERO-Goal |
| **Patching implementation dry-run doc** | ⏳ **Next task** | See task brief below |
| Patching code implementation | ⏳ Pending | Depends on dry-run doc |
| Exhaustive sim runs (patching battery) | ⏳ Pending | Depends on patching code |

---

## Next task: Patching implementation dry-run doc

**What:** Write a design document (`status_cc/patching_implementation_dryrun.md`) that is a complete implementation plan for the KV-cache patching code on π₀.₅ + LIBERO-Goal — including references to relevant code locations and key snippets — but stops short of writing the actual patching implementation. This document should be detailed enough that implementing the code afterwards is mechanical.

**Why a dry-run doc first:** The patching architecture for π₀.₅ is meaningfully different from the SmolVLA residual-stream patching done in the real-robot experiment. Writing it out first (at multiple levels of abstraction, with code references) forces design decisions to be made explicitly before implementing.

**Design principle — mirror SmolVLA where it makes sense:**
The SmolVLA real-robot patching code is the reference implementation. The π₀.₅ implementation should mirror its structure to the extent the architectures allow (residual-stream vs. KV-cache patching differ mechanistically, so some divergence is expected). Architectural decisions — like whether to integrate patching directly into openpi's source or to write new standalone scripts alongside the existing `examples/libero/` scripts — should be made by looking at how SmolVLA does it first, then discussed with the human before committing to an approach. Do not decide unilaterally.

**What to read before writing:**
1. `status_cc/kv_cache_findings.md` — the authoritative reference. Sections 4 (cache reuse), 5 (hook point), 6 (patching options A/B/C), and 7 (summary table) are most load-bearing.
2. `status_cc/misc/kv_cache_primer.md` — conceptual background.
3. The SmolVLA patching code in `AXMech/` for comparison (full path: `/home/shubodh/claude_code_workspace/2026_AXMech/AXMech`). Read it to understand the overall script structure, clean/corrupt/patched run orchestration, and where patching hooks were inserted.

**What the doc should cover (at minimum):**
- High-level architecture: how clean / corrupt / patched runs relate (one script or three invocations?)
- Hook point and exact code location (`pi0.py:sample_actions` after cache build at line 237)
- Which token positions to patch (Option A vs B vs C from kv_cache_findings.md §6) and the chosen approach with rationale
- Cache shape and indexing arithmetic (from kv_cache_findings.md §7 summary table)
- How to handle token-count mismatch between prompts (e.g., `bowl` = 1 token vs `wine bottle` = 2 tokens)
- How patching propagates through diffusion steps (cache is read-only per step — patch once, affects all steps)
- Recovery score definition: `(patched − corrupted) / (clean − corrupted)`
- What a single successful patching trial looks like end-to-end (pseudocode or prose walkthrough)
- Open decisions that need the human's input before coding begins

**Output:** `status_cc/patching_implementation_dryrun.md`

---

## Upstream openpi

Everything outside `runpod/` and `docs/runpod_setup.md` is upstream openpi. See the upstream `README.md` for model documentation, training, and non-RunPod deployment.
