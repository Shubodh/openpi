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
| `status_cc/patching_implementation_dryrun.md` | Complete implementation plan for KV-cache patching — architecture, code locations, all open decisions resolved |
| `status_cc/patching_implementation.md` | Live implementation tracker — checklist, notes, and results (populate as work proceeds) |

---

## Experiment status

| Experiment | Status | Key output |
|------------|--------|------------|
| Prompt-ablation check (LIBERO-Object) | ✅ Complete | Language load-bearing (96% clean, 36% corrupt); suite pivot to LIBERO-Goal |
| KV-cache sanity check | ✅ Complete (2026-04-29) | `kv_cache_findings.md` — shape, token positions, hook point all confirmed |
| LIBERO-Goal baseline (clean + corrupt prompt) | ✅ Complete (2026-05-03) | Both behave as expected; model is sensitive to language on LIBERO-Goal |
| **Patching implementation dry-run doc** | ✅ **Complete (2026-05-04)** | `status_cc/patching_implementation_dryrun.md` — all O0–O6 resolved |
| **Patching code implementation** | ⏳ **Next task** | See §10 in dry-run doc for script architecture; implementation ready to begin |
| Exhaustive sim runs (patching battery) | ⏳ Pending | Depends on patching code |

---

## Next task: Patching code implementation

All open decisions resolved. Read `status_cc/patching_implementation_dryrun.md` §10 for script architecture, then implement.

**All decisions (O0–O6) from the dry-run doc:**
- O0: **Resolved: JAX** (2026-05-04) — staying on the default JAX execution path
- O1: **Resolved:** keep `main_corrupt_run_expt.py` for baselines; create new `main_patching_expt.py`
- O2: **Resolved: P1** (2026-05-04) — modify `pi0.py` directly (add `donor_kv_cache` param + `_apply_kv_patch`)
- O3: **Resolved: Option A, pos 594** (2026-05-04) — patch only the differing destination token for Phase 1
- O4: **Resolved: pre-computed donor** (2026-05-04) — harvest donor KV cache once from initial obs before rollout
- O5: **Resolved: N=25 trials** (2026-05-04)
- O6: **Resolved: contrastive pair only for Phase 1** (2026-05-04) — broaden to full suite after

---

## Upstream openpi

Everything outside `runpod/` and `docs/runpod_setup.md` is upstream openpi. See the upstream `README.md` for model documentation, training, and non-RunPod deployment.
