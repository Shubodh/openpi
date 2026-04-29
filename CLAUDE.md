# openpi — Agent Guide

This is the **Shubodh fork** of the Physical Intelligence [openpi](https://github.com/physical-intelligence/openpi) repo, extended with pod setup infrastructure for running π₀.₅ + LIBERO evaluations on RunPod.

## Added infrastructure (not in upstream)

| Directory | What's in it |
|-----------|-------------|
| `runpod/` | Bash scripts for pod lifecycle — setup, launch, evaluation |
| `docs/runpod_setup.md` | Full operational guide: agent-first setup, LIBERO eval, video download, GPU rationale |

## When you're asked to set up the pod or run LIBERO

1. **Scripts to run:** `runpod/` — start here. Each script has a comment header explaining when and how to use it.
2. **Full guide:** `docs/runpod_setup.md` — covers the complete workflow including agent permissions, parallel suite runs, and baseline verification numbers.
3. **Quick orientation:** `runpod/README.md` — one-line descriptions of each script.

## When you're asked about the research / experiments

This fork is used for the AXMech mechanistic interpretability project. Experiment design, suite choices, and high-level research status live in a separate repo (`AXMech_meta-discussion-FS`).

**Experiment-specific implementation work lives here**, in `status_cc/`. When a new experiment is ready to implement, a briefing document is added to `status_cc/` with everything an agent needs to implement and run it without reading the meta repo.

**Current active experiments:**
- `status_cc/corrupt_run_experiment.md` — ✅ Complete. Prompt-ablation check on LIBERO-Object (96% clean, 36% corrupt). Language is load-bearing but suite generalizes poorly. Pivot to LIBERO-Goal for ActPatch.
- `status_cc/kv_cache_sanity_check.md` — ⏳ **Next task for the agent here.** Read-only inspection of π₀.₅'s prefix KV cache on a LIBERO-Goal prompt pair. Output: a short findings note (`status_cc/kv_cache_findings.md`). Stops before any patching code is written. **See the "Agent plan" section at the bottom of that doc** for the phased execution plan (primer → offline static analysis → optional live pass → findings). Conceptual scaffolding for the findings lives in `status_cc/misc/kv_cache_primer.md`.
- LIBERO-Goal baseline — being run by the human in parallel; not the agent's responsibility.
- Step 4 patching implementation — design discussion happening in the AXMech_meta repo. Will be briefed back here after the human + meta agent decide on architecture.

## Upstream openpi

Everything outside `runpod/` and `docs/runpod_setup.md` is upstream openpi. See the upstream `README.md` for model documentation, training, and non-RunPod deployment.
