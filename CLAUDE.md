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

This fork is used for the AXMech mechanistic interpretability project. Experiment design, suite choices, and research status live in a separate repo (`AXMech_meta-discussion-FS`) — not here. This repo is infrastructure only.

## Upstream openpi

Everything outside `runpod/` and `docs/runpod_setup.md` is upstream openpi. See the upstream `README.md` for model documentation, training, and non-RunPod deployment.
