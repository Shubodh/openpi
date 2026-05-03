# Patching Implementation

**Status:** In progress — implementation begun 2026-05-04.

**Reference:** `status_cc/patching_implementation_dryrun.md` — authoritative source for all design decisions, indexing arithmetic, and architecture. Read that first.

---

## Implementation Checklist

### Phase 1 files to create/modify

- [ ] `src/openpi/models/pi0.py` — add `donor_kv_cache` + `patch_positions` params to `sample_actions`; add `_apply_kv_patch` method
- [ ] `examples/libero/main_patching_expt.py` — new script (copy of `main_corrupt_run_expt.py` + donor harvest + patch wiring)

### Phase 1 verification steps

- [ ] Shape debug print: confirm `kv_cache` shape is `(18, 1, 788, 1, 256)` on first patched call
- [ ] Sanity check: patch all 788 positions → behavior should match clean run
- [ ] Verify token positions on RunPod: run `inspect_kv_cache.py` on `put the bowl on the plate` / `put the bowl on the stove` pair to confirm `plate`/`stove` at absolute index 594

### Phase 1 runs (N=25 each, contrastive pair only)

- [ ] Clean baseline: `put the bowl on the plate` — target success rate ≥ 70%
- [ ] Corrupt baseline: `put the bowl on the stove` (stove prompt, plate task) — target ≤ 30%
- [ ] Patched run: stove prompt + KV patch at pos 594 — does it recover toward clean?

---

## Implementation Notes

*(Populate as implementation proceeds — code decisions, surprises, deviations from dry-run plan.)*

---

## Results

*(Populate after runs complete.)*

| Run | N | Success rate | Notes |
|-----|---|-------------|-------|
| Clean | 25 | — | |
| Corrupt | 25 | — | |
| Patched (pos 594, K+V) | 25 | — | |
