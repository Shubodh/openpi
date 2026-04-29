# KV-Cache Sanity Check — π₀.₅ on LIBERO

**Status:** ⏳ To run. ~30-min engineering task. Gates Step 4 patching implementation.

**What this is:** A read-only inspection of π₀.₅'s prefix KV cache during a single forward pass on a LIBERO-Goal task. The output of this task is a short findings note that briefs the next implementation step (the patching code). **No patching, no behavioral changes — just inspect and document.**

This is the "Task B / Option 1" decision in `HUMAN.md`: the openpi agent runs the sanity check and reports findings back; the human + meta-repo agent then design the patching code based on those findings.

---

## Why this comes before patching code

The SO-101 patching result (in the AXMech repo) hooked SmolVLA's residual stream at `(layer 0, token position 131)` — the color-word token. For π₀.₅ the analogous intervention point is the **prefix KV cache** at the object-name token position(s). Three things must be confirmed before writing intervention code:

1. **Cache shape** — what is the layout of the prefix KV cache? (layers, heads, seq positions, head_dim). We need to know the exact tensor shape so we can write a clean overwrite at one position.
2. **Token positions for the object name** — for a prompt like `"put the bowl on top of the cabinet"`, which token index(es) in the prefix correspond to `"bowl"`? Is it always a single token, or does PaliGemma's tokenizer split objects into multiple subwords?
3. **Cache is actually re-used across action steps** — π₀.₅ should compute the prefix once per episode and reuse the KV cache across the closed-loop action chunks. We need to confirm this is how the openpi inference path works (vs. recomputing prefix each step).

If any of these surprises us (e.g., object name is 3 subword tokens, or cache is recomputed each step), the patching design changes. Better to find out in 30 min than mid-implementation.

---

## Concrete steps

### 1. Pick a representative prompt pair

Use the first task pair we'll patch on (from `status_cc/misc/libero_suite_choice_detailed.md`, LIBERO-Goal):

- Clean: `"put the bowl on top of the cabinet"`
- Corrupt: `"put the wine bottle on top of the cabinet"`

These differ in the object-name token region. Note: `wine bottle` is two words — useful as a stress test for whether multi-word objects span multiple tokens.

### 2. Run a single forward pass and inspect

Load the π₀.₅ policy server and a LIBERO-Goal task scene as currently set up. Do **one** forward pass per prompt (no rollout needed — just enough to populate the KV cache once). Inspect:

- **Tokenizer output** — print the full token-id sequence and the decoded token strings for both prompts. Note the index of `"bowl"` and `"wine"` / `"bottle"`. Confirm whether these are single tokens or split.
- **Prefix length** — how long is the prefix (image tokens + text tokens combined)? What's the boundary between image-derived tokens and text tokens?
- **KV cache shape** — for each transformer layer of the language backbone, log the shape of `k` and `v` tensors. Expected something like `[batch, num_heads, seq_len, head_dim]` but confirm.
- **Cache reuse** — find where in the openpi inference code path the prefix cache is built and where it's consumed by subsequent action-decoding steps. Identify the function/module names so the patching hook has a place to live.

### 3. Write findings to `status_cc/kv_cache_findings.md`

A short note (under 1 page) with:
- Token id and string for the object-name position(s) in both prompts
- Prefix layout (image token count, text token count, total)
- KV cache tensor shape per layer
- File path + function name where prefix is built and where it's consumed
- One paragraph: "for patching, the cleanest hook point is X, because Y"
- Any surprises (multi-token splits, dynamic shapes, anything that complicates a single-position overwrite)

Do **not** start writing intervention code based on findings — the human + meta agent will use the findings to design the patching architecture.

---

## Constraints

- **Read-only.** No edits to the openpi inference path. Adding a print statement or a small inspection script under `examples/libero/` is fine; modifying the server or policy code is not.
- **One forward pass per prompt is enough.** Don't run a full rollout — we just need the cache populated.
- **Stop when the findings doc is written.** Do not proceed to patching code. The next step is a discussion in the meta repo.

---

## References

- `status_cc/corrupt_run_experiment.md` — prior LIBERO-Object experiment infrastructure (clean/corrupt run flags). Reuse the prompt-override mechanism if helpful.
- `status_cc/misc/libero_suite_choice_detailed.md` — full task list and prompt strings for LIBERO-Goal.
- `HUMAN.md` (root of this repo) — Step 4 decision framework; this doc resolves Task B.
- AXMech repo (`~/claude_code_workspace/2026_AXMech/AXMech/`) — SmolVLA patching implementation (residual-stream hook at layer 0, pos 131). Useful as a *conceptual* reference for what the analogous π₀.₅ intervention point should look like, but the mechanism is different (KV cache, not residual stream).
