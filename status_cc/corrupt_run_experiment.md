# Corrupt Run Experiment — Prompt-Ablation Check

**Status:** ✅ Complete — results recorded below

**What this is:** Step 3's prerequisite check for the AXMech ActPatch project. Before running activation-patching experiments on π₀.₅ + LIBERO-Object, we must confirm that the language prompt is actually load-bearing — i.e., changing the object name in the prompt changes what the robot picks up. If the model ignores the prompt (positional shortcut), patching the object-name token would be meaningless regardless of patching mechanism.

These are also the first two legs of ActPatch (clean run + corrupt run). Running them now builds the infrastructure that patching (Step 5) slots into directly.

---

## Scientific Question

**Does π₀.₅ follow the language instruction when it conflicts with the positional shortcut?**

LIBERO-Object tasks have deterministic object placement — the target object is always at the same position for a given task. A finetuned policy could learn "go to position Y for this scene" without consulting the language at all. Prior work (LIBERO-PRO, NeurIPS 2025) showed models persist at ~0% success under nonsensical prompts but still execute the correct trajectory — meaning the model can succeed *without* language. Our corrupt run tests whether a *valid conflicting* language signal gets followed.

---

## Experiment Design

### Task pair

- **Target task:** `pick_up_the_milk_and_place_it_in_the_basket`  
  — The `task.language` string (the prompt passed to the model) is: `"Pick up the milk and place it in the basket"`
- **Corrupt prompt:** `"Pick up the tomato sauce and place it in the basket"`

**Why this pair:** Visually verified that both the milk carton and tomato sauce bottle are physically present in the milk task scene. The corrupt run keeps the scene identical; only the language changes. Orange juice is a confirmed backup if needed (also verified present in both scenes).

### The two conditions

| Condition | Scene | Prompt passed to model | What we measure |
|-----------|-------|------------------------|-----------------|
| **Clean run** | Milk task (default initial states) | `"Pick up the milk and place it in the basket"` | Baseline success rate (should be ~98%) |
| **Corrupt run** | Same milk task, same initial states | `"Pick up the tomato sauce and place it in the basket"` | Does prompt change behavior? |

**Only the prompt changes between conditions. Scene, initial states, seed — all identical.**

### Interpreting the result

- **Corrupt success rate drops significantly** (e.g., <50%) → language is load-bearing → LIBERO-Object is a valid suite for ActPatch; proceed.
- **Corrupt success rate stays high** (~90%+) → model ignores prompt text; positional shortcut dominates. This does *not* mean activation patching will fail (we intervene at a lower level than text), but it changes how the claim is framed. Report to meta discussion; suite choice may need revisiting.

**Run at least 20–30 rollouts per condition** for a reliable signal. 25 per condition (total 50) is the practical minimum.

---

## What to Implement

The modification to `examples/libero/main.py` is minimal — two new CLI flags.

### Flag 1: `--args.corrupt-prompt`

Optional string. If set, replaces the prompt sent to the model (overrides `task_description` in the observation dict). Default: `None` (no override — existing behavior unchanged).

### Flag 2: `--args.task-name-filter`

Optional string. If set, only runs tasks where `task_description` contains the filter as a substring (case-insensitive). Allows isolating just the milk task without modifying any other logic. Default: `None` (run all tasks — existing behavior unchanged).

### Resulting `main.py` changes (conceptual)

In the `Args` dataclass, add:
```python
corrupt_prompt: Optional[str] = None  # if set, overrides the task's language prompt
task_name_filter: Optional[str] = None  # if set, only run tasks whose description contains this string
```
Note: the client runs in a Python 3.8 venv (`examples/libero/.venv`), so use `Optional[str]` from `typing`, not `str | None`.

In `eval_libero`, after `task_description` is set:
- Skip task if `args.task_name_filter` is set and not in `task_description.lower()`
- In the observation dict, use `args.corrupt_prompt if args.corrupt_prompt else str(task_description)` as the prompt value

Also update the video filename suffix to make it clear which condition produced the video — e.g., append `_corrupt` to the video directory name or filename when `corrupt_prompt` is active.

### Also create: `runpod/run_corrupt_check.sh`

A convenience script that documents the exact commands to run both conditions:

(50 trials is default)

```bash
# Clean run (25 trials on milk task only):
python examples/libero/main.py \
  --args.task-suite-name libero_object \
  --args.task-name-filter "milk" \
  --args.num-trials-per-task 25 \
  --args.video-out-path data/libero/videos/corrupt_check_clean

# Corrupt run (same, different prompt):
python examples/libero/main.py \
  --args.task-suite-name libero_object \
  --args.task-name-filter "milk" \
  --args.corrupt-prompt "Pick up the tomato sauce and place it in the basket" \
  --args.num-trials-per-task 25 \
  --args.video-out-path data/libero/videos/corrupt_check_corrupt
```

---

## What to Record

After both conditions complete:
1. **Success rates**: X/25 clean, Y/25 corrupt — record in this file under "Results"
2. **Videos**: download both video dirs from RunPod to local; inspect a sample of corrupt-run videos to see whether the robot goes toward milk or tomato sauce
3. **Qualitative note**: does the robot trajectory look identical between clean and corrupt (positional shortcut), or does it change direction?

Add results here under a `## Results` section when done. This is the gate for Step 4 (experiment design) in the meta repo.

---

## Infrastructure Context

The policy server and LIBERO client are already confirmed working (baseline: 97.8% on libero_object, A40, seed 7). The typical RunPod flow:

```bash
# On pod (after setup_pod.sh + start_libero.sh):
source /workspace/openpi/runpod/libero_env.sh

# Then run the corrupt check script:
bash /workspace/openpi/runpod/run_corrupt_check.sh
```

Full RunPod setup guide: `docs/runpod_setup.md`  
Script reference: `runpod/README.md`

---

## Results

Run: `scripts_outputs_txt/corrupt_check_20260428_095455.txt` — A40, seed 7, 25 trials per condition.

| Condition | Rollouts | Successes | Success Rate |
|-----------|----------|-----------|--------------|
| Clean (milk prompt on milk scene) | 25 | 24 | **96.0%** |
| Corrupt (tomato sauce prompt on milk scene) | 25 | 9 | **36.0%** |

**Conclusion: Language is load-bearing.** The corrupt prompt dropped success from 96% to 36% — a 60pp collapse. LIBERO-Object is a valid suite for ActPatch; proceed to Step 4.

The 36% residual (9/25 successes even under the wrong prompt) is consistent with a partial positional shortcut: the milk carton is always at the same location, so the policy sometimes picks it up despite being told to pick tomato sauce. But the language signal is clearly dominant — the model is not ignoring the prompt.

**Cross-check with baseline:** Clean run 96.0% vs full-suite baseline 97.8% (489/500, A40, seed 7). The 1.8pp gap is within expected single-task variance at n=25.
