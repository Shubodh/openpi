---
name: libero-suite-choice
description: LIBERO suite decision and rationale — current decision is LIBERO-Goal as the primary ActPatch suite after Step 3; earlier LIBERO-Object analysis is preserved as historical rationale. Covers general overview of all four suites, π₀.₅ benchmark numbers, the finetuning/train-test question (Q1.1), zero-shot vs finetuned / distributional vs behavioral generalization / sample efficiency (Q1.2), the positional-shortcut concern with LIBERO-PRO empirical confirmation (Q2), how LIBERO evaluation works and what "successful intervention" means in sim (Q3), within-task rollout invariance (Q4), why LIBERO-Object was rejected after prompt-ablation/generalization checks, and why LIBERO-10 is unsuitable as primary for activation patching. Use when deciding which LIBERO suite to use or when explaining the suite choice to collaborators.
---

# LIBERO Suite Choice

**Current decision: LIBERO-Goal is the primary ActPatch suite.**

Historical note: the first-pass Step 1 decision favored LIBERO-Object as the primary suite, with LIBERO-10 as a later generalization check. Step 3 superseded that after the prompt-ablation/generalization checks: LIBERO-Object language is load-bearing within a task, but cross-task patching is confounded by layout-to-object memorization. The active plan is therefore LIBERO-Goal first, then consider broader/generalization checks after the mechanism is established.

---

## Key terms (flashcard-style)

**Step** — one control cycle: the policy receives an observation, outputs an action, and the simulator advances one physics tick. LIBERO is built on robosuite, which defaults to **20 Hz** (1 step = 50 ms). At that rate, LIBERO-Object's 280-step window ≈ 14 seconds of real time, and its longest training demo (254 steps) ≈ 12.7 seconds. Verify the exact frequency when spinning up the environment — openpi may override the robosuite default.

**Episode window** — the maximum number of *steps* a single rollout is allowed to run before being counted as failure. Set above the longest training demo to give the policy slack without letting a stuck robot run indefinitely.

**Done flag** — a binary signal fired by the MuJoCo simulator when the gripper achieves the task's success state (correct contact, correct placement). This is the sole success criterion across all LIBERO suites — no continuous reward, just 0 or 1 per episode.

**Rollout** — one full execution of the policy in sim from reset to either success (`done=True`) or timeout (episode window exceeded). LIBERO evaluates over 50 rollouts per task.

---

## LIBERO — Overview

LIBERO (Lifelong Robot Learning Benchmark) is a simulation benchmark introduced at NeurIPS 2023 for evaluating knowledge transfer and generalization in robot manipulation policies. It is built on the MuJoCo physics engine with a tabletop manipulation setup, and all tasks are specified via natural language instructions.

The benchmark is organized around four task suites, each designed to isolate a different axis of generalization. All suites share a similar physical setup (tabletop, fixed camera, parallel-jaw gripper), so differences across suites are controlled to the language and task structure level.

### LIBERO-Spatial (10 tasks)

Tests **spatial reasoning**. The same set of objects appears in all tasks, but their placement and the spatial relationship described in the language instruction change. Example: "pick up the bowl on the left" vs. "pick up the bowl on the right." The object is the same; what varies is where it is and how the instruction refers to its location. Useful for studying how policies ground spatial language ("left," "behind," "on top of") to visual layout.

All 10 tasks share a single scene. Initial object placement is deterministic (fixed initial states), and evaluation runs 50 rollouts per task with a 220-step episode window (the longest training demo is 193 steps). Success is binary — a done flag fires when the gripper achieves the target placement. Because the objects are identical across tasks and only spatial language varies, this suite isolates spatial grounding most cleanly but offers the least language diversity of the four suites.

### LIBERO-Object (10 tasks)

Tests **object category generalization**. All tasks follow the same pick-place procedure, but the object named in the instruction changes across tasks. Example: "pick up the red cube" vs. "pick up the yellow cup." The spatial layout and task structure are held constant; what varies is object identity as named in language. This is the cleanest suite for studying how policies encode and act on object-identifying language.

All 10 tasks share a single scene. Each task introduces a different target object; the pick-place procedure is identical across all 10. Episode window is 280 steps (longest demo: 254 steps); 50 rollouts per task, binary success. Important caveat for interpretability experiments: initial object placement is deterministic per task, meaning within a single task the target object is always at the same position. This opens the possibility of a positional shortcut — a policy could learn "go to position Y" without consulting the language token "red cube." See the dedicated section below on this concern.

### LIBERO-Goal (10 tasks) — current primary suite

Tests **goal/instruction understanding**. Objects and spatial layout are familiar, but the goal instruction changes — requiring the policy to interpret different procedural objectives with the same visual context. Example: same objects, different target ("place X in the bowl" vs. "place X on the plate"). Useful for studying how policies encode goal semantics separate from object or spatial grounding.

All 10 tasks share a single scene and the same object set. What varies is the goal/destination, not object identity or spatial arrangement. Episode window is 300 steps (longest demo: 270 steps); 50 rollouts per task, binary success.

Earlier notes treated this as less clean than LIBERO-Object because the language variation can be procedural ("in the bowl" vs. "on the plate") rather than a single nominal object token. After Step 3, that tradeoff flipped: identical layout across tasks is more important for the first π₀.₅ ActPatch result, because it makes clean/corrupt/patched failures interpretable instead of confounded with cross-task layout memorization.

### LIBERO-10 (LIBERO-Long, 10 tasks)

Tests **compositional long-horizon manipulation**. Tasks are more complex and multi-step, with entangled variation across object identity, spatial relationships, and goal instructions simultaneously. Example: "put the black bowl in the bottom drawer and close it." Success requires chaining multiple sub-tasks. This is the hardest suite (lowest success rates across all evaluated models) and is used as a stress test for generalization across all axes at once.

All 10 tasks share a single scene. Episode window is 520 steps (longest demo: 505 steps) — roughly 2.6× longer than LIBERO-Spatial and ~2× longer than LIBERO-Object. Each task involves an estimated 2–4 sequential sub-tasks (e.g., grasp object → open drawer → place object → close drawer). Success is binary: the full multi-step sequence must complete. 50 rollouts per task. The entanglement of object, spatial, and goal variation across sub-tasks makes causal attribution via patching significantly noisier than in the single-axis suites.

### Suite summary

| Suite | Primary axis | What's fixed | What varies | Complexity |
|-------|-------------|-------------|-------------|-----------|
| LIBERO-Spatial | Spatial grounding | Objects, procedure | Spatial relations in language | Low |
| LIBERO-Object | Object identity | Procedure, layout | Object name in language | Low |
| LIBERO-Goal | Goal semantics | Objects, layout | Goal instruction | Low |
| LIBERO-10 | All axes combined | Nothing | Object + spatial + goal | High |

---

## The four LIBERO suites — π₀.₅ benchmark numbers

| Suite | π₀.₅ success |
|-------|--------------|
| LIBERO-Spatial | 98.8% |
| LIBERO-Object | 98.2% |
| LIBERO-Goal | 98.0% |
| LIBERO-10 (LIBERO-Long) | 92.4% |

---

## Notes on finetuning and evaluation (Q&A)

### Q1.1 — When we say "π₀.₅ gets 98% on LIBERO-Object," what dataset is it finetuned on? Are there train/test splits?

LIBERO ships with **50 human teleoperated demonstrations per task** for each suite (teleoperation in sim = a human controlling the simulated robot in real time via keyboard/gamepad/3D mouse while watching the rendered scene — same idea as physical teleoperation, no hardware involved). The standard protocol is: finetune the pretrained model on those 50 demos, then evaluate behavioral success rate over 50 rollouts per task in the same simulator. The dataset is available via HuggingFace in RLDS format (`modified_libero_rlds`).

There is **no held-out test dataset** in the traditional ML sense. The "test" is not generalization to unseen data — it is whether the finetuned policy can successfully complete the task in simulation. The benchmark was originally designed for lifelong learning (can you transfer knowledge from task A to task B without forgetting A?), so "generalization" in LIBERO means behavioral generalization across tasks, not distributional generalization across data splits. The 98% number means: after finetuning on 50 demos, the policy completes the task on 49 out of 50 rollout episodes.

To be precise about the two senses of "generalization" that often get conflated: **distributional generalization** means the model performs well on data drawn from a different distribution than training — the standard train/test split paradigm in supervised ML. **Behavioral/task generalization** means the policy successfully executes the task in the environment — it may have seen every relevant situation during training, and that's fine. LIBERO tests only the second kind: given the same tasks the model trained on, can it execute them reliably? The cross-task transfer aspect of LIBERO (its original lifelong-learning motivation) is a variant of the second kind too — can a policy trained on task A still execute task A after training on task B? Still behavioral, still not distributional.

The finetuning is **imitation learning (behavior cloning)**: the model is trained to predict the human's actions given the observations at each timestep — exactly the same as finetuning on real-robot demos, just with sim-recorded data. Total demos per suite: 50 per task × 10 tasks = **500 demos per suite**. On the overfitting question: you're not conflating the concept, but the concern doesn't apply here the way it does in traditional ML. In traditional ML, overfitting means the model memorizes training data and fails on a held-out test set. Here there is no held-out test set — the "test" is whether the robot successfully executes the task in sim. Since the task is deterministic (same scene, same goal every rollout), a model that perfectly memorized the 50 demos would probably score 100% — and that would be fine, that's the goal. The benchmark is not measuring distributional generalization; it is measuring whether the policy works. So "overfitting to the demos" is not a failure mode in LIBERO's evaluation scheme — it would only matter if you cared about generalizing to new demonstrations or unseen initial conditions, which LIBERO (by design) does not test.

### Q1.2 — If LIBERO only tests behavioral generalization, is zero-shot evaluation (no finetuning) a valid alternative? Is the field moving toward distributional generalization?

Yes on both counts, and the distinction matters for interpreting the 98% numbers.

**Zero-shot evaluation is valid and is a distributional generalization test.** If you take a strong pretrained model and evaluate it on LIBERO without any task-specific finetuning, you're asking: did pretraining cover enough of the world that the model can handle these tasks out of the box? That's distributional generalization — the model was never explicitly trained on these tasks, yet must execute them. LIBERO's original design doesn't require this, but nothing stops you from doing it, and increasingly papers report both zero-shot and finetuned numbers.

**π₀.₅ zero-shot numbers on LIBERO**: clean per-suite zero-shot numbers for the standard 4 suites are not publicly reported by Physical Intelligence. What is known: on LIBERO-90 (a harder 90-task set outside the standard suites), π₀.₅ without task-specific finetuning scores approximately **18%**. Some community reproductions of the base checkpoint on the standard suites report near-zero success. The ~97–98% numbers in our docs are all post-finetuning. So the finetuning step is doing substantial work — the gap between 18% (or lower) and 98% is the value added by 500 demos of imitation learning.

**The field is actively moving toward distributional generalization.** Your historical framing is correct: robotics was hard enough that "given 50 demos, can the policy reliably execute this one task?" was a real and non-trivial challenge. With large pretrained VLAs, that bottleneck has shifted. The field now increasingly asks: few-shot performance (how little data does it need?), zero-shot performance (can it handle tasks it's never seen?), and out-of-distribution evaluation (new objects, scenes, instructions). That third question is exactly distributional generalization.

**Sample efficiency is the bridge.** Even when zero-shot performance becomes non-trivial, finetuned evaluation remains meaningful — not as "can it do the task at all?" but as "how quickly does it adapt?" A model that reaches 98% from 18% using only 500 demos is demonstrating strong sample efficiency, which is a property worth measuring independently of zero-shot capability. LIBERO's 50-demo-per-task protocol is well-suited to measuring exactly this, even if it wasn't originally designed with that framing in mind.

### Q2 — If layout is fixed in LIBERO-Object, can't the model just hardcode "red = go right"?

Yes, this is a real concern and directly analogous to our SO-101 direction/color confound. The agent confirmed that **initial object placement is deterministic per task** — fixed initial states mean within a single task (e.g., "pick up the red cube"), the red cube is always at the same position. A finetuned policy could in principle learn "when I see this tabletop configuration, go to position Y" without ever consulting the language token "red cube."

What works against this shortcut: (a) the pretrained π₀.₅ was trained on diverse data and likely has robust language grounding before finetuning; (b) across the 10 LIBERO-Object tasks, different objects are at different positions, so there is no single positional shortcut that works globally; (c) with only 50 demos per task, finetuning may not be strong enough to fully overwrite pretrained language grounding.

**Empirical confirmation — LIBERO-PRO (Oct 2025):** A paper systematically tested what happens when object positions are perturbed slightly from their fixed defaults. Result: all models that score ~98% on standard LIBERO collapse to **0% success** under even modest position shifts. π₀.₅ is more resilient than OpenVLA/π₀ (holds up to 0.4 unit displacement vs 0.2 units before collapse), but still drops to zero. This is strong empirical evidence that current finetuned models are indeed exploiting the fixed-position shortcut rather than relying primarily on language. It directly validates the concern raised in Q2 and makes the Step 3 prompt-ablation check even more important.

**Implication for our patching experiment:** We verified that language is load-bearing on LIBERO-Object before investing in patching there. The prompt-ablation check succeeded in that narrow sense (clean milk prompt: 96%; corrupt tomato-sauce prompt on the same scene: 36%). However, LIBERO-Object still failed the broader ActPatch suitability requirement because cross-task comparisons remain confounded by layout-to-object memorization. This is why the primary suite moved to LIBERO-Goal.

### Q3 — How does LIBERO evaluate success automatically? And what does that mean for our "successful intervention" metric?

LIBERO evaluation is fully automated — no human annotation needed. The MuJoCo simulator fires a binary **`done` flag** when the gripper achieves the target state (e.g., gripper contacts the correct object and placement is confirmed). Success rate is simply the fraction of rollouts where the `done` flag fires within the episode window. The evaluation loop in openpi's `examples/libero/main.py` runs `num_trials_per_task` (default 50) rollouts per task, collects the done flags, and reports the average. That's the 98.2% number.

**For our patching experiment**, the natural intervention metric is the same structure: run N rollouts (say 20–50 per condition) under each patching condition and report the fraction where the robot picks the *intended* object:

| Condition | Expected outcome |
|-----------|-----------------|
| No patch (corrupted prompt) | picks wrong object → ~0% on the "correct" definition |
| Patched at (layer, pos) | picks correct object → success rate goes up |
| No patch (clean prompt) | picks correct object → ~98% (ceiling) |

The "successful intervention" is not a new metric — it reuses the same binary done flag, just applied to a new question: after patching, does the robot behave as if it received the clean prompt? This is exactly what we measured on SO-101 (3/3 episodes), now done at scale in sim with statistical rigor.

One subtlety: in LIBERO-Object, the done flag fires for the *task's* target object, not "whichever object the language instruction names." If we patch a "yellow → violet" prompt, we need to confirm the success criterion matches what we're measuring (did it pick the object named in the *clean* prompt?). This is a bookkeeping detail in the evaluation script, not a design problem — worth noting for Step 4.

### Q4 — Within a single task, is everything (scene, object positions, instruction) fixed across all rollouts? And is the Q2 positional-shortcut concern specific to LIBERO-Object?

**Yes, within a single task everything is fixed across all rollouts — and this is true for all four LIBERO suites, not just LIBERO-Object.** A given task (say task 5 of LIBERO-Object: "pick up the yellow cup") always starts with the same scene, the same object positions, and the same instruction, for every one of the 50 rollouts. The only source of variation across rollouts within a task is physics noise (seed-sensitive MuJoCo dynamics — objects drop slightly differently after reset). The visual layout and instruction are identical.

**The Q2 positional-shortcut concern is not about rollout-to-rollout variation.** It is a sharper question: is the language instruction causally necessary at all, or would the visual scene alone suffice? Concretely, for task 5 ("pick up the yellow cup"), the yellow cup is always at position P5. A finetuned policy could learn, from vision alone, "when I see the yellow cup at position P5, reach for P5" — without ever consulting the language token "yellow cup." If so, feeding in the wrong instruction ("pick up the red cube") might not change behavior, because the policy is ignoring language entirely and navigating by visual recognition + fixed position.

This is why the prompt-ablation check in Step 3 matters: it tests whether the language instruction is doing any causal work. The concern is not that positions vary across rollouts (they don't); it is that positions are fixed enough across tasks that a model could learn object-identity → fixed-position shortcuts and ignore language entirely. Across the 10 tasks in LIBERO-Object, each object has a different fixed position, so there is no single global shortcut — but a per-task shortcut (one learned position per task) is still possible.

---

> For full technical details on episode lengths, evaluation setup, and raw agent findings, see `misc/libero_suite_choice_detailed.md`.

---

## Why LIBERO-Object was the first candidate, but is no longer primary

Our technique is **activation patching**: run a clean prompt ("pick up the violet brick") and a corrupted prompt ("pick up the yellow brick") with identical visual input, patch a single activation at one (layer, token position) from the clean run into the corrupted run, and claim that position causally mediates the behavioral difference.

For this to work cleanly, the original argument for LIBERO-Object was:

1. **Language should be the single varying factor.** LIBERO-Object is the suite where object identity in the language instruction is the primary variable — directly analogous to our SO-101 result (color word → pos131 → direction command). No spatial or procedural entanglement.

2. **High success rate reduces noise.** 98.2% means the unpatched baseline is reliable. Causal attribution is cleanest when the model works almost perfectly before intervention — you can attribute behavioral changes to the patch, not to random failures.

3. **Single pick-place procedure.** The task structure is simple: the patch has an unambiguous, interpretable effect (robot picks the wrong object). No multi-step confounds.

Step 3 showed the weakness in that argument. The prompt-ablation check confirmed that language matters within a task, but LIBERO-Object cross-task patching still cannot distinguish "patching failed" from "the model memorized a different task layout." So LIBERO-Object is now useful as evidence that valid language can perturb behavior, not as the primary ActPatch suite.

---

## Why LIBERO-Goal is now the primary suite

LIBERO-Goal gives the cleaner causal setup for the next experiment:

1. **Identical layout across tasks.** The same objects and spatial arrangement are reused, so clean/corrupt/patched comparisons are not primarily testing whether the model generalizes to a new layout.

2. **Contrastive prompts are available.** Several task pairs differ in a small number of language tokens while sharing the same scene, e.g. `put_the_bowl_on_top_of_the_cabinet` vs. `put_the_wine_bottle_on_top_of_the_cabinet`.

3. **Published baseline is still high.** π₀.₅ is reported at 98.0% on LIBERO-Goal, so a clean baseline should be near ceiling if the local setup is healthy.

4. **Failures are more interpretable.** If clean works, corrupt changes behavior, and a KV-cache patch restores clean-like behavior, the result is less vulnerable to the LIBERO-Object layout memorization confound.

Immediate next check: reproduce the LIBERO-Object-style clean/corrupt prompt sanity check on LIBERO-Goal before writing or modifying patching code.

---

## Why not LIBERO-10 as primary

Collaborator suggested LIBERO-10; this was pushed back on for a specific reason tied to the technique difference from Kaylene et al.:

- **Kaylene et al. used LIBERO-Long (= LIBERO-10) for activation *steering*.** Steering doesn't need a clean/corrupted pair — you nudge a direction scalar and observe. Task complexity is less of a problem there.
- **Our technique is patching.** We need the language-behavior mapping to be as clean as possible. LIBERO-10 tasks are long-horizon and compositional ("put the black bowl in the bottom drawer and close it") — multiple objects, multiple actions, multiple language tokens all relevant. It is harder to isolate which patch dimension is doing what.
- **Lower success rate (92.4%)** means more noise in causal estimates.

---

## Role of LIBERO-10 in the experiment plan

LIBERO-10 is not discarded — it serves as a **generalization check**. The argument structure is:

1. Establish the mechanism cleanly on LIBERO-Goal (identical layout, contrastive prompt pairs).
2. Show the patching result holds on LIBERO-10 (complex, long-horizon).

This is a stronger paper claim than LIBERO-10 alone: you explain the mechanism first, then demonstrate robustness to complexity.

---

## Sources

- LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning (NeurIPS 2023)
- Kaylene et al., "Mechanistic Interpretability for Steering Vision-Language-Action Models," CoRL 2025
- openpi π₀.₅ benchmark numbers: `examples/libero/` README
