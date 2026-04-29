---
name: libero-suite-choice-detailed
description: Full technical reference dump for the four LIBERO suites — per-suite spec tables (episode windows, demo lengths, success criteria, positional shortcut risks), cross-suite constants (evaluation protocol, randomization, dataset format), finetuning protocol, and notes from Kaylene et al. (CoRL 2025). Use when you need specific numbers or implementation details about LIBERO, not when you need the decision rationale (see libero_suite_choice.md for that).
---

# LIBERO Suite Choice — Detailed Technical Reference

This document captures the full raw findings from a detailed search of the LIBERO benchmark paper, the openpi repo (`examples/libero/`), and Kaylene et al. (CoRL 2025). It is a reference dump — not curated for reading order. For the synthesized decision, see `libero_suite_choice.md`.

---

## Sources consulted

- `openpi/examples/libero/main.py` — evaluation entry point; confirmed `num_trials_per_task`, episode window, seed handling
- `openpi/examples/libero/README.md` — π₀.₅ benchmark numbers, suite names
- LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning (NeurIPS 2023)
- Kaylene et al., "Mechanistic Interpretability for Steering Vision-Language-Action Models," CoRL 2025
- HuggingFace dataset: `modified_libero_rlds`

---

## Per-suite technical specs (from openpi source + paper)

### LIBERO-Spatial

| Field | Value |
|-------|-------|
| Number of tasks | 10 |
| Shared scene | Yes — 1 scene across all 10 tasks |
| Primary variation | Spatial relationship in language ("left," "right," "on top of") |
| Object set | Same objects across all tasks |
| Episode window | 220 steps |
| Longest training demo | 193 steps |
| Evaluation rollouts | 50 per task (default `num_trials_per_task`) |
| Success criterion | Binary done flag (gripper achieves target placement) |
| Initial state | Deterministic per task (fixed initial states) |
| π₀.₅ success rate | 98.8% |

### LIBERO-Object

| Field | Value |
|-------|-------|
| Number of tasks | 10 |
| Shared scene | Yes — 1 scene across all 10 tasks |
| Primary variation | Object identity named in instruction ("pick up X") |
| Object set | Different target object per task (red cube, yellow cup, etc.) |
| Procedure | Identical pick-place across all 10 tasks |
| Episode window | 280 steps |
| Longest training demo | 254 steps |
| Evaluation rollouts | 50 per task |
| Success criterion | Binary done flag (gripper contacts correct target object) |
| Initial state | Deterministic per task (fixed initial states) |
| π₀.₅ success rate | 98.2% |
| Positional shortcut risk | Yes — within a single task, the target object is always at the same position. Policy could learn "go to position Y" without using language. Verify language is load-bearing before relying on this suite for patching. |

### LIBERO-Goal

| Field | Value |
|-------|-------|
| Number of tasks | 10 |
| Shared scene | Yes — 1 scene across all 10 tasks |
| Primary variation | Goal/destination instruction ("in the bowl" vs. "on the plate") |
| Object set | Same objects across all tasks |
| Spatial layout | Fixed |
| Episode window | 300 steps |
| Longest training demo | 270 steps |
| Evaluation rollouts | 50 per task |
| Success criterion | Binary done flag (object reaches correct target location) |
| Initial state | Deterministic per task |
| π₀.₅ success rate | 98.0% |
| Note for patching | Language variation is procedural/relational, not nominal — causal locus likely spread across multiple tokens rather than a single object-name token. Less clean for pos131-style patching. |

### LIBERO-10 (LIBERO-Long)

| Field | Value |
|-------|-------|
| Number of tasks | 10 |
| Shared scene | Yes — 1 scene across all 10 tasks |
| Primary variation | All axes simultaneously (object + spatial + goal) |
| Episode window | 520 steps |
| Longest training demo | 505 steps |
| Relative length | ~2.6× longer than Spatial, ~1.85× longer than Object |
| Estimated sub-tasks per task | 2–4 sequential sub-tasks |
| Example task | "put the black bowl in the bottom drawer and close it" |
| Evaluation rollouts | 50 per task |
| Success criterion | Binary done flag for full multi-step sequence |
| Initial state | Deterministic per task |
| π₀.₅ success rate | 92.4% |
| Used by Kaylene et al.? | Yes — their primary sim suite for OpenVLA activation steering |

---

## Task names per suite

Task names determine video filenames (`rollout_{task_name}_{success|failure}.mp4`) and are useful for separating outputs when running multiple suites to the same directory.

### LIBERO-Spatial (10 tasks)
All tasks share the same object (`black_bowl`) — variation is purely spatial relationship in the instruction.

| Task name (= bddl filename without extension) |
|-----------------------------------------------|
| pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate |
| pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate |
| pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate |
| pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate |
| pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate |
| pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate |
| pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate |
| pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate |
| pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate |
| pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate |

**Distinguishing pattern:** all contain `black_bowl`.

### LIBERO-Object (10 tasks)
All tasks share the same procedure (`pick up X → place in basket`) — variation is object identity.

| Task name |
|-----------|
| pick_up_the_alphabet_soup_and_place_it_in_the_basket |
| pick_up_the_bbq_sauce_and_place_it_in_the_basket |
| pick_up_the_butter_and_place_it_in_the_basket |
| pick_up_the_chocolate_pudding_and_place_it_in_the_basket |
| pick_up_the_cream_cheese_and_place_it_in_the_basket |
| pick_up_the_ketchup_and_place_it_in_the_basket |
| pick_up_the_milk_and_place_it_in_the_basket |
| pick_up_the_orange_juice_and_place_it_in_the_basket |
| pick_up_the_salad_dressing_and_place_it_in_the_basket |
| pick_up_the_tomato_sauce_and_place_it_in_the_basket |

**Distinguishing pattern:** all end with `in_the_basket`.

### LIBERO-Goal (10 tasks)
Variation is goal/destination. Note: only 9 entries in bddl_files (tasks_info.txt is counted separately).

| Task name |
|-----------|
| open_the_middle_drawer_of_the_cabinet |
| open_the_top_drawer_and_put_the_bowl_inside |
| push_the_plate_to_the_front_of_the_stove |
| put_the_bowl_on_the_plate |
| put_the_bowl_on_the_stove |
| put_the_bowl_on_top_of_the_cabinet |
| put_the_cream_cheese_in_the_bowl |
| put_the_wine_bottle_on_the_rack |
| put_the_wine_bottle_on_top_of_the_cabinet |
| turn_on_the_stove |

### LIBERO-10 (10 tasks)
Multi-step tasks across multiple scenes. Names are prefixed with scene ID.

| Task name |
|-----------|
| KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it |
| KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it |
| KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it |
| KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket |
| LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket |
| LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket |
| LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate |
| LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate |
| STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy |

**Distinguishing pattern:** all prefixed with `KITCHEN_SCENE*`, `LIVING_ROOM_SCENE*`, or `STUDY_SCENE*`.

---

## Cross-suite constants

- **Physics engine:** MuJoCo
- **Setup:** Tabletop manipulation, fixed camera, parallel-jaw gripper
- **Demonstrations:** 50 human teleoperated demos per task, available via HuggingFace (`modified_libero_rlds`) in RLDS format
- **Evaluation metric:** Binary success rate (done flag), not a continuous reward
- **Rollout standard:** 50 rollouts per task across all suites (configurable via `num_trials_per_task` in `main.py`)
- **Randomization:** Initial states are deterministic (fixed), but physics simulation is seed-sensitive — simulator drops objects for ~10 steps after reset; a wait period is required before control. Seeds affect outcomes even with fixed initial states.
- **All suites:** 1 shared tabletop scene per suite (not per task — same scene for all 10 tasks within a suite)

### Evaluation metric math

The reported success rate is a single flat aggregate across all tasks and all episodes:

```
success_rate = total_successes / total_episodes
total_episodes = num_tasks × num_trials_per_task
```

For the standard full-suite evaluation:
- 10 tasks × 50 episodes = **500 total episodes**
- Each episode: binary — `done=True` (success) or `done=False` (failure), set by LIBERO's task completion check
- `main.py` accumulates `total_successes` and `total_episodes` across all tasks and logs the ratio at the end

Worked examples:
| Run | Tasks | Trials/task | Total episodes | Result | Meaning |
|-----|-------|------------|----------------|--------|---------|
| libero_object baseline (A40, seed 7) | 10 | 50 | 500 | **97.8%** | 489/500 successes |
| libero_spatial baseline (A40, seed 7) | 10 | 50 | 500 | **99.2%** | 496/500 successes |
| corrupt check, one condition | 1 (milk only) | 25 | 25 | X/25 | per-task, single-task rate |

**Key implication for the corrupt check:** with only 1 task × 25 episodes = 25 total rollouts, each success/failure is worth 4 percentage points. A result of ~40% on the clean run means ~10/25 — suspiciously low vs the 97.8% baseline and warrants investigation (is the filter working? is the prompt reaching the model correctly?).

---

## Dataset and finetuning protocol

- **Training data:** 50 human teleoperated demos per task, per suite
- **Finetuning target:** Pretrained π₀.₅ checkpoint finetuned on the suite's demo dataset
- **No train/test split:** LIBERO does not have a held-out test dataset. "Evaluation" = behavioral success rate on the same tasks used for finetuning, measured via rollouts in sim
- **Benchmark philosophy:** Originally a lifelong learning benchmark — "generalization" means knowledge transfer across tasks, not distributional generalization to unseen data

---

## Notes from Kaylene et al. (CoRL 2025) relevant to LIBERO setup

- Used **LIBERO-Long (= LIBERO-10)** for their sim experiments with OpenVLA
- Technique: activation *steering* (FFN neuron override with scalar α) — does not require clean/corrupted pairs
- Task complexity was not a concern for them because steering is a global perturbation, not a causal attribution
- GPU: H100 for sim experiments
- OpenVLA finetuned on a LIBERO-Long checkpoint (pre-existing; not trained from scratch by them)
- They showed action tokens appear in all layers (not just final), consistent with our pos131 result at layer 0

---

## Open question flagged for Step 3 (exploration)

Before committing to LIBERO-Object as the primary patching suite, run a **prompt-ablation baseline**:
- Same visual input, wrong/blank instruction → does success rate drop?
- If yes: language is load-bearing → LIBERO-Object is valid
- If no: positional shortcut present → reconsider suite choice or design a de-biased variant
