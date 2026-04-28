# openpi Scripts Primer — Corrupt Run Experiment Context

This document is a prereq reading for implementing and running the corrupt-run experiment (see `status_cc/corrupt_run_experiment.md`). It explains what each relevant script does, how they connect, and—at the end—evaluates whether the proposed two-flag implementation is the right approach.

---

## System Architecture: Two Processes

The entire evaluation system is a **client–server split**:

```
┌─────────────────────────────────────┐     WebSocket     ┌─────────────────────────────┐
│   LIBERO client (Python 3.8 venv)   │ ────────────────> │  π₀.₅ policy server (uv)    │
│   examples/libero/main.py           │  obs dict (msgpack)│  scripts/serve_policy.py    │
│                                     │ <──────────────── │                             │
│   - runs MuJoCo sim                 │  action chunk      │  - loads checkpoint         │
│   - sends observations each step    │                    │  - runs transformer forward │
│   - executes returned actions       │                    │  - returns 7-dim actions    │
└─────────────────────────────────────┘                    └─────────────────────────────┘
```

The **prompt (language instruction) lives entirely on the client side** and is sent as part of every observation dict. The server never touches the LIBERO environment or task descriptions — it just tokenizes whatever string is in `"prompt"` and feeds it to the model.

---

## File-by-File Breakdown

### `examples/libero/main.py` — The Evaluation Client

This is the only file that needs to change for the corrupt-run experiment.

**`Args` dataclass** (`main.py:22-46`):
All CLI flags live here, parsed by `tyro`. Current flags:
- `host`, `port` — where the policy server is
- `resize_size` (224) — image resize for model input
- `replan_steps` (5) — how many actions from each chunk to execute before querying again
- `task_suite_name` — which LIBERO suite to run (e.g., `libero_object`)
- `num_steps_wait` (10) — simulator warm-up steps before acting (objects drop and settle)
- `num_trials_per_task` (50) — rollouts per task
- `video_out_path` — where episode MP4s go
- `seed` (7) — random seed for reproducibility

`tyro` automatically converts Python field names (`task_suite_name`) to CLI flags (`--args.task-suite-name`).

**`eval_libero(args)` function** (`main.py:48-185`):
The main evaluation loop:
1. Loads the task suite via LIBERO's benchmark API (`main.py:53-54`)
2. For each task in the suite, calls `_get_libero_env` to get the MuJoCo env and `task_description` (`main.py:85`)
3. Loops over `num_trials_per_task` episodes; within each episode, loops over timesteps
4. Every `replan_steps` steps, builds an observation dict and calls `client.infer(element)` (`main.py:130-148`)
5. The observation dict structure (`main.py:130-141`):
   ```python
   element = {
       "observation/image":       img,          # 224×224×3 uint8, agentview (rotated 180°)
       "observation/wrist_image": wrist_img,     # 224×224×3 uint8, wrist cam (rotated 180°)
       "observation/state":       np.concatenate([eef_pos, axisangle, gripper_qpos]),  # 8-dim
       "prompt":                  str(task_description),   # ← THE LANGUAGE INSTRUCTION
   }
   ```
6. Saves an MP4 per episode (`main.py:168-174`). The filename is `rollout_{task_description}_{success|failure}.mp4`.

**`_get_libero_env(task, resolution, seed)`** (`main.py:189-196`):
Creates the MuJoCo environment from the task's `.bddl` file and returns `(env, task_description)`. The `task_description` comes from `task.language` — a string attribute on the LIBERO task object (e.g., `"Pick up the milk and place it in the basket"`). This is the only place `task_description` is set; it is never updated after this.

**`_quat2axisangle`** (`main.py:199-214`):
Utility — converts quaternion to axis-angle for the robot state. Not relevant to our changes.

**Entry point** (`main.py:217-219`):
```python
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
```
`tyro.cli` wraps `eval_libero` — it reads `Args`, parses CLI flags into it, and calls `eval_libero(args)`. This is why new fields added to `Args` automatically become CLI flags.

---

### `src/openpi/policies/libero_policy.py` — Server-Side Input Transform

**`LiberoInputs.__call__`** (`libero_policy.py:42-83`):
The server-side transform applied to incoming observations. It reads `data["prompt"]` and passes it to the model as the language condition (`libero_policy.py:80-81`):
```python
if "prompt" in data:
    inputs["prompt"] = data["prompt"]
```
The server does zero special handling of the prompt string — it takes whatever string the client sends. This confirms that overriding `"prompt"` in the client's observation dict is sufficient to change what language the model conditions on.

---

### `scripts/serve_policy.py` — The Policy Server

Starts the WebSocket server that loads the π₀.₅ checkpoint and serves action predictions.

**`Args.default_prompt`** (`serve_policy.py:45-47`):
```python
default_prompt: str | None = None
```
This is a server-side fallback: if the client sends an observation with no `"prompt"` key, the server uses this default. Since `main.py` always sends `"prompt"`, this flag is irrelevant for our experiment. **It does NOT let you override the per-task prompt from the server side.**

Launched via: `uv run scripts/serve_policy.py --env LIBERO`

---

### `packages/openpi-client/src/openpi_client/websocket_client_policy.py` — Client Transport

**`WebsocketClientPolicy.infer(obs)`** (`websocket_client_policy.py:47-54`):
Serializes the observation dict to msgpack, sends it over the WebSocket, and deserializes the response. It is a transparent pipe — whatever dict the client builds is what the server receives. No prompt modification happens here.

---

### `runpod/` — Pod Lifecycle Scripts

These handle infrastructure, not experiment logic.

| Script | What it does |
|--------|-------------|
| `setup_pod.sh` | Run after every pod restart: installs system packages (tmux, cmake, egl), reinstalls uv (wiped on stop), exports `OPENPI_DATA_HOME`, recreates the Python 3.8 venv and installs all deps |
| `start_libero.sh [suite]` | Opens a tmux session: pane 0 launches the policy server (`serve_policy.py`), pane 1 opens a clean shell with a reminder to source `libero_env.sh` once server is up |
| `libero_env.sh` | **Source** (not run) this to activate the venv, set `PYTHONPATH` and `MUJOCO_GL=egl`, and print the baseline python command |
| `run_libero_client.sh [suite] [trials] [seed] [video_path]` | Convenience wrapper that activates the venv and calls `main.py` with positional args. Used for adding parallel suite runs. |

The typical pod workflow is: `setup_pod.sh` → `start_libero.sh` → (server is up) → `source libero_env.sh` → `python examples/libero/main.py ...`

---

## Data Flow Summary (for one timestep)

```
MuJoCo sim
  └─ obs["agentview_image"]          ──rotate 180°──> img  ─────────────────────┐
  └─ obs["robot0_eye_in_hand_image"] ──rotate 180°──> wrist_img ────────────────┤
  └─ obs["robot0_eef_pos/quat/..."]  ──concatenate──> state (8-dim) ────────────┤
                                                                                 │
task.language ──────────────────────────────────────────────────────────────────>├── element dict
                                                                                 │
                                          client.infer(element) ─────────────────┘
                                                │
                                         WebSocket (msgpack)
                                                │
                                       serve_policy.py
                                                │
                                       LiberoInputs.__call__()
                                                │
                                       π₀.₅ transformer
                                                │
                                       LiberoOutputs.__call__()
                                                │
                                       action_chunk (shape: [chunk_len, 7])
                                                │
                                    first replan_steps actions ──> MuJoCo env.step()
```

The prompt string (`task.language`) enters the dict once and gets tokenized by the server's transformer backbone. Changing that string is all that's required to corrupt the language signal.

---

## Evaluation: Is the Proposed Two-Flag Implementation Correct?

### Is there an existing argument that already does what we need?

**No.** Looking at the current `Args` dataclass (`main.py:22-46`):
- There is no filter flag — `eval_libero` always iterates `range(num_tasks_in_suite)`, running all 10 tasks in `libero_object`.
- There is no prompt-override flag — the prompt is always `str(task_description)` from `task.language` (`main.py:140`).
- `serve_policy.py --default-prompt` is server-side and only applies when the client sends no `"prompt"` key at all — it cannot override a per-task prompt.

### Is the proposed implementation correct?

**Yes, and it's the minimal correct approach.** The two flags are:

**Flag 1 — `task_name_filter`:**
Without this, running `--args.num-trials-per-task 25` on `libero_object` would run 25 trials on *each* of the 10 tasks (250 total). We only want the milk task. The filter needs to be applied before the episode loop, by checking whether `task_description` contains the filter string. The right insertion point is after `_get_libero_env` returns (`main.py:85`) and before the episode loop begins (`main.py:89`).

**Flag 2 — `corrupt_prompt`:**
The prompt is set exactly once per task, at `main.py:140`: `"prompt": str(task_description)`. Replacing this with `args.corrupt_prompt if args.corrupt_prompt else str(task_description)` is sufficient. No other code path touches the prompt — `_get_libero_env` returns it, but nothing downstream re-reads `task_description` after the obs dict is built.

**One implementation note:** The video filename at `main.py:169-172` uses `task_description` (the *original* task language, not the corrupt prompt) as the segment name. This is fine — it identifies the scene, not the condition. But to distinguish clean vs. corrupt videos at a glance, a `_corrupt` suffix on the `video_out_path` (or a flag-derived subdirectory) is the right approach, which is exactly what the experiment brief proposes (separate `--args.video-out-path` values per run).

**Python 3.8 compatibility:** The client venv runs Python 3.8. `str | None` syntax is not available; `Optional[str]` from `typing` must be used for both new fields. `tyro` handles `Optional[str]` correctly for CLI parsing.

### Summary verdict

The proposed two-flag approach is correct, minimal, and hits the right insertion points. No existing mechanism covers either need. The implementation changes are entirely local to `main.py`'s `Args` dataclass and the task loop in `eval_libero`.
