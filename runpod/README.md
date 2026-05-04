# runpod/

Bash scripts for pod lifecycle management on RunPod. Full guide → [`docs/runpod_setup.md`](../docs/runpod_setup.md) ([GitHub](https://github.com/Shubodh/openpi/blob/main/docs/runpod_setup.md)).

## Scripts

| Script | Run when | What it does |
|--------|----------|--------------|
| `setup_once.sh` | Once per network volume | Clones openpi, installs uv, creates Python 3.8 venv, installs all LIBERO deps |
| `setup_pod.sh` | Every pod restart (~15-20 min) | Reinstalls uv + recreates venv + reinstalls all deps (wiped on stop) |
| `setup_agents.sh` | In pane 1 after server is up, each restart if you want agents | Installs Node.js + Claude Code + Codex CLI |
| `start_libero.sh [suite]` | After `setup_pod.sh` | Launches tmux session: pane 0 = π₀.₅ policy server, pane 1 = clean shell with reminder |
| `libero_env.sh` | **Source** in pane 1 once server is up | Activates venv + exports env vars + prints python command to run |
| `run_libero_client.sh [suite] [trials] [seed] [video_path]` | To add parallel suite runs | Runs a LIBERO client against an already-running policy server |
| `run_object_suite_corrupt_check.sh` | After `source libero_env.sh`, to run the LIBERO-object prompt-ablation experiment | Clean run + corrupt run on milk task (25 trials each); logs tee'd to `scripts_outputs_txt/` |
| `run_goal_suite_corrupt_check.sh` | After `source libero_env.sh`, to run the LIBERO-goal prompt-ablation experiment | Clean run + corrupt run on bowl/cabinet task (25 trials each); logs tee'd to `scripts_outputs_txt/` |
| `run_kv_cache_inspect.sh` | After `source libero_env.sh`, to verify KV-cache tokenizer output (no model weights needed) | Tokenizes the two contrastive LIBERO-Goal prompts, prints token IDs + absolute prefix indices; logs tee'd to `scripts_outputs_txt/kv_cache_inspect/` |
| `run_patching_phase1_verify.sh` | After `source libero_env.sh`, before any patching runs | Tokenizes Phase 1 prompt pair (plate/stove), confirms `plate` and `stove` are both at absolute index 594; no GPU needed |
| `run_patching_phase1_baselines.sh` | After `source libero_env.sh` **with server running**, to establish Phase 1 baselines | Clean + corrupt runs on the plate/stove task (25 trials each); logs tee'd to `scripts_outputs_txt/patching_phase1/` |
| `patching_env.sh` | **Source** before `run_patching_phase1.sh` | Activates server venv (Python 3.11 + JAX), sets MUJOCO_GL + PYTHONPATH + OPENPI_DATA_HOME |
| `run_patching_phase1.sh` | After `source patching_env.sh`, **NO server needed** — model loads in-process | Sanity check (N=5, all-position patch) then main patched run (N=25, pos 594); logs tee'd to `scripts_outputs_txt/patching_phase1/` |

## Typical flow

```bash
# First time ever on a fresh network volume:
bash /workspace/openpi/runpod/setup_once.sh
bash /workspace/openpi/runpod/start_libero.sh
# [pane 0] wait for "listening on :8000", then in pane 1:
bash /workspace/openpi/runpod/setup_agents.sh   # Claude Code + Codex (optional)
source /workspace/openpi/runpod/libero_env.sh
python examples/libero/main_original.py --args.task-suite-name libero_object --args.video-out-path data/libero/videos/libero_object

# Every pod restart after that (most common):
bash /workspace/openpi/runpod/setup_pod.sh      # ~15-20 min
bash /workspace/openpi/runpod/start_libero.sh
# [pane 0] wait for "listening on :8000", then in pane 1:
bash /workspace/openpi/runpod/setup_agents.sh   # Claude Code + Codex (optional)
source /workspace/openpi/runpod/libero_env.sh
# Then run whatever experiment you need, e.g.:
bash /workspace/openpi/runpod/run_goal_suite_corrupt_check.sh
```

For the agent-first workflow (recommended), API key setup, and permission configuration → see [`docs/runpod_setup.md §0`](../docs/runpod_setup.md).

## When to run which experiment scripts

### Prompt-ablation / corrupt-run checks

Run these to verify that the language prompt is load-bearing (i.e. swapping the prompt degrades success rate).
Requires: policy server running in pane 0, venv active in pane 1.

```bash
# LIBERO-Goal bowl/cabinet task (25 trials × 2 conditions):
bash /workspace/openpi/runpod/run_goal_suite_corrupt_check.sh
# LIBERO-Object milk task (25 trials × 2 conditions):
bash /workspace/openpi/runpod/run_object_suite_corrupt_check.sh
```

Logs → `scripts_outputs_txt/goal_suite_check_*.txt` / `corrupt_check_*.txt`.
Results should be recorded in `status_cc/corrupt_run_experiment.md`.

### KV-cache patching — Phase 1 (plate vs stove)

**Important:** `run_patching_phase1.sh` loads the model in-process — do NOT start the policy server before running it. The baseline scripts still need the server.

**Recommended order:**

```bash
# Step 1 — verify token positions (fast, no GPU, no server):
source /workspace/openpi/runpod/libero_env.sh
bash /workspace/openpi/runpod/run_patching_phase1_verify.sh
# Check log: 'plate' and 'stove' should both appear at absolute index 594.
# If not, update --args.patch-positions in run_patching_phase1.sh before continuing.

# Step 2 — baselines (needs server in pane 0):
bash /workspace/openpi/runpod/start_libero.sh
# [pane 0] wait for "listening on :8000", then in pane 1:
source /workspace/openpi/runpod/libero_env.sh
bash /workspace/openpi/runpod/run_patching_phase1_baselines.sh
# Record D1 (clean) and D2 (corrupt) success rates in status_cc/patching_implementation.md §7.1.

# Step 3 — patched run (NO server needed; stop or ignore the server from step 2):
source /workspace/openpi/runpod/patching_env.sh   # server venv, not libero_env.sh
bash /workspace/openpi/runpod/run_patching_phase1.sh
# Sanity check runs first (N=5, all-position patch) — verify it succeeds before the main run.
# Record C3 (sanity) and D3 (patched pos 594) in status_cc/patching_implementation.md §7.1.
```

Logs → `scripts_outputs_txt/patching_phase1/verify/verify_*.txt`, `baselines/baselines_*.txt`, `patched/run_*.txt`.

**Reading the results:** The key question is whether `patched_success_rate` recovers toward `clean_success_rate`. Update `status_cc/patching_implementation.md §7.1` with all four rows (C3 sanity + D1 clean + D2 corrupt + D3 patched).

### KV-cache tokenizer inspection (no model weights needed)

Run this to verify the exact token IDs and absolute prefix indices for the object-name words
(`bowl`, `wine`, `bottle`) in the π₀.₅ KV-cache prefix. Only the tokenizer is loaded — fast, no GPU needed.

```bash
source /workspace/openpi/runpod/libero_env.sh   # activate venv (policy server does NOT need to be up)
bash /workspace/openpi/runpod/run_kv_cache_inspect.sh
```

Logs → `scripts_outputs_txt/kv_cache_inspect/inspect_*.txt`.
Paste the output into `status_cc/kv_cache_findings.md` Section 3 to complete Phase 1 of the KV-cache sanity check.

### Downloading videos to your local machine

Run from your **local terminal** (not the pod). The pod is behind NAT — you must pull, not push.
Get SSH details from: RunPod UI → pod → "Connect" button.

```bash
# rsync (preferred — resumes interrupted transfers):
rsync -avz --progress \
  -e "ssh -p <PORT> -i ~/.ssh/id_ed25519" \
  root@<IP>:/workspace/openpi/data/libero/videos/ \
  /media/shubodh/mydisk/shubodh/Downloads/data-non-onedrive/actionXMech_project/openpi/data/libero/videos/

# scp alternative:
scp -r -P <PORT> -i ~/.ssh/id_ed25519 \
  root@<IP>:/workspace/openpi/data/libero/videos/ \
  ~/Downloads/libero_videos/
```

Full details (video filenames, separating mixed-suite videos) → [`docs/runpod_setup.md`](../docs/runpod_setup.md).

---

## Appendix: Two venvs and how `uv run` picks one

There are two virtual environments in this repo — not one venv and a system Python:

| | Location | Python | Contains |
|---|---|---|---|
| **Server venv** | `/workspace/openpi/.venv` | 3.11 | JAX, openpi, flax, orbax |
| **LIBERO client venv** | `/workspace/openpi/examples/libero/.venv` | 3.8 | LIBERO, robosuite, torch, openpi-client |

**`uv run python` ignores the activated venv.** It looks for the project's `pyproject.toml` at `/workspace/openpi/` and always uses the associated venv (`.venv` at the repo root — the server venv, Python 3.11). This is true even if `source libero_env.sh` has been run and the LIBERO venv is currently active.

**Plain `python` follows the shell's `$PATH`.** After `source libero_env.sh`, `python` points to the Python 3.8 LIBERO venv.

**Why this matters for `run_patching_phase1.sh`:**

```bash
source libero_env.sh       # activates LIBERO venv; also sets MUJOCO_GL=egl and PYTHONPATH
python ...                 # ← Python 3.8, LIBERO venv — no JAX, would fail
uv run python ...          # ← Python 3.11, server venv — has JAX + LIBERO sim deps
```

Shell-level exports (`MUJOCO_GL`, `PYTHONPATH`, `OPENPI_DATA_HOME`) are inherited by `uv run python` since they live in the shell environment, not in a venv. That is why `source libero_env.sh` is still required before running `run_patching_phase1.sh` even though the venv it activates is overridden by `uv run`.

**LIBERO simulation deps in the server venv:** `setup_pod.sh` (and `setup_once.sh`) install mujoco, imageio, etc. into the server venv, then copy robosuite directly from the LIBERO client venv. Torch is not included — not needed for env stepping.

**Why copy robosuite instead of pip-installing it?** `pip install robosuite==1.4.1` resolves to version 1.5.x (incompatible module layout — missing `single_arm_env`), even with the exact pin. Installing from the git tag `v1.4.1` has the same problem. The only reliable source of the correct robosuite is the LIBERO client venv, which already has exactly the right version. The setup scripts do `rm -rf` then `cp -r` (not just `cp -r`, which would nest the directory inside the existing one rather than replacing it).

---

## Appendix: Server venv setup for patching script — lessons learned (2026-05-04)

This documents what was learned getting `main_patching_expt.py` to run in the server venv. The setup scripts reflect the final working approach, but this log explains why.

### The problem

`main_patching_expt.py` needs JAX (server venv, Python 3.11) AND LIBERO simulation (originally Python 3.8 client venv). These can't share a Python version — openpi requires ≥3.11, LIBERO's pip-locked deps target 3.8.

Solution: install LIBERO simulation deps into the server venv, carefully.

### Pitfalls encountered

**1. `pip install` vs `uv pip install`**
The server venv is created by uv and has no `pip` binary in `.venv/bin/`. Plain `pip` after `source activate` resolves to the system pip (`/usr/local/...`), which is invisible to the server venv. Always use `uv pip install` for the server venv. After this, the `which pip` check is unreliable — use `uv pip install` unconditionally.

**2. `pip install robosuite==1.4.1` installs 1.5.x**
Despite the exact pin, PyPI's robosuite 1.4.1 resolves to a 1.5.x build with a different module layout (no `single_arm_env`). The git tag `v1.4.1` also doesn't match. The only fix: copy directly from the LIBERO client venv, which has the correct version installed. Must `rm -rf` the destination first — `cp -r src dst` when `dst` exists nests `src` inside `dst` rather than replacing it.

**3. `uv pip install -e third_party/libero` misses most LIBERO deps**
Editable install only picks up what's declared in `setup.py` install_requires. LIBERO's `requirements.txt` has additional packages (`bddl`, `easydict`, `gym`, `hydra-core`, etc.) not in setup.py. Must also install from `requirements.txt`.

**4. `requirements.txt` lines have leading spaces — grep `^pkg` patterns fail**
The file looks like ` robosuite` and ` transformers==4.21.1` with a leading space. The pattern `^robosuite` never matches. Use `^\s*robosuite` or the `^\s*(pkg1|pkg2|...)` form.

**5. `transformers==4.21.1` drags in `tokenizers==0.12.1` which requires Rust**
Both are training-only. Exclude `transformers` (and by extension tokenizers) from the requirements install.

**6. LIBERO's numpy pin (1.22.4) breaks JAX**
Installing from requirements.txt downgrades numpy from 1.26.x → 1.22.4. JAX 0.5.3 requires `np.dtypes` (added in numpy 1.25). After the requirements.txt install, restore numpy:
```bash
uv pip install "numpy>=1.22.4,<2.0.0"
```

### The working manual setup sequence (as of 2026-05-04)

Run with the server venv implicit (from `/workspace/openpi`):

```bash
cd /workspace/openpi
uv sync
uv pip install "mujoco>=3.2" imageio imageio-ffmpeg "opencv-python>=4.6" scipy tqdm pyyaml pyopengl etils tyro
uv pip install -e /workspace/openpi/packages/openpi-client
uv pip install -e /workspace/openpi/third_party/libero

# Install LIBERO requirements (excluding training-only and incompatible packages)
grep -viE "^\s*(robosuite|torch|wandb|transformers|thop|robomimic|numpy)" \
  /workspace/openpi/third_party/libero/requirements.txt | uv pip install -r /dev/stdin

# Restore numpy (LIBERO pin downgrades it, breaking JAX)
uv pip install "numpy>=1.22.4,<2.0.0"

# Copy robosuite from LIBERO client venv (pip version is wrong)
SERVER_SITE=$(/workspace/openpi/.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
rm -rf "${SERVER_SITE}/robosuite"
cp -r /workspace/openpi/examples/libero/.venv/lib/python3.8/site-packages/robosuite "${SERVER_SITE}/robosuite"
```

Then run the patching script:
```bash
source /workspace/openpi/runpod/patching_env.sh
bash /workspace/openpi/runpod/run_patching_phase1.sh
```

**Status as of 2026-05-04:** numpy downgrade fixed, robosuite working. Script has not yet completed a full run — further import errors possible. Update this section as new blockers are resolved.
