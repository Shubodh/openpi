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
