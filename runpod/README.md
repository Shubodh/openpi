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
