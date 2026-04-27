# runpod/

Bash scripts for pod lifecycle management on RunPod. Full guide → [`docs/runpod_setup.md`](../docs/runpod_setup.md) ([GitHub](https://github.com/Shubodh/openpi/blob/main/docs/runpod_setup.md)).

## Scripts

| Script | Run when | What it does |
|--------|----------|--------------|
| `setup_once.sh` | Once per network volume | Clones openpi, installs uv, creates Python 3.8 venv, installs all LIBERO deps |
| `setup_pod.sh` | Every pod restart (~1-2 min) | Reinstalls uv (wiped on stop); restores env vars — no agent install |
| `setup_agents.sh` | Once per volume, or each restart if you want agents | Installs Node.js + Claude Code + Codex CLI; prints API key and permissions setup instructions |
| `start_libero.sh [suite]` | After `setup_pod.sh` | Launches tmux session: pane 0 = π₀.₅ policy server, pane 1 = LIBERO client (waits for server) |
| `run_libero_client.sh [suite] [trials] [seed] [video_path]` | To add parallel suite runs | Runs a LIBERO client against an already-running policy server |

## Typical flow

```bash
# First time ever on a fresh network volume:
bash /workspace/openpi/runpod/setup_once.sh
bash /workspace/openpi/runpod/setup_agents.sh
bash /workspace/openpi/runpod/start_libero.sh libero_object

# Every pod restart after that:
bash /workspace/openpi/runpod/setup_agents.sh  # only if you want Claude Code / Codex
bash /workspace/openpi/runpod/setup_pod.sh
bash /workspace/openpi/runpod/start_libero.sh libero_object
```

For the agent-first workflow (recommended), API key setup, and permission configuration → see [`docs/runpod_setup.md §0`](../docs/runpod_setup.md).
