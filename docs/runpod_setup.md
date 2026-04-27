# RunPod Setup Guide — LIBERO & VLA Models

## Table of Contents

0. [Agent-First Setup (Recommended)](#0-agent-first-setup-recommended)
1. [Quick Reference](#1-quick-reference)
2. [Pod Configuration](#2-pod-configuration)
3. [π₀.₅ + LIBERO Setup (openpi) — Current Stack](#3-π05--libero-setup-openpi--current-stack)
4. [LeRobot + SmolVLA Setup](#4-lerobot--smolvla-setup)
5. [Auto-stop / Sleep Workflow](#5-auto-stop--sleep-workflow)
6. [Supplementary](#6-supplementary)
   - 6.1 [GPU Decision Rationale (A40)](#61-gpu-decision-rationale-a40)
   - 6.2 [openpi With Docker (attempted, does not work on RunPod)](#62-openpi-with-docker-attempted-does-not-work-on-runpod)
   - 6.3 [FFmpeg Install Script](#63-ffmpeg-install-script)
   - 6.4 [LeRobot on RTX 5090 (special handling)](#64-lerobot-on-rtx-5090-special-handling)
7. [Legacy / Archived](#7-legacy--archived)
   - 7.1 [OpenVLA + LIBERO Setup (superseded)](#71-openvla--libero-setup-superseded)

---

## 0. Agent-First Setup (Recommended)

Instead of manually running setup commands (~2 hrs first time, ~1 min on restarts), hand the whole thing to an AI agent.

### Step 1 — Set API keys (do this before starting the pod)

**Primary (recommended):** RunPod UI → pod config → "Environment Variables" → add before starting the pod:
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```
These inject automatically on pod start — no typing needed in the terminal.

**Fallback:** paste manually after SSH-ing in:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

### Step 2 — Install Claude Code (30 seconds, one command)

```bash
curl -fsSL https://claude.ai/install.sh | bash && source ~/.bashrc
```

### Step 2b — Install Codex CLI (optional, if you want the OpenAI agent)

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs
npm install -g @openai/codex
```

Or use the script: `bash /workspace/openpi/runpod/setup_agents.sh` (installs both).

### Step 3 — Launch the agent

**First-time (fresh network volume — openpi not yet cloned):**
```bash
cd /workspace
git clone https://github.com/Shubodh/openpi.git
cd openpi
claude --dangerously-skip-permissions
# Prompt: "Run runpod/setup_once.sh to set up the LIBERO environment"
```

**Every pod restart (network volume already has openpi + venv):**
```bash
cd /workspace/openpi
claude --dangerously-skip-permissions
# Prompt: "Run runpod/setup_pod.sh then start_libero.sh for suite libero_object"
```

The agent handles all installs, waits for the server to come up, and launches the tmux session. You can detach (`Ctrl+B D`) and walk away.

### Scripts reference (all in `/workspace/openpi/runpod/` on the pod)

| Script | When | What |
|--------|------|------|
| `setup_once.sh` | Once per network volume | Clone openpi, create venv, all pip installs |
| `setup_pod.sh` | Every pod restart (~1-2 min) | uv + Claude Code + Codex reinstall + env vars |
| `setup_agents.sh` | Once per volume | Node.js + Claude Code + Codex CLI |
| `start_libero.sh [suite]` | After setup_pod.sh | tmux: pane 0 = server, pane 1 = client |
| `run_libero_client.sh [suite] [trials] [seed]` | Add parallel suite runs | Client against running server |

---

## 1. Quick Reference

| What | Detail |
|------|--------|
| GPU (π₀.₅ + LIBERO inference) | **A40 — locked** ($0.44/hr, 48GB VRAM, ~$10.56/day) — [rationale §6.1](#61-gpu-decision-rationale-a40) |
| GPU (LeRobot/SmolVLA, future) | RTX 3090 → 4090 → A100 (upgrade if slow) |
| Network volume | 100 GB at `/workspace` (~$0.005/hr, ~$7/mo) |
| Container disk | Temporary — erased on pod stop |
| Volume disk | Persistent — erased only on pod **terminate** |
| PyTorch template (openpi/LIBERO) | Any recent template works — `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` confirmed fine |
| PyTorch template (LeRobot) | `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04` (tested) |
| openpi setup | **Without Docker** — two tmux panes (server + client). Docker does not work on RunPod; see §6.2. |
| MuJoCo headless | `export MUJOCO_GL=egl` (fallback: `MUJOCO_GL=glx`) |
| Checkpoint source (π₀.₅) | Auto-downloaded from GCS on first run — set `OPENPI_DATA_HOME` to network volume first |

> **Stopped vs Terminated:** Stopped = paused, volume survives. Terminated = deleted, volume gone.

---

## 2. Pod Configuration

- **GPU: A40** — when creating the network volume, filter by A40 to see the 2 available data centers.
- **Network volume: 100 GB** — always allocate this. Mounted at `/workspace`. Running out mid-run wastes the whole session.
- **First install on any fresh pod:**

```bash
apt-get update && apt-get install -y tmux vim
# set -w mode-keys vi   ← add to ~/.tmux.conf for vim keybindings in tmux
```

---

## 3. π₀.₅ + LIBERO Setup (openpi) — Current Stack

> **Why without Docker:** openpi recommends Docker, but RunPod pods are themselves Docker containers. Running Docker inside them (DinD) requires `--privileged` on the outer container, which RunPod does not grant. We hit cascading failures: iptables blocked, overlayfs mounts blocked, namespace creation blocked. Docker is not viable on RunPod without a privileged pod. The without-Docker path runs server and client directly on the host and works fine — openpi calls it "not recommended" only because it requires more manual setup, not because it has functional limitations.

### Automated scripts (recommended)

> **Better yet:** use the agent-first approach in §0 — let Claude Code run these scripts for you.

Scripts live at `runpod/` in this repo (i.e., `/workspace/openpi/runpod/` on the pod). They are available immediately after `git clone`.

| Script | When to run | What it does |
|--------|-------------|--------------|
| `setup_once.sh` | **Once ever** per network volume | Clone, submodules, uv, venv, all pip installs |
| `setup_pod.sh` | **Every pod restart** (~1-2 min) | Reinstall uv + agents + restore env vars (venv + checkpoint persist) |
| `start_libero.sh [suite]` | After `setup_pod.sh` | tmux session: pane 0 = server, pane 1 = client (auto-waits for server) |
| `run_libero_client.sh [suite] [trials] [seed] [video_path]` | To add parallel suite runs | Runs a client against the already-running server |

```bash
# First time ever:
bash /workspace/openpi/runpod/setup_once.sh
bash /workspace/openpi/runpod/setup_agents.sh   # install Claude Code + Codex
bash /workspace/openpi/runpod/start_libero.sh

# Every restart after that:
bash /workspace/openpi/runpod/setup_pod.sh
bash /workspace/openpi/runpod/start_libero.sh

# Different suite:
bash /workspace/openpi/runpod/start_libero.sh libero_spatial
```

To reattach to the session after detaching: `tmux attach -t libero`

---

### Manual steps (reference — what the scripts do)

**Step 1 — Clone and init submodules**

Clone to `/workspace` (network volume — persistent). Home (`/root`) is container disk, wiped on pod stop.

```bash
cd /workspace
git clone https://github.com/Shubodh/openpi.git
cd openpi
git submodule update --init --recursive  # required — libero submodule is empty without this
```

> **Note — LFS:** The openpi README recommends `GIT_LFS_SKIP_SMUDGE=1` before cloning because LeRobot (a dependency) has LFS files. For the π₀.₅ + LIBERO setup, LeRobot LFS files are not used — our client env only installs `examples/libero/requirements.txt` + `third_party/libero` + `openpi-client`. If you ever see LFS-related errors during pip installs, this is the likely cause.

**Step 1b — Install EGL library (required for headless rendering)**

```bash
apt-get install -y libegl1-mesa
```

Without this, `MUJOCO_GL=egl` fails with `AttributeError: 'NoneType' object has no attribute 'eglQueryString'`.

**Step 2 — Install uv**

openpi uses `uv` for package management, not pip/conda.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

**Step 3 — Point checkpoint cache to network volume**

Checkpoint is auto-downloaded from GCS on first server start. Default cache is `~/.cache/openpi` (container disk — ephemeral). Redirect to network volume so it persists:

```bash
export OPENPI_DATA_HOME=/workspace/openpi_assets
mkdir -p /workspace/openpi_assets
echo 'export OPENPI_DATA_HOME=/workspace/openpi_assets' >> ~/.bashrc
```

**Step 4 — Run server and client (two tmux panes)**

Open two tmux panes. First run will download the checkpoint from GCS — takes a while.

```bash
# Pane 1: Policy server
cd /workspace/openpi
uv run scripts/serve_policy.py --env LIBERO

# Pane 2: LIBERO client (default suite: libero_object)
cd /workspace/openpi
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=egl   # headless rendering; fallback: MUJOCO_GL=glx

# Pause now till Pane 1 server is up: Would say something like "Server listening on port 8000" when ready.
python examples/libero/main.py

# To run a different suite:
python examples/libero/main.py --args.task-suite-name libero_spatial
```

**Step 5 — Running multiple suites in parallel**

The policy server is stateless per request — multiple clients can connect simultaneously. To run a second suite while one is already going, open a new tmux pane (no reinstall needed, venv already exists):

```bash
# New tmux pane:
cd /workspace/openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=egl
python examples/libero/main.py --args.task-suite-name libero_object
```

> **Default suite is `libero_spatial`**, not libero_object. Always pass `--args.task-suite-name` explicitly. Our primary suite for patching experiments is `libero_object`.

Videos land in `data/libero/videos/` (relative to where you ran the command, i.e. `/workspace/openpi/data/libero/videos/`) — one `.mp4` per episode, auto-saved. Video filenames are `rollout_{task_name}_{success|failure}.mp4`. Task names and their distinguishing patterns per suite → `docs/` section "Task names per suite" in `libero_suite_choice_detailed.md` (in AXMech_meta).

**Separating mixed-suite videos** (if two suites wrote to the same dir):
```bash
# spatial: all filenames contain black_bowl
mkdir -p data/libero/videos/libero_spatial
mv data/libero/videos/*black_bowl*.mp4 data/libero/videos/libero_spatial/

# object: all filenames contain in_the_basket
mkdir -p data/libero/videos/libero_object
mv data/libero/videos/*in_the_basket*.mp4 data/libero/videos/libero_object/
```

**Downloading videos to your local machine** — run this from your **local terminal** (not the pod). The pod can't push to your local machine (it's behind NAT); you must pull from local:
```bash
# Get SSH details: RunPod UI → pod → "Connect" button
rsync -avz --progress \
  -e "ssh -p <PORT> -i ~/.ssh/id_ed25519" \
  root@<IP>:/workspace/openpi/data/libero/videos/ \
  ~/Downloads/libero_videos/

# Or with scp:
scp -r -P <PORT> -i ~/.ssh/id_ed25519 \
  root@<IP>:/workspace/openpi/data/libero/videos/ \
  ~/Downloads/libero_videos/
```

**`main.py` arguments reference:**

| Argument | Default | Notes |
|----------|---------|-------|
| `--args.task-suite-name` | `libero_spatial` | Options: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90` |
| `--args.num-trials-per-task` | `50` | Rollouts per task — 50 matches published protocol |
| `--args.seed` | `7` | Random seed; affects object positions even with fixed initial states |
| `--args.video-out-path` | `data/libero/videos` | Relative to `/workspace/openpi/` |
| `--args.host` | `0.0.0.0` | Policy server host |
| `--args.port` | `8000` | Policy server port |
| `--args.replan-steps` | `5` | How often to replan (every N steps) |

Use `run_libero_client.sh` (in `runpod/`) to run a client with these args conveniently:

```bash
bash /workspace/openpi/runpod/run_libero_client.sh                          # libero_object, 50 trials
bash /workspace/openpi/runpod/run_libero_client.sh libero_spatial           # spatial, 50 trials
bash /workspace/openpi/runpod/run_libero_client.sh libero_object 10         # quick 10-trial check
bash /workspace/openpi/runpod/run_libero_client.sh libero_object 50 42      # different seed
```

**Step 7 — Baseline verification**

Confirm π₀.₅ achieves ~98% on libero_object before running any experiments. Run 50 rollouts per task to match the published protocol.

**Our numbers (A40, seed 7):** libero_spatial **99.2%** (496/500), libero_object **97.8%** (489/500). Published numbers: libero_spatial 98.8%, libero_object 98.2%, libero_goal 98.0%, libero_10 92.4%.

> **EGL cleanup errors at exit** — harmless. After all episodes complete, Python prints `EGLError: EGL_NOT_INITIALIZED` during garbage collection (`__del__`). This is the EGL context being destroyed after the display is already torn down. Does not affect results. Fix if annoying: `apt-get install -y libglu1-mesa`.

**TODO — fill in as Step 2 progresses:**
- [x] Confirm uv install works on RunPod PyTorch template ✓
- [ ] Confirm OPENPI_DATA_HOME redirect works — checkpoint lands at /workspace/openpi_assets
- [ ] Confirm checkpoint download size and time on first run
- [x] Confirm MUJOCO_GL=egl works headless — **requires `apt-get install -y libegl1-mesa`** first; without it: `AttributeError: 'NoneType' object has no attribute 'eglQueryString'` ✓
- [x] Baseline eval numbers confirmed on our server — **libero_spatial: 99.2% (496/500)**, better than published 98.8% ✓
- [x] libero_object baseline confirmed on our server — **97.8% (489/500)**, vs published 98.2% ✓

---

## 4. LeRobot + SmolVLA Setup

**Step 1 — FFmpeg**

```bash
apt-get update && apt-get install -y software-properties-common
add-apt-repository -y ppa:ubuntuhandbook1/ffmpeg7
apt-get update && apt-get install -y ffmpeg git-lfs
git lfs install
```

Full install script → [§6.3](#63-ffmpeg-install-script)

For RTX 5090 → [§6.4](#64-lerobot-on-rtx-5090-special-handling) (do before pod creation)

**Step 2 — LeRobot + SmolVLA**

```bash
git clone https://github.com/shubodh/lerobot.git
cd lerobot
pip install -e ".[smolvla]"
```

**Step 3 — Credentials**

```bash
git config --global credential.helper store
hf auth login   # say 'y' to add token as git credential
pip install wandb weave && wandb login
```

**Next:** Training — see SO-101 training and inference notes.

---

## 5. Auto-stop / Sleep Workflow

```bash
# Stop when done (pod paused, volume survives):
python train.py && runpodctl stop pod $RUNPOD_POD_ID

# Terminate when done (pod + volume deleted — use carefully):
python train.py && runpodctl remove pod $RUNPOD_POD_ID
```

Use **stop** for overnight runs. Use **terminate** only when fully done.

---

## 6. Supplementary

### 6.1 GPU Decision Rationale (A40)

Locked 2026-04-26. RTX 3090 became unavailable; evaluated alternatives:

| GPU | Price | VRAM | GPU avail. | Network vol. data centers | Full day (~24hr) |
|-----|-------|------|------------|---------------------------|------------------|
| **A40** | **$0.44/hr** | **48GB** | **Med** | **2 (2nd highest overall)** | **~$10.56** |
| RTX 4090 | $0.69/hr | 24GB | Med | High | ~$16.56 |
| RTX 5090 | $0.99/hr | 32GB | High | High | ~$23.76 |
| RTX 3090 | $0.46/hr | 24GB | Unavailable | 1 data center only | — |

A40 wins: cheapest viable option (~$6/day cheaper than 4090), most VRAM (48GB vs 24GB — useful for holding multiple activation caches during patching), and 2nd highest network volume data center availability on RunPod. openpi inference only needs >8GB so we have ample headroom.

### 6.2 openpi With Docker (attempted, does not work on RunPod)

> **Do not attempt.** Documented here so we don't re-investigate next time.

RunPod pods are themselves Docker containers. Running Docker inside them (Docker-in-Docker / DinD) requires `--privileged` on the outer container, which RunPod does not grant. We hit three cascading failures even with workaround flags:
1. `--iptables=false` needed because iptables/nftables are blocked
2. `--bridge=none` needed because bridge network creation is blocked
3. Even with both flags, overlayfs bind mounts fail (`unshare: operation not permitted`) — namespace creation is blocked at the kernel level

RunPod's own DinD tutorial uses Bazel (`rules_oci`) which builds OCI images without a daemon — that's for building images, not running them. There is no pod template or config that unlocks DinD without a privileged pod.

The Docker path commands are preserved below for reference only:

```bash
# Install Docker (requires sudo first: apt-get install -y sudo)
bash scripts/docker/install_docker_ubuntu22.sh
# Start daemon with workaround flags:
dockerd --iptables=false --bridge=none --storage-driver=vfs &
# Install NVIDIA toolkit:
bash scripts/docker/install_nvidia_container_toolkit.sh
pkill dockerd && dockerd --iptables=false --bridge=none --storage-driver=vfs &
# Run (single command, both containers):
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

### 6.3 FFmpeg Install Script

```bash
#!/bin/bash
set -e
echo "--- Starting Installation ---"
apt-get update
apt-get install -y software-properties-common
add-apt-repository -y ppa:ubuntuhandbook1/ffmpeg7
apt-get update
apt-get install -y ffmpeg git-lfs
git lfs install
echo "Installed FFmpeg version:"
ffmpeg -version | head -n 1
echo "Checking for SVT-AV1 support..."
if ffmpeg -encoders 2>/dev/null | grep -i svt; then
    echo "SUCCESS: V....D libsvtav1 detected."
else
    echo "WARNING: libsvtav1 not found."
fi
echo "--- Done ---"
```

Save as `install_ffmpeg.sh`, then `chmod +x install_ffmpeg.sh && ./install_ffmpeg.sh`.

### 6.4 LeRobot on RTX 5090 (special handling)

Do this **before** firing up the pod. The standard `pyproject.toml` caps torch at `<=2.5.1`, incompatible with the 5090:

```
In pyproject.toml → dependencies:
  torch >= 2.1.0, <= 2.5.1   →   torch >= 2.1.0
  (also remove upper bounds for torchvision and torchcodec)
```

Do on a separate branch (e.g. `rtx5090`). Then follow §4 normally.

Note: RTX 5090 caused too many bugs with LeRobot as of Dec 2025; switched to 4090 at the time.

---

## 7. Legacy / Archived

### 7.1 OpenVLA + LIBERO Setup (superseded)

> **Superseded** — current stack is π₀.₅ + openpi (§3). Kept for reference only.

```bash
apt-get update && apt-get install -y git git-lfs ffmpeg

git clone https://github.com/Shubodh/openvla.git
cd openvla
pip install -e .

# Flash Attention 2 (if fails: pip cache remove flash_attn first)
pip install packaging ninja
ninja --version; echo $?  # should return 0
pip install "flash-attn==2.5.5" --no-build-isolation

# LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt

# Optional: 10GB dataset (not required for evals)
# git clone git@hf.co:datasets/openvla/modified_libero_rlds

# Headless rendering
apt-get update && apt-get install -y libgl1-mesa-glx libegl1-mesa patchelf
export MUJOCO_GL=egl
pip install "numpy<2"
pip install 'huggingface_hub[cli,torch]'

# Sample eval
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 2
```
