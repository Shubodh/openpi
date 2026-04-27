# docs/

Operational documentation for this fork. Upstream openpi docs live in the upstream repo.

## Index

| File | What it covers |
|------|---------------|
| [`runpod_setup.md`](runpod_setup.md) | **Primary guide** — full RunPod setup for π₀.₅ + LIBERO |
| [`docker.md`](docker.md) | Docker setup (upstream) — not usable on RunPod; see §6.2 of runpod_setup.md |
| [`norm_stats.md`](norm_stats.md) | Normalization statistics for model checkpoints (upstream) |
| [`remote_inference.md`](remote_inference.md) | Running openpi models remotely (upstream) |

---

## `runpod_setup.md` — section map

| Section | What's in it |
|---------|-------------|
| **§0 Agent-First Setup** | The recommended entry point: API keys → install Claude Code + Codex → permissions file reference → launch agent with one prompt. |
| **§1 Quick Reference** | GPU choice, volume config, template, MuJoCo GL setting |
| **§2 Pod Configuration** | GPU (A40 locked), network volume size, first-boot commands |
| **§3 π₀.₅ + LIBERO Setup** | Why no Docker; automated scripts table; full manual steps; parallel suite runs; video separation and download; `main.py` args reference; baseline verification numbers |
| **§4 LeRobot + SmolVLA** | FFmpeg, lerobot install, credentials (HuggingFace, wandb) |
| **§5 Auto-stop / Sleep** | `runpodctl stop/remove` patterns for overnight runs |
| **§6 Supplementary** | A40 GPU rationale; Docker failure post-mortem; FFmpeg script; RTX 5090 special handling |
| **§7 Legacy / Archived** | OpenVLA + LIBERO setup (superseded) |

**Confirmed baseline numbers (A40, seed 7):**
- `libero_spatial`: 99.2% (496/500) — published 98.8%
- `libero_object`: 97.8% (489/500) — published 98.2%
