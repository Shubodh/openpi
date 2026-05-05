#!/bin/bash
# setup_agents.sh — install Codex CLI on the pod
# Run once per network volume, after setup_once.sh.
# Also usable standalone on any fresh pod before setup_once.sh if you want the agent
# to run setup_once.sh for you.
set -e

echo "=== [1/2] Installing Node.js ==="
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y -q nodejs

echo "=== [2/2] Installing Codex CLI ==="
npm install -g @openai/codex

echo ""
echo "=== setup_agents.sh complete ==="
echo ""
echo "Set API keys — choose one method:"
echo ""
echo "  Primary (set before pod starts — no typing needed in terminal):"
echo "    RunPod UI → pod config → Environment Variables → add:"
echo "      OPENAI_API_KEY=sk-..."
echo ""
echo "  Fallback (paste manually after SSH-ing in):"
echo "      export OPENAI_API_KEY=sk-..."
echo ""
echo "Permission files reference (see runpod/runpod_setup.md §0 Step 3):"
echo "  Codex CLI: ~/.codex/config.toml or .codex/config.toml in project"
echo ""
echo "Launch Codex agent:"
echo "  cd /workspace/openpi && codex 'your task here'"
echo ""
echo "Suggested prompts:"
echo "  First-time: 'Run runpod/setup_once.sh to set up the LIBERO environment'"
echo "  Restart:    'Run runpod/setup_pod.sh then start_libero.sh for suite libero_object'"
