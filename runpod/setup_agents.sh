#!/bin/bash
# setup_agents.sh — install AI coding agents (Claude Code + Codex CLI) on the pod
# Run once per network volume, after setup_once.sh.
# Also usable standalone on any fresh pod before setup_once.sh if you want the agent
# to run setup_once.sh for you.
set -e

echo "=== [1/3] Installing Node.js ==="
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y -q nodejs

echo "=== [2/3] Installing Claude Code ==="
curl -fsSL https://claude.ai/install.sh | bash
source "$HOME/.local/bin/env" 2>/dev/null || true

echo "=== [3/3] Installing Codex CLI ==="
npm install -g @openai/codex

echo ""
echo "=== setup_agents.sh complete ==="
echo ""
echo "Set API keys — choose one method:"
echo ""
echo "  Primary (set before pod starts — no typing needed in terminal):"
echo "    RunPod UI → pod config → Environment Variables → add:"
echo "      ANTHROPIC_API_KEY=sk-ant-..."
echo "      OPENAI_API_KEY=sk-..."
echo ""
echo "  Fallback (paste manually after SSH-ing in):"
echo "      export ANTHROPIC_API_KEY=sk-ant-..."
echo "      export OPENAI_API_KEY=sk-..."
echo ""
echo "Launch Claude Code agent:"
echo "  cd /workspace/openpi"
echo "  claude --dangerously-skip-permissions"
echo ""
echo "Suggested prompts:"
echo "  First-time: 'Run runpod/setup_once.sh to set up the LIBERO environment'"
echo "  Restart:    'Run runpod/setup_pod.sh then start_libero.sh for suite libero_object'"
