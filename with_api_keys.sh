#!/usr/bin/env bash
set -euo pipefail

export OPENAI_API_KEY=$(cat ~/.openai_key)
export ANTHROPIC_API_KEY=$(cat ~/.claude_key)
export GEMINI_API_KEY=$(cat ~/.gemini_key)

exec "$@"
