#!/usr/bin/env zsh
# One-shot environment setup for macOS (Apple Silicon). Safe to re-run.
set -e
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$PATH"

command -v python3.11 >/dev/null || brew install python@3.11
command -v node >/dev/null || brew install node

[ -d .venv ] || python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip freeze > requirements.lock.txt

(cd frontend && NODE_ENV=development npm install --include=dev)
echo "Setup complete."
