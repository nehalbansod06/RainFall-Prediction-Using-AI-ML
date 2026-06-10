#!/usr/bin/env bash
set -euo pipefail

# Cross-platform shell launcher for the project
ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"

if [ ! -x "$PYTHON" ]; then
  echo "Virtual environment not found — creating .venv using system python..."
  python3 -m venv "$VENV" || python -m venv "$VENV"
fi

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r "$ROOT/requirements.txt"

echo "Starting Flask app on http://127.0.0.1:5000"
"$PYTHON" "$ROOT/app_flask.py"
