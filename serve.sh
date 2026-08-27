#!/bin/bash
# Serve the writeup locally. No venv, no deps, no hosted anything.
#   ./serve.sh [port]
cd "$(dirname "$0")"
PORT="${1:-8080}"
# uv installs to ~/.local/bin, which is not on PATH in a non-login shell (e.g. ssh).
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
if command -v uv >/dev/null 2>&1; then
  RUN=(uv run --no-project python -m http.server)
else
  echo "note: uv not found, falling back to system python3"
  RUN=(python3 -m http.server)
fi
echo "docs on http://localhost:$PORT   and http://$(scutil --get LocalHostName).local:$PORT"
echo "ctrl-c to stop"
exec "${RUN[@]}" "$PORT" --directory docs --bind 0.0.0.0
