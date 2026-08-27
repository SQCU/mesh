#!/bin/bash
# Serve the writeup locally. No venv, no deps, no hosted anything.
#   ./serve.sh [port]
cd "$(dirname "$0")"
PORT="${1:-8080}"
echo "docs on http://localhost:$PORT  (ctrl-c to stop)"
exec uv run --no-project python -m http.server "$PORT" --directory docs
