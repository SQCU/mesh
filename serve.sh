#!/bin/bash
cd "$(dirname "$0")"
PORT="${1:-8080}"
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
command -v uv >/dev/null 2>&1 && RUN=(uv run --no-project python -m http.server) || RUN=(python3 -m http.server)
echo "docs on http://localhost:$PORT and http://$(scutil --get LocalHostName).local:$PORT"
exec "${RUN[@]}" "$PORT" --directory docs --bind 0.0.0.0
