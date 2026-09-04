#!/bin/bash
cd "$(dirname "$0")"
PORT="${1:-8080}"
RUN=(./bin/mesh-python -m http.server)
echo "docs on http://localhost:$PORT and http://$(scutil --get LocalHostName).local:$PORT"
exec "${RUN[@]}" "$PORT" --directory docs --bind 0.0.0.0
