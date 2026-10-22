#!/bin/bash
cd "$(dirname "$0")"
PORT="${1:-8080}"
RUN=(./bin/mesh-python -m http.server)
echo "docs on http://localhost:$PORT and http://$(ipconfig getifaddr "$(route -n get default 2>/dev/null | awk '/interface:/{print $2}')"):$PORT"
exec "${RUN[@]}" "$PORT" --directory docs --bind 0.0.0.0
