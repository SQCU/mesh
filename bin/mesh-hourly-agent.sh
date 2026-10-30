#!/bin/bash
# The hourly as the node's own (rdma/mesh_hourly.py, design/workstation-and-collective-2026-10-29.md): the user's launchd
# agent io.mesh.hourly starts `mesh_hourly.py node` from CHECKOUT at etc/mesh-nodes.json's hourly_minute, every hour.
# No machine launches another's run: each node takes main, holds its part's election and, in the elected part, runs
# its own rank and keeps its own record (~/.cache/mesh/hourly). A definition this script writes never changes with what
# it runs, and a loaded agent is left running (bin/mesh-launchd-agent.sh).
# Run as the user:  mesh-hourly-agent.sh CHECKOUT  (bin/mesh-user.sh runs it)
set -u
[ $# -eq 1 ] || { echo "mesh-hourly-agent.sh CHECKOUT" >&2; exit 64; }
source=$(cd "$1" && pwd -P) || exit 1
venv=$HOME/.venv-mesh-uv
minute=$("$venv/bin/python" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["hourly_minute"]))' \
  "$source/etc/mesh-nodes.json") || exit 1
agents=$HOME/Library/LaunchAgents; mkdir -p "$agents" "$HOME/Library/Logs"
cat > "$agents/io.mesh.hourly.plist.new" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!-- the node's hourly (bin/mesh-hourly-agent.sh): rdma/mesh_hourly.py node, each hour at minute $minute -->
<plist version="1.0"><dict><key>Label</key><string>io.mesh.hourly</string>
<key>ProgramArguments</key><array><string>$venv/bin/python</string><string>$source/rdma/mesh_hourly.py</string><string>node</string></array>
<key>StartCalendarInterval</key><dict><key>Minute</key><integer>$minute</integer></dict>
<key>LimitLoadToSessionType</key><string>Background</string>
<key>EnvironmentVariables</key><dict><key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/local/mesh/bin:/usr/bin:/bin:/usr/sbin:/sbin</string></dict>
<key>StandardOutPath</key><string>$HOME/Library/Logs/io.mesh.hourly.log</string>
<key>StandardErrorPath</key><string>$HOME/Library/Logs/io.mesh.hourly.log</string></dict></plist>
PL
exec "$source/bin/mesh-launchd-agent.sh" "$agents/io.mesh.hourly.plist.new"
