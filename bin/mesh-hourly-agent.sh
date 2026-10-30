#!/bin/bash
# The hourly as the node's own (rdma/mesh_hourly.py, design/workstation-and-collective-2026-10-29.md): the system launchd
# job io.mesh.hourly runs `mesh_hourly.py node` from CHECKOUT as this user at etc/mesh-nodes.json's hourly_minute, every
# hour. A system job, not the user's agent: on a headless node a user agent's calendar never fires with no login
# (2026-10-29 20:28: no node's ran), and a system job is loaded at boot and is outside Local Network privacy, which its
# election's mDNS and its ranks' fabric connections need. No machine launches another's run: each node takes main, holds
# its part's election and, in the elected part, runs its own rank and keeps its own record (~/.cache/mesh/hourly). A
# definition this script writes never changes with what it runs, and a loaded job is left running
# (bin/mesh-launchd-agent.sh).
# Run as the user, with sudo -n:  mesh-hourly-agent.sh CHECKOUT  (bin/mesh-user.sh runs it)
set -u
[ $# -eq 1 ] || { echo "mesh-hourly-agent.sh CHECKOUT" >&2; exit 64; }
source=$(cd "$1" && pwd -P) || exit 1
venv=$HOME/.venv-mesh-uv
minute=$("$venv/bin/python" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["hourly_minute"]))' \
  "$source/etc/mesh-nodes.json") || exit 1
mkdir -p "$HOME/Library/Logs"
new=$(mktemp) || exit 1
cat > "$new" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!-- the node's hourly (bin/mesh-hourly-agent.sh): rdma/mesh_hourly.py node as $(id -un), each hour at minute $minute -->
<plist version="1.0"><dict><key>Label</key><string>io.mesh.hourly</string>
<key>UserName</key><string>$(id -un)</string>
<key>ProgramArguments</key><array><string>$venv/bin/python</string><string>$source/rdma/mesh_hourly.py</string><string>node</string></array>
<key>StartCalendarInterval</key><dict><key>Minute</key><integer>$minute</integer></dict>
<key>EnvironmentVariables</key><dict><key>HOME</key><string>$HOME</string><key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/local/mesh/bin:/usr/bin:/bin:/usr/sbin:/sbin</string></dict>
<key>StandardOutPath</key><string>$HOME/Library/Logs/io.mesh.hourly.log</string>
<key>StandardErrorPath</key><string>$HOME/Library/Logs/io.mesh.hourly.log</string></dict></plist>
PL
sudo -n install -m 644 -o root -g wheel "$new" /Library/LaunchDaemons/io.mesh.hourly.plist.new || { rm -f "$new"; exit 1; }
rm -f "$new"
exec sudo -n "$source/bin/mesh-launchd-agent.sh" /Library/LaunchDaemons/io.mesh.hourly.plist.new system
