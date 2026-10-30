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
# its log where the system's newsyslog rotates the mesh's user logs (etc/newsyslog-mesh.conf), else the user's Logs
log=/usr/local/mesh/log/user/io.mesh.hourly.log
[ -w /usr/local/mesh/log/user ] || { mkdir -p "$HOME/Library/Logs"; log=$HOME/Library/Logs/io.mesh.hourly.log; }
# every repository the hourly reads main from is readable here, else this node's every hourly fails at its first
# fetch: said now, at setup (metal-microbench is private: docs/nodes.md step 3, its deploy key)
"$venv/bin/python" - "$source/etc/mesh-nodes.json" <<'PY'
import json, os, subprocess, sys
for name, url in json.load(open(sys.argv[1]))['upstreams'].items():
    try:
        ok = subprocess.run(['git', 'ls-remote', url, 'refs/heads/main'], capture_output=True, timeout=30,
                            env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}).returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    if not ok:
        print(f'io.mesh.hourly: {name} ({url}) is not readable here; this node cannot run the hourly until it is')
PY
new=$(mktemp) || exit 1
cat > "$new" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!-- the node's hourly (bin/mesh-hourly-agent.sh): rdma/mesh_hourly.py node as $(id -un), each hour at minute $minute -->
<plist version="1.0"><dict><key>Label</key><string>io.mesh.hourly</string>
<key>UserName</key><string>$(id -un)</string>
<key>ProgramArguments</key><array><string>$venv/bin/python</string><string>$source/rdma/mesh_hourly.py</string><string>node</string></array>
<key>StartCalendarInterval</key><dict><key>Minute</key><integer>$minute</integer></dict>
<key>EnvironmentVariables</key><dict><key>HOME</key><string>$HOME</string><key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/local/mesh/bin:/usr/bin:/bin:/usr/sbin:/sbin</string></dict>
<key>StandardOutPath</key><string>$log</string>
<key>StandardErrorPath</key><string>$log</string></dict></plist>
PL
sudo -n install -m 644 -o root -g wheel "$new" /Library/LaunchDaemons/io.mesh.hourly.plist.new || { rm -f "$new"; exit 1; }
rm -f "$new"
exec sudo -n "$source/bin/mesh-launchd-agent.sh" /Library/LaunchDaemons/io.mesh.hourly.plist.new system
