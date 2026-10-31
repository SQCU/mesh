#!/bin/bash
# The substrate as its user's (design/transport.md): the user's mesh checkout built and installed (make all install:
# the bridge and the libraries, each image that changed renamed into rdma/), ~/.venv-mesh-uv with its base requirements,
# and the bridge as the user's launchd agent io.mesh.bridge, run from the checkout with the node's bridge configuration, the operand CONFIGURATION (a JSON, TOML or YAML document,
# etc/bridge/README.md): rdma/bridge_configuration.py writes its canonical form, which the launchd agent's
# definition names, so the keeper's load at boot keeps it.  There is no default configuration.
# A definition this script writes never changes with what it runs: a loaded launchd agent is left running (bin/mesh-launchd-agent.sh).  The document's path is recorded
# beside its canonical form.  The bridge's definition names its
# program and its configuration's path alone: a new binary, libmesh or configuration is installed by rename, and the
# bridge's heartbeat, sent SIGHUP, restarts the bridge on them, its region kept (design/transport.md section 2).  The install records what it installed
# (~/.cache/mesh/facts/io.mesh.bridge, which node-info publishes).
# Run as the user:  mesh-user.sh CONFIGURATION [CHECKOUT]  (the checkout holding this script: ~/mesh, the node's one)
set -u
if [ $# -lt 1 ] || [ $# -gt 2 ] || [ ! -f "$1" ]; then
  echo "mesh-user.sh CONFIGURATION [CHECKOUT]: the node's bridge configuration document (JSON, TOML or YAML; etc/bridge/README.md) is required, the bridge has no default" >&2
  exit 64
fi
document=$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1")
source=$(cd "${2:-$(dirname "$0")/..}" && pwd -P) || exit 1
python3 "$source/rdma/mesh_source.py" verify "$source" || exit 1
venv=$HOME/.venv-mesh-uv
support="$HOME/Library/Application Support"; state=$HOME/.cache/mesh
uv=$(command -v uv || echo /usr/local/mesh/bin/uv)
installed=$(make -s -C "$source/rdma" all install) || exit 1
[ -x "$venv/bin/python" ] || "$uv" venv -q --python "$(cat "$source/.python-version")" "$venv" || exit 1
"$uv" pip install -q --python "$venv/bin/python" -r "$source/rdma/requirements.txt" || exit 1
dependency_root=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))') || exit 1
for dependency_patch in "$source/rdma/torch-2.14.patch" "$source/rdma/torch-wrapper-storage.patch"; do
  if ! patch -R -s -f -C -F 0 -p1 -d "$dependency_root" < "$dependency_patch" >/dev/null 2>&1; then
    patch -f -F 0 -p1 -d "$dependency_root" < "$dependency_patch" || exit 1
  fi
done
configuration="$support/io.mesh.bridge/configuration.plist"; renew=
"$venv/bin/python" "$source/rdma/bridge_configuration.py" "$document" "$configuration.new" || exit 1
if cmp -s "$configuration.new" "$configuration"; then rm -f "$configuration.new"
else [ ! -f "$configuration" ] || renew=1; mv -f "$configuration.new" "$configuration" || exit 1; fi
printf '%s\n' "$document" > "$support/io.mesh.bridge/document"
launchd_agents=$HOME/Library/LaunchAgents; mkdir -p "$launchd_agents" "$HOME/Library/Logs" "$state/facts"
images_changed=$(printf '%s\n' "$installed" | grep -E 'rdma/(mesh-heartbeat|mesh-flow|libmesh.dylib) installed' || true)
[ -z "$renew" ] || { cp "$source/rdma/mesh-flow" "$source/rdma/mesh-flow.renew" && mv -f "$source/rdma/mesh-flow.renew" "$source/rdma/mesh-flow"; } || exit 1
cat > "$launchd_agents/io.mesh.bridge.plist.new" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!-- the node's bridge (bin/mesh-user.sh): its heartbeat (rdma/mesh-heartbeat), which opens and closes the bridge; a new binary or configuration is installed by rename and the heartbeat's SIGHUP restarts both on them, the region kept; SIGUSR1 releases the bridge and its region, SIGTERM releases them and ends the heartbeat (design/transport.md section 2) -->
<plist version="1.0"><dict><key>Label</key><string>io.mesh.bridge</string>
<key>ProgramArguments</key><array><string>$source/rdma/mesh-heartbeat</string><string>$source/rdma/mesh-flow</string><string>$configuration</string></array>
<key>LimitLoadToSessionType</key><string>Background</string><key>AbandonProcessGroup</key><true/>
<key>RunAtLoad</key><true/><key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict><key>ThrottleInterval</key><integer>60</integer><key>ExitTimeOut</key><integer>60</integer>
<key>StandardErrorPath</key><string>$HOME/Library/Logs/io.mesh.bridge.log</string></dict></plist>
PL
# the loaded bridge is this checkout's only when its job runs this checkout's heartbeat: another checkout's images did
# not change here, so its bridge is not signalled (its job takes this definition at its next load)
loaded=
program=$(launchctl print "user/$(id -u)/io.mesh.bridge" 2>/dev/null | awk '/^[[:space:]]*program = /{print $3; exit}')
[ -n "$program" ] && [ "$(cd "$(dirname "$program")" 2>/dev/null && pwd -P)/$(basename "$program")" = "$source/rdma/mesh-heartbeat" ] && loaded=1
out=$("$source/bin/mesh-launchd-agent.sh" "$launchd_agents/io.mesh.bridge.plist.new") || { printf '%s\n' "$out"; exit 1; }
[ -z "$out" ] || printf '%s\n' "$out"
{ [ -z "$loaded" ] || { [ -z "$renew" ] && [ -z "$images_changed" ]; }; } \
  || { launchctl kill SIGHUP "user/$(id -u)/io.mesh.bridge" 2>/dev/null; \
       echo "io.mesh.bridge: its new binaries or configuration are installed; its heartbeat restarts the bridge on them (SIGHUP), the region kept"; }
# the hourly's system job (its failure said, never the bridge's install's)
out=$("$source/bin/mesh-hourly-agent.sh" "$source" 2>&1) || out="io.mesh.hourly not installed: $out"
[ -z "$out" ] || printf '%s\n' "$out"
printf 'revision=%s\n' "$(python3 "$source/rdma/mesh_source.py" revision "$source")" > "$state/facts/io.mesh.bridge"
