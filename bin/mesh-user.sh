#!/bin/bash
# The substrate as its user's (design/transport.md): the user's mesh checkout built and installed (make all install:
# the bridge and the libraries, each image that changed renamed into rdma/) with its Python bindings, ~/.venv-mesh-uv
# with torch-mesh and the checkout's library on its path, and the bridge as the user's launchd agent io.mesh.bridge, run from
# the checkout with the node's bridge configuration, the operand CONFIGURATION (a JSON, TOML or YAML document,
# etc/bridge/README.md): rdma/bridge_configuration.py writes its canonical form, which the launchd agent's
# definition names, so the keeper's load at boot keeps it.  There is no default configuration.
# A definition this script writes never changes with what it runs: a loaded launchd agent is left running (bin/mesh-launchd-agent.sh).  The document's path is recorded
# beside its canonical form.  The bridge's definition names its
# program and its configuration's path alone: a new binary, libmesh or configuration is installed by rename, and the
# bridge's heartbeat, sent SIGHUP, restarts the bridge on them, its region kept (design/transport.md section 2).  The install records what it installed
# (~/.cache/mesh/facts/io.mesh.bridge, which node-info publishes).
# Run as the user, again after every pull:  mesh-user.sh CONFIGURATION [CHECKOUT]  (~/mesh)
set -u
if [ $# -lt 1 ] || [ $# -gt 2 ] || [ ! -f "$1" ]; then
  echo "mesh-user.sh CONFIGURATION [CHECKOUT]: the node's bridge configuration document (JSON, TOML or YAML; etc/bridge/README.md) is required, the bridge has no default" >&2
  exit 64
fi
document=$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1")
source=$(cd "${2:-$HOME/mesh}" && pwd -P) || exit 1
# Exported sources carry a verified committed-file manifest instead of a mutable .git checkout.
if [ -f "$source/.mesh-source.json" ]; then
  python3 "$source/rdma/mesh_source.py" verify "$source" || exit 1
fi
venv=$HOME/.venv-mesh-uv
support="$HOME/Library/Application Support"; state=$HOME/.cache/mesh
uv=$(command -v uv || echo /usr/local/mesh/bin/uv)
make -C "$source/rdma" all install >/dev/null || exit 1
[ -x "$venv/bin/python" ] || "$uv" venv -q --python "$(cat "$source/.python-version")" "$venv" || exit 1
"$uv" pip install -q --python "$venv/bin/python" -r "$source/rdma/requirements.txt" || exit 1
bound=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))') || exit 1
csum=$({ printf '%s\n' "$bound"; cat "$source/rdma/mesh-plan.h" "$source/rdma/nccl.h" "$source/rdma/mesh_c_build.py"; } | shasum)
[ -f "$source/rdma/_mesh_c$bound" ] && [ "$(cat "$venv/mesh-c.sum" 2>/dev/null)" = "$csum" ] \
  || { "$venv/bin/python" "$source/rdma/mesh_c_build.py" >/dev/null && printf '%s\n' "$csum" > "$venv/mesh-c.sum"; } || exit 1
"$venv/bin/python" -c 'import sys, sysconfig; open(sysconfig.get_paths()["purelib"] + "/mesh.pth", "w").write(sys.argv[1] + "\n")' "$source/rdma" || exit 1
sum=$({ printf '%s\n' "$source"; cat "$source/rdma/libnccl-mesh.dylib"; cd "$source/rdma/torch-mesh" && python3 "$source/rdma/mesh_source.py" files "$source" rdma/torch-mesh | xargs -0 cat; } 2>/dev/null | shasum)
if [ "$(cat "$venv/torch-mesh.sum" 2>/dev/null)" != "$sum" ]; then
  build=$(mktemp -d) || exit 1
  (cd "$source/rdma/torch-mesh" && python3 "$source/rdma/mesh_source.py" files "$source" rdma/torch-mesh | xargs -0 tar -cf -) | tar -xf - -C "$build" || exit 1
  MESH_RDMA="$source/rdma" "$uv" pip install -q --python "$venv/bin/python" --no-build-isolation --reinstall-package torch-mesh "$build"
  status=$?; rm -rf "$build"
  [ "$status" = 0 ] || exit "$status"
  printf '%s\n' "$sum" > "$venv/torch-mesh.sum"
fi
configuration="$support/io.mesh.bridge/configuration.plist"; renew=
"$venv/bin/python" "$source/rdma/bridge_configuration.py" "$document" "$configuration.new" || exit 1
if cmp -s "$configuration.new" "$configuration"; then rm -f "$configuration.new"
else [ ! -f "$configuration" ] || renew=1; mv -f "$configuration.new" "$configuration" || exit 1; fi
printf '%s\n' "$document" > "$support/io.mesh.bridge/document"
launchd_agents=$HOME/Library/LaunchAgents; mkdir -p "$launchd_agents" "$HOME/Library/Logs" "$state/facts"
images=$(cat "$source/rdma/mesh-heartbeat" "$source/rdma/mesh-flow" "$source/rdma/libmesh.dylib" | shasum -a 256 | cut -d' ' -f1)
before=$(sed -n 's/^images=//p' "$state/facts/io.mesh.bridge" 2>/dev/null)
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
out=$("$source/bin/mesh-launchd-agent.sh" "$launchd_agents/io.mesh.bridge.plist.new") || { printf '%s\n' "$out"; exit 1; }
[ -z "$out" ] || printf '%s\n' "$out"
{ [ -z "$renew" ] && [ "$before" = "$images" ]; } \
  || { launchctl kill SIGHUP "user/$(id -u)/io.mesh.bridge" 2>/dev/null; \
       echo "io.mesh.bridge: its new binaries or configuration are installed; its heartbeat restarts the bridge on them (SIGHUP), the region kept"; }
printf 'revision=%s\nimages=%s\n' "$(python3 "$source/rdma/mesh_source.py" revision "$source" 2>/dev/null || echo unknown)" "$images" > "$state/facts/io.mesh.bridge"
