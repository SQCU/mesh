#!/bin/bash
# The substrate as its user's (design/bridge-and-ipc.md): the user's mesh checkout built (the bridge, the library and its
# Python bindings), ~/.venv-mesh-uv with torch-mesh and the checkout's library on its path, and the bridge as the user's
# agent io.mesh.bridge, run from the checkout with the node's bridge configuration, the operand CONFIGURATION (a JSON,
# TOML or YAML document, etc/bridge/README.md): rdma/bridge_configuration.py writes its canonical form, which the agent's
# definition names, so the keeper's load at boot keeps it.  There is no default configuration.
# Run as the user, again after every pull:  mesh-user.sh CONFIGURATION [CHECKOUT]  (~/mesh)
set -u
if [ $# -lt 1 ] || [ ! -f "$1" ]; then
  echo "mesh-user.sh CONFIGURATION [CHECKOUT]: the node's bridge configuration document (JSON, TOML or YAML; etc/bridge/README.md) is required; the bridge has no default" >&2
  exit 64
fi
document=$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1")
source=$(cd "${2:-$HOME/mesh}" && pwd -P) || exit 1
venv=$HOME/.venv-mesh-uv
uv=$(command -v uv || echo /usr/local/mesh/bin/uv)
make -C "$source/rdma" all >/dev/null || exit 1
[ -x "$venv/bin/python" ] || "$uv" venv -q --python "$(cat "$source/.python-version")" "$venv" || exit 1
"$uv" pip install -q --python "$venv/bin/python" -r "$source/rdma/requirements.txt" || exit 1
"$venv/bin/python" "$source/rdma/mesh_c_build.py" >/dev/null || exit 1
"$venv/bin/python" -c 'import sys, sysconfig; open(sysconfig.get_paths()["purelib"] + "/mesh.pth", "w").write(sys.argv[1] + "\n")' "$source/rdma" || exit 1
sum=$(cat "$source"/rdma/libnccl-mesh.dylib "$source"/rdma/torch-mesh/setup.py "$source"/rdma/torch-mesh/torch_mesh/* 2>/dev/null | shasum)
if [ "$(cat "$venv/torch-mesh.sum" 2>/dev/null)" != "$sum" ]; then
  build=$(mktemp -d) || exit 1
  cp -R "$source/rdma/torch-mesh/." "$build/"
  MESH_RDMA="$source/rdma" "$uv" pip install -q --python "$venv/bin/python" --no-build-isolation --reinstall-package torch-mesh "$build"
  status=$?; rm -rf "$build"
  [ "$status" = 0 ] || exit "$status"
  printf '%s\n' "$sum" > "$venv/torch-mesh.sum"
fi
configuration="$HOME/Library/Application Support/io.mesh.bridge/configuration.plist"
"$venv/bin/python" "$source/rdma/bridge_configuration.py" "$document" "$configuration" || exit 1
agents=$HOME/Library/LaunchAgents; mkdir -p "$agents" "$HOME/Library/Logs"
cat > "$agents/io.mesh.bridge.plist.new" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!-- mesh-flow $(shasum -a 256 "$source/rdma/mesh-flow" | cut -d' ' -f1), configuration $(shasum -a 256 "$configuration" | cut -d' ' -f1) from $document -->
<plist version="1.0"><dict><key>Label</key><string>io.mesh.bridge</string>
<key>ProgramArguments</key><array><string>$source/rdma/mesh-flow</string><string>$configuration</string></array>
<key>LimitLoadToSessionType</key><string>Background</string>
<key>RunAtLoad</key><true/><key>KeepAlive</key><true/><key>ThrottleInterval</key><integer>10</integer><key>ExitTimeOut</key><integer>60</integer>
<key>StandardErrorPath</key><string>$HOME/Library/Logs/io.mesh.bridge.log</string></dict></plist>
PL
"$source/bin/mesh-agent.sh" "$agents/io.mesh.bridge.plist.new"
