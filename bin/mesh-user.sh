#!/bin/bash
# The substrate as its user's (design/bridge-and-ipc.md): the user's mesh checkout built (the bridge, the library and its
# Python bindings), ~/.venv-mesh-uv with torch-mesh and the checkout's library on its path, and the bridge as the user's
# agent io.mesh.bridge, run from the checkout with the node's bridge configuration, the operand CONFIGURATION (a JSON,
# TOML or YAML document, etc/bridge/README.md): rdma/bridge_configuration.py writes its canonical form, which the agent's
# definition names, so the keeper's load at boot keeps it.  There is no default configuration.  With CONVERGENCE, the
# node's convergence document (etc/converge/README.md), the checkout follows its upstream by itself: the user's agent
# io.mesh.converge runs bin/mesh-converge.sh, which builds each upstream revision in a stage and runs this script again on
# the staged images (MESH_USER_BUILT) in a turn between calls (README.md, Convergence).  Each document's path is recorded
# beside its canonical form, so the convergence and the keeper keep the node's own.  The bridge's install records what it
# runs (~/.cache/mesh/facts/io.mesh.bridge, which node-info publishes) and the inputs it converged to.
# Run as the user, again after every pull:  mesh-user.sh CONFIGURATION [CHECKOUT [CONVERGENCE]]  (~/mesh)
set -u
if [ $# -lt 1 ] || [ ! -f "$1" ] || { [ $# -ge 3 ] && [ ! -f "$3" ]; }; then
  echo "mesh-user.sh CONFIGURATION [CHECKOUT [CONVERGENCE]]: the node's bridge configuration document (JSON, TOML or YAML; etc/bridge/README.md) is required, the bridge has no default; CONVERGENCE, where given, is the node's convergence document (etc/converge/README.md)" >&2
  exit 64
fi
document=$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1")
source=$(cd "${2:-$HOME/mesh}" && pwd -P) || exit 1
convergence=; [ $# -lt 3 ] || convergence=$(cd "$(dirname "$3")" && pwd -P)/$(basename "$3")
venv=$HOME/.venv-mesh-uv
support="$HOME/Library/Application Support"; state=$HOME/.cache/mesh
uv=$(command -v uv || echo /usr/local/mesh/bin/uv)
[ -n "${MESH_USER_BUILT:-}" ] || make -C "$source/rdma" all >/dev/null || exit 1
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
configuration="$support/io.mesh.bridge/configuration.plist"
"$venv/bin/python" "$source/rdma/bridge_configuration.py" "$document" "$configuration" || exit 1
printf '%s\n' "$document" > "$support/io.mesh.bridge/document"
if [ -n "$convergence" ]; then
  mkdir -p "$support/io.mesh.converge"
  "$venv/bin/python" - "$source/rdma" "$convergence" "$support/io.mesh.converge/configuration.plist" <<'PY' || exit 1
import plistlib, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from bridge_configuration import READERS
path = Path(sys.argv[2])
reader = READERS.get(path.suffix.lower())
if reader is None:
    sys.exit(f'{path}: a convergence document is a .json, .toml or .yaml document')
document = reader(path.read_bytes())
keys = {'period_s', 'quiet_s', 'notice_s', 'wait_s', 'gpu_percent'}
if not isinstance(document, dict) or set(document) != keys | {'purpose'}:
    sys.exit(f'{path}: a convergence document states its purpose and {", ".join(sorted(keys))}, and nothing else')
if not isinstance(document['purpose'], str) or not document['purpose'].strip():
    sys.exit(f'{path}: purpose, a statement of what the convergence is for, is required')
for key in keys:
    if isinstance(document[key], bool) or not isinstance(document[key], int) or document[key] < 0:
        sys.exit(f'{path}: {key} is a whole number, at least 0')
if document['period_s'] < 1 or document['wait_s'] < document['quiet_s']:
    sys.exit(f'{path}: period_s is at least 1 and wait_s at least quiet_s')
Path(sys.argv[3]).write_bytes(plistlib.dumps({'name': path.stem, **{k: document[k] for k in sorted(keys)}, 'purpose': ' '.join(document['purpose'].split())}))
PY
  printf '%s\n' "$convergence" > "$support/io.mesh.converge/document"
fi
agents=$HOME/Library/LaunchAgents; mkdir -p "$agents" "$HOME/Library/Logs" "$state/facts" "$state/converge"
images=$(cat "$source/rdma/mesh-flow" "$source/rdma/libmesh.dylib" | shasum -a 256 | cut -d' ' -f1)
cat > "$agents/io.mesh.bridge.plist.new" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!-- mesh-flow and libmesh $images, configuration $(shasum -a 256 "$configuration" | cut -d' ' -f1) from $document -->
<plist version="1.0"><dict><key>Label</key><string>io.mesh.bridge</string>
<key>ProgramArguments</key><array><string>$source/rdma/mesh-flow</string><string>$configuration</string></array>
<key>LimitLoadToSessionType</key><string>Background</string>
<key>RunAtLoad</key><true/><key>KeepAlive</key><true/><key>ThrottleInterval</key><integer>10</integer><key>ExitTimeOut</key><integer>60</integer>
<key>StandardErrorPath</key><string>$HOME/Library/Logs/io.mesh.bridge.log</string></dict></plist>
PL
"$source/bin/mesh-agent.sh" "$agents/io.mesh.bridge.plist.new" || exit 1
printf 'revision=%s\nimages=%s\n' "$(git -C "$source" rev-parse HEAD 2>/dev/null || echo unknown)" "$images" > "$state/facts/io.mesh.bridge"
period=$(plutil -extract period_s raw -o - "$support/io.mesh.converge/configuration.plist" 2>/dev/null)
if [ -n "$period" ]; then
  cat > "$agents/io.mesh.converge.plist.new" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict><key>Label</key><string>io.mesh.converge</string>
<key>ProgramArguments</key><array><string>/bin/bash</string><string>$source/bin/mesh-converge.sh</string></array>
<key>LimitLoadToSessionType</key><string>Background</string><key>StartInterval</key><integer>$period</integer>
<key>EnvironmentVariables</key><dict><key>PATH</key><string>/usr/local/mesh/bin:/usr/bin:/bin:/usr/sbin:/sbin</string></dict>
<key>StandardOutPath</key><string>$HOME/Library/Logs/io.mesh.converge.log</string><key>StandardErrorPath</key><string>$HOME/Library/Logs/io.mesh.converge.log</string></dict></plist>
PL
  "$source/bin/mesh-agent.sh" "$agents/io.mesh.converge.plist.new" || exit 1
  "$source/bin/mesh-converge.sh" stamp HEAD > "$state/converge/io.mesh.bridge"
else
  echo "mesh-user.sh: no convergence document recorded: $source moves only when this script is run again (etc/converge/README.md)"
fi
