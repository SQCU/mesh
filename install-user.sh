#!/bin/bash


set -u
R="$(cd "$(dirname "$0")" && pwd)"
D="$HOME/.local/mesh"; A="$HOME/Library/LaunchAgents"
ok(){ printf '  \033[32mOK  \033[0m %s\n' "$*"; }
failures=0; retained=0; replaced=0
bad(){ failures=$((failures+1)); printf '  \033[31mFAIL\033[0m %s\n' "$*"; }
try(){ d="$1"; shift; "$@" >/dev/null 2>&1 && ok "$d" || bad "$d"; }
mkdir -p "$D/bin" "$D/etc" "$D/log" "$D/viz" "$A" "$HOME/.local/bin"
# telemetry and the observer run only when someone runs them (mesh-top starts its own sampler); copies loaded before
# 2026-10-29 are booted out and removed
for L in io.mesh.telemetry io.mesh.observer; do launchctl bootout "gui/$UID/$L" >/dev/null 2>&1; rm -f "$A/$L.plist" "$A/$L.plist.new"; done
install -C -m 755 "$R/bin/mesh-nodeinfo.sh" "$D/bin/mesh-nodeinfo.sh"
install -C -m 755 "$R/user/mesh-telemetry.py" "$D/bin/mesh-telemetry.py"
install -C -m 755 "$R/bin/mesh-peers.sh" "$R/bin/mesh-rdma.sh" "$D/bin/"
install -C -m 755 "$R/bin/mesh-observe.py" "$D/bin/mesh-observe.py"
install -C -m 755 "$R/bin/mesh-top.py" "$D/bin/mesh-top.py"
install -C -m 644 "$R/rdma/mesh_disk.py" "$R/rdma/mesh-disk.h" "$R/rdma/peers.py" "$D/bin/"
install -C -m 644 "$R/etc/mesh-capacity.json" "$D/etc/mesh-capacity.json"
install -C -m 644 "$R/etc/mesh-nodes.json" "$D/etc/mesh-nodes.json"
install -C -m 644 "$R/keys/authorized_keys" "$D/etc/authorized_keys"
[ -f "$D/etc/observer.json" ] || install -m 644 "$R/etc/observer.json" "$D/etc/observer.json"
install -C -m 755 "$R/viz/serve.py" "$D/viz/serve.py"
install -C -m 644 "$R/viz/index.html" "$D/viz/index.html"
install -C -m 755 "$R/bin/mesh-stat" "$D/bin/mesh-stat" && rm -f "$D/bin/libmesh.dylib" || bad "mesh-stat installed"
try "memory bandwidth sampler built" xcrun clang -fobjc-arc -framework Foundation -lIOReport "$R/user/mesh-bandwidth.m" -o "$R/user/mesh-bandwidth"
try "memory bandwidth sampler installed" install -m 755 "$R/user/mesh-bandwidth" "$D/bin/mesh-bandwidth"
install -C -m 755 "$R/bin/mesh-runtime-install.sh" "$D/bin/mesh-runtime-install.sh"
try "UV runtime realized" "$D/bin/mesh-runtime-install.sh" "$D" "$R"
ln -sf "$D/bin/mesh-python" "$HOME/.local/bin/mesh-python"
ln -sf "$D/bin/mesh-observe.py" "$HOME/.local/bin/mesh-observe"
ln -sf "$D/bin/mesh-top.py" "$HOME/.local/bin/mesh-top"
ok "$D/bin populated"
ok "mesh-peers: $D/bin/mesh-peers.sh"
printf '%s %s\n' "${MESH_BRANCH:-main}" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$D/revision"
printf 'reported_failures=%s retained_jobs=%s replaced_jobs=%s\n' "$failures" "$retained" "$replaced" > "$D/install-status"
echo "installer completed: $failures reported failures; $retained loaded jobs current, $replaced loaded now or with a new definition for their next load"
