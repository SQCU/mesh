#!/bin/bash


set -u
R="$(cd "$(dirname "$0")" && pwd)"
D="$HOME/.local/mesh"; A="$HOME/Library/LaunchAgents"
ok(){ printf '  \033[32mOK  \033[0m %s\n' "$*"; }
failures=0; retained=0
bad(){ failures=$((failures+1)); printf '  \033[31mFAIL\033[0m %s\n' "$*"; }
try(){ d="$1"; shift; "$@" >/dev/null 2>&1 && ok "$d" || bad "$d"; }
load(){
  label=$1
  launchctl enable "gui/$UID/$label" >/dev/null 2>&1
  if launchctl print "gui/$UID/$label" >/dev/null 2>&1; then
    retained=$((retained+1)); ok "$label retained; launchd definition refresh pending"
    return
  fi
  for _ in 1 2 3; do launchctl bootstrap "gui/$UID" "$A/$label.plist" >/dev/null 2>&1; launchctl print "gui/$UID/$label" >/dev/null 2>&1 && break; sleep 1; done
  launchctl print "gui/$UID/$label" >/dev/null 2>&1 && ok "$label" || bad "$label"
}
system_owns(){
  label=$1; port=$2; url=$3
  pid=$(launchctl print "system/$label" 2>/dev/null | awk '/^[[:space:]]*pid = /{print $3; exit}')
  [ -n "$pid" ] && /usr/sbin/lsof -nP -a -p "$pid" -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1 \
    && curl -fsS --max-time 3 "$url" | "$PY" -c 'import json,sys; raise SystemExit(0 if isinstance(json.load(sys.stdin),dict) else 1)'
}
provide_http(){
  label=$1; port=$2; url=$3
  if system_owns "$label" "$port" "$url"; then
    retained=$((retained+1))
    ok "$label provided by system domain on :$port"
  else
    load "$label"
  fi
}
mkdir -p "$D/bin" "$D/etc" "$D/log" "$D/viz" "$A" "$HOME/.local/bin"
install -m 755 "$R/bin/mesh-nodeinfo.sh" "$D/bin/mesh-nodeinfo.sh"
install -m 755 "$R/user/mesh-nodeinfod.py" "$D/bin/mesh-nodeinfod.py"
install -m 755 "$R/user/mesh-telemetry.py" "$D/bin/mesh-telemetry.py"
install -m 755 "$R/user/mesh-update-user.sh" "$D/bin/mesh-update-user.sh"
install -m 755 "$R/bin/mesh-peers.sh" "$D/bin/mesh-peers.sh"
install -m 755 "$R/bin/mesh-run.sh" "$D/bin/mesh-run.sh"
install -m 755 "$R/bin/mesh-observe.py" "$D/bin/mesh-observe.py"
install -m 644 "$R/etc/mesh-capacity.json" "$D/etc/mesh-capacity.json"
install -m 644 "$R/etc/mesh-nodes.json" "$D/etc/mesh-nodes.json"
install -m 644 "$R/keys/authorized_keys" "$D/etc/authorized_keys"
install -m 755 "$R/viz/serve.py" "$D/viz/serve.py"
install -m 644 "$R/viz/index.html" "$D/viz/index.html"
try "mesh-stat built" make -C "$R/rdma" mesh-stat
try "mesh-stat installed" install -m 755 "$R/rdma/mesh-stat" "$D/bin/mesh-stat"
try "memory bandwidth sampler built" xcrun clang -fobjc-arc -framework Foundation -lIOReport "$R/user/mesh-bandwidth.m" -o "$R/user/mesh-bandwidth"
try "memory bandwidth sampler installed" install -m 755 "$R/user/mesh-bandwidth" "$D/bin/mesh-bandwidth"
install -m 755 "$R/bin/mesh-runtime-install.sh" "$D/bin/mesh-runtime-install.sh"
try "UV runtime realized" "$D/bin/mesh-runtime-install.sh" "$D" "$R"
ln -sf "$D/bin/mesh-python" "$HOME/.local/bin/mesh-python"
ln -sf "$D/bin/mesh-observe.py" "$HOME/.local/bin/mesh-observe"
ok "$D/bin populated"
PY=$D/bin/mesh-python
cat > "$A/io.mesh.nodeinfo.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>io.mesh.nodeinfo</string>
  <key>ProgramArguments</key><array><string>$PY</string><string>$D/bin/mesh-nodeinfod.py</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>5</integer>
  <key>StandardErrorPath</key><string>$D/log/io.mesh.nodeinfo.err.log</string>
</dict></plist>
PL
cat > "$A/io.mesh.telemetry.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>io.mesh.telemetry</string>
  <key>ProgramArguments</key><array><string>$PY</string><string>$D/bin/mesh-telemetry.py</string></array>
  <key>EnvironmentVariables</key><dict>
    <key>MESH_BANDWIDTH</key><string>$D/bin/mesh-bandwidth</string>
    <key>MESH_STAT</key><string>$D/bin/mesh-stat</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>5</integer>
  <key>StandardOutPath</key><string>$D/log/io.mesh.telemetry.out.log</string>
  <key>StandardErrorPath</key><string>$D/log/io.mesh.telemetry.err.log</string>
</dict></plist>
PL
cat > "$A/io.mesh.observer.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>io.mesh.observer</string>
  <key>ProgramArguments</key><array><string>$PY</string><string>$D/viz/serve.py</string></array>
  <key>WorkingDirectory</key><string>$D</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>5</integer>
  <key>StandardOutPath</key><string>$D/log/io.mesh.observer.out.log</string>
  <key>StandardErrorPath</key><string>$D/log/io.mesh.observer.err.log</string>
</dict></plist>
PL
load io.mesh.nodeinfo
provide_http io.mesh.telemetry 8788 http://127.0.0.1:8788/v1/latest
provide_http io.mesh.observer 8787 http://127.0.0.1:8787/latest.json
sleep 2
printf x | nc -G 2 -w 2 ::1 8100 >/dev/null 2>&1 && ok "responder answering on :8100" || bad "responder not answering"
curl -fsS --max-time 3 http://127.0.0.1:8788/v1/latest >/dev/null 2>&1 && ok "telemetry answering on :8788" || bad "telemetry not answering"
curl -fsS --max-time 3 http://127.0.0.1:8787/latest.json >/dev/null 2>&1 && ok "observer answering on :8787" || bad "observer not answering"
cat > "$A/io.mesh.update.user.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>io.mesh.update.user</string>
  <key>ProgramArguments</key><array><string>$D/bin/mesh-update-user.sh</string></array>
  <key>StartInterval</key><integer>900</integer>
</dict></plist>
PL
load io.mesh.update.user
ok "mesh-peers: $D/bin/mesh-peers.sh"
printf '%s\n' "${MESH_BRANCH:-main}" > "$D/branch"
printf '%s %s\n' "${MESH_BRANCH:-main}" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$D/revision"
printf 'reported_failures=%s retained_jobs=%s\n' "$failures" "$retained" > "$D/install-status"
echo "installer completed: $failures reported failures; $retained loaded jobs retained; live-generation convergence not asserted"
