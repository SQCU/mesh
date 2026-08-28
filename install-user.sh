#!/bin/bash
# No sudo. Makes this machine a fully identified fabric partner using only the
# user domain. See README "Joining without root".
set -u
R="$(cd "$(dirname "$0")" && pwd)"
D="$HOME/.local/mesh"; A="$HOME/Library/LaunchAgents"
ok(){ printf '  \033[32mOK  \033[0m %s\n' "$*"; }
mkdir -p "$D/bin" "$A"
install -m 755 "$R/bin/mesh-nodeinfo.sh" "$D/bin/mesh-nodeinfo.sh"
install -m 755 "$R/user/mesh-nodeinfod.py" "$D/bin/mesh-nodeinfod.py"
install -m 755 "$R/bin/mesh-peers.sh" "$D/bin/mesh-peers.sh"
install -m 755 "$R/bin/mesh-run.sh" "$D/bin/mesh-run.sh"
ok "$D/bin populated"
PY=$(command -v python3 || echo /usr/bin/python3)
cat > "$A/io.mesh.nodeinfo.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>io.mesh.nodeinfo</string>
  <key>ProgramArguments</key><array><string>$PY</string><string>$D/bin/mesh-nodeinfod.py</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>5</integer>
  <key>StandardErrorPath</key><string>$D/nodeinfod.err</string>
</dict></plist>
PL
launchctl bootout "gui/$UID/io.mesh.nodeinfo" 2>/dev/null; sleep 1
launchctl bootstrap "gui/$UID" "$A/io.mesh.nodeinfo.plist" 2>/dev/null
sleep 2
printf x | nc -G 2 -w 2 ::1 8100 >/dev/null 2>&1 && ok "responder answering on :8100" \
  || printf '  \033[31mFAIL\033[0m responder not answering\n'
ok "mesh-peers: $D/bin/mesh-peers.sh"
