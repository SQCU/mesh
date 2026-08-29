#!/bin/bash
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CONF="${MESH_CONF:-/usr/local/mesh/bridge.conf}"
[ -f "$CONF" ] || CONF="$ROOT/etc/bridge.conf"
LABEL=io.mesh.bridge
BIN="$ROOT/rdma/mesh-flow"
STAT="$ROOT/rdma/mesh-stat"

mesh_pct=25; app_pct=25; node=0; peer=""; region=/mesh0
# shellcheck disable=SC1090
[ -f "$CONF" ] && . "$CONF"

if [ "$(id -u)" != 0 ] && sudo -n true 2>/dev/null; then
  exec sudo -n MESH_CONF="$CONF" "$0" "$@"
fi
if [ "$(id -u)" = 0 ]; then DOM="system"; PLIST=/Library/LaunchDaemons/$LABEL.plist
else DOM="gui/$(id -u)"; PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"; fi

wire_check() {
  ram=$(sysctl -n hw.memsize)
  want_pct=$(awk -v a="$mesh_pct" -v b="$app_pct" 'BEGIN{print a+b}')
  if awk -v w="$want_pct" 'BEGIN{exit !(w > 90)}'; then
    echo "mesh-bridge: refusing mesh_pct=$mesh_pct + app_pct=$app_pct = $want_pct%" >&2
    echo "  wiring more than 90% of RAM can leave this node unable to boot." >&2
    return 78
  fi
  want=$(awk -v r="$ram" -v w="$want_pct" 'BEGIN{printf "%.0f", r*w/100}')
  [ "$want" -gt "$(sysctl -n vm.global_user_wire_limit)" ] &&
    sysctl -w vm.global_user_wire_limit="$want" >/dev/null 2>&1
  return 0
}

write_plist() {
  mkdir -p "$(dirname "$PLIST")"
  cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>$LABEL</string>
<key>ProgramArguments</key><array>
<string>$BIN</string><string>-I</string><string>$node</string>
<string>-M</string><string>$mesh_pct</string><string>-s</string><string>$region</string>
$( [ -n "$peer" ] && printf '<string>%s</string>' "$peer" )
</array>
<key>RunAtLoad</key><true/>
<key>KeepAlive</key><true/>
<key>ExitTimeOut</key><integer>0</integer>
<key>StandardOutPath</key><string>/tmp/$LABEL.log</string>
<key>StandardErrorPath</key><string>/tmp/$LABEL.log</string>
</dict></plist>
PL
}

pid_of() { launchctl print "$DOM/$LABEL" 2>/dev/null | awk '/^\tpid = /{print $3}'; }

do_stop() {
  p=$(pid_of)
  launchctl bootout "$DOM/$LABEL" >/dev/null 2>&1
  [ -z "$p" ] && { echo "mesh-bridge: stopped"; return 0; }
  for _ in $(seq 1 200000); do kill -0 "$p" 2>/dev/null || break; done
  kill -0 "$p" 2>/dev/null && { echo "mesh-bridge: $LABEL still running as $p after 1.5s; not escalating" >&2; return 1; }
  echo "mesh-bridge: stopped"
}

do_start() {
  wire_check || return $?
  [ -n "$(pid_of)" ] && { echo "mesh-bridge: already running as $(pid_of)"; return 0; }
  write_plist
  launchctl bootstrap "$DOM" "$PLIST" 2>/dev/null || launchctl load "$PLIST" 2>/dev/null
  for _ in $(seq 1 400); do [ -n "$(pid_of)" ] && break; done
  p=$(pid_of)
  [ -z "$p" ] && { echo "mesh-bridge: failed to start; see /tmp/$LABEL.log" >&2; return 1; }
  echo "mesh-bridge: running as $p, mesh ${mesh_pct}% app ${app_pct}%"
}

do_status() {
  p=$(pid_of)
  [ -n "$p" ] && echo "$LABEL pid $p" || echo "$LABEL not running"
  [ -x "$STAT" ] && "$STAT" "$region"
}

case "${1:-status}" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop; do_start ;;
  status)  do_status ;;
  *) echo "usage: $0 {start|stop|restart|status}" >&2; exit 64 ;;
esac
