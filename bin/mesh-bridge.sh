#!/bin/bash
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CONF="${MESH_CONF:-/usr/local/mesh/bridge.conf}"
[ -f "$CONF" ] || CONF="$HOME/.mesh-bridge.conf"
[ -f "$CONF" ] || CONF="$ROOT/etc/bridge.conf"
LABEL=io.mesh.bridge
BIN="$ROOT/rdma/mesh-flow"
STAT="$ROOT/rdma/mesh-stat"

scope=gui; mesh_pct=; node=0; peer=""; region=/mesh0
mesh_arena_pages=; mesh_receive_pages=

if [ -f "$CONF" ]; then . "$CONF"; fi
mesh_arena_pages="${MESH_ARENA_PAGES:-$mesh_arena_pages}"
mesh_receive_pages="${MESH_RECEIVE_PAGES:-$mesh_receive_pages}"
geometry=(-A "$mesh_arena_pages" -R "$mesh_receive_pages")
[ -n "$mesh_pct" ] && geometry+=(-M "$mesh_pct")

case "$scope" in
system)
  if [ "$(id -u)" != 0 ]; then exec sudo -n MESH_CONF="$CONF" MESH_ARENA_PAGES="$mesh_arena_pages" MESH_RECEIVE_PAGES="$mesh_receive_pages" "$0" "$@"; fi
  DOM=system; PLIST=/Library/LaunchDaemons/$LABEL.plist
  LOGDIR="${MESH_LOG_DIR:-/usr/local/mesh/log}" ;;
gui)
  DOM="gui/$(id -u)"; PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
  LOGDIR="${MESH_LOG_DIR:-$HOME/.mesh-logs}" ;;
*) echo "mesh-bridge: invalid configured scope $scope" >&2; exit 64 ;;
esac

wire_check() {
  want=$("$BIN" --layout "${geometry[@]}") || return $?
  if [ "$want" -gt "$(sysctl -n vm.global_user_wire_limit)" ]; then
    if [ "$(id -u)" = 0 ]; then sysctl -w vm.global_user_wire_limit="$want"
    else sudo -n sysctl -w vm.global_user_wire_limit="$want"; fi
  fi
}

write_plist() {
  mkdir -p "$(dirname "$PLIST")" "$LOGDIR"
  cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>$LABEL</string>
<key>ProgramArguments</key><array>
<string>$BIN</string><string>-I</string><string>$node</string>
$( printf '<string>%s</string>' "${geometry[@]}" )
<string>-s</string><string>$region</string>
$( [ -n "$peer" ] && printf '<string>%s</string>' "$peer" )
</array>
<key>RunAtLoad</key><true/>
<key>KeepAlive</key><true/>
<key>ExitTimeOut</key><integer>0</integer>
<key>EnvironmentVariables</key><dict><key>MESH_LOG_DIR</key><string>$LOGDIR</string></dict>
<key>StandardOutPath</key><string>$LOGDIR/$LABEL.log</string>
<key>StandardErrorPath</key><string>$LOGDIR/$LABEL.log</string>
</dict></plist>
PL
}

pid_of() { launchctl print "$DOM/$LABEL" 2>/dev/null | awk '/^\tpid = /{print $3}'; }

do_stop() {
  p=$(pid_of)
  launchctl bootout "$DOM/$LABEL" >/dev/null 2>&1 & request=$!
  for _ in $(seq 1 300); do
    if ! kill -0 "$request" 2>/dev/null && { [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; }; then break; fi
    sleep 0.1
  done
  if kill -0 "$request" 2>/dev/null || { [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }; then
    echo "mesh-bridge: stop pending after 30s (bridge=${p:-none}, request=$request); not escalating or replacing its region" >&2
    return 1
  fi
  wait "$request" 2>/dev/null
  echo "mesh-bridge: stopped"
}

do_start() {
  [ -n "$(pid_of)" ] && { echo "mesh-bridge: already running as $(pid_of)"; return 0; }
  wire_check || return $?
  write_plist
  launchctl bootstrap "$DOM" "$PLIST"
  for _ in $(seq 1 400); do [ -n "$(pid_of)" ] && break; sleep 0.01; done
  p=$(pid_of)
  [ -z "$p" ] && { echo "mesh-bridge: failed to start; see $LOGDIR/$LABEL.log" >&2; return 1; }
  for _ in $(seq 1 1200); do
    if "$STAT" --ready "$region" >/dev/null 2>&1; then
      echo "mesh-bridge: running as $p, registered $want bytes"
      return 0
    fi
    sleep 0.1
  done
  echo "mesh-bridge: setup incomplete; see $LOGDIR/$LABEL.log" >&2
  return 1
}

do_status() {
  p=$(pid_of)
  [ -n "$p" ] && echo "$LABEL pid $p" || echo "$LABEL not running"
  [ -x "$STAT" ] && "$STAT" "$region"
}

case "${1:-status}" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop && do_start ;;
  status)  do_status ;;
  ready)   "$STAT" --ready "$region" ;;
  *) echo "usage: $0 {start|stop|restart|status|ready}" >&2; exit 64 ;;
esac
