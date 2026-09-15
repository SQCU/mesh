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

scope=gui; mesh_pct=; node=0; links=(); region=/mesh0
mesh_arena_pages=; mesh_block_pages=; mesh_qps=1

if [ -f "$CONF" ]; then . "$CONF"; fi
mesh_qps="${MESH_QPS:-$mesh_qps}"
mesh_arena_pages="${MESH_ARENA_PAGES:-$mesh_arena_pages}"
mesh_block_pages="${MESH_BLOCK_PAGES:-$mesh_block_pages}"
geometry=(-A "$mesh_arena_pages" -B "$mesh_block_pages")
link_args=()
for link in "${links[@]}"; do link_args+=(--link "$link"); done
[ -n "$mesh_pct" ] && geometry+=(-M "$mesh_pct")

case "$scope" in
system)
  if [ "$(id -u)" != 0 ]; then exec sudo -n MESH_CONF="$CONF" MESH_ARENA_PAGES="$mesh_arena_pages" MESH_BLOCK_PAGES="$mesh_block_pages" MESH_QPS="$mesh_qps" "$0" "$@"; fi
  DOM=system; PLIST=/Library/LaunchDaemons/$LABEL.plist
  LOGDIR="${MESH_LOG_DIR:-/usr/local/mesh/log}" ;;
gui)
  DOM="gui/$(id -u)"; PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
  LOGDIR="${MESH_LOG_DIR:-$HOME/.mesh-logs}" ;;
*) echo "mesh-bridge: invalid configured scope $scope" >&2; exit 64 ;;
esac

wire_check() {
  want=$("$BIN" --layout -I "$node" "${geometry[@]}" "${link_args[@]}") || return $?
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
$( for arg in "${link_args[@]}"; do printf '<string>%s</string>' "$arg"; done )
</array>
<key>RunAtLoad</key><true/>
<key>KeepAlive</key><true/>
<key>ExitTimeOut</key><integer>0</integer>
<key>EnvironmentVariables</key><dict><key>MESH_LOG_DIR</key><string>$LOGDIR</string><key>MESH_QPS</key><string>$mesh_qps</string></dict>
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
  if launchctl print "$DOM/$LABEL" >/dev/null 2>&1; then
    have=$("$STAT" "$region" 2>/dev/null | tr ',' '\n' | grep -E '"(rows|block|qps|links)"' | tr -d '" ' | tr '\n' ' ')
    want="rows:$mesh_arena_pages block:$mesh_block_pages qps:$mesh_qps links:${#links[@]}"
    if [ -n "$(pid_of)" ] && echo "$have" | grep -q "rows:$mesh_arena_pages " && echo "$have" | grep -q "block:$mesh_block_pages " && echo "$have" | grep -q "qps:$mesh_qps " && echo "$have" | grep -q "links:${#links[@]} "; then
      echo "mesh-bridge: already running as $(pid_of) with $want"; return 0
    fi
    if "$STAT" "$region" 2>/dev/null | grep -qE '"client":[1-9]'; then
      echo "mesh-bridge: running with other geometry ($have) and a client attached; not restarting" >&2; return 1
    fi
    echo "mesh-bridge: running with other geometry ($have); restarting for $want"
    do_stop || return $?
  fi
  wire_check || return $?
  write_plist
  launchctl bootstrap "$DOM" "$PLIST" || return $?
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
