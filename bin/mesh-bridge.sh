#!/bin/bash
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CONF="${MESH_CONF:-/usr/local/mesh/bridge.conf}"
[ -f "$CONF" ] || CONF="$HOME/.mesh-bridge.conf"
[ -f "$CONF" ] || CONF="$ROOT/etc/bridge.conf"
LABEL=io.mesh.bridge
BIN="${MESH_BIN:-/usr/local/mesh/bin/mesh-flow}"; [ -x "$BIN" ] || BIN="$ROOT/rdma/mesh-flow"
STAT="$(dirname "$BIN")/mesh-stat"; [ -x "$STAT" ] || STAT="$ROOT/rdma/mesh-stat"
case "$BIN" in *-wt/*|*/.build/*|/tmp/*|/private/tmp/*) echo "mesh-bridge: refusing to launch the bridge from a worktree, build or scratch path ($BIN); run make install-bridge" >&2; exit 65 ;; esac

scope=gui; mesh_pct=; node=0; links=(); region=/mesh0; flags=()
mesh_arena_pages=; mesh_block_pages=; mesh_qps=1

if [ -f "$CONF" ]; then . "$CONF"; fi
case "$region" in /mesh0) ;; *) LABEL="io.mesh.bridge.${region#/}" ;; esac
mesh_qps="${MESH_QPS:-$mesh_qps}"
mesh_arena_pages="${MESH_ARENA_PAGES:-$mesh_arena_pages}"
mesh_block_pages="${MESH_BLOCK_PAGES:-$mesh_block_pages}"
geometry=(-A "$mesh_arena_pages" -B "$mesh_block_pages")
link_args=()
for link in ${links[@]+"${links[@]}"}; do link_args+=(--link "$link"); done
[ -n "$mesh_pct" ] && geometry+=(-M "$mesh_pct")
geometry+=(${flags[@]+"${flags[@]}"})

case "$scope" in
system)
  if [ "$(id -u)" != 0 ]; then exec sudo -n MESH_CONF="$CONF" MESH_ARENA_PAGES="$mesh_arena_pages" MESH_BLOCK_PAGES="$mesh_block_pages" MESH_QPS="$mesh_qps" MESH_BIN="$BIN" MESH_LOG_DIR="${MESH_LOG_DIR:-}" MESH_LOG="${MESH_LOG:-}" MESH_LEDGER="${MESH_LEDGER:-}" "$0" "$@"; fi
  DOM=system; PLIST=/Library/LaunchDaemons/$LABEL.plist
  LOGDIR="${MESH_LOG_DIR:-/usr/local/mesh/log}" ;;
gui)
  DOM="gui/$(id -u)"; PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
  LOGDIR="${MESH_LOG_DIR:-$HOME/.mesh-logs}" ;;
*) echo "mesh-bridge: invalid configured scope $scope" >&2; exit 64 ;;
esac
LOG="${MESH_LOG:-$LOGDIR/$LABEL.log}"

wire_check() {
  want=$("$BIN" --layout -I "$node" "${geometry[@]}" ${link_args[@]+"${link_args[@]}"}) || return $?
  if [ "$want" -gt "$(sysctl -n vm.global_user_wire_limit)" ]; then
    if [ "$(id -u)" = 0 ]; then sysctl -w vm.global_user_wire_limit="$want"
    else sudo -n sysctl -w vm.global_user_wire_limit="$want"; fi
  fi
}

write_plist() {
  mkdir -p "$(dirname "$PLIST")" "$(dirname "$LOG")"
  cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>$LABEL</string>
<key>ProgramArguments</key><array>
<string>$BIN</string><string>-I</string><string>$node</string>
$( printf '<string>%s</string>' "${geometry[@]}" )
<string>-s</string><string>$region</string>
$( for arg in ${link_args[@]+"${link_args[@]}"}; do printf '<string>%s</string>' "$arg"; done )
</array>
<key>RunAtLoad</key><true/>
<key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
<key>ThrottleInterval</key><integer>1</integer>
<key>ExitTimeOut</key><integer>60</integer>
<key>EnvironmentVariables</key><dict><key>MESH_LOG_DIR</key><string>$LOGDIR</string><key>MESH_QPS</key><string>$mesh_qps</string>$( [ -n "${MESH_LEDGER:-}" ] && printf '<key>MESH_LEDGER</key><string>%s</string>' "$MESH_LEDGER" )</dict>
<key>StandardOutPath</key><string>$LOG</string>
<key>StandardErrorPath</key><string>$LOG</string>
</dict></plist>
PL
}

pid_of() { launchctl print "$DOM/$LABEL" 2>/dev/null | awk '/^\tpid = /{print $3}'; }
# The job's last exit status as launchd reports it (empty: it has not exited).
last_exit_of() { launchctl print "$DOM/$LABEL" 2>/dev/null | awk -F' = ' '/^\tlast exit code = /{print $2}' | grep -E '^[0-9]+$'; }
# Whether a client is attached to the region (a client's bridge stopped then exits 75 and launchd starts the next).
attached() { "$STAT" "$region" 2>/dev/null | grep -q '"attached":true'; }

# SIGTERM to the job's bridge, its exit waited for however long it takes (never escalated: a line every 30 s while it
# runs), then the job booted out.  A bridge with clients attached is not stopped: its clients' processes end first.
do_stop() {
  if attached; then
    echo "mesh-bridge: a client of $region is attached; not stopping its bridge (end the client first)" >&2
    return 1
  fi
  p=$(pid_of)
  if [ -n "$p" ]; then
    kill -TERM "$p" 2>/dev/null
    n=0
    while kill -0 "$p" 2>/dev/null; do
      sleep 0.1; n=$((n+1))
      [ $((n % 300)) -eq 0 ] && echo "mesh-bridge: bridge $p still running $((n/10)) s after its SIGTERM; not escalating" >&2
    done
  fi
  launchctl bootout "$DOM/$LABEL" >/dev/null 2>&1
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
  # ready, or its bridge exited (launchd's last exit code): observed, never a clock
  while :; do
    if "$STAT" --ready "$region" >/dev/null 2>&1; then
      echo "mesh-bridge: running as $(pid_of), registered $want bytes"
      return 0
    fi
    if [ -z "$(pid_of)" ] && [ -n "$(last_exit_of)" ]; then
      echo "mesh-bridge: the bridge exited ($(last_exit_of)) before it was ready; see $LOG" >&2
      return 1
    fi
    sleep 0.1
  done
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
  remove)  do_stop && rm -f "$PLIST" && echo "mesh-bridge: $PLIST removed" ;;
  status)  do_status ;;
  ready)   "$STAT" --ready "$region" ;;
  *) echo "usage: $0 {start|stop|restart|remove|status|ready}" >&2; exit 64 ;;
esac
