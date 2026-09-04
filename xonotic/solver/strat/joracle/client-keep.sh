#!/bin/sh
ADDR="${1:-127.0.0.1:26042}"
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
XONOTIC=$(cd -- "$HERE/../.." && pwd)
BIN=${JORACLE_CLIENT_BIN:-$XONOTIC/darkplaces-work/darkplaces-sdl}
BASE=${JORACLE_BASEDIR:-$HOME/dox/xonotic/Xonotic}
USERDIR=${JORACLE_CLIENT_USERDIR:-/tmp/mesh-joracle/client-userdir}
LOG=${JORACLE_CLIENT_LOG:-/tmp/mesh-joracle/logs/client.log}
EVT=${JORACLE_CLIENT_EVENTS:-/tmp/mesh-joracle/logs/client.events}
STATE=${JORACLE_CLIENT_STATE:-/tmp/mesh-joracle/client-engine.pid}
SESSION=${JORACLE_CLIENT_SESSIONID:-joracle-client}
ASSETS=${JORACLE_ASSET_MANIFEST:-/tmp/mesh-joracle/assets.manifest}
ACTIVE=${STATE%.pid}.active-log
HEALTH=${STATE%.pid}.health

mkdir -p "$USERDIR/data" "$(dirname "$LOG")" "$(dirname "$STATE")"

log_evt() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$EVT"; }
binary_inode() { stat -f %i "$BIN" 2>/dev/null || stat -c %i "$BIN" 2>/dev/null; }
process_inode() {
  lsof -a -p "$1" -d txt -F in 2>/dev/null | awk -v path="n$BIN" '/^i/{inode=substr($0,2)} $0==path{print inode; exit}'
}
managed() {
  pid=$1
  kill -0 "$pid" 2>/dev/null || return 1
  [ -n "$(process_inode "$pid")" ] || return 1
  cmd=$(ps -p "$pid" -o command= 2>/dev/null)
  case "$cmd" in
    *"-basedir $BASE"*"-userdir $USERDIR"*"+connect $ADDR"*) return 0 ;;
  esac
  return 1
}
current_binary() { [ "$(process_inode "$1")" = "$(binary_inode)" ]; }
active_log() { cat "$ACTIVE" 2>/dev/null; }
connected() {
  log=$(active_log)
  [ -n "$log" ] && awk '/Connection [[:alpha:]]+ to /{up=NR} /Connection timed out|Server disconnected|Host_Error/{down=NR} END{exit !(up>down)}' "$log" 2>/dev/null
}
disconnected() {
  log=$(active_log)
  [ -n "$log" ] && awk '/Connection [[:alpha:]]+ to /{up=NR} /Connection timed out|Server disconnected|Host_Error/{down=NR} END{exit !(down>=up && down>0)}' "$log" 2>/dev/null
}
renderer_bad() {
  log=$(active_log)
  [ -n "$log" ] && grep -qiE 'file loaded but decode failed|GLSL shader.*failed|failed to compile.*shader|fallback.*non-PBR|division by zero|zero-size image' "$log" 2>/dev/null
}
write_health() {
  pid=$(cat "$STATE" 2>/dev/null)
  state=starting
  if [ -n "$pid" ] && managed "$pid"; then
    if renderer_bad; then state=renderer_degraded; elif connected; then state=healthy; else state=connecting; fi
  else
    state=process_absent
  fi
  asset_id=$(cksum "$ASSETS" 2>/dev/null | awk '{print $1 ":" $2}')
  printf 'state=%s\npid=%s\nsession=%s\nbinary_inode=%s\nexpected_inode=%s\nasset_manifest_id=%s\nlog=%s\n' \
    "$state" "${pid:-}" "$SESSION" "$(process_inode "${pid:-0}")" "$(binary_inode)" "${asset_id:-}" "$(active_log)" > "$HEALTH"
}
terminate() {
  pid=$1
  managed "$pid" || return
  kill -TERM "$pid" 2>/dev/null || return
  while kill -0 "$pid" 2>/dev/null; do sleep 1; done
  log_evt "client_stopped pid=$pid"
}
cleanup() {
  pid=$(cat "$STATE" 2>/dev/null)
  [ -n "$pid" ] && terminate "$pid"
  write_health
}
trap cleanup EXIT
trap 'exit 0' INT TERM

launch() {
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  genlog="$LOG.$stamp"
  nohup "$BIN" -xonotic -condebug -sessionid "$SESSION" -basedir "$BASE" -userdir "$USERDIR" \
        +vid_fullscreen 0 +vid_width 1600 +vid_height 900 +vid_vsync 1 \
        +developer 1 +con_debug 1 +r_texture_dds_load_logfailure 1 \
        +hud_panel_radar 0 +hud_panel_modicons_payload_ribbon 0 \
        +_cl_name observer +_termsofservice_accepted "${JORACLE_TOS_ACCEPTED:-1}" \
        +connect "$ADDR" +defer 10 "cmd spectate; togglemenu" >> "$genlog" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" > "$STATE"
  printf '%s\n' "$genlog" > "$ACTIVE"
  log_evt "client_start pid=$pid inode=$(binary_inode) session=$SESSION addr=$ADDR log=$genlog"
}

reconcile() {
  adopted=
  for lock in "$USERDIR"/lock*; do
    [ -e "$lock" ] || continue
    for pid in $(lsof -t "$lock" 2>/dev/null | sort -u); do
      managed "$pid" || continue
      if [ -z "$adopted" ] && current_binary "$pid"; then
        adopted=$pid
        printf '%s\n' "$pid" > "$STATE"
        [ -f "$LOG" ] && printf '%s\n' "$LOG" > "$ACTIVE"
        log_evt "client_adopt pid=$pid inode=$(process_inode "$pid") addr=$ADDR"
      else
        log_evt "client_replace pid=$pid inode=$(process_inode "$pid") expected=$(binary_inode)"
        terminate "$pid"
      fi
    done
  done
  [ -n "$adopted" ] || launch
}

log_evt "supervisor_start addr=$ADDR session=$SESSION"
reconcile
degraded=
while :; do
  sleep 15
  pid=$(cat "$STATE" 2>/dev/null)
  if [ -z "$pid" ] || ! managed "$pid"; then
    log_evt "client_absent reason=process_gone -> relaunch"
    launch
    degraded=
  elif ! current_binary "$pid"; then
    log_evt "client_stale pid=$pid inode=$(process_inode "$pid") expected=$(binary_inode) -> replace"
    terminate "$pid"
    launch
    degraded=
  elif disconnected; then
    log_evt "client_disconnected pid=$pid log=$(active_log) -> reconnect"
    terminate "$pid"
    launch
    degraded=
  elif renderer_bad && [ -z "$degraded" ]; then
    log_evt "client_renderer_degraded pid=$pid log=$(active_log)"
    degraded=1
  fi
  write_health
done
