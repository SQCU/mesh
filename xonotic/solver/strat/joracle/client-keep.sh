#!/bin/sh
# Keep an on-device Xonotic client attached to the demo server.
#
# Server restarts are CORRECT and expected (matches cycle; the supervisor
# relaunches). A client left unattached across one is not. This closes the
# gap where every bring-up printed a connect command instead of making the
# connection, so a human had nothing to look at after the first cycle.
ADDR="${1:-127.0.0.1:26042}"
BIN=/Users/mdot/dox/xonotic/Xonotic/Xonotic.app/Contents/MacOS/xonotic-osx-sdl-bin
BASE=/Users/mdot/dox/xonotic/Xonotic
USERDIR=/tmp/xonrun-client
LOG=/tmp/xonclient-keep.log
EVT=/tmp/xonclient-keep.events
STATE=/tmp/xonclient-keep.pid

log_evt() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$EVT"; }

launch() {
  : > "$LOG"                       # truncate so a stale timeout never re-matches
  nohup "$BIN" -basedir "$BASE" -userdir "$USERDIR" \
        +vid_fullscreen 0 +connect "$ADDR" >> "$LOG" 2>&1 &
  echo $! > "$STATE"
  log_evt "client_start pid=$(cat "$STATE") addr=$ADDR"
}

log_evt "supervisor_start addr=$ADDR"
while :; do
  pid=$(cat "$STATE" 2>/dev/null)
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    log_evt "client_absent reason=process_gone -> relaunch"
    launch
  elif tail -4 "$LOG" 2>/dev/null | grep -qiE 'connection timed out|disconnected|host_error'; then
    log_evt "client_dropped reason=$(tail -4 "$LOG" | grep -ioE 'connection timed out|disconnected|host_error' | tail -1) -> relaunch"
    kill "$pid" 2>/dev/null; sleep 2; launch
  fi
  sleep 10
done
