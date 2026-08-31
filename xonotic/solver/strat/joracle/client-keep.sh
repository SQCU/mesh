#!/bin/sh
# Keep an on-device Xonotic client attached to the demo server.
#
# Server restarts are CORRECT (matches cycle). A client left unattached across
# one is not. Three distinct failure modes, three distinct detections:
#   1. process gone            -> relaunch
#   2. dropped after connecting -> relaunch (log says timed out / disconnected)
#   3. launched while the server was down, sits at a menu forever -> relaunch
# (3) must NOT fire while the client is still loading: a 166 MB megamap takes
# minutes to precache, and an impatient timer turns the supervisor into a
# relaunch thrash loop that leaks client processes. So (3) requires BOTH no
# connection AND a log that has stopped growing (client no longer making
# progress) for QUIET seconds.
ADDR="${1:-127.0.0.1:26042}"
QUIET="${JORACLE_CLIENT_QUIET:-120}"
BIN=/Users/mdot/dox/xonotic/Xonotic/Xonotic.app/Contents/MacOS/xonotic-osx-sdl-bin
BASE=/Users/mdot/dox/xonotic/Xonotic
USERDIR=/tmp/xonrun-client
LOG=/tmp/xonclient-keep.log
EVT=/tmp/xonclient-keep.events
STATE=/tmp/xonclient-keep.pid

log_evt() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$EVT"; }
connected() { grep -qiE 'entered the game|observer connected|\bconnected\b' "$LOG" 2>/dev/null; }
log_age()   { echo $(( $(date +%s) - $(stat -f %m "$LOG" 2>/dev/null || date +%s) )); }

launch() {
  # never leave an orphan behind
  old=$(cat "$STATE" 2>/dev/null); [ -n "$old" ] && kill "$old" 2>/dev/null
  pkill -f 'xonotic-osx-sdl-bin' 2>/dev/null; sleep 1
  : > "$LOG"
  nohup "$BIN" -basedir "$BASE" -userdir "$USERDIR" \
        +vid_fullscreen 0 +connect "$ADDR" >> "$LOG" 2>&1 &
  echo $! > "$STATE"
  log_evt "client_start pid=$(cat "$STATE") addr=$ADDR"
}

log_evt "supervisor_start addr=$ADDR quiet=${QUIET}s"
launch
while :; do
  sleep 15
  pid=$(cat "$STATE" 2>/dev/null)
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    log_evt "client_absent reason=process_gone -> relaunch"; launch
  elif connected && tail -4 "$LOG" 2>/dev/null | grep -qiE 'connection timed out|disconnected|host_error'; then
    log_evt "client_dropped reason=lost_connection -> relaunch"; launch
  elif ! connected && [ "$(log_age)" -gt "$QUIET" ]; then
    log_evt "client_never_attached reason=no_connect_and_log_quiet_${QUIET}s -> relaunch"; launch
  fi
done
