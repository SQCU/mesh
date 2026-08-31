#!/bin/sh
set -u
LOG=$1
shift
EVT=${JORACLE_SERVER_EVENTS:-/tmp/mesh-joracle/logs/server.events}
child=
stop() {
  if [ -n "$child" ] && kill -0 "$child" 2>/dev/null; then
    kill -TERM "$child" 2>/dev/null
    wait "$child"
  fi
  exit 0
}
trap stop INT TERM
while :; do
  printf '%s server_start\n' "$(date -u +%FT%TZ)" >> "$EVT"
  "$@" >> "$LOG" 2>&1 &
  child=$!
  wait "$child"
  status=$?
  printf '%s server_exit status=%s restart=1\n' "$(date -u +%FT%TZ)" "$status" >> "$EVT"
  child=
  sleep 1
done
