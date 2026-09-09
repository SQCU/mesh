#!/bin/zsh
set -u
ROOT=${0:A:h:h}
PCT=${MESH_LOOPBACK_PCT:-4}
A=${MESH_LOOPBACK_A:-127.0.0.1:18601}
B=${MESH_LOOPBACK_B:-127.0.0.1:18602}
LOG=${MESH_LOG_DIR:-$HOME/.mesh-logs}
mkdir -p "$LOG"
case ${1:-start} in
start)
  "$ROOT/rdma/mesh-flow" -I 0 -M "$PCT" -s /loop0 --link "udp,1,$A,$B" > "$LOG/loop0.log" 2>&1 &
  echo $! > "$LOG/loop0.pid"
  "$ROOT/rdma/mesh-flow" -I 1 -M "$PCT" -s /loop1 --link "udp,0,$B,$A" > "$LOG/loop1.log" 2>&1 &
  echo $! > "$LOG/loop1.pid"
  print "loopback bridges: /loop0 pid $(cat "$LOG/loop0.pid"), /loop1 pid $(cat "$LOG/loop1.pid"); logs in $LOG";;
stop)
  for n in loop0 loop1; do [ -f "$LOG/$n.pid" ] && kill -TERM "$(cat "$LOG/$n.pid")" 2>/dev/null; rm -f "$LOG/$n.pid"; done
  print "loopback bridges signalled";;
status)
  for n in loop0 loop1; do "$ROOT/rdma/mesh-stat" "/$n" 2>&1 | head -3; done;;
esac
