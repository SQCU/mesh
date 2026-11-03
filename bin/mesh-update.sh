#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
LOG=/usr/local/mesh/log/update.log
FLOOR=$(awk '$1=="#define" && $2=="MESH_DISK_FLOOR_GIB"{print $3}' /usr/local/mesh/bin/mesh-disk.h 2>/dev/null)
FREE=$(df -k /usr/local/mesh/log 2>/dev/null | awk 'NR==2{print $4}')
if [ -n "$FLOOR" ] && [ -n "$FREE" ] && [ "$FREE" -lt $((FLOOR << 20)) ]; then
  echo "mesh-disk: $LOG not written: /usr/local/mesh/log has $((FREE * 1024)) bytes free, under the floor of $FLOOR GiB (mesh-disk.h)" >&2
  exec >/dev/null 2>&1
else [ -d /usr/local/mesh/log ] && exec >>"$LOG" 2>&1; fi
ts(){ date '+%F %T'; }
BRANCH=$(cat /usr/local/mesh/branch 2>/dev/null)
[ -n "$BRANCH" ] || BRANCH=$(awk '{print $1}' /usr/local/mesh/revision 2>/dev/null)
BRANCH=${BRANCH:-main}
echo "[$(ts)] converging from $BRANCH"
curl -fsSL "https://raw.githubusercontent.com/SQCU/mesh/$BRANCH/bootstrap.sh" \
  | MESH_BRANCH="$BRANCH" bash
