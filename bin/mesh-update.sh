#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
LOG=/usr/local/mesh/log/update.log
[ -d /usr/local/mesh/log ] && exec >>"$LOG" 2>&1
ts(){ date '+%F %T'; }
BRANCH=$(awk '{print $1}' /usr/local/mesh/revision 2>/dev/null)
BRANCH=${BRANCH:-main}
echo "[$(ts)] converging from $BRANCH"
curl -fsSL "https://raw.githubusercontent.com/SQCU/mesh/$BRANCH/bootstrap.sh" \
  | MESH_BRANCH="$BRANCH" bash
