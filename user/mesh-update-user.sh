#!/bin/bash
set -o pipefail
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin
D="$HOME/.local/mesh"; LOG="$D/update.log"
SRC="$D/source-current"
[ -f "$SRC/install-user.sh" ] || SRC="$D/src"
# rdma/mesh-disk.h's numbers (the current source's): the log moves to update.log.1 past the cap; a source
# generation (about 150 MB) is fetched only while the disk keeps the floor free after it
FLOOR=$(awk '$1=="#define" && $2=="MESH_DISK_FLOOR_GIB"{print $3}' "$SRC/rdma/mesh-disk.h" 2>/dev/null)
CAP=$(awk '$1=="#define" && $2=="MESH_LOG_CAP_MIB"{print $3}' "$SRC/rdma/mesh-disk.h" 2>/dev/null)
[ -n "$CAP" ] && [ "$(stat -f %z "$LOG" 2>/dev/null || echo 0)" -gt $((CAP << 20)) ] && mv -f "$LOG" "$LOG.1"
exec >>"$LOG" 2>&1
ts(){ date '+%F %T'; }
BRANCH=$(cat "$D/branch" 2>/dev/null); BRANCH=${BRANCH:-main}
mkdir -p "$D/sources"
FREE=$(df -k "$D/sources" 2>/dev/null | awk 'NR==2{print $4}')
if [ -n "$FLOOR" ] && [ -n "$FREE" ] && [ "$FREE" -lt $(( (FLOOR << 20) + (256 << 10) )) ]; then
  echo "[$(ts)] mesh-disk: a source generation not fetched (268435456 bytes): $D has $((FREE * 1024)) bytes free, under the floor of $FLOOR GiB (mesh-disk.h); keeping $SRC"
  STAGE=
else
  STAGE=$(mktemp -d "$D/sources/source.XXXXXX")
fi
if [ -n "$STAGE" ] && curl -fsSL "https://codeload.github.com/${MESH_REPO:-SQCU/mesh}/tar.gz/$BRANCH" -o "$STAGE/source.tgz"; then
  echo "[$(ts)] fetched $BRANCH"
  if cmp -s "$STAGE/source.tgz" "$SRC/source.tgz"; then
    rm -f "$STAGE/source.tgz"; rmdir "$STAGE"
  elif tar xzf "$STAGE/source.tgz" -C "$STAGE" --strip-components=1 && [ -f "$STAGE/install-user.sh" ]; then
    ln -s "$STAGE" "$STAGE/current"
    mv -fh "$STAGE/current" "$D/source-current"
    SRC="$STAGE"
  else
    echo "[$(ts)] extraction failed, keeping $SRC"
  fi
elif [ -n "$STAGE" ]; then
  echo "[$(ts)] fetch failed, keeping $SRC"
fi
# every generation but the current one deleted (271 of them, 35 GB, on the M5 by 2026-09-30)
KEEP=$(basename "$(readlink "$D/source-current" 2>/dev/null)" 2>/dev/null)
case "$KEEP" in
  source.?*) for GENERATION in "$D"/sources/source.*; do
               [ "$(basename "$GENERATION")" = "$KEEP" ] || rm -rf "$GENERATION"
             done ;;
esac
MESH_BRANCH="$BRANCH" bash "$SRC/install-user.sh"
