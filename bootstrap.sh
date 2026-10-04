#!/bin/bash
set -o pipefail
REPO="${MESH_REPO:-SQCU/mesh}"
BRANCH="${MESH_BRANCH:-main}"
DEST="${MESH_DEST:-/usr/local/src/mesh}"
[ "$(id -u)" -eq 0 ] || exec sudo "$0" "$@"

echo "==> 1/3  opening the lifeline"
launchctl enable system/com.openssh.sshd 2>/dev/null
launchctl bootstrap system /System/Library/LaunchDaemons/ssh.plist 2>/dev/null
nc -z -G 2 127.0.0.1 22 >/dev/null 2>&1 \
  && echo "    sshd listening on :22" \
  || echo "    WARNING sshd did not come up, do not walk away yet"

echo "==> 2/3  fetching $REPO@$BRANCH"
mkdir -p "$DEST/generations"
SOURCE="$DEST/current"
[ -f "$SOURCE/install.sh" ] || SOURCE="$DEST"
# rdma/mesh-disk.h's floor (the current source's): a generation (about 150 MB) is fetched only while the disk
# keeps the floor free after it
FLOOR=$(awk '$1=="#define" && $2=="MESH_DISK_FLOOR_GIB"{print $3}' "$SOURCE/rdma/mesh-disk.h" 2>/dev/null)
FREE=$(df -k "$DEST/generations" 2>/dev/null | awk 'NR==2{print $4}')
if [ -n "$FLOOR" ] && [ -n "$FREE" ] && [ "$FREE" -lt $(( (FLOOR << 20) + (256 << 10) )) ]; then
  echo "mesh-disk: a source generation not fetched (268435456 bytes): $DEST has $((FREE * 1024)) bytes free, under the floor of $FLOOR GiB (mesh-disk.h); retaining $SOURCE"
  STAGE=
else
  STAGE=$(mktemp -d "$DEST/generations/source.XXXXXX")
fi
if [ -n "$STAGE" ] && curl -fsSL "https://codeload.github.com/$REPO/tar.gz/$BRANCH" -o "$STAGE/source.tgz"; then
  if cmp -s "$STAGE/source.tgz" "$SOURCE/source.tgz"; then
    rm -f "$STAGE/source.tgz"; rmdir "$STAGE"
  elif tar xzf "$STAGE/source.tgz" -C "$STAGE" --strip-components=1 && [ -f "$STAGE/install.sh" ]; then
    ln -s "$STAGE" "$STAGE/current"
    mv -fh "$STAGE/current" "$DEST/current"
    SOURCE="$STAGE"
  else
    echo "    extraction failed; retaining $SOURCE"
  fi
elif [ -n "$STAGE" ]; then
  echo "    fetch failed; retaining $SOURCE"
fi
# every generation but the current one deleted (each update that found the source changed had kept one:
# 327 of them, 46 GB, on the M4 Pro by 2026-09-30)
KEEP=$(basename "$(readlink "$DEST/current" 2>/dev/null)" 2>/dev/null)
case "$KEEP" in
  source.?*) for GENERATION in "$DEST"/generations/source.*; do
               [ "$(basename "$GENERATION")" = "$KEEP" ] || rm -rf "$GENERATION"
             done ;;
esac

echo "==> 3/3  provisioning"
MESH_REPO="$REPO" MESH_BRANCH="$BRANCH" exec bash "$SOURCE/install.sh"
