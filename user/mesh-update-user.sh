#!/bin/bash
set -o pipefail
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin
D="$HOME/.local/mesh"; LOG="$D/update.log"
exec >>"$LOG" 2>&1
ts(){ date '+%F %T'; }
BRANCH=$(cat "$D/branch" 2>/dev/null); BRANCH=${BRANCH:-main}
mkdir -p "$D/sources"
STAGE=$(mktemp -d "$D/sources/source.XXXXXX")
SRC="$D/source-current"
[ -f "$SRC/install-user.sh" ] || SRC="$D/src"
if curl -fsSL "https://codeload.github.com/${MESH_REPO:-SQCU/mesh}/tar.gz/$BRANCH" -o "$STAGE/source.tgz"; then
  echo "[$(ts)] fetched $BRANCH"
  if cmp -s "$STAGE/source.tgz" "$SRC/source.tgz"; then
    rm -f "$STAGE/source.tgz"; rmdir "$STAGE"
  elif tar xzf "$STAGE/source.tgz" -C "$STAGE" --strip-components=1 && [ -f "$STAGE/install-user.sh" ]; then
    ln -s "$STAGE" "$STAGE/current"
    mv -fh "$STAGE/current" "$D/source-current"
    SRC="$STAGE"
  else
    echo "[$(ts)] extraction failed, keeping $SRC; partial fetch at $STAGE"
  fi
else
  echo "[$(ts)] fetch failed, keeping $SRC; partial fetch at $STAGE"
fi
MESH_BRANCH="$BRANCH" bash "$SRC/install-user.sh"
