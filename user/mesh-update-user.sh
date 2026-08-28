#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin
D="$HOME/.local/mesh"; SRC="$D/src"; LOG="$D/update.log"
exec >>"$LOG" 2>&1
ts(){ date '+%F %T'; }
BRANCH=$(cat "$D/branch" 2>/dev/null); BRANCH=${BRANCH:-main}
mkdir -p "$SRC"
if curl -fsSL "https://codeload.github.com/SQCU/mesh/tar.gz/$BRANCH" | tar xz -C "$SRC" --strip-components=1; then
  echo "[$(ts)] fetched $BRANCH"
  printf '%s %s\n' "$BRANCH" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$D/revision"
  bash "$SRC/install-user.sh"
else
  echo "[$(ts)] fetch failed, keeping current tree"
fi
