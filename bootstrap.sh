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
STAGE=$(mktemp -d "$DEST/generations/source.XXXXXX")
SOURCE="$DEST/current"
[ -f "$SOURCE/install.sh" ] || SOURCE="$DEST"
if curl -fsSL "https://codeload.github.com/$REPO/tar.gz/$BRANCH" -o "$STAGE/source.tgz"; then
  if cmp -s "$STAGE/source.tgz" "$SOURCE/source.tgz"; then
    rm -f "$STAGE/source.tgz"; rmdir "$STAGE"
  elif tar xzf "$STAGE/source.tgz" -C "$STAGE" --strip-components=1 && [ -f "$STAGE/install.sh" ]; then
    ln -s "$STAGE" "$STAGE/current"
    mv -fh "$STAGE/current" "$DEST/current"
    SOURCE="$STAGE"
  else
    echo "    extraction failed; retaining $SOURCE; partial fetch at $STAGE"
  fi
else
  echo "    fetch failed; retaining $SOURCE; partial fetch at $STAGE"
fi

echo "==> 3/3  provisioning"
MESH_BRANCH="$BRANCH" exec bash "$SOURCE/install.sh"
