#!/bin/bash
REPO="${MESH_REPO:-SQCU/mesh}"
REF="${MESH_REF:-main}"
case "$REF" in
  main) sha=$(curl -fsSL "https://api.github.com/repos/$REPO/commits/main" 2>/dev/null \
              | awk -F'"' '/"sha"/{print $4; exit}')
        [ -n "$sha" ] && REF="$sha" ;;
esac
DEST="${MESH_DEST:-/usr/local/src/mesh}"
[ "$(id -u)" -eq 0 ] || exec sudo "$0" "$@"

echo "==> 1/3  opening the lifeline"
launchctl enable system/com.openssh.sshd 2>/dev/null
launchctl bootstrap system /System/Library/LaunchDaemons/ssh.plist 2>/dev/null
nc -z -G 2 127.0.0.1 22 >/dev/null 2>&1 \
  && echo "    sshd listening on :22" \
  || echo "    WARNING sshd did not come up, do not walk away yet"

echo "==> 2/3  fetching $REPO@$REF"
rm -rf "$DEST"; mkdir -p "$DEST"
curl -fsSL "https://codeload.github.com/$REPO/tar.gz/$REF" | tar xz -C "$DEST" --strip-components=1 \
  || { echo "    fetch failed; sshd is up, finish remotely"; exit 1; }

echo "==> 3/3  provisioning"
exec bash "$DEST/install.sh"
