#!/bin/bash
# bootstrap.sh -- the last thing you ever type on this machine's keyboard.
#
# Run during the single physical visit, AFTER enabling RDMA in Recovery:
#   curl -fsSL https://raw.githubusercontent.com/OWNER/mesh/main/bootstrap.sh | sudo bash
#
# Deliberately does NOT use git: on a virgin Mac, invoking git triggers the
# Command Line Tools GUI installer. curl + tarball has no such dependency.
set -u
REPO="${MESH_REPO:-OWNER/mesh}"
REF="${MESH_REF:-main}"
DEST="${MESH_DEST:-/usr/local/src/mesh}"

[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }

echo "==> 1/3  opening the lifeline first, so a later failure can't strand this box"
launchctl enable system/com.openssh.sshd 2>/dev/null
launchctl bootstrap system /System/Library/LaunchDaemons/ssh.plist 2>/dev/null
if nc -z -G 2 127.0.0.1 22 >/dev/null 2>&1; then echo "    sshd listening on :22"
else echo "    WARNING: sshd did not come up -- do not walk away yet"; fi

echo "==> 2/3  fetching $REPO@$REF"
rm -rf "$DEST"; mkdir -p "$DEST"
if ! curl -fsSL "https://codeload.github.com/$REPO/tar.gz/$REF" | tar xz -C "$DEST" --strip-components=1; then
  echo "    FAILED to fetch repo. sshd is up; finish remotely."; exit 1
fi

echo "==> 3/3  provisioning"
exec bash "$DEST/install.sh"
