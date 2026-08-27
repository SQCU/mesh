#!/bin/bash
# enable-autologin.sh -- boot straight into a GUI session, no keyboard.
#
# WHY THIS EXISTS
# Every mesh service is a LaunchDaemon and starts fine at the login window, so the
# node is reachable after a reboot without this. But a workload that needs a user
# session -- Metal/GPU in some configurations, Docker Desktop, anything touching
# WindowServer -- will silently not run. The node stays up, stays reachable, keeps
# beaconing, and does no work. That is a false negative, which is the failure this
# fleet cares about most. See THREAT-MODEL.md.
#
# WHAT IT COSTS
# macOS stores the auto-login password in /etc/kcpassword obfuscated with a fixed
# 11-byte XOR key. Treat it as plaintext to anyone with disk access. That is
# consistent with this fleet: FileVault is already off, and the security boundary
# is physical access to the room. Do NOT run this on a `portable` node, which
# leaves that boundary.
#
#   sudo ./enable-autologin.sh [username]
#
# Undo with:  sudo ./enable-autologin.sh --disable

set -u
[ "$(id -u)" -eq 0 ] || { echo "must run as root: sudo ./enable-autologin.sh"; exit 1; }

LW=/Library/Preferences/com.apple.loginwindow

if [ "${1:-}" = "--disable" ]; then
  defaults delete "$LW" autoLoginUser 2>/dev/null
  rm -f /etc/kcpassword
  echo "auto-login disabled; this node will boot to the login window"
  exit 0
fi

USER_NAME="${1:-${SUDO_USER:-$(stat -f%Su /dev/console)}}"
id "$USER_NAME" >/dev/null 2>&1 || { echo "no such user: $USER_NAME"; exit 1; }

PROFILE=$(cat /usr/local/mesh/profile 2>/dev/null || echo appliance)
if [ "$PROFILE" != appliance ]; then
  echo "refusing: this node's profile is '$PROFILE'."
  echo "A portable node leaves the room, and with it the physical access control"
  echo "that makes a plaintext-equivalent stored password acceptable."
  exit 1
fi

if [ "$(fdesetup status)" != "FileVault is Off." ]; then
  echo "warning: FileVault is on. Auto-login cannot bypass the pre-boot unlock,"
  echo "so this machine will still demand a keyboard at every boot."
fi

printf 'password for %s (not echoed): ' "$USER_NAME"
stty -echo; IFS= read -r PW; stty echo; printf '\n'
[ -n "$PW" ] || { echo "empty password, aborting"; exit 1; }

# Verify BEFORE writing, so a typo cannot leave a node that boots to a rejected
# login and quietly never reaches a session.
if ! dscl . -authonly "$USER_NAME" "$PW" >/dev/null 2>&1; then
  echo "that password is not valid for $USER_NAME -- nothing written"
  exit 1
fi

PW="$PW" python3 - "$USER_NAME" <<'PY'
import os, sys
key = [0x7D,0x89,0x52,0x23,0xD2,0xBC,0xDD,0xEA,0xA3,0xB9,0x1F]
pw  = list(os.environ['PW'].encode('utf-8'))
pad = 12 - (len(pw) % 12) if len(pw) % 12 else 12
pw += [0] * pad
enc = bytes(b ^ key[i % len(key)] for i, b in enumerate(pw))
fd = os.open('/etc/kcpassword', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
os.write(fd, enc); os.close(fd)
os.chown('/etc/kcpassword', 0, 0)
PY
unset PW

defaults write "$LW" autoLoginUser -string "$USER_NAME"
chmod 600 /etc/kcpassword; chown root:wheel /etc/kcpassword

echo
echo "auto-login enabled for $USER_NAME"
echo "  $(ls -l /etc/kcpassword)"
echo "  autoLoginUser = $(defaults read "$LW" autoLoginUser)"
echo
echo "This takes effect on the NEXT boot. Verify with a real reboot while you can"
echo "still reach the machine, not by trusting this message."
