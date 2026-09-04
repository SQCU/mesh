#!/bin/bash
LW=/Library/Preferences/com.apple.loginwindow
R="$(cd "$(dirname "$0")" && pwd)"
[ "$(id -u)" -eq 0 ] || exec sudo "$0" "$@"

[ "${1:-}" = "--disable" ] && {
  defaults delete "$LW" autoLoginUser 2>/dev/null
  rm -f /etc/kcpassword
  echo "auto-login disabled"
  exit 0; }

USER_NAME="${1:-${SUDO_USER:-$(stat -f%Su /dev/console)}}"
id "$USER_NAME" >/dev/null 2>&1 || USER_NAME=$(dscl . -read /Groups/admin GroupMembership \
  | tr ' ' '\n' | grep -v 'GroupMembership:\|^root$\|^_\|^$' | head -1)

[ "$(fdesetup status)" = "FileVault is Off." ] \
  || echo "warning: FileVault is on, this machine will still demand a keyboard at boot"

printf 'password for %s (not echoed): ' "$USER_NAME"
stty -echo; IFS= read -r PW; stty echo; printf '\n'

dscl . -authonly "$USER_NAME" "$PW" >/dev/null 2>&1 || {
  echo "password validation failed for $USER_NAME, nothing written"
  echo "an unvalidated write would boot this node to a login it cannot pass"
  exit 1; }

PW="$PW" "$R/bin/mesh-python" - <<'PY'
import os
key = [0x7D,0x89,0x52,0x23,0xD2,0xBC,0xDD,0xEA,0xA3,0xB9,0x1F]
pw  = list(os.environ['PW'].encode('utf-8'))
pw += [0] * (12 - (len(pw) % 12) if len(pw) % 12 else 12)
fd = os.open('/etc/kcpassword', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
os.write(fd, bytes(b ^ key[i % len(key)] for i, b in enumerate(pw))); os.close(fd)
os.chown('/etc/kcpassword', 0, 0)
PY
unset PW

defaults write "$LW" autoLoginUser -string "$USER_NAME"
chmod 600 /etc/kcpassword; chown root:wheel /etc/kcpassword
echo "auto-login enabled for $USER_NAME, effective next boot"
echo "  $(ls -l /etc/kcpassword)"
