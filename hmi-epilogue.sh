#!/bin/bash
[ "$(id -u)" -eq 0 ] || exec sudo "$0" "$@"
FW=/usr/libexec/ApplicationFirewall/socketfilterfw
MARK=/usr/local/mesh/hmi
SLEEP=${MESH_HMI_SLEEP:-10}
DISPLAYSLEEP=${MESH_HMI_DISPLAYSLEEP:-5}
FIREWALL=${MESH_HMI_FIREWALL:-on}
LOCK=${MESH_HMI_SCREENLOCK:-300}
USER_NAME="${MESH_USER:-${SUDO_USER:-$(stat -f%Su /dev/console)}}"
ok(){ printf '  \033[32mOK  \033[0m %s\n' "$*"; }
printf '\n\033[1m== HMI epilogue: %s ==\033[0m\n' "$(scutil --get LocalHostName)"
mkdir -p /usr/local/mesh
printf 'sleep=%s displaysleep=%s firewall=%s screenlock=%s set=%s by=%s\n' \
  "$SLEEP" "$DISPLAYSLEEP" "$FIREWALL" "$LOCK" "$(date '+%F %T')" "$USER_NAME" > "$MARK"
ok "marker $MARK written (keeper stops re-asserting power/firewall)"
pmset -a sleep "$SLEEP";               ok "sleep $SLEEP"
pmset -a displaysleep "$DISPLAYSLEEP"; ok "displaysleep $DISPLAYSLEEP"
pmset -a disksleep 10;                 ok "disksleep 10"
pmset -b powermode 0;                  ok "battery powermode 0"
[ "$FIREWALL" = on ] && { "$FW" --setglobalstate on >/dev/null 2>&1; ok "application firewall on"; }
sudo -u "$USER_NAME" defaults -currentHost write com.apple.screensaver idleTime -int "$LOCK" 2>/dev/null
ok "screensaver idleTime $LOCK ($USER_NAME)"
launchctl bootout system/io.mesh.caffeinate 2>/dev/null
launchctl disable system/io.mesh.caffeinate 2>/dev/null
ok "caffeinate daemon stopped"
printf '\n  Untouched, and still re-asserted every 60s: sshd, screen sharing, beacon,\n'
printf '  Thunderbolt fabric, network time, RDMA alarm.\n'
printf '  Undo with: sudo rm %s && sudo /usr/local/mesh/bin/mesh-keeper.sh\n\n' "$MARK"
