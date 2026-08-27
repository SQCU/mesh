#!/bin/bash
# Watchdog: re-assert the no-downtime invariants every 60s. Logs only on drift.
LOG=/usr/local/mesh/log/keeper.log
[ -f "$LOG" ] && [ "$(stat -f%z "$LOG")" -gt 10485760 ] && { tail -c 2000000 "$LOG" > "$LOG.tmp"; mv "$LOG.tmp" "$LOG"; }
exec >>"$LOG" 2>&1
ts(){ date '+%F %T'; }
drift=0
PROFILE=$(cat /usr/local/mesh/profile 2>/dev/null || echo appliance)

# A portable node is allowed to sleep and keeps its defences; only an appliance
# owes us continuous presence.
if [ "$PROFILE" = appliance ]; then
  for kv in sleep=0 displaysleep=0 disksleep=0 standby=0 autorestart=1 womp=1 powermode=2; do
    k=${kv%%=*}; want=${kv##*=}
    have=$(pmset -g custom | awk -v k="$k" '$1==k {print $2; exit}')
    [ -n "$have" ] || continue
    if [ "$have" != "$want" ]; then
      echo "[$(ts)] DRIFT $k=$have -> $want"; pmset -a "$k" "$want"; drift=1
    fi
  done
fi
# The two lifelines. Losing either strands the node.
if ! /usr/bin/nc -z -G 2 127.0.0.1 22 >/dev/null 2>&1; then
  echo "[$(ts)] DRIFT sshd not listening -> re-bootstrapping"
  launchctl enable system/com.openssh.sshd >/dev/null 2>&1
  launchctl bootstrap system /System/Library/LaunchDaemons/ssh.plist >/dev/null 2>&1
  drift=1
fi
if ! /usr/bin/nc -z -G 2 127.0.0.1 5900 >/dev/null 2>&1; then
  echo "[$(ts)] DRIFT screensharing down -> re-bootstrapping"
  launchctl enable system/com.apple.screensharing >/dev/null 2>&1
  launchctl bootstrap system /System/Library/LaunchDaemons/com.apple.screensharing.plist >/dev/null 2>&1
  drift=1
fi
# The beacon is how the rest of the mesh learns this node still exists. Its silence
# is the alarm signal, so it must never be silent for a reason we could have fixed.
if ! pgrep -qf "dns-sd -R"; then
  echo "[$(ts)] DRIFT beacon not advertising -> restarting"
  launchctl kickstart -k system/io.mesh.beacon >/dev/null 2>&1
  drift=1
fi
# Clock drift breaks ssh and TLS silently, which reads as a dead node.
if [ "$(systemsetup -getusingnetworktime 2>/dev/null | awk '{print $NF}')" != "On" ]; then
  echo "[$(ts)] DRIFT network time off -> re-enabling"
  systemsetup -setusingnetworktime on >/dev/null 2>&1; drift=1
fi
# On an appliance the firewall can only ever subtract reachability, and the room
# is the real boundary. A portable node keeps its firewall -- it leaves the room.
if [ "$PROFILE" = appliance ] \
   && /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate 2>/dev/null | grep -q 'State = 1'; then
  echo "[$(ts)] DRIFT app firewall enabled itself -> disabling"
  /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off >/dev/null 2>&1; drift=1
fi
# RDMA is verify-only: enable/disable is Recovery-OS-gated, so this can never self-heal.
if [ "$(/usr/bin/rdma_ctl status 2>&1)" != "enabled" ]; then
  echo "[$(ts)] ALARM rdma disabled - needs physical Recovery visit, not fixable remotely"
  drift=1
fi
[ "$drift" -eq 0 ] || echo "[$(ts)] pass complete"
