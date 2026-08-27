#!/bin/bash
LOG=/usr/local/mesh/log/keeper.log
[ -f "$LOG" ] && [ "$(stat -f%z "$LOG")" -gt 10485760 ] && { tail -c 2000000 "$LOG" >"$LOG.tmp"; mv "$LOG.tmp" "$LOG"; }
exec >>"$LOG" 2>&1
ts(){ date '+%F %T'; }
FW=/usr/libexec/ApplicationFirewall/socketfilterfw
drift=0
for kv in sleep=0 displaysleep=0 disksleep=0 standby=0 autorestart=1 womp=1 powermode=2; do
  k=${kv%%=*}; want=${kv##*=}
  have=$(pmset -g custom | awk -v k="$k" '$1==k{print $2;exit}')
  [ -n "$have" ] && [ "$have" != "$want" ] && { echo "[$(ts)] DRIFT $k=$have -> $want"; pmset -a "$k" "$want"; drift=1; }
done
nc -z -G 2 127.0.0.1 22 >/dev/null 2>&1 || {
  echo "[$(ts)] DRIFT sshd down"
  launchctl enable system/com.openssh.sshd >/dev/null 2>&1
  launchctl bootstrap system /System/Library/LaunchDaemons/ssh.plist >/dev/null 2>&1
  drift=1; }
nc -z -G 2 127.0.0.1 5900 >/dev/null 2>&1 || {
  echo "[$(ts)] DRIFT screensharing down"
  launchctl enable system/com.apple.screensharing >/dev/null 2>&1
  launchctl bootstrap system /System/Library/LaunchDaemons/com.apple.screensharing.plist >/dev/null 2>&1
  drift=1; }
pgrep -qf "dns-sd -R" || {
  echo "[$(ts)] DRIFT beacon down"
  launchctl kickstart -k system/io.mesh.beacon >/dev/null 2>&1
  drift=1; }
[ "$(systemsetup -getusingnetworktime 2>/dev/null | awk '{print $NF}')" = On ] || {
  echo "[$(ts)] DRIFT ntp off"; systemsetup -setusingnetworktime on >/dev/null 2>&1; drift=1; }
"$FW" --getglobalstate 2>/dev/null | grep -q 'State = 1' && {
  echo "[$(ts)] DRIFT firewall on"; "$FW" --setglobalstate off >/dev/null 2>&1; drift=1; }
[ "$(rdma_ctl status 2>&1)" = enabled ] || {
  echo "[$(ts)] ALARM rdma disabled, physical Recovery OS visit required"; drift=1; }
[ "$drift" -eq 0 ] || echo "[$(ts)] pass complete"
exit 0
