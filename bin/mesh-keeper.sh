#!/bin/bash
LOG=/usr/local/mesh/log/keeper.log
[ -f "$LOG" ] && [ "$(stat -f%z "$LOG")" -gt 10485760 ] && { tail -c 2000000 "$LOG" >"$LOG.tmp"; mv "$LOG.tmp" "$LOG"; }
exec >>"$LOG" 2>&1
ts(){ date '+%F %T'; }
FW=/usr/libexec/ApplicationFirewall/socketfilterfw
drift=0
POLICY=/usr/local/mesh/policy
[ -f "$POLICY" ] || cp /usr/local/mesh/policy.default "$POLICY" 2>/dev/null
while read -r k v; do
  [ -n "$k" ] || continue
  have=$(pmset -g custom | awk -v k="$k" '$1==k{print $2;exit}')
  [ -n "$have" ] && [ "$have" != "$v" ] && { echo "[$(ts)] DRIFT $k=$have -> $v"; pmset -a "$k" "$v"; drift=1; }
done < <(grep -v '^firewall' "$POLICY")
fw_want=$(awk '$1=="firewall"{print $2;exit}' "$POLICY")
fw_have=$("$FW" --getglobalstate 2>/dev/null | grep -q 'State = 1' && echo on || echo off)
[ "$fw_have" = "$fw_want" ] || { echo "[$(ts)] DRIFT firewall=$fw_have -> $fw_want"; "$FW" --setglobalstate "$fw_want" >/dev/null 2>&1; drift=1; }
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
[ -x /usr/local/mesh/bin/mesh-fabric-init.sh ] && /usr/local/mesh/bin/mesh-fabric-init.sh
if [ -x /usr/local/mesh/bin/mesh-proxy-announce.sh ] && ! /usr/local/mesh/bin/mesh-proxy-announce.sh; then
  echo "[$(ts)] proxy set changed -> restarting router"
  launchctl kickstart -k system/io.mesh.router >/dev/null 2>&1; drift=1
fi
pgrep -qf /usr/local/mesh/bin/babeld || {
  echo "[$(ts)] DRIFT babeld down"; launchctl kickstart -k system/io.mesh.router >/dev/null 2>&1; drift=1; }
[ "$(rdma_ctl status 2>&1)" = enabled ] || {
  echo "[$(ts)] ALARM rdma disabled, physical Recovery OS visit required"; drift=1; }
[ "$drift" -eq 0 ] || echo "[$(ts)] pass complete"
exit 0
