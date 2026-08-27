#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
LOG=/usr/local/mesh/log/fabric.log
[ -d /usr/local/mesh/log ] && exec >>"$LOG" 2>&1
ts(){ date '+%F %T'; }
drift=0
ports=$(ibv_devices 2>/dev/null | awk 'NR>2 && $1!=""{sub(/^rdma_/,"",$1);print $1}')
[ -n "$ports" ] || exit 0
networksetup -listallnetworkservices 2>/dev/null | grep -q '^Thunderbolt Bridge$' && {
  echo "[$(ts)] DRIFT tb bridge service enabled"
  networksetup -setnetworkserviceenabled "Thunderbolt Bridge" off >/dev/null 2>&1; drift=1; }
for br in $(ifconfig -l | tr ' ' '\n' | grep '^bridge'); do
  mem=$(ifconfig "$br" 2>/dev/null | awk '/member:/{print $2}')
  [ -n "$mem" ] || continue
  for m in $mem; do echo "$ports" | grep -qx "$m" || continue 2; done
  echo "[$(ts)] DRIFT $br bridges fabric ports $(echo $mem)"
  for m in $mem; do ifconfig "$br" deletem "$m" >/dev/null 2>&1; done
  ifconfig "$br" destroy >/dev/null 2>&1; drift=1
done
for p in $ports; do
  mac=$(ifconfig "$p" 2>/dev/null | awk '/ether/{print $2}')
  [ -n "$mac" ] || continue
  set -- $(echo "$mac" | tr ':' ' ')
  ll="fe80::$(printf '%02x' $((0x$1 ^ 0x02)))${2}:${3}ff:fe${4}:${5}${6}"
  ifconfig "$p" | grep -q UP || { echo "[$(ts)] DRIFT $p down"; ifconfig "$p" up; drift=1; }
  ifconfig "$p" 2>/dev/null | grep -qi "inet6 ${ll}" || {
    echo "[$(ts)] DRIFT $p missing $ll"
    ifconfig "$p" inet6 "$ll" prefixlen 64 >/dev/null 2>&1; drift=1; }
done
[ "$(sysctl -n net.inet6.ip6.forwarding 2>/dev/null)" = 1 ] || {
  echo "[$(ts)] DRIFT ip6 forwarding off"
  sysctl -w net.inet6.ip6.forwarding=1 >/dev/null 2>&1; drift=1; }
[ "$drift" -eq 0 ] || echo "[$(ts)] fabric converged"
exit 0
