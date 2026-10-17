#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
LOG=/usr/local/mesh/log/fabric.log
[ -d /usr/local/mesh/log ] && exec >>"$LOG" 2>&1
ts(){ date '+%F %T'; }
drift=0
ports=$(ibv_devices 2>/dev/null | awk 'NR>2 && $1!=""{sub(/^rdma_/,"",$1);print $1}')
[ -n "$ports" ] && networksetup -listallnetworkservices 2>/dev/null | grep -q '^Thunderbolt Bridge$' && {
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
now=$(date +%s)
rids=$(ioreg -l -w0 -r -c AppleThunderboltRDMAInterface 2>/dev/null | awk '/\+-o rdma_/{n=substr($2,6)} /"RID" =/{print n, $NF}')
peers=" $(ioreg -l -w0 -r -c AppleThunderboltRDMAPeerInterface 2>/dev/null | awk '/"RID" =/{printf "%s ", $NF}')"
: > /var/run/mesh-fabric.$$
for p in $ports; do
  f=/var/run/mesh-link.$p
  link=$(ifconfig "$p" 2>/dev/null | awk '/status:/{print $2}')
  rid=$(echo "$rids" | awk -v p="$p" '$1==p{print $2}')
  case "$peers" in *" ${rid:-none} "*) peer=present ;; *) peer=absent ;; esac
  echo "port=$p link=${link:-unknown} rdma_peer=$peer" >> /var/run/mesh-fabric.$$
  if [ "$link" != active ] || [ "$peer" = present ]; then
    [ -f "$f" ] && { read -r t0 n tb < "$f"
      [ "$peer" = present ] && { echo "[$(ts)] RECOVERED $p rdma_peer=present after $((now - ${t0:-$now})) s, ${n:-0} bounces"
        logger -p user.warning -t io.mesh.fabric "$p RDMA peer present again after $((now - ${t0:-$now})) s, ${n:-0} interface bounces"; }
      rm -f "$f"; }
    continue
  fi
  [ -f "$f" ] || { echo "$now 0 0" > "$f"; echo "[$(ts)] DOWN $p link=active rdma_peer=absent"; drift=1; continue; }
  read -r t0 n tb < "$f"; t0=${t0:-$now} n=${n:-0} tb=${tb:-0}
  w=$n; [ "$w" -gt 5 ] && w=5; wait=$((60 << w)); [ "$wait" -gt 1800 ] && wait=1800
  [ $((now - t0)) -ge 60 ] && [ $((now - tb)) -ge "$wait" ] || continue
  n=$((n + 1)); echo "$t0 $n $now" > "$f"
  echo "[$(ts)] BOUNCE $p ($n) link=active rdma_peer=absent for $((now - t0)) s"
  logger -p user.warning -t io.mesh.fabric "$p cable link up but no RDMA peer for $((now - t0)) s: interface bounced ($n)"
  ifconfig "$p" down; sleep 2; ifconfig "$p" up; drift=1
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
# a router's end-system model is the weak one (RFC 1122 3.3.4.2): its fabric identity, the ULA on lo0
# (mesh-router-init.sh), is the source of what it sends along babeld's routes and is accepted on whichever cable it
# arrives; macOS's strong model chose a cable's link-local as the source and refused a datagram bound to the ULA (no
# route: scoped to lo0), so the ULAs carried nothing (metal-microbench docs/measurement.md 2026-10-16)
for k in check_interface select_src_strong_end; do
  [ "$(sysctl -n net.inet6.ip6.$k 2>/dev/null)" = 0 ] || {
    echo "[$(ts)] DRIFT ip6 $k on"
    sysctl -w net.inet6.ip6.$k=0 >/dev/null 2>&1; drift=1; }
done
mv -f /var/run/mesh-fabric.$$ /var/run/mesh-fabric; chmod 644 /var/run/mesh-fabric
[ "$drift" -eq 0 ] || echo "[$(ts)] fabric converged"
exit 0
