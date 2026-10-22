#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
LOG=/usr/local/mesh/log/fabric.log
[ -d /usr/local/mesh/log ] && exec >>"$LOG" 2>&1
ts(){ date '+%F %T'; }
drift=0
rdma=$("$(dirname "$0")/mesh-rdma.sh")
ports=$(printf '%s\n' "$rdma" | awk 'NF{print $1}')
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
: > /var/run/mesh-fabric.$$
for p in $ports; do
  f=/var/run/mesh-link.$p
  link=$(ifconfig "$p" 2>/dev/null | awk '/status:/{print $2}')
  peer=$(printf '%s\n' "$rdma" | awk -v p="$p" '$1==p{sub(/^rdma_peer=/,"",$4); print $4}')
  echo "port=$p link=${link:-unknown} rdma_peer=$peer" >> /var/run/mesh-fabric.$$
  t0=
  [ -f "$f" ] && read -r t0 _ < "$f"
  if [ "$link" != active ] || [ "$peer" = present ]; then
    [ -f "$f" ] && [ "$peer" = present ] && { echo "[$(ts)] RECOVERED $p rdma_peer=present after $((now - ${t0:-$now})) s"
      logger -p user.warning -t io.mesh.fabric "$p RDMA peer present again after $((now - ${t0:-$now})) s"; }
    rm -f "$f"; continue
  fi
  # design/rubrics.md M3 (10-20): a port is never brought down here; one taken down under the bridge's open device
  # terminates the device beneath its objects
  t0=${t0:-$now}; echo "$t0" > "$f"; drift=1
  echo "[$(ts)] HELD $p link=active rdma_peer=absent for $((now - t0)) s"
  logger -p user.warning -t io.mesh.fabric "$p cable link up but no RDMA peer for $((now - t0)) s"
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
# a router's end-system model is the weak one (RFC 1122 3.3.4.2): its fabric identity, the ULA on each fabric port
# (mesh-router-init.sh), is the source of what it sends along babeld's routes and is accepted on whichever cable it
# arrives; macOS's strong model had chosen a cable's link-local as the source (metal-microbench docs/measurement.md
# 2026-10-16)
for k in check_interface select_src_strong_end; do
  [ "$(sysctl -n net.inet6.ip6.$k 2>/dev/null)" = 0 ] || {
    echo "[$(ts)] DRIFT ip6 $k on"
    sysctl -w net.inet6.ip6.$k=0 >/dev/null 2>&1; drift=1; }
done
mv -f /var/run/mesh-fabric.$$ /var/run/mesh-fabric; chmod 644 /var/run/mesh-fabric
[ "$drift" -eq 0 ] || echo "[$(ts)] fabric converged"
exit 0
