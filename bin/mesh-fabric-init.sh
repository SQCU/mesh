#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
LOG=/usr/local/mesh/log/fabric.log
[ -d /usr/local/mesh/log ] && exec >>"$LOG" 2>&1
ts(){ date '+%F %T'; }
drift=0
ports=$(ioreg -r -c AppleThunderboltRDMAInterface -d 1 2>/dev/null | awk '/^\+-o rdma_/{print substr($2,6)}')
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
lan=$(route -n get default 2>/dev/null | awk '/interface:/{print $2}')
lanok=0; [ -n "$lan" ] && [ -n "$(ipconfig getifaddr "$lan" 2>/dev/null)" ] && lanok=1
active=$(for p in $ports; do [ "$(ifconfig "$p" 2>/dev/null | awk '/status:/{print $2}')" = active ] && echo "$p"; done)
: > /var/run/mesh-fabric.$$
for p in $ports; do
  f=/var/run/mesh-link.$p
  link=$(ifconfig "$p" 2>/dev/null | awk '/status:/{print $2}')
  rid=$(echo "$rids" | awk -v p="$p" '$1==p{print $2}')
  case "$peers" in *" ${rid:-none} "*) peer=present ;; *) peer=absent ;; esac
  echo "port=$p link=${link:-unknown} rdma_peer=$peer" >> /var/run/mesh-fabric.$$
  t0= n=
  [ -f "$f" ] && read -r t0 n _ < "$f"
  if [ "$link" != active ] || [ "$peer" = present ]; then
    [ -f "$f" ] && [ "$peer" = present ] && { echo "[$(ts)] RECOVERED $p rdma_peer=present after $((now - ${t0:-$now})) s, ${n:-0} bounces"
      logger -p user.warning -t io.mesh.fabric "$p RDMA peer present again after $((now - ${t0:-$now})) s, ${n:-0} interface bounces"; }
    rm -f "$f"; continue
  fi
  t0=${t0:-$now} n=${n:-0}; other=$lanok
  for q in $active; do [ "$q" = "$p" ] || other=1; done
  if [ "$other" = 0 ]; then
    echo "$t0 $n" > "$f"; drift=1
    echo "[$(ts)] HELD $p link=active rdma_peer=absent for $((now - t0)) s: its link is the node's only one, so it is not bounced"
    continue
  fi
  n=$((n + 1)); echo "$t0 $n" > "$f"
  echo "[$(ts)] BOUNCE $p ($n) link=active rdma_peer=absent for $((now - t0)) s"
  logger -p user.warning -t io.mesh.fabric "$p cable link up but no RDMA peer for $((now - t0)) s: interface bounced ($n)"
  ifconfig "$p" down; ifconfig "$p" up; drift=1
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
