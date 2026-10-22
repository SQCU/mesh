#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
MESH_ROOT=/usr/local/mesh
PREFIX=fd6d:6573:68
LOG=$MESH_ROOT/log/router.log
[ -d "$MESH_ROOT/log" ] && exec >>"$LOG" 2>&1
ts(){ date '+%F %T'; }
ports=$("$(dirname "$0")/mesh-rdma.sh" | awk '{print $1}')
uuid=$(ioreg -rd1 -c IOPlatformExpertDevice | awk -F'"' '/IOPlatformUUID/{print $4}')
h=$(printf '%s' "$uuid" | shasum -a 256 | cut -c1-20)
g(){ printf '%x' $(( 0x${1} | 0x1000 )); }
ULA="$PREFIX:$(g ${h:0:4}):$(g ${h:4:4}):$(g ${h:8:4}):$(g ${h:12:4}):$(g ${h:16:4})"
# the node's fabric identity on each fabric port, not lo0: macOS scopes a datagram's route to the interface that holds
# its source, so a ULA on lo0 found no route along a cable (metal-microbench docs/measurement.md 2026-10-16); a stale
# identity, or this one on lo0, is removed
for i in lo0 $ports; do
  cfg=$(ifconfig "$i" 2>/dev/null)
  for old in $(printf '%s\n' "$cfg" | awk -v p="$PREFIX" '$1=="inet6" && $2 ~ "^"p":"{print $2}'); do
    [ "$old" = "$ULA" ] && [ "$i" != lo0 ] && continue
    echo "[$(ts)] identity $old on $i -> removed"
    ifconfig "$i" inet6 "$old" -alias >/dev/null 2>&1
  done
  [ "$i" = lo0 ] || printf '%s\n' "$cfg" | grep -q "inet6 ${ULA} " || {
    echo "[$(ts)] identity $ULA -> $i"
    ifconfig "$i" inet6 "$ULA" prefixlen 128 alias >/dev/null 2>&1; }
done
CONF=$MESH_ROOT/babeld.conf
{ for p in $ports; do echo "interface $p type wired"; done
  echo "redistribute local ip ${ULA}/128 eq 128 allow"
  echo "redistribute deny"; } > "$CONF"
pgrep -qf "$MESH_ROOT/bin/babeld" || echo "[$(ts)] starting babeld on $(echo $ports)"
exec "$MESH_ROOT/bin/babeld" -c "$CONF" -S "$MESH_ROOT/babel-state" -I "" -L "$LOG"
