#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
MESH_ROOT=/usr/local/mesh
PREFIX=fd6d:6573:68
LOG=$MESH_ROOT/log/router.log
[ -d "$MESH_ROOT/log" ] && exec >>"$LOG" 2>&1
ts(){ date '+%F %T'; }
ports=$(ibv_devices 2>/dev/null | awk 'NR>2 && $1!=""{sub(/^rdma_/,"",$1);print $1}')
[ -n "$ports" ] || exit 0
uuid=$(ioreg -rd1 -c IOPlatformExpertDevice | awk -F'"' '/IOPlatformUUID/{print $4}')
h=$(printf '%s' "$uuid" | shasum -a 256 | cut -c1-20)
g(){ printf '%x' $(( 0x${1} | 0x1000 )); }
ULA="$PREFIX:$(g ${h:0:4}):$(g ${h:4:4}):$(g ${h:8:4}):$(g ${h:12:4}):$(g ${h:16:4})"
for old in $(ifconfig lo0 | awk -v p="$PREFIX" '$1=="inet6" && $2 ~ "^"p":"{print $2}'); do
  [ "$old" = "$ULA" ] && continue
  echo "[$(ts)] stale identity $old -> removed"
  ifconfig lo0 inet6 "$old" -alias >/dev/null 2>&1
done
ifconfig lo0 | grep -q "inet6 ${ULA} " || {
  echo "[$(ts)] identity $ULA -> lo0"
  ifconfig lo0 inet6 "$ULA" prefixlen 128 alias >/dev/null 2>&1; }
CONF=$MESH_ROOT/babeld.conf
{ for p in $ports; do echo "interface $p"; done
  echo "redistribute local ip ${ULA}/128 eq 128 allow"
  echo "redistribute deny"; } > "$CONF"
pgrep -qf "$MESH_ROOT/bin/babeld" || echo "[$(ts)] starting babeld on $(echo $ports)"
exec "$MESH_ROOT/bin/babeld" -c "$CONF" -S "$MESH_ROOT/babel-state" -I "" -L "$LOG"
