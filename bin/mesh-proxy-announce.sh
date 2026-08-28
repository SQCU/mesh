#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
PREFIX=fd6d:6573:68
D=${MESH_DEADLINE:-2}
STATE=/usr/local/mesh/proxied
LOG=/usr/local/mesh/log/proxy.log
[ -d /usr/local/mesh/log ] && exec 2>>"$LOG"
ts(){ date '+%F %T'; }
probe(){ t=$(mktemp)
  { printf x | nc -G "$D" -w "$D" "$1" 8099 2>/dev/null > "$t.a"; } &
  { printf x | nc -G "$D" -w "$D" "$1" 8100 2>/dev/null > "$t.b"; } &
  wait
  i=$(cat "$t.a" 2>/dev/null); [ -s "$t.a" ] || i=$(cat "$t.b" 2>/dev/null)
  rm -f "$t" "$t.a" "$t.b"; printf '%s' "$i"; }
self=$(ifconfig lo0 2>/dev/null | awk -v p="$PREFIX" '$1=="inet6" && $2 ~ "^"p":"{print $2;exit}')
ports=$(ibv_devices 2>/dev/null | awk 'NR>2 && $1!=""{sub(/^rdma_/,"",$1);print $1}')
mine=$(ifconfig 2>/dev/null | awk '$1=="inet6"{split($2,x,"%"); print x[1]}')
want=$(mktemp)
for p in $ports; do
  for n in $(ndp -an 2>/dev/null | awk -v i="$p" '$3==i && $1 ~ /^fe80::/ && $2 != "(incomplete)"{print $1}'); do
    echo "$mine" | grep -qx "${n%%\%*}" && continue
    ula=$(probe "$n" | awk -F= '/^ula=/{print $2;exit}')
    case "$ula" in "$PREFIX":*) ;; *) continue ;; esac
    [ "$ula" = "$self" ] && continue
    netstat -rn -f inet6 2>/dev/null | awk '{print $1}' | grep -q "^${ula}$" && continue
    printf '%s %s\n' "$ula" "$n" >> "$want"
  done
done
sort -u "$want" -o "$want"
touch "$STATE"
while read -r ula nh; do
  [ -n "$ula" ] || continue
  grep -q "^$ula " "$STATE" || { echo "[$(ts)] proxy-announce $ula via $nh" >&2
    route -n add -inet6 "$ula/128" "$nh" >/dev/null 2>&1; }
done < "$want"
while read -r ula nh; do
  [ -n "$ula" ] || continue
  grep -q "^$ula " "$want" || { echo "[$(ts)] withdraw $ula" >&2
    route -n delete -inet6 "$ula/128" >/dev/null 2>&1; }
done < "$STATE"
if ! cmp -s "$want" "$STATE"; then cp "$want" "$STATE"; rm -f "$want"; exit 1; fi
rm -f "$want"; exit 0
