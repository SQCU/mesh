#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
PREFIX=fd6d:6573:68
D=${MESH_DEADLINE:-2}
probe(){ t=$(mktemp)
  { printf x | nc -G "$D" -w "$D" "$1" 8099 2>/dev/null | tr '
' ' ' > "$t.a"; } &
  { printf x | nc -G "$D" -w "$D" "$1" 8100 2>/dev/null | tr '
' ' ' > "$t.b"; } &
  wait
  i=$(cat "$t.a" 2>/dev/null); [ -n "$i" ] || i=$(cat "$t.b" 2>/dev/null)
  rm -f "$t" "$t.a" "$t.b"; printf '%s' "$i"; }
self=$(ifconfig lo0 2>/dev/null | awk -v p="$PREFIX" '$1=="inet6" && $2 ~ "^"p":"{print $2;exit}')
ports=$(ibv_devices 2>/dev/null | awk 'NR>2 && $1!=""{sub(/^rdma_/,"",$1);print $1}')
nodes=$(netstat -rn -f inet6 2>/dev/null | awk -v p="$PREFIX" '$1 ~ "^"p":" {split($1,f,"%"); print f[1]" "$NF}' | sort -u)
mine=$(ifconfig 2>/dev/null | awk '$1=="inet6"{split($2,f,"%"); print f[1]}')
IFS='
'
for e in $nodes; do
  a=${e%% *}; via=${e##* }
  [ "$a" = "$self" ] && via=self
  printf 'node %s via=%s\n' "$a" "$via"
done
neigh=""
for p in $ports; do
  for n in $(ndp -an 2>/dev/null | awk -v i="$p" '$3==i && $1 ~ /^fe80::/ && $2 != "(incomplete)"{print $1}'); do
    b=${n%%\%*}
    echo "$mine" | grep -qx "$b" && continue
    printf 'neigh %s on=%s\n' "$n" "$p"
    neigh="$neigh$n
"
  done
done
for e in $nodes; do a=${e%% *}; { i=$(probe "$a"); [ -n "$i" ] && printf 'info %s %s\n' "$a" "$i"; } & done
for n in $neigh; do [ -n "$n" ] && { i=$(probe "$n"); [ -n "$i" ] && printf 'info %s %s\n' "$n" "$i"; } & done
wait
