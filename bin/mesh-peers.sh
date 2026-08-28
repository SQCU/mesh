#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
PREFIX=fd6d:6573:68
D=${MESH_DEADLINE:-2}
self=$(ifconfig lo0 2>/dev/null | awk -v p="$PREFIX" '$1=="inet6" && $2 ~ "^"p":"{print $2;exit}')
nodes=$(netstat -rn -f inet6 2>/dev/null | awk -v p="$PREFIX" '$1 ~ "^"p":" {split($1,f,"%"); print f[1]" "$NF}' | sort -u)
IFS='
'
for e in $nodes; do
  a=${e%% *}; via=${e##* }
  [ "$a" = "$self" ] && via=self
  printf 'node %s via=%s\n' "$a" "$via"
done
for e in $nodes; do
  a=${e%% *}
  { i=$(printf x | nc -G "$D" -w "$D" "$a" 8099 2>/dev/null | tr '\n' ' ')
    [ -n "$i" ] && printf 'info %s %s\n' "$a" "$i"; } &
done
wait
