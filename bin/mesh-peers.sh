#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
PREFIX=fd6d:6573:68
printf '\n  %-44s %-10s %s\n' "NODE" "STATE" "VIA"
n=0; peers=0
while read -r a via; do
  [ -n "$a" ] || continue
  n=$((n+1))
  if [ "$via" = lo0 ]; then printf '  %-44s %-10s %s\n' "$a" "self" "-"
  else peers=$((peers+1)); printf '  %-44s \033[32m%-10s\033[0m %s\n' "$a" "reachable" "$via"; fi
done < <(netstat -rn -f inet6 2>/dev/null | awk -v p="$PREFIX" '$1 ~ "^"p":" {split($1,f,"%"); print f[1], $NF}' | sort -u)
[ "$n" -eq 0 ] && printf '  \033[31mnothing in the routing table -- babeld down, or no fabric cable\033[0m\n'
printf '\n  %s node(s), %s peer(s)\n\n' "$n" "$peers"
