#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
PREFIX=fd6d:6573:68
self=$(ifconfig lo0 2>/dev/null | awk -v p="$PREFIX" '$1=="inet6" && $2 ~ "^"p":"{print $2;exit}')
printf '\n  %-18s %-12s %-5s %-6s %-9s %-13s %-38s %s\n' NODE MODEL CORES MEM RDMA REV ADDRESS VIA
n=0
for e in $(netstat -rn -f inet6 2>/dev/null | awk -v p="$PREFIX" '$1 ~ "^"p":" {split($1,f,"%"); print f[1]"|"$NF}' | sort -u); do
  a=${e%%|*}; via=${e##*|}; n=$((n+1))
  info=$(printf 'x' | nc -w 2 "$a" 8099 2>/dev/null)
  g(){ printf '%s' "$info" | awk -F= -v k="$1" '$1==k{print $2;exit}'; }
  name=$(g name); model=$(g model); cores=$(g cores); mem=$(g memgb); rd=$(g rdma); rev=$(g rev)
  [ -n "$info" ] || { name="?"; rd="unreachable"; }
  [ "$a" = "$self" ] && via=self
  printf '  %-18s %-12s %-5s %-6s %-9s %-13s %-38s %s\n' \
    "${name:-?}" "${model:-?}" "${cores:-?}" "${mem:+${mem}G}" "${rd:-?}" "${rev:-?}" "$a" "$via"
done
[ "$n" -eq 0 ] && printf '  \033[31mnothing in the routing table -- babeld down, or no fabric cable\033[0m\n'
printf '\n  %s node(s)\n\n' "$n"
