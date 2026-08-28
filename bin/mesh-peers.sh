#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
PREFIX=fd6d:6573:68
self=$(ifconfig lo0 2>/dev/null | awk -v p="$PREFIX" '$1=="inet6" && $2 ~ "^"p":"{print $2}')
printf '\n  %-44s %-8s %s\n' "NODE" "STATE" "HOPS"
n=0
for a in $(netstat -rn -f inet6 2>/dev/null | awk -v p="$PREFIX" '$1 ~ "^"p":" {print $1}' | cut -d% -f1 | sort -u); do
  n=$((n+1))
  if [ "$a" = "$self" ]; then printf '  %-44s %-8s %s\n' "$a" "self" "-"; continue; fi
  if ping6 -c1 -W 1500 "$a" >/dev/null 2>&1; then st=up; else st=$'\033[33mrouted, silent\033[0m'; fi
  hops=$(netstat -rn -f inet6 2>/dev/null | awk -v a="$a" '$1==a{print $NF; exit}')
  printf '  %-44s %-8b %s\n' "$a" "$st" "${hops:-?}"
done
[ "$n" -eq 0 ] && printf '  \033[31mno nodes in the routing table -- babeld down, or no fabric cable\033[0m\n'
echo
