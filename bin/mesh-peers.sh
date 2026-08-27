#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
SECS="${1:-4}"
TMP=$(mktemp); trap 'rm -f "$TMP" "$TMP".a' EXIT
dns-sd -B _meshnode._tcp local >"$TMP" 2>&1 &
P=$!
sleep "$SECS"; kill "$P" 2>/dev/null; wait "$P" 2>/dev/null
addr_of(){
  dns-sd -t 2 -G v4v6 "$1.local" 2>/dev/null | awk '$2=="Add"{print $6}' >"$TMP.a"
  grep -m1 '^169\.254\.' "$TMP.a" || grep -m1 -E '^[0-9]+\.[0-9]+\.' "$TMP.a" || grep -m1 . "$TMP.a"
}
printf '\n  %-26s %-40s %s\n' NODE ADDRESS STATE
n=0
while read -r name; do
  [ -n "$name" ] || continue
  n=$((n+1)); a=$(addr_of "$name" | head -1)
  printf '  %-26s %-40s %s\n' "$name" "${a:-?}" "$([ -n "$a" ] && echo up || echo UNRESOLVABLE)"
done < <(awk '$2=="Add"{print $NF}' "$TMP" | sort -u)
[ "$n" -gt 0 ] || printf '  \033[31mno nodes answered in %ss, the fabric is dark\033[0m\n' "$SECS"
echo
