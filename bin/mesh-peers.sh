#!/bin/bash
# List every node currently announcing itself on the fabric.
# ABSENCE FROM THIS LIST IS THE ALARM. Run it from any machine on the mesh.
#   mesh-peers [seconds]
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
SECS="${1:-4}"
TMP=$(mktemp)
dns-sd -B _meshnode._tcp local >"$TMP" 2>&1 &
P=$!
sleep "$SECS"
kill "$P" 2>/dev/null; wait "$P" 2>/dev/null
printf '\n  %-26s %-34s %s\n' "NODE" "ADDRESS" "STATE"
n=0
while read -r name; do
  [ -n "$name" ] || continue
  n=$((n+1))
  addr=$(dns-sd -t 2 -G v4v6 "$name.local" 2>/dev/null | awk '$2=="Add"{print $6; exit}')
  if [ -n "$addr" ]; then st="up"; else st="ANNOUNCING BUT UNRESOLVABLE"; fi
  printf '  %-26s %-34s %s\n' "$name" "${addr:-?}" "$st"
done < <(awk '$2=="Add"{print $NF}' "$TMP" | sort -u)
rm -f "$TMP"
[ "$n" -eq 0 ] && printf '  \033[31mno nodes answered in %ss -- the fabric is dark\033[0m\n' "$SECS"
echo
