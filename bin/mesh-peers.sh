#!/bin/bash
# List every node currently announcing itself on the fabric.
#
# ABSENCE FROM THIS LIST IS THE ALARM -- but only for appliance nodes. A portable
# node that is asleep in a bag is expected to be missing, and conflating the two
# would make the alarm meaningless within a week. The profile is carried in each
# node's announcement so the distinction survives without a central registry.
#
#   mesh-peers [seconds]
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
SECS="${1:-4}"
TMP=$(mktemp); trap 'rm -f "$TMP" "$TMP".*' EXIT

dns-sd -B _meshnode._tcp local >"$TMP" 2>&1 &
P=$!
sleep "$SECS"
kill "$P" 2>/dev/null; wait "$P" 2>/dev/null

# Prefer the fabric address: link-local IPv4 first, then any other v4, then v6.
# Taking dns-sd's first answer picks internal interfaces like anpi2 and misleads.
best_addr(){
  dns-sd -t 2 -G v4v6 "$1.local" 2>/dev/null | awk '$2=="Add"{print $6}' > "$TMP.a"
  grep -m1 '^169\.254\.'          "$TMP.a" && return
  grep -m1 -E '^[0-9]+\.[0-9]+\.' "$TMP.a" && return
  grep -m1 -v '^$'                "$TMP.a"
}
profile_of(){
  dns-sd -t 2 -L "$1" _meshnode._tcp local 2>/dev/null \
    | tr ' ' '\n' | sed -n 's/^profile=//p' | head -1
}

printf '\n  %-24s %-10s %-22s %s\n' "NODE" "PROFILE" "ADDRESS" "STATE"
n=0
while read -r name; do
  [ -n "$name" ] || continue
  n=$((n+1))
  addr=$(best_addr "$name" | head -1)
  prof=$(profile_of "$name"); prof=${prof:-unknown}
  if [ -n "$addr" ]; then st="up"; else st=$'\033[33mannouncing, unresolvable\033[0m'; fi
  printf '  %-24s %-10s %-22s %b\n' "$name" "$prof" "${addr:-?}" "$st"
done < <(awk '$2=="Add"{print $NF}' "$TMP" | sort -u)

if [ "$n" -eq 0 ]; then
  printf '  \033[31mno nodes answered in %ss -- the fabric is dark\033[0m\n' "$SECS"
fi
echo
