#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
D=${MESH_DEADLINE:-3}
P=${MESH_PROBE_DEADLINE:-4}
norm(){ printf '%s' "$1" | tr 'A-Z' 'a-z' | sed 's/%<0>$//' \
        | awk -F: 'NF>2{o="";for(i=1;i<=NF;i++){g=$i;sub(/^0+/,"",g);if(g=="")g="0";o=o (i>1?":":"") g}print o;next}{print}'; }
ports=$(ibv_devices 2>/dev/null | awk 'NR>2 && $1!=""{sub(/^rdma_/,"",$1);print $1}')
me=$(scutil --get LocalHostName 2>/dev/null)
B=$(mktemp); trap 'rm -f "$B" "$B".*' EXIT
dns-sd -B _meshnode._tcp local > "$B" 2>&1 &
sleep "$D"; kill %1 2>/dev/null; wait 2>/dev/null
for n in $(awk '$2=="Add"{print $NF}' "$B" | sort -u); do
  [ -n "$n" ] || continue
  [ "$n" = "$me" ] && printf 'node %s self\n' "$n" || printf 'node %s peer\n' "$n"
  (
    dns-sd -t "$D" -G v4v6 "$n.local" 2>/dev/null | awk '$2=="Add"{print $6}' | sort -u > "$B.$n"
    best=""; bestk=""
    while read -r raw; do
      a=$(norm "$raw")
      case "$a" in
        0.0.0.0|127.0.0.1|::1|fe80::1%lo0|"") continue ;;
        10.*|192.168.*|172.1[6-9].*|172.2[0-9].*|172.3[01].*) k=lan ;;
        169.254.*) k=fabric-v4ll ;;
        fd6d:6573:68:*) k=fabric-routed ;;
        2*:*) k=lan-v6 ;;
        fe80:*%*) i=${a##*%}; k=skip
                  for p in $ports; do [ "$i" = "$p" ] && k=fabric-adjacent; done ;;
        *) k=skip ;;
      esac
      [ "$k" = skip ] && continue
      printf 'path %s kind=%s addr=%s\n' "$n" "$k" "$a"
      case "$k" in
        lan)             [ "$bestk" = lan ] || { best=$a; bestk=lan; } ;;
        fabric-adjacent) [ -n "$best" ] || { best=$a; bestk=$k; } ;;
        lan-v6)          [ -n "$best" ] || { best=$a; bestk=$k; } ;;
      esac
    done < "$B.$n"
    [ -n "$best" ] || exit 0
    t=$(mktemp)
    { printf x | nc -G "$P" -w "$P" "$best" 8099 2>/dev/null | tr '\n' ' ' > "$t.a"; } &
    { printf x | nc -G "$P" -w "$P" "$best" 8100 2>/dev/null | tr '\n' ' ' > "$t.b"; } &
    wait
    i=$(cat "$t.a" 2>/dev/null); [ -n "$i" ] || i=$(cat "$t.b" 2>/dev/null)
    rm -f "$t" "$t".*
    case "$i" in
      "") : ;;
      "mesh1 "*)
        want=${i#mesh1 }; want=${want%% *}
        body=${i#mesh1 * }
        if [ "${#body}" -ge "$want" ]; then printf 'info %s via=%s %s\n' "$n" "$best" "$body"
        else printf 'partial %s via=%s got=%s want=%s\n' "$n" "$best" "${#body}" "$want"; fi ;;
      *"end "*|*"end") printf 'info %s via=%s %s\n' "$n" "$best" "$i" ;;
      *) printf 'partial %s via=%s bytes=%s unframed\n' "$n" "$best" "${#i}" ;;
    esac
  ) &
done
wait
