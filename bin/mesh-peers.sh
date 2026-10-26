#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
s=$0; [ -L "$s" ] && s=$(readlink "$s"); here=${s%/*}
norm(){ printf '%s' "$1" | tr 'A-Z' 'a-z' | sed 's/%<0>$//' \
        | awk -F: 'NF>2{o="";for(i=1;i<=NF;i++){g=$i;sub(/^0+/,"",g);if(g=="")g="0";o=o (i>1?":":"") g}print o;next}{print}'; }
fnv(){ local h=${1//-/} n=0 i; for ((i = 0; i < 32; i += 2)); do n=$(( (n * 0x01000193 & 0xffffffff) ^ 0x${h:i:2} )); done
       n=$((n & 0x7fffffff)); echo $(( n ? n : 1 )); }
ports=$("$here/mesh-rdma.sh" | awk '{print $1}')
me=$(fnv "$(ioreg -a -rd1 -c IOPlatformExpertDevice | plutil -extract 0.IOPlatformUUID raw -o - -)")
T=$(mktemp -d); : > "$T/named" "$T/pids" "$T/names"
trap 'kill $(cat "$T/pids") 2>/dev/null; rm -rf "$T"' EXIT; trap 'exit 1' INT TERM
jq -r '.nodes // {} | to_entries[] | select((.key | test("^[0-9A-Fa-f]{8}(-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$")) and .value.name) | "\(.key) \(.value.name)"' \
  "$here/../etc/mesh-nodes.json" 2>/dev/null | while read -r id nm; do printf '%s %s\n' "$(fnv "$id")" "$nm"; done > "$T/names"
named(){ grep -qxF "$1" "$T/named" && return 0; printf '%s\n' "$1" >> "$T/named"
  nm=$(awk -v n="$1" '$1 == n {$1 = ""; sub(/^ /, ""); print; exit}' "$T/names")
  printf 'node %s %s%s\n' "$1" "$([ "$1" = "$me" ] && echo self || echo peer)" "${nm:+ name=$nm}"; }
named "$me"
stat=$("$here/mesh-stat" 2>/dev/null | tail -1)
i=0
while n=$(printf '%s' "$stat" | plutil -extract "nodes.$i.node" raw -o - - 2>/dev/null); do
  r=nodes.$i; i=$((i+1))
  named "$n"
  printf 'info %s via=bridge model=%s memory=%s region=%s\n' "$n" \
    "$(printf '%s' "$stat" | plutil -extract "$r.model" raw -o - - 2>/dev/null)" \
    "$(printf '%s' "$stat" | plutil -extract "$r.memory" raw -o - - 2>/dev/null)" \
    "$(printf '%s' "$stat" | plutil -extract "$r.region" raw -o - - 2>/dev/null)"
done
addresses(){
  rm -f "$T/$2.g"; mkfifo "$T/$2.g"; dns-sd -G v4v6 "$1" > "$T/$2.g" 2>/dev/null & echo $! >> "$T/pids"; echo $! >> "$T/$2.pids"
  while read -r _ ar _ _ _ raw _; do
    [ "$ar" = Add ] || continue
    a=$(norm "$raw")
    case "$a" in
      0.0.0.0|127.0.0.1|::1|fe80::1%lo0|"") continue ;;
      10.*|192.168.*|172.1[6-9].*|172.2[0-9].*|172.3[01].*) k=lan ;;
      169.254.*) k=fabric-v4ll ;;
      fd6d:6573:68:*) k=fabric-routed ;;
      2*:*) k=lan-v6 ;;
      fe80:*%*) i=${a##*%}; k=; for p in $ports; do [ "$i" = "$p" ] && k=fabric-adjacent; done; [ -n "$k" ] || continue ;;
      *) continue ;;
    esac
    printf 'path %s kind=%s addr=%s\n' "$3" "$k" "$a"
  done < "$T/$2.g"
}
follow(){
  rm -f "$T/$2.l"; mkfifo "$T/$2.l"; dns-sd -L "$1" _meshnode._tcp local > "$T/$2.l" 2>/dev/null & echo $! >> "$T/pids"; echo $! >> "$T/$2.pids"
  target= id=
  while IFS= read -r line; do
    case "$line" in
      *" can be reached at "*)
        t=${line#* can be reached at }; t=${t%%:*}; t=${t%.}
        [ -n "$t" ] && [ "$t" != "$target" ] || continue
        target=$t; [ -z "$id" ] || addresses "$target" "$2" "$id" & ;;
      " "*=*)
        n=$(printf '%s\n' $line | sed -n 's/^node=//p' | head -1)
        n=${n:-unidentified-$2}
        [ "$n" = "$id" ] || { id=$n; printf '%s\n' "$id" > "$T/$2.id"; named "$id"; [ -z "$target" ] || addresses "$target" "$2" "$id" & }
        printf 'info %s via=mdns%s\n' "$id" "$line" ;;
    esac
  done < "$T/$2.l"
}
mkfifo "$T/browse"; dns-sd -B _meshnode._tcp local > "$T/browse" 2>/dev/null & echo $! >> "$T/pids"
while read -r _ ar _ _ _ _ inst; do
  case "$ar" in Add|Rmv) ;; *) continue ;; esac
  k=$(printf '%s' "$inst" | shasum | cut -c1-16); c=$(cat "$T/$k.count" 2>/dev/null || echo 0)
  if [ "$ar" = Add ]; then
    echo $((c+1)) > "$T/$k.count"; [ "$c" = 0 ] && follow "$inst" "$k" &
  else
    [ "$c" -gt 1 ] && { echo $((c-1)) > "$T/$k.count"; continue; }
    rm -f "$T/$k.count"; kill $(cat "$T/$k.pids" 2>/dev/null) 2>/dev/null
    printf 'gone %s\n' "$(cat "$T/$k.id" 2>/dev/null || echo "unidentified-$k")"
    rm -f "$T/$k.pids" "$T/$k.l" "$T/$k.g" "$T/$k.id"
  fi
done < "$T/browse"
