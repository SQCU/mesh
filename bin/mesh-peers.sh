#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
s=$0; [ -L "$s" ] && s=$(readlink "$s"); here=${s%/*}
norm(){ printf '%s' "$1" | tr 'A-Z' 'a-z' | sed 's/%<0>$//' \
        | awk -F: 'NF>2{o="";for(i=1;i<=NF;i++){g=$i;sub(/^0+/,"",g);if(g=="")g="0";o=o (i>1?":":"") g}print o;next}{print}'; }
ports=$(ibv_devices 2>/dev/null | awk 'NR>2 && $1!=""{sub(/^rdma_/,"",$1);print $1}')
me=$(scutil --get LocalHostName 2>/dev/null)
T=$(mktemp -d); : > "$T/named" "$T/pids"
trap 'kill $(cat "$T/pids") 2>/dev/null; rm -rf "$T"' EXIT; trap 'exit 1' INT TERM
named(){ grep -qxF "$1" "$T/named" && return 0; printf '%s\n' "$1" >> "$T/named"
  [ "$1" = "$me" ] && printf 'node %s self\n' "$1" || printf 'node %s peer\n' "$1"; }
stat=$("$here/mesh-stat" 2>/dev/null | tail -1)
own=$(printf '%s' "$stat" | plutil -extract node raw -o - - 2>/dev/null)
i=0
while name=$(printf '%s' "$stat" | plutil -extract "nodes.$i.name" raw -o - - 2>/dev/null); do
  r=nodes.$i; i=$((i+1)); n=$(printf '%s' "$stat" | plutil -extract "$r.node" raw -o - - 2>/dev/null)
  [ "$n" = "$own" ] && me=$name
  named "$name"
  printf 'info %s via=bridge node=%s model=%s memory=%s window=%s\n' "$name" "$n" \
    "$(printf '%s' "$stat" | plutil -extract "$r.model" raw -o - - 2>/dev/null)" \
    "$(printf '%s' "$stat" | plutil -extract "$r.memory" raw -o - - 2>/dev/null)" \
    "$(printf '%s' "$stat" | plutil -extract "$r.window" raw -o - - 2>/dev/null)"
done
addresses(){
  mkfifo "$T/$2.g"; dns-sd -G v4v6 "$1.local" > "$T/$2.g" 2>/dev/null & echo $! >> "$T/pids"; echo $! >> "$T/$2.pids"
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
    printf 'path %s kind=%s addr=%s\n' "$1" "$k" "$a"
  done < "$T/$2.g"
}
follow(){
  mkfifo "$T/$2.l"; dns-sd -L "$1" _meshnode._tcp local > "$T/$2.l" 2>/dev/null & echo $! >> "$T/pids"; echo $! >> "$T/$2.pids"
  host=
  while IFS= read -r line; do
    case "$line" in
      *" can be reached at "*)
        h=${line#* can be reached at }; h=${h%%:*}; h=${h%.}; h=${h%.local}
        [ -n "$h" ] && [ "$h" != "$host" ] || continue
        host=$h; named "$host"; addresses "$host" "$2" & ;;
      " "*=*) [ -n "$host" ] && printf 'info %s via=mdns%s\n' "$host" "$line" ;;
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
    rm -f "$T/$k.count"; kill $(cat "$T/$k.pids" 2>/dev/null) 2>/dev/null; rm -f "$T/$k.pids" "$T/$k.l" "$T/$k.g"
    printf 'gone %s\n' "$inst"
  fi
done < "$T/browse"
