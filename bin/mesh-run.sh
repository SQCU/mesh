#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
PREFIX=fd6d:6573:68
D=${MESH_DEADLINE:-2}
U="${MESH_SSH_USER:-$(id -un)}"
[ "$#" -ge 2 ] || { echo "usage: mesh-run [<name>|all|others] <command...>" >&2; exit 1; }
target=$1; shift
probe(){ i=$(printf x | nc -G "$D" -w "$D" "$1" 8099 2>/dev/null | tr '\n' ' ')
         [ -n "$i" ] || i=$(printf x | nc -G "$D" -w "$D" "$1" 8100 2>/dev/null | tr '\n' ' ')
         printf '%s' "$i"; }
f(){ printf '%s' "$2" | tr ' ' '\n' | sed -n "s/^$1=//p" | head -1; }
self=$(ifconfig lo0 2>/dev/null | awk -v p="$PREFIX" '$1=="inet6" && $2 ~ "^"p":"{print $2;exit}')
ports=$(ibv_devices 2>/dev/null | awk 'NR>2 && $1!=""{sub(/^rdma_/,"",$1);print $1}')
mine=$(ifconfig 2>/dev/null | awk '$1=="inet6"{split($2,x,"%"); print x[1]}')
cand=$(netstat -rn -f inet6 2>/dev/null | awk -v p="$PREFIX" '$1 ~ "^"p":" {split($1,x,"%"); print x[1]}')
for p in $ports; do
  for n in $(ndp -an 2>/dev/null | awk -v i="$p" '$3==i && $1 ~ /^fe80::/ && $2 != "(incomplete)"{print $1}'); do
    echo "$mine" | grep -qx "${n%%\%*}" || cand="$cand
$n"
  done
done
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT; k=0
IFS='
'
for a in $cand; do
  [ -n "$a" ] || continue
  k=$((k+1))
  { i=$(probe "$a"); [ -n "$i" ] && printf '%s\t%s\t%s\n' "$(f ula "$i")" "$(f name "$i")" "$a" > "$T/$k"; } &
done
wait
for row in $(cat "$T"/* 2>/dev/null | sort -u -k1,1); do
  ula=${row%%	*}; rest=${row#*	}; name=${rest%%	*}; addr=${rest#*	}
  case "$target" in
    all) ;;
    others) [ "$ula" = "$self" ] && continue ;;
    *) [ "$name" = "$target" ] || [ "$ula" = "$target" ] || continue ;;
  esac
  tag=${name:-$addr}
  if [ "$ula" = "$self" ]; then bash -c "$*" 2>&1 | sed "s|^|$tag |" &
  else ssh -o BatchMode=yes -o ConnectTimeout="$D" -o StrictHostKeyChecking=accept-new \
         "$U@$addr" "$@" 2>&1 | sed "s|^|$tag |" & fi
done
wait
