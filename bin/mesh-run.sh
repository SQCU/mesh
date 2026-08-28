#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
PREFIX=fd6d:6573:68
D=${MESH_DEADLINE:-2}
U="${MESH_SSH_USER:-$(id -un)}"
[ "$#" -ge 2 ] || { echo "usage: mesh-run [<name>|<addr>|all|others] <command...>" >&2; exit 1; }
target=$1; shift
self=$(ifconfig lo0 2>/dev/null | awk -v p="$PREFIX" '$1=="inet6" && $2 ~ "^"p":"{print $2;exit}')
nodes=$(netstat -rn -f inet6 2>/dev/null | awk -v p="$PREFIX" '$1 ~ "^"p":" {split($1,f,"%"); print f[1]}' | sort -u)
IFS='
'
for a in $nodes; do
  case "$target" in
    all) ;;
    others) [ "$a" = "$self" ] && continue ;;
    "$a") ;;
    *) n=$(printf x | nc -G "$D" -w "$D" "$a" 8099 2>/dev/null | awk -F= '/^name=/{print $2;exit}')
       [ "$n" = "$target" ] || continue ;;
  esac
  if [ "$a" = "$self" ]; then
    bash -c "$*" 2>&1 | sed "s|^|$a |" &
  else
    ssh -o BatchMode=yes -o ConnectTimeout="$D" -o StrictHostKeyChecking=accept-new \
        "$U@$a" "$@" 2>&1 | sed "s|^|$a |" &
  fi
done
wait
