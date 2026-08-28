#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
PREFIX=fd6d:6573:68
USER_NAME="${MESH_SSH_USER:-$(id -un)}"
[ "$#" -ge 1 ] || { echo "usage: mesh-run [<name>|all|others] <command...>" >&2; exit 1; }
target=$1; shift
[ "$#" -ge 1 ] || { echo "usage: mesh-run [<name>|all|others] <command...>" >&2; exit 1; }
self=$(ifconfig lo0 2>/dev/null | awk -v p="$PREFIX" '$1=="inet6" && $2 ~ "^"p":"{print $2;exit}')
addrs=$(netstat -rn -f inet6 2>/dev/null | awk -v p="$PREFIX" '$1 ~ "^"p":" {split($1,f,"%"); print f[1]}' | sort -u)
rc=0
for a in $addrs; do
  case "$target" in
    all) ;;
    others) [ "$a" = "$self" ] && continue ;;
    *) n=$(printf 'x' | nc -w 2 "$a" 8099 2>/dev/null | awk -F= '/^name=/{print $2;exit}')
       [ "$n" = "$target" ] || [ "$a" = "$target" ] || continue ;;
  esac
  name=$(printf 'x' | nc -w 2 "$a" 8099 2>/dev/null | awk -F= '/^name=/{print $2;exit}')
  printf '\033[1m--- %s (%s)\033[0m\n' "${name:-$a}" "$a"
  if [ "$a" = "$self" ]; then bash -c "$*" || rc=$?
  else ssh -o BatchMode=yes -o ConnectTimeout=8 "$USER_NAME@$a" "$@" || rc=$?; fi
done
exit $rc
