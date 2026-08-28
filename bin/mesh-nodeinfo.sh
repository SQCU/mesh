#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
PREFIX=fd6d:6573:68
ports=$(ibv_devices 2>/dev/null | awk 'NR>2 && $1!=""{sub(/^rdma_/,"",$1);print $1}')
printf 'name=%s\n' "$(scutil --get LocalHostName 2>/dev/null || hostname -s)"
printf 'ula=%s\n' "$(ifconfig lo0 2>/dev/null | awk -v p="$PREFIX" '$1=="inet6" && $2 ~ "^"p":"{print $2;exit}')"
printf 'model=%s\n' "$(sysctl -n hw.model 2>/dev/null)"
printf 'cores=%s\n' "$(sysctl -n hw.ncpu 2>/dev/null)"
printf 'memgb=%s\n' "$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))"
printf 'macos=%s\n' "$(sw_vers -productVersion 2>/dev/null)"
lan=$(route -n get default 2>/dev/null | awk '/interface:/{print $2}')
lanok=0; [ -n "$lan" ] && [ -n "$(ipconfig getifaddr "$lan" 2>/dev/null)" ] && lanok=1
fabok=0
for d in $ports; do
  [ "$(ibv_devinfo -d "rdma_$d" 2>/dev/null | awk '/state:/{print $2}')" = PORT_ACTIVE ] && fabok=1
done
printf 'planes=%s lan=%s fabric=%s\n' "$((lanok+fabok))" "${lan:-none}" "$fabok"
printf 'rdma=%s\n' "$(rdma_ctl status 2>&1)"
printf 'sdk=%s\n' "$([ -f "$(xcrun --show-sdk-path 2>/dev/null)/usr/include/infiniband/verbs.h" ] && echo yes || echo no)"
printf 'branch=%s\n' "$(awk '{print $1}' /usr/local/mesh/revision 2>/dev/null || echo unknown)"
printf 'converged=%s\n' "$(awk '{print $2}' /usr/local/mesh/revision 2>/dev/null || echo unknown)"
printf 'uptime=%s\n' "$(uptime | sed 's/.*up //;s/,[^,]*users.*//' | tr -d ' ')"
for p in $ports; do
  st=$(ibv_devinfo -d "rdma_$p" 2>/dev/null | awk '/state:/{print $2}')
  gid=$(ibv_devinfo -d "rdma_$p" -v 2>/dev/null | awk '/GID\[  0\]/{print $NF}')
  peer=$(ndp -an 2>/dev/null | awk -v i="$p" '$3==i && $1 !~ /ff02|^$/ && $2 != "(incomplete)"{print $1}' | grep -v "$gid" | head -1)
  printf 'port=%s state=%s gid=%s peer=%s\n' "$p" "${st:-UNKNOWN}" "${gid:-none}" "${peer:-none}"
done
netstat -rn -f inet6 2>/dev/null | awk -v p="$PREFIX" '$1 ~ "^"p":" && $NF!="lo0"{split($1,f,"%"); print "route="f[1]" via="$NF}' | sort -u
printf 'end\n'
