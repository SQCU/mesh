#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
PREFIX=fd6d:6573:68
printf 'name=%s\n' "$(scutil --get LocalHostName 2>/dev/null || hostname -s)"
printf 'node_id=%s\n' "$(ioreg -rd1 -c IOPlatformExpertDevice | awk -F'"' '/IOPlatformUUID/{print $(NF-1)}')"
printf 'boot_id=%s\n' "$(sysctl -n kern.bootsessionuuid 2>/dev/null)"
printf 'ula=%s\n' "$(ifconfig lo0 2>/dev/null | awk -v p="$PREFIX" '$1=="inet6" && $2 ~ "^"p":"{print $2;exit}')"
printf 'model=%s\n' "$(sysctl -n hw.model 2>/dev/null)"
printf 'cores=%s\n' "$(sysctl -n hw.ncpu 2>/dev/null)"
printf 'memgb=%s\n' "$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))"
/usr/local/mesh/bin/mesh-stat 2>/dev/null | sed -n 's/.*"window":\([1-9][0-9]*\).*/window=\1/p'
printf 'last_reset=%s\n' "$(sysctl -n kern.shutdownreason 2>/dev/null | awk '{print $1}')"
printf 'macos=%s\n' "$(sw_vers -productVersion 2>/dev/null)"
printf 'rdma=%s\n' "$(rdma_ctl status 2>&1)"
printf 'sdk=%s\n' "$([ -f "$(xcrun --show-sdk-path 2>/dev/null)/usr/include/infiniband/verbs.h" ] && echo yes || echo no)"
printf 'branch=%s\n' "$(awk '{print $1}' /usr/local/mesh/revision 2>/dev/null || echo unknown)"
printf 'converged=%s\n' "$(awk '{print $2}' /usr/local/mesh/revision 2>/dev/null || echo unknown)"
grep -E '^(disk_free_gib|idle_blockers)=' /var/run/mesh-health 2>/dev/null
printf 'health_warnings=%s\n' "$([ -f /var/run/mesh-health ] && awk '/^warn=/{n++} END{print n+0}' /var/run/mesh-health || echo unmeasured)"
printf 'uptime=%s\n' "$(uptime | sed 's/.*up //;s/,[^,]*users.*//' | tr -d ' ')"
grep '^port=' /var/run/mesh-fabric 2>/dev/null
netstat -rn -f inet6 2>/dev/null | awk -v p="$PREFIX" '$1 ~ "^"p":" && $NF!="lo0"{split($1,f,"%"); print "route="f[1]" via="$NF}' | sort -u
printf 'end\n'
