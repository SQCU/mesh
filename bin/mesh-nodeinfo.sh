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
"${0%/*}/mesh-stat" 2>/dev/null | sed -n 's/.*"window":\([1-9][0-9]*\).*/window=\1/p'
printf 'last_reset=%s\n' "$(sysctl -n kern.shutdownreason 2>/dev/null | awk '{print $1}')"
printf 'macos=%s\n' "$(sw_vers -productVersion 2>/dev/null)"
printf 'rdma=%s\n' "$(rdma_ctl status 2>&1)"
printf 'sdk=%s\n' "$([ -f "$(xcrun --show-sdk-path 2>/dev/null)/usr/include/infiniband/verbs.h" ] && echo yes || echo no)"
printf 'branch=%s\n' "$(awk '{print $1}' /usr/local/mesh/revision 2>/dev/null || echo unknown)"
printf 'converged=%s\n' "$(awk '{print $2}' /usr/local/mesh/revision 2>/dev/null || echo unknown)"
U=$(cat /usr/local/mesh/admin-user 2>/dev/null || id -un); H=$(eval echo "~$U")
printf 'account=%s\n' "$U"
printf 'chip=%s\n' "$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
printf 'gpu_cores=%s\n' "$(ioreg -rc AGXAccelerator -d 1 2>/dev/null | awk -F'= ' '/"gpu-core-count"/{print $2; exit}')"
printf 'instances=%s\n' "$(E=$H/.cache/mesh/instances.json; [ -f "$E" ] && printf '%s/%s' "$(plutil -extract node raw -o - "$E" 2>/dev/null)" "$(plutil -extract cores raw -o - "$E" 2>/dev/null)")"
printf 'revisions=%s\n' "$(for p in "$H"/Library/LaunchAgents/*.plist; do
  [ "$(plutil -extract LimitLoadToSessionType raw "$p" 2>/dev/null)" = Background ] || continue
  r=$(git -C "$(dirname "$(plutil -extract ProgramArguments.0 raw "$p" 2>/dev/null)")" rev-parse --short=12 HEAD 2>/dev/null) && printf '%s:%s,' "$(basename "$p" .plist)" "$r"
done | sed 's/,$//')"
printf 'models=%s\n' "$(cd "$H/models" 2>/dev/null && stat -f '%Sp %z %N' -- */* 2>/dev/null | awk '$1 ~ /^-/ && NF == 3 && $3 !~ /[,:]/{printf "%s%s:%s", n++ ? "," : "", $3, $2}')"
grep -E '^(disk_free_gib|idle_blockers)=' /var/run/mesh-health 2>/dev/null
printf 'health_warnings=%s\n' "$([ -f /var/run/mesh-health ] && awk '/^warn=/{n++} END{print n+0}' /var/run/mesh-health || echo unmeasured)"
printf 'uptime=%s\n' "$(uptime | sed 's/.*up //;s/,[^,]*users.*//' | tr -d ' ')"
grep '^port=' /var/run/mesh-fabric 2>/dev/null
netstat -rn -f inet6 2>/dev/null | awk -v p="$PREFIX" '$1 ~ "^"p":" && $NF!="lo0"{split($1,f,"%"); print "route="f[1]" via="$NF}' | sort -u
printf 'end\n'
