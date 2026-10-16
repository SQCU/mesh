#!/bin/bash
# The node's facts, one key=value token each (rdma/peers.py observation splits a line on whitespace, so every value is
# one token: a space in it is written _); README.md, node-info.  Beside its own, it publishes what each layer of the node
# states of itself under ~/.cache/mesh (bin/mesh-converge.sh): facts/<label> (revision=, images= and other lines, the
# layer's install writing them after its restart), converge/<label>.state (the layer's last outcome) and converging/<label>
# (a layer's turn under way).
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
PREFIX=fd6d:6573:68
fact(){ printf '%s=%s\n' "$1" "$(printf '%s' "$2" | tr -s '[:space:]' _)"; }
U=$(cat /usr/local/mesh/admin-user 2>/dev/null || id -un); H=$(eval echo "~$U"); S=$H/.cache/mesh
booted(){
  fact node_id "$(ioreg -rd1 -c IOPlatformExpertDevice | awk -F'"' '/IOPlatformUUID/{print $(NF-1)}')"
  fact model "$(sysctl -n hw.model 2>/dev/null)"
  fact cores "$(sysctl -n hw.ncpu 2>/dev/null)"
  fact memgb "$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))"
  fact macos "$(sw_vers -productVersion 2>/dev/null)"
  fact chip "$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  fact gpu_cores "$(ioreg -rc AGXAccelerator -d 1 2>/dev/null | awk -F'= ' '/"gpu-core-count"/{print $2; exit}')"
}
own(){
  boot=$(sysctl -n kern.bootsessionuuid 2>/dev/null); B=${HOME:-/nonexistent}/.cache/mesh/nodeinfo.boot
  fact name "$(scutil --get LocalHostName 2>/dev/null || hostname -s)"
  fact boot_id "$boot"
  if [ -n "$boot" ] && [ "$(head -1 "$B" 2>/dev/null)" = "boot_id=$boot" ]; then sed 1d "$B"
  else mkdir -p "${B%/*}" 2>/dev/null && { fact boot_id "$boot"; booted; } > "$B.$$" && mv -f "$B.$$" "$B" && sed 1d "$B" || booted; fi
  fact ula "$(ifconfig lo0 2>/dev/null | awk -v p="$PREFIX" '$1=="inet6" && $2 ~ "^"p":"{print $2;exit}')"
  "${0%/*}/mesh-stat" 2>/dev/null | sed -n 's/.*"window":\([1-9][0-9]*\).*/window=\1/p'
  fact last_reset "$(sysctl -n kern.shutdownreason 2>/dev/null | awk '{print $1}')"
  fact rdma "$(rdma_ctl status 2>&1)"
  fact sdk "$([ -f "$(xcrun --show-sdk-path 2>/dev/null)/usr/include/infiniband/verbs.h" ] && echo yes || echo no)"
  fact branch "$(awk '{print $1}' /usr/local/mesh/revision 2>/dev/null || echo unknown)"
  fact converged "$(awk '{print $2}' /usr/local/mesh/revision 2>/dev/null || echo unknown)"
  fact account "$U"
  E=$H/.cache/mesh/instances.json
  fact instances "$([ -f "$E" ] && printf '%s/%s' "$(plutil -extract node raw -o - "$E" 2>/dev/null)" "$(plutil -extract cores raw -o - "$E" 2>/dev/null)")"
  fact revisions "$({ for f in "$S"/facts/*; do [ -f "$f" ] && sed -n "s|^revision=\(.\{1,12\}\).*|${f##*/}:\1|p" "$f"; done
    for p in "$H"/Library/LaunchAgents/*.plist; do l=$(basename "$p" .plist); [ ! -f "$S/facts/$l" ] || continue
      [ "$(plutil -extract LimitLoadToSessionType raw "$p" 2>/dev/null)" = Background ] || continue
      r=$(git -C "$(dirname "$(plutil -extract ProgramArguments.0 raw "$p" 2>/dev/null)")" rev-parse --short=12 HEAD 2>/dev/null) && printf '%s:%s\n' "$l" "$r"
    done; } | paste -sd, -)"
  fact images "$(for f in "$S"/facts/*; do [ -f "$f" ] && sed -n "s|^images=\(.\{1,12\}\).*|${f##*/}:\1|p" "$f"; done | paste -sd, -)"
  fact converge "$(for f in "$S"/converge/*.state; do [ -f "$f" ] && l=${f##*/} && printf '%s:%s\n' "${l%.state}" "$(awk 'NR==1{sub(/:$/,"",$1); print $1}' "$f")"; done | paste -sd, -)"
  fact converging "$(now=$(date +%s); for f in "$S"/converging/*; do [ -f "$f" ] && read -r e _ < "$f" && [ "${e:-0}" -gt "$now" ] 2>/dev/null && printf '%s\n' "${f##*/}"; done | paste -sd, -)"
  fact models "$(cd "$H/models" 2>/dev/null && stat -L -f '%Sp %z %N' -- */* 2>/dev/null | awk '$1 ~ /^-/ && NF == 3 && $3 !~ /[,:]/{printf "%s%s:%s", n++ ? "," : "", $3, $2}')"
  grep -E '^(disk_free_gib|idle_blockers)=' /var/run/mesh-health 2>/dev/null
  fact health_warnings "$([ -f /var/run/mesh-health ] && awk '/^warn=/{n++} END{print n+0}' /var/run/mesh-health || echo unmeasured)"
  fact uptime "$(uptime | sed 's/.*up //;s/,[^,]*users.*//' | tr -d ' ')"
  grep '^port=' /var/run/mesh-fabric 2>/dev/null
  netstat -rn -f inet6 2>/dev/null | awk -v p="$PREFIX" '$1 ~ "^"p":" && $NF!="lo0"{split($1,f,"%"); print "route="f[1]" via="$NF}' | sort -u
}
out=$(own)
printf '%s\n' "$out"
for f in "$S"/facts/*; do [ -f "$f" ] && awk 1 "$f"; done 2>/dev/null \
  | grep -E '^[a-z][a-z0-9_]*=' | grep -vE '^(revision|images)=' | tr -s ' \t' __ \
  | awk -F= 'NR==FNR{own[$0]=1; next} !($1 in own)' <(printf '%s\n' "$out" | tr ' ' '\n' | cut -d= -f1) -
printf 'end\n'
