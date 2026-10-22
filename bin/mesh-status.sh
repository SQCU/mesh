#!/bin/bash
s=$0; [ -L "$s" ] && s=$(readlink "$s"); HERE="$(cd "$(dirname "$s")" && pwd)"
b(){ printf '\n\033[1m%s\033[0m\n' "$*"; }
b "NODE"; printf "  branch %s converged %s\n" "$(awk '{print $1}' /usr/local/mesh/revision 2>/dev/null || echo unknown)" "$(awk '{print $2}' /usr/local/mesh/revision 2>/dev/null || echo unknown)"; _id=$(ioreg -a -rd1 -c IOPlatformExpertDevice | plutil -extract 0.IOPlatformUUID raw -o - -)
echo "  $(plutil -extract "nodes.$_id.name" raw -o - "$HERE/../etc/mesh-nodes.json" 2>/dev/null || echo unnamed) / host UUID $_id  up $(uptime | sed 's/.*up //;s/,.*users.*//')"
b "POWER"; pmset -g custom | awk '/^ (sleep|displaysleep|disksleep|standby|autorestart|womp|powermode)/{printf "  %-22s %s\n",$1,$2}'
b "ASSERTIONS"; pmset -g assertions | awk '/UserIsActive|PreventUserIdleDisplaySleep|PreventUserIdleSystemSleep|PreventSystemSleep/ && NF==2 {printf "  %-30s %s\n",$1,$2}'
b "HEALTH"; [ -f /var/run/mesh-health ] && sed -e 's/^warn=/WARN /' -e 's/^/  /' /var/run/mesh-health || echo "  no report: /var/run/mesh-health is written by the keeper (mesh-health.sh)"
b "REMOTE"; for p in 22:ssh 5900:screensharing; do
  nc -z 127.0.0.1 "${p%%:*}" >/dev/null 2>&1 && echo "  ${p##*:} (${p%%:*}): OPEN" || echo "  ${p##*:} (${p%%:*}): CLOSED"
done
T="/usr/local/mesh/bin/tailscale --socket=/var/run/tailscaled.socket"; [ -d /Applications/Tailscale.app ] && T=/Applications/Tailscale.app/Contents/MacOS/tailscale
echo "  tailnet: $($T status --json 2>/dev/null | plutil -extract BackendState raw -o - - 2>/dev/null || echo 'not answering') $($T ip -4 2>/dev/null)$([ -d /Applications/Tailscale.app ] && echo ' (Tailscale.app)')"
b "NETWORK"; for i in en0 en1 bridge0; do echo "  $i: $(ipconfig getifaddr $i 2>/dev/null || echo '<none>')"; done
b "LINKS"
_lan=$(route -n get default 2>/dev/null | awk '/interface:/{print $2}')
_lanok=0; [ -n "$_lan" ] && [ -n "$(ipconfig getifaddr "$_lan" 2>/dev/null)" ] && _lanok=1
_fab=0
rdma=$("$HERE/mesh-rdma.sh")
for _d in $(printf '%s\n' "$rdma" | awk 'NF{sub(/^device=/,"",$2); print $2}'); do
  [ "$(ifconfig "${_d#rdma_}" 2>/dev/null | awk '/status:/{print $2}')" = active ] && _fab=1
done
printf "  wifi/lan     : %s\n" "$([ $_lanok = 1 ] && echo "up via $_lan" || echo DOWN)"
printf "  fabric link  : %s (ifconfig; RDMA peers from the IORegistry in the RDMA section)\n" "$([ $_fab = 1 ] && echo "up" || echo DOWN)"
if [ $((_lanok+_fab)) -lt 2 ]; then
  printf "  \033[31mDEGRADED: %s of 2 planes. One more failure strands this node.\033[0m\n" "$((_lanok+_fab))"
  printf "  \033[31mRepair before running anything that can disrupt the remaining plane.\033[0m\n"
else printf "  redundant: 2 of 2\n"; fi

b "RDMA"; echo "  rdma_ctl: $(/usr/bin/rdma_ctl status 2>&1)   nvram: $(nvram rdma-enable 2>/dev/null | awk '{print $2}')"
for d in $(printf '%s\n' "$rdma" | awk 'NF{sub(/^device=/,"",$2); print $2}'); do
  printf '  %-10s link=%s rdma_peer=%s%s\n' "$d" \
    "$(ifconfig "${d#rdma_}" 2>/dev/null | awk '/status:/{print $2}')" \
    "$(awk -v p="port=${d#rdma_}" '$1==p{sub(/rdma_peer=/,"",$3); print $3}' /var/run/mesh-fabric 2>/dev/null)" \
    "$([ -f /var/run/mesh-link.${d#rdma_} ] && read -r t0 _ < /var/run/mesh-link.${d#rdma_} && echo " (no peer for $(( $(date +%s) - t0 )) s)")"
done
b "THUNDERBOLT"; system_profiler SPThunderboltDataType 2>/dev/null | awk '/Device Name:|Speed:|Status:/{gsub(/^ +/,"");print "  "$0}' | head -12
b "DAEMONS"; for spec in io.mesh.beacon:resident io.mesh.keeper:periodic io.mesh.fabric:oneshot io.mesh.router:resident io.mesh.rdma-init:oneshot io.mesh.update:periodic; do
  L=${spec%%:*}; kind=${spec##*:}
  info=$(launchctl print "system/$L" 2>/dev/null)
  pid=$(printf '%s' "$info" | awk -F'= ' '/^\tpid /{gsub(/ /,"",$2);print $2;exit}')
  [ -z "$info" ] && { printf '  %-22s %-9s \033[31mNOT LOADED\033[0m\n' "$L" "$kind"; continue; }
  [ "$kind" = resident ] && [ -z "$pid" ] && { printf '  %-22s %-9s \033[31mDOWN\033[0m\n' "$L" "$kind"; continue; }
  printf '  %-22s %-9s %s\n' "$L" "$kind" "${pid:+pid $pid}"
done
b "FABRIC PEERS"; st=$("$HERE/mesh-stat" 2>/dev/null | tail -1); own=$(printf '%s' "$st" | plutil -extract node raw -o - - 2>/dev/null); i=0
fnv(){ local h=${1//-/} n=0 k; for ((k = 0; k < 32; k += 2)); do n=$(( (n * 0x01000193 & 0xffffffff) ^ 0x${h:k:2} )); done
       n=$((n & 0x7fffffff)); echo $(( n ? n : 1 )); }
names=$(jq -r '.nodes // {} | to_entries[] | select((.key | test("^[0-9A-Fa-f]{8}(-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$")) and .value.name) | "\(.key) \(.value.name)"' \
  "$HERE/../etc/mesh-nodes.json" 2>/dev/null | while read -r id nm; do printf '%s %s\n' "$(fnv "$id")" "$nm"; done)
while n=$(printf '%s' "$st" | plutil -extract "nodes.$i.node" raw -o - - 2>/dev/null); do
  nm=$(printf '%s\n' "$names" | awk -v n="$n" '$1 == n {$1 = ""; sub(/^ /, ""); print; exit}')
  printf '  %-24s node %-10s %s%s\n' "${nm:-node $n}" "$n" "$(printf '%s' "$st" | plutil -extract "nodes.$i.model" raw -o - - 2>/dev/null)" "$([ "$n" = "$own" ] && echo '  (this node)')"
  i=$((i+1))
done
[ "$i" -gt 0 ] || echo "  this node's bridge holds no record (mesh-stat errno $(printf '%s' "$st" | plutil -extract errno raw -o - - 2>/dev/null || echo unread))"
echo
