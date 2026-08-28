#!/bin/bash
b(){ printf '\n\033[1m%s\033[0m\n' "$*"; }
b "NODE"; printf "  branch %s converged %s\n" "$(awk '{print $1}' /usr/local/mesh/revision 2>/dev/null || echo unknown)" "$(awk '{print $2}' /usr/local/mesh/revision 2>/dev/null || echo unknown)"; echo "  $(scutil --get ComputerName) / $(scutil --get LocalHostName).local  up $(uptime | sed 's/.*up //;s/,.*users.*//')"
b "POWER"; pmset -g custom | awk '/^ (sleep|displaysleep|disksleep|standby|autorestart|womp|powermode)/{printf "  %-22s %s\n",$1,$2}'
b "ASSERTIONS"; pmset -g assertions | awk '/PreventUserIdleSystemSleep|PreventSystemSleep/ && NF==2 {printf "  %-30s %s\n",$1,$2}'
b "REMOTE"; for p in 22:ssh 5900:screensharing; do
  nc -z -G 2 127.0.0.1 "${p%%:*}" >/dev/null 2>&1 && echo "  ${p##*:} (${p%%:*}): OPEN" || echo "  ${p##*:} (${p%%:*}): CLOSED"
done
b "NETWORK"; for i in en0 en1 bridge0; do echo "  $i: $(ipconfig getifaddr $i 2>/dev/null || echo '<none>')"; done
b "RDMA"; echo "  rdma_ctl: $(/usr/bin/rdma_ctl status 2>&1)   nvram: $(nvram rdma-enable 2>/dev/null | awk '{print $2}')"
for d in $(/usr/bin/ibv_devices 2>/dev/null | awk 'NR>2&&$1!=""{print $1}'); do
  printf '  %-10s state=%s mtu=%s\n' "$d" \
    "$(/usr/bin/ibv_devinfo -d "$d" 2>/dev/null | awk '/state:/{print $2}')" \
    "$(/usr/bin/ibv_devinfo -d "$d" 2>/dev/null | awk '/active_mtu:/{print $2}')"
done
b "THUNDERBOLT"; system_profiler SPThunderboltDataType 2>/dev/null | awk '/Device Name:|Speed:|Status:/{gsub(/^ +/,"");print "  "$0}' | head -12
b "DAEMONS"; for spec in io.mesh.caffeinate:resident io.mesh.beacon:resident io.mesh.keeper:periodic io.mesh.fabric:oneshot io.mesh.router:resident io.mesh.nodeinfo:socket io.mesh.rdma-init:oneshot; do
  L=${spec%%:*}; kind=${spec##*:}
  info=$(launchctl print "system/$L" 2>/dev/null)
  pid=$(printf '%s' "$info" | awk -F'= ' '/^\tpid /{gsub(/ /,"",$2);print $2;exit}')
  [ -z "$info" ] && { printf '  %-22s %-9s \033[31mNOT LOADED\033[0m\n' "$L" "$kind"; continue; }
  [ "$kind" = resident ] && [ -z "$pid" ] && { printf '  %-22s %-9s \033[31mDOWN\033[0m\n' "$L" "$kind"; continue; }
  printf '  %-22s %-9s %s\n' "$L" "$kind" "${pid:+pid $pid}"
done
b "FABRIC PEERS"; /usr/local/mesh/bin/mesh-peers.sh 3 2>/dev/null | sed '/^$/d'
echo
