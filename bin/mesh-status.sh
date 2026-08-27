#!/bin/bash
# One-shot health report for this mesh node.
b(){ printf '\n\033[1m%s\033[0m\n' "$*"; }
b "NODE";    echo "  $(scutil --get ComputerName) / $(scutil --get LocalHostName).local  up $(uptime | sed 's/.*up //;s/,.*users.*//')"
b "POWER";   pmset -g custom | awk '/^ (sleep|displaysleep|disksleep|standby|autorestart|womp|highpowermode)/ {printf "  %-22s %s\n",$1,$2}'
b "ASSERTIONS"; pmset -g assertions | awk '/PreventUserIdleSystemSleep|PreventSystemSleep/ && NF==2 {printf "  %-30s %s\n",$1,$2}'
b "REMOTE";  for p in 22:SSH 5900:ScreenSharing; do
               n=${p%%:*}; l=${p##*:}
               nc -z -G 2 127.0.0.1 "$n" >/dev/null 2>&1 && echo "  $l ($n): OPEN" || echo "  $l ($n): CLOSED"
             done
b "NETWORK"; for i in en0 en1 bridge0; do
               ip=$(ipconfig getifaddr $i 2>/dev/null); echo "  $i: ${ip:-<none>}"
             done
b "RDMA";    echo "  rdma_ctl: $(/usr/bin/rdma_ctl status 2>&1)   nvram: $(nvram rdma-enable 2>/dev/null | awk '{print $2}')"
             /usr/bin/ibv_devices 2>/dev/null | awk 'NR>2 && $1!="" {printf "  device %s\n",$1}'
             for d in $(/usr/bin/ibv_devices 2>/dev/null | awk 'NR>2 && $1!=""{print $1}'); do
               s=$(/usr/bin/ibv_devinfo -d "$d" 2>/dev/null | awk '/state:/{print $2,$3}')
               m=$(/usr/bin/ibv_devinfo -d "$d" 2>/dev/null | awk '/active_mtu:/{print $2}')
               echo "    $d  state=$s mtu=$m"
             done
b "THUNDERBOLT"; system_profiler SPThunderboltDataType 2>/dev/null | awk '/Device Name:|Speed:|Status:/{gsub(/^ +/,"");print "  "$0}' | head -12
b "DAEMONS"
# Report LIVE state, not the last exit code. `launchctl list` shows the previous
# run's status, so a KeepAlive daemon that correctly restarted reads as a failure
# -- and, worse, a dead one can read as fine. Misreporting liveness is precisely
# the false negative this fleet exists to avoid.
#   resident = must be running right now; periodic/oneshot = loaded is correct
for spec in io.mesh.caffeinate:resident io.mesh.beacon:resident \
            io.mesh.keeper:periodic io.mesh.rdma-init:oneshot; do
  L=${spec%%:*}; kind=${spec##*:}
  info=$(launchctl print "system/$L" 2>/dev/null)
  if [ -z "$info" ]; then printf '  %-22s %-9s \033[31mNOT LOADED\033[0m\n' "$L" "$kind"; continue; fi
  pid=$(printf '%s' "$info" | awk -F'= ' '/^\tpid /{gsub(/ /,"",$2);print $2; exit}')
  if [ "$kind" = resident ]; then
    if [ -n "$pid" ]; then printf '  %-22s %-9s running  pid %s\n' "$L" "$kind" "$pid"
    else                   printf '  %-22s %-9s \033[31mDOWN -- should be running\033[0m\n' "$L" "$kind"; fi
  else
    printf '  %-22s %-9s loaded%s\n' "$L" "$kind" "${pid:+, pid $pid}"
  fi
done

b "FABRIC PEERS"
# Absence from this list is the alarm.
if [ -x /usr/local/mesh/bin/mesh-peers.sh ]; then
  /usr/local/mesh/bin/mesh-peers.sh 3 2>/dev/null | sed '/^$/d'
else
  echo "  mesh-peers not installed"
fi
echo
