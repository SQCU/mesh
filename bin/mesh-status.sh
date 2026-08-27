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
b "JOBS";    launchctl list 2>/dev/null | awk 'NR==1 || /io\.mesh/ {printf "  %s\n",$0}'
echo
