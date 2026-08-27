#!/bin/bash
exec >>/usr/local/mesh/log/rdma-init.log 2>&1
ts(){ date '+%F %T'; }
echo "[$(ts)] status=$(/usr/bin/rdma_ctl status 2>&1) nvram=$(nvram rdma-enable 2>/dev/null | awk '{print $2}')"
for _ in $(seq 1 15); do
  for d in $(/usr/bin/ibv_devices 2>/dev/null | awk 'NR>2&&$1!=""{print $1}'); do
    [ "$(/usr/bin/ibv_devinfo -d "$d" 2>/dev/null | awk '/state:/{print $2}')" = PORT_ACTIVE ] && {
      echo "[$(ts)] fabric up: $d mtu=$(/usr/bin/ibv_devinfo -d "$d" | awk '/active_mtu:/{print $2}')"
      exit 0; }
  done
  sleep 2
done
echo "[$(ts)] no ACTIVE port after 30s"
exit 0
