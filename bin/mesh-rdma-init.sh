#!/bin/bash
exec >>/usr/local/mesh/log/rdma-init.log 2>&1
ts(){ date '+%F %T'; }
echo "[$(ts)] status=$(/usr/bin/rdma_ctl status 2>&1) nvram=$(nvram rdma-enable 2>/dev/null | awk '{print $2}')"
for _ in $(seq 1 15); do
  for d in $(/usr/bin/ibv_devices 2>/dev/null | awk 'NR>2&&$1!=""{print $1}'); do
    [ "$(ifconfig "${d#rdma_}" 2>/dev/null | awk '/status:/{print $2}')" = active ] && {
      echo "[$(ts)] fabric link active: $d rdma_state=unmeasured"
      exit 0; }
  done
  sleep 2
done
echo "[$(ts)] no active fabric link after 30s rdma_state=unmeasured"
exit 0
