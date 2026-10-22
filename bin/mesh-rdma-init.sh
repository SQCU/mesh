#!/bin/bash
exec >>/usr/local/mesh/log/rdma-init.log 2>&1
ts(){ date '+%F %T'; }
echo "[$(ts)] status=$(/usr/bin/rdma_ctl status 2>&1) nvram=$(nvram rdma-enable 2>/dev/null | awk '{print $2}')"
for d in $("$(dirname "$0")/mesh-rdma.sh" | awk '{sub(/^device=/,"",$2); print $2}'); do
  echo "[$(ts)] $d link=$(ifconfig "${d#rdma_}" 2>/dev/null | awk '/status:/{print $2}') rdma_peer=$(awk -v p="port=${d#rdma_}" '$1==p{sub(/rdma_peer=/,"",$3); print $3}' /var/run/mesh-fabric 2>/dev/null)"
done
exit 0
