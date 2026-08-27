#!/bin/bash
# Boot-time RDMA verification.
# HARD CONSTRAINT: rdma_ctl enable/disable ONLY works from Recovery OS (exit 77 in
# full macOS). The enable bit lives in NVRAM (rdma-enable) and persists across
# reboots, so at runtime we can only VERIFY and alarm -- never remediate.
# A failure here means a physical 1TR/Recovery visit is required.
LOG=/usr/local/mesh/log/rdma-init.log
exec >>"$LOG" 2>&1
ts(){ date '+%F %T'; }
st=$(/usr/bin/rdma_ctl status 2>&1)
nv=$(nvram rdma-enable 2>/dev/null | awk '{print $2}')
echo "[$(ts)] rdma status=$st nvram=$nv"
if [ "$st" != "enabled" ]; then
  echo "[$(ts)] ALARM: RDMA DISABLED and unfixable from macOS."
  echo "[$(ts)] ALARM: requires physical Recovery OS (1TR) visit -> rdma_ctl enable"
  exit 1
fi
# Wait for a port to come ACTIVE. A node sitting with no peer cable is NOT an error
# -- it may legitimately be in a bag between rooms.
for i in $(seq 1 15); do
  for d in $(/usr/bin/ibv_devices 2>/dev/null | awk 'NR>2 && $1!=""{print $1}'); do
    s=$(/usr/bin/ibv_devinfo -d "$d" 2>/dev/null | awk '/state:/{print $2}')
    if [ "$s" = "PORT_ACTIVE" ]; then
      m=$(/usr/bin/ibv_devinfo -d "$d" 2>/dev/null | awk '/active_mtu:/{print $2}')
      echo "[$(ts)] fabric up: $d PORT_ACTIVE mtu=$m"; exit 0
    fi
  done
  sleep 2
done
echo "[$(ts)] no ACTIVE port after 30s (no peer attached?) - RDMA itself is enabled"
exit 0
