#!/bin/bash
# Start the mesh bridge from configuration. Changing the configuration and
# restarting this is the supported way to change how much of a node the mesh
# holds; nothing has to be tuned while it runs.
#
# An API user picks two percentages: what the mesh holds, and what their own
# programs expect to use. Everything else is derived.
set -u
CONF=${MESH_CONF:-/usr/local/mesh/bridge.conf}
[ -f "$CONF" ] || CONF="$(dirname "$0")/../etc/bridge.conf"
mesh_pct=25; app_pct=25; node=0; peer=""; region=/mesh0
# shellcheck disable=SC1090
[ -f "$CONF" ] && . "$CONF"

ram=$(sysctl -n hw.memsize)
want_pct=$(awk -v a="$mesh_pct" -v b="$app_pct" 'BEGIN{print a+b}')

# The one refusal. Wiring so much that the kernel and everything outside the
# mesh cannot fit leaves a node that fails on boot and needs hands on it, and
# this fleet is meant never to need that. 90% is the ceiling.
if awk -v w="$want_pct" 'BEGIN{exit !(w > 90)}'; then
  echo "mesh-bridge: refusing mesh_pct=$mesh_pct + app_pct=$app_pct = $want_pct%" >&2
  echo "  more than 90% of RAM would be wired, which can leave this node unable" >&2
  echo "  to boot without someone physically present. Lower one of them in $CONF." >&2
  exit 78
fi

want=$(awk -v r="$ram" -v w="$want_pct" 'BEGIN{printf "%.0f", r*w/100}')
have=$(sysctl -n vm.global_user_wire_limit)
if [ "$want" -gt "$have" ]; then
  sysctl -w vm.global_user_wire_limit="$want" >/dev/null 2>&1 \
    || { echo "mesh-bridge: cannot raise wire limit (need root)" >&2; exit 77; }
fi
printf 'mesh-bridge: mesh %s%% app %s%% wire limit %.1f GB of %.1f GB\n' \
  "$mesh_pct" "$app_pct" "$(awk -v w="$want" 'BEGIN{print w/1e9}')" \
  "$(awk -v r="$ram" 'BEGIN{print r/1e9}')"
exec "$(dirname "$0")/../rdma/mesh-flow" -I "$node" -M "$mesh_pct" -s "$region" ${peer:+"$peer"}
