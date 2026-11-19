#!/bin/sh
# design/node-addressed-transfers.md "Testing": the two-simulated-node test.  Builds the simulator (tests/sim-wire.c),
# the libnccl-mesh communicator test (tests/sim-collectives.c) and the peer service's test (tests/peer-groups.c) against
# rdma/.build (make -C rdma all first), runs the simulator with CABLES cables of LANES queue pairs (default 3 and 2, the
# M5 Ultra pair), then every peer-groups mode between its two nodes, then a two-rank communicator (and, with COMMS > 1,
# that many at once sharing the order).  Every client is bounded by WAIT seconds.  Exit 0 only if every run passed.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
out=${TMPDIR:-/tmp}/mesh-sim.$$
mkdir -p "$out" "$here/.build/tests"
cables=${CABLES:-3} lanes=${LANES:-2} comms=${COMMS:-1} wait=${WAIT:-60}
cc -O2 -Wall -Wextra -Werror -o "$here/.build/tests/sim-wire" "$here/tests/sim-wire.c" -L"$here/.build" -lmesh -Wl,-rpath,"$here/.build" || exit 1
cc -O2 -Wall -Wextra -Werror -o "$here/.build/tests/peer-groups" "$here/tests/peer-groups.c" -L"$here/.build" -lmesh -Wl,-rpath,"$here/.build" || exit 1
cc -O2 -Wall -Wextra -Werror -o "$here/.build/tests/sim-collectives" "$here/tests/sim-collectives.c" -L"$here/.build" -lnccl-mesh -Wl,-rpath,"$here/.build" || exit 1
bounded() { perl -e "alarm $wait; exec @ARGV" "$@"; }
"$here/.build/tests/sim-wire" 11 12 "$cables" "$lanes" 768 > "$out/sim.out" 2> "$out/sim.err" &
sim=$!
trap 'kill $sim 2>/dev/null' EXIT
while ! grep -q ready "$out/sim.out" 2>/dev/null; do sleep 0.1; done
failed=0 group=1000
for mode in "" concurrent sequential blocked packet late-receive late-receive-window; do
  group=$((group + 100))
  MESH_REGION=/mesh-sim-11 bounded "$here/.build/tests/peer-groups" send 12 $group $mode > "$out/send" 2>&1 &
  sender=$!
  MESH_REGION=/mesh-sim-12 bounded "$here/.build/tests/peer-groups" receive 11 $group $mode > "$out/receive" 2>&1
  received=$?
  wait $sender; sent=$?
  echo "peer-groups ${mode:-default}: send $sent receive $received"; cat "$out/send" "$out/receive"
  [ $sent = 0 ] && [ $received = 0 ] || failed=1
done
port=$((20000 + $$ % 20000))
MESH_REGION=/mesh-sim-11 NCCL_COMM_ID="::1:$port" bounded "$here/.build/tests/sim-collectives" 0 "$comms" > "$out/rank0" 2>&1 &
rank0=$!
MESH_REGION=/mesh-sim-12 NCCL_COMM_ID="::1:$port" bounded "$here/.build/tests/sim-collectives" 1 "$comms" > "$out/rank1" 2>&1
second=$?
wait $rank0; first=$?
echo "collectives: rank 0 $first rank 1 $second"; cat "$out/rank0" "$out/rank1"
[ $first = 0 ] && [ $second = 0 ] || failed=1
kill $sim; wait $sim 2>/dev/null
cat "$out/sim.err"
echo "two simulated nodes: $([ $failed = 0 ] && echo passed || echo FAILED)"
exit $failed
