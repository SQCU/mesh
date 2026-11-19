#!/bin/sh
# design/declared-programs.md "Testing": the two-simulated-node test.  Builds the simulator (tests/sim-wire.c over
# tests/wire.h: the bridge's own steps on each node's region, a wire that enforces receive-before-send, each queue pair's
# post order, equal lengths and the same piece at both ends), the libnccl-mesh communicator test (tests/sim-collectives.c),
# the run-ahead test (tests/sim-runahead.c) and the peer service's test (tests/peer-groups.c) against rdma/.build (make -C
# rdma all first).  It runs the simulator with CABLES cables of LANES queue pairs (default 3 and 2, the M5 Ultra pair),
# then every peer-groups mode between its two nodes, then a two-rank communicator and COMMS (default 2) at once in one
# program, the same through the NumPy bindings (tests/sim-mpi.py, with PYTHON's cffi: mesh_c_build.py bound to rdma/.build),
# then a one-way WRITE plan and an AlltoAllV of uneven counts declared (tests/sim-programs.c),
# then a two-rank communicator's program on the Metal path (tests/sim-metal.m, with this build's kernels beside
# the library: made here, none installed); then, over a simulator whose queue pair LAG (default 1) is served once in 64 turns, PROGRAMS (default 2)
# process pairs each running its own one-step program of CALLS (default 12000) Broadcasts ahead of its receiver, the
# programs sharing the two nodes by rounds.  Every client is bounded by WAIT seconds.  Exit 0 only if every run passed.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
out=${TMPDIR:-/tmp}/mesh-sim.$$
mkdir -p "$out" "$here/.build/tests"
cables=${CABLES:-3} lanes=${LANES:-2} comms=${COMMS:-2} wait=${WAIT:-120} lag=${LAG:-1} programs=${PROGRAMS:-2} calls=${CALLS:-12000}
for t in sim-wire peer-groups; do
  cc -O2 -Wall -Wextra -Werror -o "$here/.build/tests/$t" "$here/tests/$t.c" -L"$here/.build" -lmesh -Wl,-rpath,"$here/.build" || exit 1
done
for t in sim-collectives sim-runahead sim-programs; do
  cc -O2 -Wall -Wextra -Werror -o "$here/.build/tests/$t" "$here/tests/$t.c" -L"$here/.build" -lnccl-mesh -Wl,-rpath,"$here/.build" || exit 1
done
cc -fobjc-arc -O2 -Wall -Wextra -Werror -o "$here/.build/tests/sim-metal" "$here/tests/sim-metal.m" -L"$here/.build" -lnccl-mesh \
  -Wl,-rpath,"$here/.build" -framework Metal -framework Foundation || exit 1
make -s -C "$here" .build/libnccl-mesh.metallib .build/libnccl-mesh.binarchive || exit 1
python=${PYTHON:-$HOME/.venv-mesh-uv/bin/python}
mkdir -p "$out/py"
"$python" -I "$here/mesh_c_build.py" "$out/py" "$here/.build" > /dev/null 2>&1 || { echo "sim-two-nodes: $python cannot build _mesh_c (cffi)"; exit 1; }
kernels=$(cat "$here/.build/libnccl-mesh.sha256")
for f in metallib binarchive; do cp "$here/.build/libnccl-mesh.$f" "$here/.build/libnccl-mesh.$kernels.$f" || exit 1; done
bounded() { perl -e "alarm $wait; exec @ARGV" "$@"; }
sim=
simulate() {
  LAG=$1 "$here/.build/tests/sim-wire" 11 12 "$cables" "$lanes" 768 > "$out/sim.out" 2> "$out/sim.err" &
  sim=$!
  while ! grep -q ready "$out/sim.out" 2>/dev/null; do sleep 0.1; done
}
finish() { kill $sim; wait $sim 2>/dev/null; cat "$out/sim.err"; rm -f "$out/sim.out"; }
trap 'kill $sim 2>/dev/null' EXIT
failed=0 group=1000
simulate ""
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
for n in 1 "$comms"; do
  port=$((port + 8))
  MESH_REGION=/mesh-sim-11 NCCL_COMM_ID="::1:$port" bounded "$here/.build/tests/sim-collectives" 0 "$n" 4 > "$out/rank0" 2>&1 &
  rank0=$!
  MESH_REGION=/mesh-sim-12 NCCL_COMM_ID="::1:$port" bounded "$here/.build/tests/sim-collectives" 1 "$n" 4 > "$out/rank1" 2>&1
  second=$?
  wait $rank0; first=$?
  echo "collectives over $n communicator(s), one program: rank 0 $first rank 1 $second ($(cat "$out/rank0" "$out/rank1" | grep -c '"checked":true') checked)"
  grep -v '"checked":true' "$out/rank0" "$out/rank1"
  [ $first = 0 ] && [ $second = 0 ] || failed=1
done
port=$((port + 8))
MESH_REGION=/mesh-sim-11 NCCL_COMM_ID="::1:$port" bounded "$here/.build/tests/sim-programs" 0 > "$out/programs0" 2>&1 &
rank0=$!
MESH_REGION=/mesh-sim-12 NCCL_COMM_ID="::1:$port" bounded "$here/.build/tests/sim-programs" 1 > "$out/programs1" 2>&1
second=$?
wait $rank0; first=$?
echo "a WRITE plan and an AlltoAllV, declared: rank 0 $first rank 1 $second"; cat "$out/programs0" "$out/programs1"
[ $first = 0 ] && [ $second = 0 ] || failed=1
port=$((port + 8))
PYTHONPATH="$out/py" MESH_REGION=/mesh-sim-11 NCCL_COMM_ID="::1:$port" bounded "$python" "$here/tests/sim-mpi.py" 0 > "$out/mpi0" 2>&1 &
rank0=$!
PYTHONPATH="$out/py" MESH_REGION=/mesh-sim-12 NCCL_COMM_ID="::1:$port" bounded "$python" "$here/tests/sim-mpi.py" 1 > "$out/mpi1" 2>&1
second=$?
wait $rank0; first=$?
echo "mesh_mpi: rank 0 $first rank 1 $second"; tail -3 "$out/mpi0" "$out/mpi1"
[ $first = 0 ] && [ $second = 0 ] || failed=1
port=$((port + 8))
MESH_REGION=/mesh-sim-11 NCCL_COMM_ID="::1:$port" bounded "$here/.build/tests/sim-metal" 0 > "$out/metal0" 2>&1 &
rank0=$!
MESH_REGION=/mesh-sim-12 NCCL_COMM_ID="::1:$port" bounded "$here/.build/tests/sim-metal" 1 > "$out/metal1" 2>&1
second=$?
wait $rank0; first=$?
echo "Metal path: rank 0 $first rank 1 $second"; cat "$out/metal0" "$out/metal1"
[ $first = 0 ] && [ $second = 0 ] || failed=1
finish
simulate "$lag"
rm -rf "$out/barrier"; mkdir -p "$out/barrier"
pids=
p=0
while [ $p -lt "$programs" ]; do
  port=$((port + 1))
  for rank in 0 1; do
    MESH_REGION=/mesh-sim-1$((rank + 1)) NCCL_COMM_ID="::1:$port" BARRIER="$out/barrier" PROGRAMS="$programs" \
      bounded "$here/.build/tests/sim-runahead" $rank $p "$calls" > "$out/ahead.$p.$rank" 2>&1 &
    pids="$pids $!"
  done
  p=$((p + 1))
done
for pid in $pids; do wait $pid || failed=1; done
echo "run-ahead: $programs programs sharing the nodes, $calls Broadcasts each, queue pair $lag lagging:"
cat "$out"/ahead.*
finish
echo "two simulated nodes: $([ $failed = 0 ] && echo passed || echo FAILED)"
exit $failed
