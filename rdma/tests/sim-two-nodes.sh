#!/bin/sh
# design/declared-programs.md "Testing": the two-simulated-node test.  Builds the simulator (tests/sim-wire.c: the
# switch's own steps, mesh-switched.h, on each node's region, a wire that enforces receive-before-send, each queue pair's
# post order and one frame count), the libnccl-mesh communicator test (tests/sim-collectives.c),
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
for t in sim-wire peer-groups sim-dangling; do
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
# programs in turn toward the same node: two process pairs, each on leases of pages a process wrote before (DIRTY), the
# second's crossings larger than the first's; then one process pair redeclaring its program at four sizes, growing.
# Every piece either end posts must be the other's same piece (the wire ends at any mismatch)
for sizes in 6144 98304 6144,98304,393216,1572864; do
  port=$((port + 8))
  DIRTY=1 MESH_REGION=/mesh-sim-11 NCCL_COMM_ID="::1:$port" bounded "$here/.build/tests/sim-metal" 0 4 "$sizes" > "$out/metal0" 2>&1 &
  rank0=$!
  DIRTY=1 MESH_REGION=/mesh-sim-12 NCCL_COMM_ID="::1:$port" bounded "$here/.build/tests/sim-metal" 1 4 "$sizes" > "$out/metal1" 2>&1
  second=$?
  wait $rank0; first=$?
  echo "Metal path in turn, $sizes bytes: rank 0 $first rank 1 $second ($(cat "$out/metal0" "$out/metal1" | grep -c '"checked":true') checked)"
  grep -v '"checked":true' "$out/metal0" "$out/metal1"
  kill -0 $sim 2>/dev/null || { echo "the simulator ended (its wire's check):"; cat "$out/sim.err"; failed=1; break; }
  [ $first = 0 ] && [ $second = 0 ] || failed=1
done
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
# a cable lost while a program runs (sim-wire SIGUSR2: its queue pairs and every unit on them gone, the cable up again
# UP ms later): the pieces on it are switched again over the other cables, so the program runs to its end at both ranks
# and the next one runs; nothing fails and nothing restarts (design/switched-multipath-dma.md).  A pair of sim-programs
# processes is the clients (rounds: how long it runs)
simulate ""
programs_pair() {   # programs_pair NAME ROUNDS: ranks 0 and 1 in the background (r0, r1: the clients' own processes)
  port=$((port + 8))
  env MESH_REGION=/mesh-sim-11 NCCL_COMM_ID="::1:$port" perl -e "alarm $wait; exec @ARGV" "$here/.build/tests/sim-programs" 0 "$2" > "$out/$1.0" 2>&1 &
  r0=$!
  env MESH_REGION=/mesh-sim-12 NCCL_COMM_ID="::1:$port" perl -e "alarm $wait; exec @ARGV" "$here/.build/tests/sim-programs" 1 "$2" > "$out/$1.1" 2>&1 &
  r1=$!
}
pair_end() {   # both ranks 0, the simulator still running
  wait $r0; first=$?; wait $r1; second=$?
  alive=1; kill -0 $sim 2>/dev/null || alive=0
  echo "cable loss, $1: rank 0 $first rank 1 $second, the wire $([ $alive = 1 ] && echo running || echo ENDED)"
  [ $first = 0 ] && [ $second = 0 ] && [ $alive = 1 ] || { failed=1; cat "$out/$1.0" "$out/$1.1"; for n in 11 12; do MESH_REGION=/mesh-sim-$n "$here/.build/mesh-order-state" 2; done; }
}
programs_pair lost 40; sleep 1
kill -USR2 $sim; sleep 1; kill -USR2 $sim
pair_end lost
programs_pair after 6; pair_end after
finish
# a program left with a piece never sent (its last SEND's gate never opened: tests/sim-dangling.c), closed at the
# executions it issued and its clients gone: no queue pair waits at that piece (no piece is placed on one ahead of
# time), so the pair declared after it runs at once; then a gone pair's last pieces still crossing as it ends
# ("inflight"), and the pair after it runs behind them
dangling_pair() {   # dangling_pair NAME EXECUTIONS MODE: both ranks, ended before it returns
  MESH_REGION=/mesh-sim-11 bounded "$here/.build/tests/sim-dangling" 12 "$2" "$3" > "$out/$1.0" 2>&1 &
  r0=$!
  MESH_REGION=/mesh-sim-12 bounded "$here/.build/tests/sim-dangling" 11 "$2" "$3" > "$out/$1.1" 2>&1
  second=$?; wait $r0; first=$?
  echo "dangling, $1: rank 0 $first rank 1 $second"; [ $first = 0 ] && [ $second = 0 ] || { failed=1; cat "$out/$1.0" "$out/$1.1"; }
}
simulate ""
dangling_pair dangling 16 dangling
t0=$(date +%s); programs_pair behind 6; pair_end behind; echo "dangling, behind ended after $(( $(date +%s) - t0 )) s"
dangling_pair inflight 200000 inflight
programs_pair following 6; pair_end following
finish
echo "two simulated nodes: $([ $failed = 0 ] && echo passed || echo FAILED)"
exit $failed
