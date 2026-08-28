#!/bin/sh
set -e
cd "$(dirname "$0")"
cc -O2 -std=c11 -Wall -Wextra -o fakesolver ../solver/fakesolver.c
cc -O2 -std=c11 -Wall -Wextra -o meshtest meshtest.c
HOST=$(hostname -s)
echo "host $HOST  $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
for N in 512 4096 16384 65536; do
  echo "--- $N floats, 60 Hz, 600 ticks, pipelined, solver work 0 us"
  ./meshtest /mesh_p$N $N 600 60 0 ./fakesolver 0 2>/dev/null | grep -v '^region\|^nul-'
  echo "--- $N floats, 60 Hz, 600 ticks, in-frame blocking, solver work 0 us"
  ./meshtest /mesh_b$N $N 600 60 0 ./fakesolver 1 2>/dev/null | grep -v '^region\|^nul-'
done
echo "--- 4096 floats, 60 Hz, 600 ticks, pipelined, solver work 5000 us (slower than the tick)"
./meshtest /mesh_slow 4096 600 60 5000 ./fakesolver 0 2>/dev/null | grep -v '^region\|^nul-'
echo "--- 4096 floats, 60 Hz, solver exits after 3 s, pipelined"
MESH_SOLVER_RUNFOR=3 ./meshtest /mesh_die 4096 600 60 0 ./fakesolver 0 2>/dev/null | grep -v '^region\|^nul-'
echo "--- 4096 floats, 60 Hz, solver exits after 3 s, in-frame blocking capped at 4000 us"
MESH_SOLVER_RUNFOR=3 ./meshtest /mesh_die2 4096 600 60 0 ./fakesolver 1 4000 2>/dev/null | grep -v '^region\|^nul-'
echo "--- 4096 floats, unpaced pipelined, 200000 ticks (seqlock torture)"
./meshtest /mesh_tort 4096 200000 1000000 0 ./fakesolver 0 2>/dev/null | grep -v '^region\|^nul-'
echo "--- 4096 floats, saturation (no tick pacing), 20000 ticks"
./meshtest /mesh_sat 4096 20000 1000000 0 ./fakesolver 1 2>/dev/null | grep -v '^region\|^nul-'
echo "--- 65536 floats, saturation (no tick pacing), 5000 ticks"
./meshtest /mesh_sat2 65536 5000 1000000 0 ./fakesolver 1 2>/dev/null | grep -v '^region\|^nul-'
