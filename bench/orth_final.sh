#!/bin/sh
D=$(dirname "$0")
MESH_PY=${MESH_PY:-$D/../bin/mesh-python}
for R in 512 1024 2048 4096; do
  case $R in 512|1024) REPS=${N1:-11};; *) REPS=${N2:-7};; esac
  perl -e "alarm 1800; exec @ARGV" "$MESH_PY" "$D/orth.py" 4096 $R cpu_chol,cpu_hybrid,bchol,cgs2 $REPS '{"b":512,"iters":8}'
done
