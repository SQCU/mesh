#!/bin/sh
D=$(dirname "$0")
MESH_PY=${MESH_PY:-$D/../bin/mesh-python}
for R in 2048 4096; do
  for B in 256 512 1024; do
    for IT in 5 8 10 14; do
      perl -e "alarm 1800; exec @ARGV" "$MESH_PY" "$D/orth.py" 4096 $R cgs2,bchol_ns 5 "{\"b\":$B,\"iters\":$IT}" | grep -v normal_matrix_only
    done
  done
done
