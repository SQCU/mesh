#!/bin/sh
UV=${UV:-uv}
D=$(dirname "$0")
for R in 512 1024 2048 4096; do
  case $R in 512|1024) REPS=${N1:-11};; *) REPS=${N2:-7};; esac
  perl -e "alarm 1800; exec @ARGV" "$UV" run --quiet --with mlx python "$D/orth.py" 4096 $R cpu_chol,cpu_hybrid,bchol,cgs2 $REPS '{"b":512,"iters":8}'
done
