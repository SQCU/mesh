#!/bin/sh
UV=${UV:-uv}
D=$(dirname "$0")
for R in 2048 4096; do
  for B in 256 512 1024; do
    for IT in 5 8 10 14; do
      perl -e "alarm 1800; exec @ARGV" "$UV" run --quiet --with mlx python "$D/orth.py" 4096 $R cgs2,bchol_ns 5 "{\"b\":$B,\"iters\":$IT}" | grep -v gram_only
    done
  done
done
