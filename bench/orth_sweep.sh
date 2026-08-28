#!/bin/sh
UV=${UV:-uv}
D=$(dirname "$0")
M=${METHODS:-cpu_chol,cpu_hybrid,bchol,bchol_ns,ns,cgs2}
for R in 512 1024 2048 4096; do
  case $R in 512|1024) REPS=${N1:-21};; 2048) REPS=${N2:-15};; *) REPS=${N3:-9};; esac
  perl -e "alarm 3600; exec @ARGV" "$UV" run --quiet --with mlx python "$D/orth.py" 4096 $R $M $REPS "$1" || echo "{\"R\":$R,\"fail\":1}"
done
