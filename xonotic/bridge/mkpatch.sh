#!/bin/sh
HERE="$(cd "$(dirname "$0")" && pwd)"
ORIG="${1:-/Applications/Xonotic/source/darkplaces}"
PATCHED="$2"
OUT="$HERE/engine/0001-mesh-shm-builtins.patch"
: > "$OUT"
for f in prvm_cmds.c prvm_cmds.h svvm_cmds.c clvm_cmds.c; do
  diff -u "$ORIG/$f" "$PATCHED/$f" | sed "s|^--- .*$f|--- a/$f|; s|^+++ .*$f|+++ b/$f|" >> "$OUT"
done
wc -l "$OUT"
