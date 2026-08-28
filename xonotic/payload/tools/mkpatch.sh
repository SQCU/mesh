#!/usr/bin/env bash
set -eu
PRISTINE=${1:?pristine qcsrc}
PATCHED=${2:?patched qcsrc}
OUT=${3:?output patch}
FILES="common/gamemodes/gamemode/_mod.inc common/gamemodes/gamemode/_mod.qh common/scores.qh common/stats.qh common/mutators/mutator/waypoints/all.inc"
WORK=$(mktemp -d)
for f in $FILES; do
  mkdir -p "$WORK/a/$(dirname "$f")" "$WORK/b/$(dirname "$f")"
  cp "$PRISTINE/$f" "$WORK/a/$f"
  cp "$PATCHED/$f" "$WORK/b/$f"
done
( cd "$WORK" && diff -ru a b ) > "$OUT" || true
rm -rf "$WORK"
wc -l "$OUT"
