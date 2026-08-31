#!/usr/bin/env bash
set -eu
payload_root=$(cd "$(dirname "$0")" && pwd)
mesh_root=$(cd "$payload_root/../.." && pwd)
output=${1:-$mesh_root/xonotic/payload-build}
compiler=${QCC:-$mesh_root/xonotic/gmqcc-work/gmqcc}
mkdir -p "$output"
output=$(cd "$output" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
make -C "$mesh_root/xonotic/qcsrc" QCC="$compiler" QCCFLAGS_WATERMARK=payload PROGS_OUT="$output" WORKDIR="$work" qc

# THE BUILD IS A SET, NOT THREE FILES.
#
# progs.dat, csprogs.dat and menu.dat are compiled separately but must agree on
# things that live in the built data rather than in the code: STAT indices,
# entity field offsets, string tables, compile-time cvar defaults. A progs.dat
# from one build with a csprogs.dat from another reads a DIFFERENT stat slot --
# e.g. STAT(PAYLOAD_PUSH_PACKED) decodes as another field's value -- silently,
# with no error on either side. That is a genuine disagreement between two
# programs, not one program lagging itself, which is why these are recorded and
# checked together and never file by file.
#
# A .dat also cannot always be regenerated: built from uncommitted source that
# has since changed, it is a program that can be measured and never rebuilt. So
# the manifest names the source, and says plainly when the source was dirty.
manifest=$output/BUILD_MANIFEST
{
  echo "built=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "commit=$(git -C "$mesh_root" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "dirty=$(git -C "$mesh_root" status --porcelain -- xonotic/qcsrc | wc -l | tr -d ' ')"
  echo "compiler_id=$(cksum "$compiler" | awk '{print $1 ":" $2}')"
  echo "qcsrc_id=$(find "$mesh_root/xonotic/qcsrc" \( -name '*.qc' -o -name '*.qh' -o -name '*.inc' \) \
      | sort | xargs cksum 2>/dev/null | cksum | awk '{print $1}')"
  for f in progs.dat csprogs.dat menu.dat; do
    [ -f "$output/$f" ] && echo "$f=$(cksum "$output/$f" | awk '{print $1 ":" $2}')"
  done
  echo "set_id=$(cat "$output"/progs.dat "$output"/csprogs.dat "$output"/menu.dat 2>/dev/null | cksum | awk '{print $1}')"
} > "$manifest"
echo "build set $(sed -n 's/^set_id=//p' "$manifest") from commit $(sed -n 's/^commit=//p' "$manifest")$([ "$(sed -n 's/^dirty=//p' "$manifest")" != 0 ] && echo ' (DIRTY)')"
