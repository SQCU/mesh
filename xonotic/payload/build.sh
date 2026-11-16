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
cp "$mesh_root/xonotic/teams-k5-assets/effectinfo.txt" "$output/effectinfo.txt"
manifest=$output/BUILD_MANIFEST
{
  echo "built=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "branch=$(git -C "$mesh_root" symbolic-ref --short HEAD 2>/dev/null || echo main)"
  echo "dirty=$(git -C "$mesh_root" status --porcelain -- xonotic/qcsrc | wc -l | tr -d ' ')"
  echo "compiler_id=$(cksum "$compiler" | awk '{print $1 ":" $2}')"
  echo "qcsrc_id=$(find "$mesh_root/xonotic/qcsrc" \( -name '*.qc' -o -name '*.qh' -o -name '*.inc' \) \
      | sort | xargs cksum 2>/dev/null | cksum | awk '{print $1}')"
  for f in progs.dat csprogs.dat menu.dat effectinfo.txt; do
    [ -f "$output/$f" ] && echo "$f=$(cksum "$output/$f" | awk '{print $1 ":" $2}')"
  done
  echo "set_id=$(cat "$output"/progs.dat "$output"/csprogs.dat "$output"/menu.dat 2>/dev/null | cksum | awk '{print $1}')"
} > "$manifest"
echo "build set $(sed -n 's/^set_id=//p' "$manifest") from branch $(sed -n 's/^branch=//p' "$manifest")$([ "$(sed -n 's/^dirty=//p' "$manifest")" != 0 ] && echo ' (DIRTY)')"
"$mesh_root/bin/mesh-python" "$payload_root/runtime-package.py" "$output" "$payload_root/cfg/gamemodes-payload.cfg" "$output/zzzzzz-mesh-runtime.pk3"
