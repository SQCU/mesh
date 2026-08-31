#!/usr/bin/env bash
set -eu
payload_root=$(cd "$(dirname "$0")" && pwd)
mesh_root=$(cd "$payload_root/../.." && pwd)
output=${1:-$mesh_root/xonotic/payload-build}
compiler=${QCC:-$mesh_root/xonotic/gmqcc-work/gmqcc}
build_root=$(mktemp -d)
trap 'rm -rf "$build_root"' EXIT
rsync -a "$mesh_root/xonotic/payload-build/qcsrc/" "$build_root/qcsrc/"
rsync -a "$payload_root/qcsrc/" "$build_root/qcsrc/"
make -C "$build_root/qcsrc" QCC="$compiler" QCCFLAGS_WATERMARK=payload qc
mkdir -p "$output"
cp "$build_root/progs.dat" "$output/progs.dat"
cp "$build_root/csprogs.dat" "$output/csprogs.dat"
