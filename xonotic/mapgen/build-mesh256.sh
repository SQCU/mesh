#!/bin/sh
set -eu
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
OUT=${MESH256_MAPDIR:-/tmp/mesh256-map/data/maps}
"$HERE/../../bin/mesh-python" "$HERE/../payload/tools/mapgen.py" 256 --rooms=128 --teams=256 --carts=32 --out="$OUT"
mkdir -p "$HERE/build"
cp "$OUT/genarena256.pk3" "$HERE/build/mesh256.pk3"
