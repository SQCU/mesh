#!/bin/bash
set -o pipefail
mesh_repo=$(cd "$(dirname "$0")/.." && pwd)
mesh_build=${Q3MAP2_BUILD_ROOT:-$mesh_repo/.build/netradiant}
mesh_remote=${Q3MAP2_REPO:-https://github.com/Garux/netradiant-custom.git}
mesh_branch=${Q3MAP2_BRANCH:-master}
mkdir -p "$mesh_build"
mesh_build=$(cd "$mesh_build" && pwd)
mesh_stage=$(mktemp -d "$mesh_build/source.XXXXXX")
echo "building compiler from $mesh_remote branch $mesh_branch in $mesh_stage"
if git clone --depth 1 --single-branch --branch "$mesh_branch" "$mesh_remote" "$mesh_stage" \
    && (git -C "$mesh_stage" apply "$mesh_repo/vendor/netradiant-capacity.patch" \
        || git -C "$mesh_stage" apply --reverse --check "$mesh_repo/vendor/netradiant-capacity.patch") \
    && make -C "$mesh_stage" -j "${MESH_BUILD_JOBS:-2}" binaries-q3map2 \
        CC=clang CXX=clang++ DEPENDENCIES_CHECK=off DOWNLOAD_GAMEPACKS=no \
        MACLIBDIR="${MESH_MACLIBDIR:-/opt/homebrew/lib}"; then
  ln -s "$mesh_stage" "$mesh_stage/current"
  mv -fh "$mesh_stage/current" "$mesh_build/current"
  echo "compiler ready: $mesh_build/current/install/q3map2"
else
  echo "compiler build failed; previous compiler retained; diagnostic generation: $mesh_stage" >&2
  exit 1
fi
