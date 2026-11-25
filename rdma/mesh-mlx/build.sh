#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RDMA_DIR="${RDMA_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY="${PY:-$HOME/gemma-e2b-pareto/.venv/bin/python}"
NB="${NB:-$HOME/mesh-resume-mlx/nanobind}"
OVERLAY="${OVERLAY:-$HOME/mesh-resume-mlx/mlx-overlay}"
OUT_DIR="${OUT_DIR:-$HOME/gemma-e2b-pareto-switched}"
BUILD_DIR="${BUILD_DIR:-/tmp/meshmlx-flat-build}"

make -C "$RDMA_DIR" .build/libmesh.dylib >/dev/null
LIBMESH_DIR="$RDMA_DIR/.build"
PYINC="$($PY -c "import sysconfig; print(sysconfig.get_paths()['include'])")"
MLX="$($PY -c "import os, mlx.core as m; print(os.path.dirname(m.__file__))")"

mkdir -p "$BUILD_DIR/obj"
if [ ! -f "$BUILD_DIR/libnanobind-static.a" ]; then
  NBF="-O3 -DNDEBUG -std=gnu++20 -arch arm64 -fPIC -fvisibility=hidden -fno-strict-aliasing -fno-stack-protector -DNB_BUILD -DNB_COMPACT_ASSERTIONS -I$PYINC -I$NB/include -isystem $NB/ext/robin_map/include"
  pids=()
  for s in nb_internals nb_func nb_type nb_enum nb_ndarray nb_static_property nb_datetime common error trampoline implicit; do
    clang++ $NBF -c "$NB/src/$s.cpp" -o "$BUILD_DIR/obj/$s.o" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  ar qc "$BUILD_DIR/libnanobind-static.a" "$BUILD_DIR"/obj/nb_*.o "$BUILD_DIR"/obj/common.o "$BUILD_DIR"/obj/error.o "$BUILD_DIR"/obj/trampoline.o "$BUILD_DIR"/obj/implicit.o
  ranlib "$BUILD_DIR/libnanobind-static.a"
fi

clang++ -O3 -DNDEBUG -std=gnu++20 -flto=thin -arch arm64 -fPIC -fvisibility=hidden -fno-stack-protector \
  -DACCELERATE_NEW_LAPACK -D_METAL_ -DNB_DOMAIN=mlx -D_meshmlx_EXPORTS \
  -I"$RDMA_DIR" -I"$PYINC" -I"$NB/include" -I"$OVERLAY" \
  -isystem "$MLX/include/metal_cpp" -isystem "$MLX/include" \
  -c "$SCRIPT_DIR/meshmlx.cpp" -o "$BUILD_DIR/obj/meshmlx.o"

clang++ -O3 -DNDEBUG -flto=thin -arch arm64 -bundle -Wl,-headerpad_max_install_names \
  -Wl,@"$NB/cmake/darwin-ld-cpython.sym" -Wl,-dead_strip -Wl,-x -Wl,-S \
  -o "$BUILD_DIR/_meshmlx.cpython-313-darwin.so" \
  "$BUILD_DIR/obj/meshmlx.o" "$BUILD_DIR/libnanobind-static.a" \
  -L"$LIBMESH_DIR" -L"$MLX/lib" -Wl,-rpath,"$LIBMESH_DIR" -Wl,-rpath,"$MLX/lib" \
  -lmlx -lmesh -framework Metal -framework Foundation -framework QuartzCore -framework Accelerate

mkdir -p "$OUT_DIR"
cp "$BUILD_DIR/_meshmlx.cpython-313-darwin.so" "$OUT_DIR/_meshmlx.cpython-313-darwin.so"
cd "$OUT_DIR" && "$PY" -c "import mlx.core, _meshmlx as mm; print('import ok, flat-DMA meshmlx ready:', hasattr(mm, 'fused_sum_norm'))"
