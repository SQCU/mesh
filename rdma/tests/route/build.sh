#!/bin/sh
# builds libroute.dylib (the environment and deployed forward in C, the int8 network's Metal library, the Swift
# trainer behind trainer.h), which the Python driver (py/route) loads; tests/switch.c's policy router; int8check.
# Run from tests/route.
set -e
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-26.4.1.app/Contents/Developer}
xcrun -sdk macosx cc -std=c11 -O2 -DACCELERATE_NEW_LAPACK -I ../.. -I .. -c sim.c -o sim.o
xcrun -sdk macosx cc -std=c11 -O2 -c telemetry.c -o telemetry.o
xcrun -sdk macosx cc -std=c11 -O2 -DACCELERATE_NEW_LAPACK -I ../.. -I .. -ffp-contract=off -c forward.c -o forward.o
# the exchange: a client of this node's installed libnccl-mesh (MESH_RDMA, default ~/mesh/rdma: its header and library)
MESH_RDMA=${MESH_RDMA:-$HOME/mesh/rdma}
xcrun -sdk macosx cc -std=c11 -O2 -I "$MESH_RDMA" -c exchange.c -o exchange.o
xcrun -sdk macosx metal -std=metal4.0 -O3 -I . -c int8.metal -o int8.air
xcrun -sdk macosx metal -std=metal4.0 -O3 -I . -c net.metal -o net.air
xcrun -sdk macosx metallib int8.air net.air -o route.metallib
swiftc -O -parse-as-library -emit-library -module-name route -import-objc-header trainer.h policy.swift learn.swift net.swift runs.swift trainer.swift \
  sim.o forward.o exchange.o telemetry.o -framework Accelerate -L "$MESH_RDMA" -lnccl-mesh -Xlinker -rpath -Xlinker "$MESH_RDMA" \
  -Xlinker -install_name -Xlinker @rpath/libroute.dylib -o libroute.dylib
swiftc -O -import-objc-header int8mm.h int8check.swift -o int8check
xcrun -sdk macosx cc -std=c11 -O2 -DROUTE_POLICY -DACCELERATE_NEW_LAPACK -I ../.. -I .. -I . -ffp-contract=off ../switch.c sim.c forward.c telemetry.c -framework Accelerate -lm -o switch-policy
# the OCaml plugin (py/ocaml): where OCaml is installed; elsewhere the one built beside it is used (it carries its runtime)
if command -v ocamlopt >/dev/null; then py/ocaml/build.sh; fi
