#!/bin/sh
# builds libroute.dylib (the environment, its array kernels and the deployed forward in C, the int8 network's Metal
# library and its interface in Swift: kernels.h), which the Python driver (py/route) loads; tests/switch.c's policy router;
# stages (the products against the roofline, the network's build-up stage by stage, its checks).
# Run from tests/route.
set -e
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-26.4.1.app/Contents/Developer}
xcrun -sdk macosx cc -std=c11 -O2 -DACCELERATE_NEW_LAPACK -I ../.. -I .. -c sim.c -o sim.o
xcrun -sdk macosx cc -std=c11 -O2 -I ../.. -I .. -c sims.c -o sims.o
xcrun -sdk macosx cc -std=c11 -O2 -c telemetry.c -o telemetry.o
xcrun -sdk macosx cc -std=c11 -O2 -DACCELERATE_NEW_LAPACK -I ../.. -I .. -ffp-contract=off -c forward.c -o forward.o
# the exchange: a client of this node's installed libnccl-mesh (MESH_RDMA, default ~/mesh/rdma: its header and library)
MESH_RDMA=${MESH_RDMA:-$HOME/mesh/rdma}
xcrun -sdk macosx cc -std=c11 -O2 -I "$MESH_RDMA" -c exchange.c -o exchange.o
# without fast math: act.h's bit-for-bit contract with forward.c (credit.metal: no product, fast)
for k in mm fwd bwd opt; do xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math -I . -c $k.metal -o $k.air; done
xcrun -sdk macosx metal -std=metal4.0 -O3 -I . -c credit.metal -o credit.air
xcrun -sdk macosx metallib mm.air fwd.air bwd.air opt.air credit.air -o route.metallib
swiftc -O -parse-as-library -emit-library -module-name route -import-objc-header swift.h policy.swift learn.swift model.swift netapi.swift \
  sim.o sims.o forward.o exchange.o telemetry.o -framework Accelerate -L "$MESH_RDMA" -lnccl-mesh -Xlinker -rpath -Xlinker "$MESH_RDMA" \
  -Xlinker -install_name -Xlinker @rpath/libroute.dylib -o libroute.dylib
swiftc -O -import-objc-header swift.h -I ../.. -I .. stages.swift -L . -lroute -Xlinker -rpath -Xlinker @executable_path -o stages
xcrun -sdk macosx cc -std=c11 -O2 -DROUTE_POLICY -DACCELERATE_NEW_LAPACK -I ../.. -I .. -I . -ffp-contract=off ../switch.c sim.c forward.c telemetry.c -framework Accelerate -lm -o switch-policy
# the OCaml plugin (py/ocaml): where OCaml is installed; elsewhere the one built beside it is used (it carries its runtime)
if command -v ocamlopt >/dev/null; then py/ocaml/build.sh; fi
