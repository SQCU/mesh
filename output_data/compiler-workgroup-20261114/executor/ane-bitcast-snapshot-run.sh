#!/bin/bash
set -u
D=/Users/mdot/dox/mesh/output_data/compiler-workgroup-20261114/executor
F=(/Users/mdot/dox/mesh/compiler/mesh_compiler/{operations,decompose,export,ir,native,metal,schedule,bridge,lower_metal,driver,runtime}.py
   /Users/mdot/dox/mesh/rdma/mesh.py /Users/mdot/dox/mesh/rdma/libmesh.dylib
   /Users/mdot/metal-microbench/.build/libmetal_recording.dylib /Users/mdot/metal-microbench/.build/libmetal_ir.dylib
   /tmp/mesh-ane-snapshot.py)
shasum -a 256 "${F[@]}" > $D/ane-bitcast-snapshot.before.sha256
( cd /Users/mdot/dox/mesh && git rev-parse HEAD ) > $D/ane-bitcast-snapshot.heads
( cd /Users/mdot/metal-microbench && git rev-parse HEAD ) >> $D/ane-bitcast-snapshot.heads
start=$(date +%s)
env PYTHONPATH=/Users/mdot/dox/mesh/compiler:/Users/mdot/dox/mesh/rdma TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  DYLD_INSERT_LIBRARIES=/Users/mdot/metal-microbench/.build/libmetal_recording.dylib \
  DYLD_LIBRARY_PATH=/Users/mdot/metal-microbench/.build:/Users/mdot/dox/mesh/rdma \
  /Users/mdot/.venv-mesh-uv/bin/python -u /tmp/mesh-ane-snapshot.py $D/ane-bitcast-snapshot.json \
  > $D/ane-bitcast-snapshot.stdout 2> $D/ane-bitcast-snapshot.stderr
echo "exit=$? seconds=$(( $(date +%s) - start ))" > $D/ane-bitcast-snapshot.exit
shasum -a 256 "${F[@]}" > $D/ane-bitcast-snapshot.after.sha256
cmp -s $D/ane-bitcast-snapshot.before.sha256 $D/ane-bitcast-snapshot.after.sha256 && echo hashes-unchanged || echo hashes-CHANGED
cat $D/ane-bitcast-snapshot.exit
