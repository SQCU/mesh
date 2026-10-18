BINS = mesh-flow mesh-stat libmesh.dylib libnccl-mesh.dylib libmesh-coreml.dylib libmesh_swift.dylib
# the Swift module (../swift: the numerical executor's API, mesh-call.h) and its callers (../examples)
SWIFT = ../swift/Mesh.swift ../swift/Bounds.swift ../swift/Topology.swift
EXAMPLES = linear-chain sync-on-remote-fill bounds-table coreml-chain indexed-gather gram-chain
MESH_CODESIGN_IDENTITY ?= Mesh Bridge
# every warning an error, but a deprecation in the installed binaries, which each node builds with its own SDK (a newer
# one's deprecation must not leave a node's bridge unbuilt); native-audit keeps every warning an error
WARN = -Wall -Wextra -Werror -Wno-error=deprecated-declarations
all: $(BINS)
examples: $(EXAMPLES)
native-audit: .build/mesh-link.s .build/mesh-program.s .build/mesh-call.s .build/mesh-dataflow.s .build/Mesh.s
.build/Mesh.s: $(SWIFT) mesh.h mesh-plan.h mesh-call.h mesh-collective.h mesh-dataflow.h module.modulemap Makefile
	mkdir -p .build
	swiftc -O -whole-module-optimization -parse-as-library -emit-assembly -module-name Mesh -I . $(SWIFT) -o $@
.build/%.s: %.c mesh.h mesh-plan.h mesh-verbs.h mesh-bridge.h mesh-dataflow.h mesh-call.h Makefile
	mkdir -p .build
	cc $(CPPFLAGS) -fblocks -O2 -Wall -Wextra -Werror -S -o $@ $<
# the bridge and mesh-stat read the region through libmesh: beside them here, and at ../rdma from $MESH_ROOT/bin
LIBMESH = -L. -lmesh -Wl,-rpath,@loader_path -Wl,-rpath,@loader_path/../rdma
# design/bridge-and-ipc.md#files: the bridge, mesh-link.c and mesh-program.c over mesh-bridge.h
mesh-flow: mesh-link.c mesh-program.c mesh-bridge.h mesh-verbs.h mesh.h mesh-plan.h mesh-dataflow.h mesh-call.h libmesh.dylib Makefile
	mkdir -p .build
	cc $(CPPFLAGS) -fblocks -O2 $(WARN) -o .build/$@ mesh-link.c mesh-program.c $(LIBMESH) -lrdma -framework SystemConfiguration -framework CoreFoundation -framework IOKit
	codesign --force --sign "$(MESH_CODESIGN_IDENTITY)" --identifier io.mesh.bridge .build/$@ || codesign --force --sign - --identifier io.mesh.bridge .build/$@
	mv -f .build/$@ $@
mesh-stat: mesh-stat.c mesh.h mesh-plan.h libmesh.dylib Makefile
	cc -O2 $(WARN) -o $@ mesh-stat.c $(LIBMESH)
# the one-way stream readings (design/transport-one-gib-2026-09-10.md): the bridge's raw stream and its own path, a
# client of libmesh, at .build/mesh-stream (it finds libmesh at ../rdma from there); built on its own, not in `all`
mesh-stream: .build/mesh-stream
.build/mesh-stream: mesh-stream.c mesh.h mesh-plan.h mesh-dataflow.h mesh-call.h mesh-collective.h libmesh.dylib Makefile
	mkdir -p .build
	cc -O2 $(WARN) -o $@ mesh-stream.c $(LIBMESH)
.PHONY: mesh-stream
# design/collectives-dataflow.md#advisories: the compiled collectives' advisories (mesh_program_advise) over the loopback
# fabric's maps, at .build/mesh-advise with libmesh's sources compiled in (the installed libmesh.dylib untouched); not in `all`
mesh-advise: .build/mesh-advise
.build/mesh-advise: mesh-advise.c mesh-dataflow.c mesh-dataflow.h mesh-call.c mesh-call.h mesh-collective.c mesh-collective.h mesh.h mesh-plan.h Makefile
	mkdir -p .build
	cc -fblocks -O2 -Wall -Wextra -Werror -o $@ mesh-advise.c mesh-dataflow.c mesh-call.c mesh-collective.c
.PHONY: mesh-advise
# design/bridge-dataflow.md#the-loopback-fabric: N bridges on one host over loopverbs.c (shared memory in place of RDMA),
# each its configuration's node, region and loopback ports: never a production bridge (no RDMA, ad-hoc signed); the
# deployed mesh-flow is built from the same source without MESH_LOOP_FABRIC and stays as it is
LOOP = .build/loop
mesh-flow-loop: $(LOOP)/mesh-flow-loop
$(LOOP)/mesh-flow-loop: mesh-link.c mesh-program.c mesh-bridge.h mesh-verbs.h mesh.h mesh-plan.h mesh-dataflow.h mesh-dataflow.c mesh-call.h mesh-call.c mesh-collective.h \
  mesh-collective.c loopverbs.c Makefile
	mkdir -p $(LOOP)
	cc $(CPPFLAGS) -fblocks -O2 $(WARN) -DMESH_LOOP_FABRIC -o $@ mesh-link.c mesh-program.c mesh-dataflow.c mesh-call.c mesh-collective.c loopverbs.c \
	  -framework SystemConfiguration -framework CoreFoundation -framework IOKit
	codesign --force --sign - --identifier io.mesh.bridge.loop $@
# host: the loop's clients: libmesh bound to a loop bridge (mesh_loop_bind), libnccl-mesh with the walk's kernels on the
# CPU (nccl-mesh-host.c), both at .build/host under the libraries' install names, and the loop's driver (mesh-loop.c)
HOST = .build/host
host: $(HOST)/libmesh.dylib $(HOST)/libnccl-mesh.dylib $(HOST)/mesh-loop $(LOOP)/mesh-flow-loop
$(HOST)/libmesh.dylib: mesh-dataflow.c mesh-dataflow.h mesh-call.c mesh-call.h mesh-collective.c mesh-collective.h mesh.h mesh-plan.h Makefile
	mkdir -p $(HOST)
	cc -fblocks -O2 $(WARN) -DMESH_LOOP_FABRIC -dynamiclib -Wl,-install_name,@rpath/libmesh.dylib -o $@ mesh-dataflow.c mesh-call.c mesh-collective.c
$(HOST)/libnccl-mesh.dylib: nccl-mesh.c nccl-mesh-host.c nccl-mesh-metal.h mesh-metal.h nccl.h mesh-collective.h mesh-call.h mesh-dataflow.h \
  mesh.h mesh-plan.h $(HOST)/libmesh.dylib Makefile
	cc -fblocks -O2 $(WARN) -DMESH_HOST_EXECUTOR -DMESH_LOOP_FABRIC -dynamiclib -Wl,-install_name,@rpath/libnccl-mesh.dylib \
	  -Wl,-rpath,@loader_path $(foreach f,$(NCCL_API),-Wl,-alias,_$(f),_p$(f)) -o $@ nccl-mesh.c nccl-mesh-host.c -L$(HOST) -lmesh
$(HOST)/mesh-loop: mesh-loop.c nccl.h mesh-collective.h mesh-call.h mesh-dataflow.h mesh.h mesh-plan.h $(HOST)/libmesh.dylib \
  $(HOST)/libnccl-mesh.dylib Makefile
	cc -fblocks -O2 $(WARN) -DMESH_LOOP_FABRIC -o $@ mesh-loop.c -L$(HOST) -lmesh -lnccl-mesh -Wl,-rpath,@loader_path
.PHONY: mesh-flow-loop host
# NCCL's API (nccl.h) on the prepared transfers and the collective planner of libmesh.dylib; every function also
# under its pncclX profiling name, as NCCL exports both
NCCL_API = ncclMemAlloc ncclMemFree ncclGetVersion ncclGetUniqueId ncclCommInitRankConfig ncclCommInitRank \
  ncclCommInitAll ncclCommFinalize ncclCommDestroy ncclCommAbort ncclCommSplit ncclCommShrink ncclGetErrorString \
  ncclGetLastError ncclCommGetAsyncError ncclCommCount ncclCommCuDevice ncclCommUserRank ncclCommRegister \
  ncclCommDeregister ncclRedOpCreatePreMulSum ncclRedOpDestroy ncclReduce ncclBcast ncclBroadcast ncclAllReduce \
  ncclReduceScatter ncclAllGather ncclAlltoAll ncclGather ncclScatter ncclAllReduceConfig ncclBroadcastConfig \
  ncclReduceConfig ncclAllGatherConfig ncclReduceScatterConfig ncclAlltoAllConfig ncclGatherConfig ncclScatterConfig \
  ncclSend ncclRecv ncclGroupStart ncclGroupEnd ncclCommRevoke ncclCommGetUniqueId ncclCommGrow
# rdma/mesh.metal, the GPU transport primitives, is libnccl-mesh's kernels' one copy of them: its text embedded in
# nccl-mesh-metal.m's source as the header mesh_metal.h
.build/mesh_metal.h: mesh.metal
	mkdir -p .build
	xxd -i mesh.metal > $@
libnccl-mesh.dylib: nccl-mesh.c nccl-mesh-metal.m nccl-mesh-metal.h mesh-metal.m mesh-metal.h .build/mesh_metal.h nccl.h \
  mesh-collective.h mesh-call.h mesh-dataflow.h mesh.h mesh-plan.h libmesh.dylib Makefile
	cc -fblocks -O2 $(WARN) -dynamiclib -I .build -Wl,-install_name,@rpath/libnccl-mesh.dylib -Wl,-rpath,@loader_path \
	  $(foreach f,$(NCCL_API),-Wl,-alias,_$(f),_p$(f)) -o $@ nccl-mesh.c nccl-mesh-metal.m mesh-metal.m -L. -lmesh \
	  -framework Metal -framework Foundation
libmesh-coreml.dylib: mesh-coreml.m Makefile
	cc -fobjc-arc -O2 $(WARN) -dynamiclib -Wl,-install_name,@rpath/libmesh-coreml.dylib -o $@ mesh-coreml.m \
	  -framework CoreML -framework Foundation
libmesh.dylib: mesh-dataflow.c mesh-dataflow.h mesh-call.c mesh-call.h mesh-collective.c mesh-collective.h mesh.h mesh-plan.h Makefile
	cc -fblocks -O2 $(WARN) -dynamiclib -Wl,-install_name,@rpath/libmesh.dylib -o $@ mesh-dataflow.c mesh-call.c mesh-collective.c
libmesh_swift.dylib: $(SWIFT) libmesh.dylib module.modulemap mesh.h mesh-plan.h mesh-call.h mesh-collective.h mesh-dataflow.h Makefile
	swiftc -O -whole-module-optimization -emit-library -emit-module -module-name Mesh -I . -L . -lmesh -Xlinker -rpath -Xlinker @loader_path -Xlinker -install_name -Xlinker @rpath/libmesh_swift.dylib $(SWIFT) -o $@
linear-chain: ../examples/linear-chain.swift libmesh_swift.dylib
	swiftc -O -parse-as-library -I . -L . -lmesh_swift -Xlinker -rpath -Xlinker @loader_path $< -o $@
sync-on-remote-fill: ../examples/sync-on-remote-fill.swift libmesh_swift.dylib
	swiftc -O -parse-as-library -I . -L . -lmesh_swift -Xlinker -rpath -Xlinker @loader_path $< -o $@
bounds-table: ../examples/bounds-table.swift libmesh_swift.dylib
	swiftc -O -parse-as-library -I . -L . -lmesh_swift -Xlinker -rpath -Xlinker @loader_path $< -o $@
coreml-chain: ../examples/coreml-chain.swift libmesh_swift.dylib
	swiftc -O -parse-as-library -I . -L . -lmesh_swift -Xlinker -rpath -Xlinker @loader_path $< -o $@
indexed-gather: ../examples/indexed-gather.swift libmesh_swift.dylib
	swiftc -O -parse-as-library -I . -L . -lmesh_swift -Xlinker -rpath -Xlinker @loader_path $< -o $@
gram-chain: ../examples/gram-chain.swift libmesh_swift.dylib
	swiftc -O -parse-as-library -Xcc -DACCELERATE_NEW_LAPACK -I . -L . -lmesh_swift -Xlinker -rpath -Xlinker @loader_path $< -o $@
clean:
	rm -f $(BINS) $(EXAMPLES) Mesh.swiftmodule Mesh.swiftdoc Mesh.swiftsourceinfo Mesh.abi.json
.PHONY: all examples clean native-audit

# _mesh_c: the headers as one Python module (mesh_c_build.py), for the interpreter that imports it
PYTHON ?= $(HOME)/.venv-mesh-uv/bin/python
mesh-c: mesh_c_build.py mesh-plan.h nccl.h libmesh.dylib libnccl-mesh.dylib
	$(PYTHON) mesh_c_build.py
.PHONY: mesh-c
