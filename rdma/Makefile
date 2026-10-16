BINS = mesh-flow mesh-stat libmesh.dylib libnccl-mesh.dylib libmesh-coreml.dylib
MESH_CODESIGN_IDENTITY ?= Mesh Bridge
# every warning an error, but a deprecation in the installed binaries, which each node builds with its own SDK (a newer
# one's deprecation must not leave a node's bridge unbuilt); native-audit keeps every warning an error
WARN = -Wall -Wextra -Werror -Wno-error=deprecated-declarations
all: $(BINS)
native-audit: .build/mesh-flow.s .build/mesh-call.s .build/mesh-dataflow.s
.build/%.s: %.c mesh.h mesh-plan.h mesh-verbs.h mesh-dataflow.h mesh-call.h Makefile
	mkdir -p .build
	cc $(CPPFLAGS) -fblocks -O2 -Wall -Wextra -Werror -S -o $@ $<
# the bridge and mesh-stat read the region through libmesh: beside them here, and at ../rdma from $MESH_ROOT/bin
LIBMESH = -L. -lmesh -Wl,-rpath,@loader_path -Wl,-rpath,@loader_path/../rdma
mesh-flow: mesh-flow.c mesh-verbs.h mesh.h mesh-plan.h mesh-dataflow.h mesh-call.h libmesh.dylib Makefile
	mkdir -p .build
	cc $(CPPFLAGS) -fblocks -O2 $(WARN) -o .build/$@ mesh-flow.c $(LIBMESH) -lrdma -framework SystemConfiguration -framework CoreFoundation -framework IOKit
	codesign --force --sign "$(MESH_CODESIGN_IDENTITY)" --identifier io.mesh.bridge .build/$@ || codesign --force --sign - --identifier io.mesh.bridge .build/$@
	mv -f .build/$@ $@
mesh-stat: mesh-stat.c mesh.h mesh-plan.h libmesh.dylib Makefile
	cc -O2 $(WARN) -o $@ mesh-stat.c $(LIBMESH)
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
clean:
	rm -f $(BINS)
.PHONY: all clean native-audit

# _mesh_c: the headers as one Python module (mesh_c_build.py), for the interpreter that imports it
PYTHON ?= $(HOME)/.venv-mesh-uv/bin/python
mesh-c: mesh_c_build.py mesh-plan.h nccl.h libmesh.dylib libnccl-mesh.dylib
	$(PYTHON) mesh_c_build.py
.PHONY: mesh-c
