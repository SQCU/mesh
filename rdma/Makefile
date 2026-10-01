BINS = mesh-flow mesh-stat libmesh.dylib libnccl-mesh.dylib
MESH_CODESIGN_IDENTITY ?= Mesh Bridge
MESH_PREFIX ?= /usr/local/mesh
all: $(BINS)
native-audit: .build/mesh-flow.s .build/mesh-dataflow.s .build/mesh-net.s
.build/%.s: %.c mesh.h mesh-verbs.h mesh-dataflow.h mesh-net.h Makefile
	mkdir -p .build
	cc $(CPPFLAGS) -fblocks -O2 -Wall -Wextra -Werror -S -o $@ $<
mesh-flow: mesh-flow.c mesh-verbs.h mesh-disk.h mesh.h mesh-dataflow.h mesh-dataflow.c Makefile
	mkdir -p .build
	cc $(CPPFLAGS) -fblocks -O2 -o .build/$@ mesh-flow.c mesh-dataflow.c -lrdma
	codesign --force --sign "$(MESH_CODESIGN_IDENTITY)" --identifier io.mesh.bridge .build/$@ || codesign --force --sign - --identifier io.mesh.bridge .build/$@
	mv -f .build/$@ $@
mesh-stat: mesh-stat.c mesh.h
	cc -O2 -o $@ mesh-stat.c
libmesh.dylib: mesh-dataflow.c mesh-dataflow.h mesh-collective.c mesh-collective.h mesh-net.c mesh-net.h mesh.h Makefile
	cc -fblocks -O2 -dynamiclib -Wl,-install_name,@rpath/libmesh.dylib -o $@ mesh-dataflow.c mesh-collective.c mesh-net.c
# NCCL's API (nccl.h) over the communicator sessions and the collective planner of libmesh.dylib; every
# function also under its pncclX profiling name, as NCCL exports both
NCCL_API = ncclMemAlloc ncclMemFree ncclGetVersion ncclGetUniqueId ncclCommInitRankConfig ncclCommInitRank \
  ncclCommInitAll ncclCommFinalize ncclCommDestroy ncclCommAbort ncclCommSplit ncclGetErrorString ncclGetLastError \
  ncclCommGetAsyncError ncclCommCount ncclCommCuDevice ncclCommUserRank ncclCommRegister ncclCommDeregister \
  ncclRedOpCreatePreMulSum ncclRedOpDestroy ncclReduce ncclBcast ncclBroadcast ncclAllReduce ncclReduceScatter \
  ncclAllGather ncclAlltoAll ncclGather ncclScatter ncclAllReduceConfig ncclBroadcastConfig ncclReduceConfig \
  ncclAllGatherConfig ncclReduceScatterConfig ncclAlltoAllConfig ncclGatherConfig ncclScatterConfig ncclSend ncclRecv \
  ncclGroupStart ncclGroupEnd
libnccl-mesh.dylib: nccl-mesh.c nccl-mesh-metal.m nccl.h mesh-net.h mesh-collective.h mesh-dataflow.h mesh.h libmesh.dylib Makefile
	cc -fblocks -O2 -Wall -Wextra -dynamiclib -Wl,-install_name,@rpath/libnccl-mesh.dylib -Wl,-rpath,@loader_path \
	  $(foreach f,$(NCCL_API),-Wl,-alias,_$(f),_p$(f)) -o $@ nccl-mesh.c nccl-mesh-metal.m -L. -lmesh -framework Metal -framework Foundation
clean:
	rm -f $(BINS)
.PHONY: all clean native-audit
install-bridge: mesh-flow mesh-stat
	install -d $(MESH_PREFIX)/bin
	install -m 755 mesh-flow mesh-stat $(MESH_PREFIX)/bin/
	codesign -dv $(MESH_PREFIX)/bin/mesh-flow 2>&1 | grep -E "^Identifier=io.mesh.bridge$$"
