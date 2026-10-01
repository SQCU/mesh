BINS = mesh-flow mesh-stat libmesh.dylib
MESH_CODESIGN_IDENTITY ?= Mesh Bridge
all: $(BINS)
native-audit: .build/mesh-flow.s .build/mesh-call.s .build/mesh-dataflow.s
.build/%.s: %.c mesh.h mesh-verbs.h mesh-dataflow.h mesh-call.h Makefile
	mkdir -p .build
	cc $(CPPFLAGS) -fblocks -O2 -Wall -Wextra -Werror -S -o $@ $<
mesh-flow: mesh-flow.c mesh-verbs.h mesh.h mesh-dataflow.h mesh-call.h mesh-dataflow.c mesh-call.c Makefile
	mkdir -p .build
	cc $(CPPFLAGS) -fblocks -O2 -Wall -Wextra -o .build/$@ mesh-flow.c mesh-dataflow.c mesh-call.c -lrdma
	codesign --force --sign "$(MESH_CODESIGN_IDENTITY)" --identifier io.mesh.bridge .build/$@ || codesign --force --sign - --identifier io.mesh.bridge .build/$@
	mv -f .build/$@ $@
mesh-stat: mesh-stat.c mesh.h
	cc -O2 -o $@ mesh-stat.c
libmesh.dylib: mesh-dataflow.c mesh-dataflow.h mesh-call.c mesh-call.h mesh-collective.c mesh-collective.h mesh.h Makefile
	cc -fblocks -O2 -dynamiclib -Wl,-install_name,@rpath/libmesh.dylib -o $@ mesh-dataflow.c mesh-call.c mesh-collective.c -lobjc
# NCCL's API (nccl.h) on the communicator streams and the collective planner of libmesh.dylib; every function
# also under its pncclX profiling name, as NCCL exports both
NCCL_API = ncclMemAlloc ncclMemFree ncclGetVersion ncclGetUniqueId ncclCommInitRankConfig ncclCommInitRank \
  ncclCommInitAll ncclCommFinalize ncclCommDestroy ncclCommAbort ncclCommSplit ncclGetErrorString ncclGetLastError \
  ncclCommGetAsyncError ncclCommCount ncclCommCuDevice ncclCommUserRank ncclCommRegister ncclCommDeregister \
  ncclRedOpCreatePreMulSum ncclRedOpDestroy ncclReduce ncclBcast ncclBroadcast ncclAllReduce ncclReduceScatter \
  ncclAllGather ncclAlltoAll ncclGather ncclScatter ncclAllReduceConfig ncclBroadcastConfig ncclReduceConfig \
  ncclAllGatherConfig ncclReduceScatterConfig ncclAlltoAllConfig ncclGatherConfig ncclScatterConfig ncclSend ncclRecv \
  ncclGroupStart ncclGroupEnd
libnccl-mesh.dylib: nccl-mesh.c nccl-mesh-metal.m nccl.h mesh-collective.h mesh-call.h mesh-dataflow.h mesh.h libmesh.dylib Makefile
	cc -fblocks -O2 -Wall -Wextra -dynamiclib -Wl,-install_name,@rpath/libnccl-mesh.dylib -Wl,-rpath,@loader_path \
	  $(foreach f,$(NCCL_API),-Wl,-alias,_$(f),_p$(f)) -o $@ nccl-mesh.c nccl-mesh-metal.m -L. -lmesh -framework Metal -framework Foundation
clean:
	rm -f $(BINS)
.PHONY: all clean native-audit
