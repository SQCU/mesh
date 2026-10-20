# design/transport.md: the node's images, built in .build by `all`, which installs nothing; `install` renames
# each that changed into rdma/, in this order: the libraries first, the bridge's binary last (bin/mesh-converge.sh swap's)
IMAGES = libmesh.dylib libnccl-mesh.dylib libmesh-coreml.dylib mesh-stat mesh-flow
MESH_CODESIGN_IDENTITY ?= Mesh Bridge
# every warning an error, but a deprecation in the installed binaries, which each node builds with its own SDK (a newer
# one's deprecation must not leave a node's bridge unbuilt); native-audit keeps every warning an error
WARN = -Wall -Wextra -Werror -Wno-error=deprecated-declarations
all: $(IMAGES:%=.build/%)
native-audit: .build/mesh-link.s .build/mesh-records.s .build/mesh-call.s .build/mesh-dataflow.s
.build/%.s: %.c mesh.h mesh-plan.h mesh-verbs.h mesh-bridge.h mesh-dataflow.h mesh-call.h Makefile
	mkdir -p .build
	cc $(CPPFLAGS) -fblocks -O2 -Wall -Wextra -Werror -S -o $@ $<
# the bridge and mesh-stat read the region through libmesh: beside them once installed, and at ../rdma from $MESH_ROOT/bin
LIBMESH = -L.build -lmesh -Wl,-rpath,@loader_path -Wl,-rpath,@loader_path/../rdma
# design/transport.md: the bridge, mesh-link.c and mesh-records.c over mesh-bridge.h
.build/mesh-flow: mesh-link.c mesh-records.c mesh-bridge.h mesh-verbs.h mesh.h mesh-plan.h mesh-dataflow.h mesh-call.h .build/libmesh.dylib Makefile
	cc $(CPPFLAGS) -fblocks -O2 $(WARN) -o $@ mesh-link.c mesh-records.c $(LIBMESH) -lrdma -framework SystemConfiguration -framework CoreFoundation -framework IOKit
	codesign --force --sign "$(MESH_CODESIGN_IDENTITY)" --identifier io.mesh.bridge $@ || codesign --force --sign - --identifier io.mesh.bridge $@
.build/mesh-stat: mesh-stat.c mesh.h mesh-plan.h .build/libmesh.dylib Makefile
	cc -O2 $(WARN) -o $@ mesh-stat.c $(LIBMESH)
# design/transport.md (README.md, Convergence; bin/mesh-converge.sh swap's form): each built image whose bytes
# differ from the installed one's copied beside it and renamed into place, in IMAGES' order, and the bridge's binary
# anew wherever libmesh changed; a running process keeps the files it mapped, and the bridge runs a new binary once
# stopped by SIGTERM (bin/mesh-user.sh; design/transport.md section 9)
install: all
	@moved=; for f in $(IMAGES); do \
	  if cmp -s .build/$$f $$f; then [ $$f = mesh-flow ] && [ -n "$$moved" ] || continue; \
	  else [ $$f != libmesh.dylib ] || moved=1; fi; \
	  cp .build/$$f $$f.install && mv -f $$f.install $$f || exit 1; echo "rdma/$$f installed"; \
	done
# design/transport.md: the compiled collectives' advisories (mesh_program_advise) over the loopback
# fabric's maps, at .build/mesh-advise with libmesh's sources compiled in (the installed libmesh.dylib untouched); not in `all`
mesh-advise: .build/mesh-advise
.build/mesh-advise: mesh-advise.c mesh-dataflow.c mesh-dataflow.h mesh-call.c mesh-call.h mesh-collective.c mesh-collective.h mesh.h mesh-plan.h \
  nccl.h Makefile
	mkdir -p .build
	cc -fblocks -O2 -Wall -Wextra -Werror -o $@ mesh-advise.c mesh-dataflow.c mesh-call.c mesh-collective.c
.PHONY: mesh-advise
# design/transport.md: N bridges on one host over loopverbs.c (shared memory in place of RDMA),
# each its configuration's node, region and loopback ports: never a production bridge (no RDMA, ad-hoc signed); the
# deployed mesh-flow is built from the same source without MESH_LOOP_FABRIC and stays as it is
LOOP = .build/loop
mesh-flow-loop: $(LOOP)/mesh-flow-loop
$(LOOP)/mesh-flow-loop: mesh-link.c mesh-records.c mesh-bridge.h mesh-verbs.h mesh.h mesh-plan.h mesh-dataflow.h mesh-dataflow.c mesh-call.h mesh-call.c mesh-collective.h \
  mesh-collective.c loopverbs.c Makefile
	mkdir -p $(LOOP)
	cc $(CPPFLAGS) -fblocks -O2 $(WARN) -DMESH_LOOP_FABRIC -o $@ mesh-link.c mesh-records.c mesh-dataflow.c mesh-call.c mesh-collective.c loopverbs.c \
	  -framework SystemConfiguration -framework CoreFoundation -framework IOKit
	codesign --force --sign - --identifier io.mesh.bridge.loop $@
# host: the loop's clients: libmesh bound to a loop bridge (mesh_loop_bind), libnccl-mesh with the walk's kernels on the
# CPU (nccl-mesh-host.c), both at .build/host under the libraries' install names, and the loop's driver (mesh-loop.c)
HOST = .build/host
host: $(HOST)/libmesh.dylib $(HOST)/libnccl-mesh.dylib $(HOST)/mesh-loop $(LOOP)/mesh-flow-loop
$(HOST)/libmesh.dylib: mesh-dataflow.c mesh-dataflow.h mesh-call.c mesh-call.h mesh-collective.c mesh-collective.h mesh.h mesh-plan.h Makefile
	mkdir -p $(HOST)
	cc -fblocks -O2 $(WARN) -DMESH_LOOP_FABRIC -dynamiclib -Wl,-install_name,@rpath/libmesh.dylib -o $@ mesh-dataflow.c mesh-call.c mesh-collective.c
$(HOST)/libnccl-mesh.dylib: nccl-mesh.c nccl-mesh-host.c nccl-mesh-metal.h nccl.h mesh-collective.h mesh-call.h mesh-dataflow.h \
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
.build/libnccl-mesh.dylib: nccl-mesh.c nccl-mesh-metal.m nccl-mesh-metal.h .build/mesh_metal.h nccl.h \
  mesh-collective.h mesh-call.h mesh-dataflow.h mesh.h mesh-plan.h .build/libmesh.dylib Makefile
	cc -fblocks -O2 $(WARN) -dynamiclib -I .build -Wl,-install_name,@rpath/libnccl-mesh.dylib -Wl,-rpath,@loader_path \
	  $(foreach f,$(NCCL_API),-Wl,-alias,_$(f),_p$(f)) -o $@ nccl-mesh.c nccl-mesh-metal.m -L.build -lmesh \
	  -framework Metal -framework Foundation
.build/libmesh-coreml.dylib: mesh-coreml.m Makefile
	mkdir -p .build
	cc -fobjc-arc -O2 $(WARN) -dynamiclib -Wl,-install_name,@rpath/libmesh-coreml.dylib -o $@ mesh-coreml.m \
	  -framework CoreML -framework Foundation
.build/libmesh.dylib: mesh-dataflow.c mesh-dataflow.h mesh-call.c mesh-call.h mesh-collective.c mesh-collective.h mesh.h mesh-plan.h Makefile
	mkdir -p .build
	cc -fblocks -O2 $(WARN) -dynamiclib -Wl,-install_name,@rpath/libmesh.dylib -o $@ mesh-dataflow.c mesh-call.c mesh-collective.c
# the built images; never an installed image (the launchd job's binary and the libraries clients map)
clean:
	rm -rf $(IMAGES:%=.build/%)
.PHONY: all install clean native-audit

# _mesh_c: the headers as one Python module (mesh_c_build.py), for the interpreter that imports it, bound to the
# installed libraries it loads (make install puts them in rdma/; this target installs nothing, so it never stops the
# bridge: design/transport.md)
PYTHON ?= $(HOME)/.venv-mesh-uv/bin/python
mesh-c: mesh_c_build.py mesh-plan.h nccl.h
	$(PYTHON) mesh_c_build.py
.PHONY: mesh-c
