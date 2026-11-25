# design/transport.md: the node's images, built in .build by `all`, which installs nothing; `install` renames each
# that changed into rdma/, in IMAGES' order: the libraries first, the bridge's binary last
IMAGES = libmesh.dylib libnccl-mesh.dylib libmesh-coreml.dylib libmesh-phase.dylib mesh-stat mesh-switch-stat mesh-cluster mesh-heartbeat mesh-flow
MESH_CODESIGN_IDENTITY ?= Mesh Bridge
# every warning an error, but a deprecation in the installed binaries, which each node builds with its own SDK (a newer
# one's deprecation must not leave a node's bridge unbuilt)
WARN = -Wall -Wextra -Werror -Wno-error=deprecated-declarations
# the region's headers every image includes (mesh.h includes the other two)
MESH_H = mesh.h mesh-view.h mesh-call.h mesh-codec.h mesh-reduce.h
# the routing policy's headers the bridge includes (mesh-switched.h includes route/route.h, which includes split.h)
ROUTE_H = route/route.h route/split.h
all: $(IMAGES:%=.build/%) .build/mesh-install-state
# the bridge and mesh-stat read the region through libmesh: beside them once installed, and at ../rdma from $MESH_ROOT/bin
LIBMESH = -L.build -lmesh -Wl,-rpath,@loader_path -Wl,-rpath,@loader_path/../rdma
# design/transport.md: the bridge, mesh-link.c and mesh-records.c over mesh-bridge.h
.build/mesh-flow: mesh-link.c mesh-records.c mesh-bridge.h mesh-verbs.h mesh-observe.h mesh-switched.h $(MESH_H) \
  .build/libmesh.dylib Makefile
	mkdir -p .build
	cc $(CPPFLAGS) -O2 $(WARN) -o $@ mesh-link.c mesh-records.c $(LIBMESH) -lrdma -framework CoreFoundation -framework IOKit -framework Accelerate
	codesign --force --sign "$(MESH_CODESIGN_IDENTITY)" --identifier io.mesh.bridge $@ || codesign --force --sign - --identifier io.mesh.bridge $@
# design/transport.md section 2: the heartbeat, the job launchd keeps alive, which opens and closes the bridge
.build/mesh-heartbeat: mesh-heartbeat.c mesh-observe.h $(MESH_H) Makefile
	mkdir -p .build
	cc -O2 $(WARN) -o $@ mesh-heartbeat.c
	codesign --force --sign "$(MESH_CODESIGN_IDENTITY)" --identifier io.mesh.bridge $@ || codesign --force --sign - --identifier io.mesh.bridge $@
.build/mesh-stat: mesh-stat.c mesh-observe.h $(MESH_H) .build/libmesh.dylib Makefile
	cc -O2 $(WARN) -o $@ mesh-stat.c $(LIBMESH)
# the switch's counters, lanes and view read from the region, one JSON line
.build/mesh-switch-stat: mesh-switch-stat.c mesh-switched.h $(MESH_H) Makefile
	mkdir -p .build
	cc -O2 $(WARN) -o $@ mesh-switch-stat.c
.build/mesh-cluster: mesh-cluster.c mesh-observe.h $(MESH_H) Makefile
	mkdir -p .build
	cc -O2 $(WARN) -o $@ mesh-cluster.c
# the install's read-only preflight (rdma/mesh_install.py builds it alike): the region's live client holders
.build/mesh-install-state: mesh-install-state.c $(MESH_H) Makefile
	mkdir -p .build
	cc -O2 $(WARN) -o $@ mesh-install-state.c
# libmesh: the region's open, client slots, page table calls, arena leases and fabric view
.build/libmesh.dylib: mesh-dataflow.c mesh-call.c mesh-collective.c $(MESH_H) Makefile
	mkdir -p .build
	cc -O2 $(WARN) -dynamiclib -Wl,-install_name,@rpath/libmesh.dylib -o $@ mesh-dataflow.c mesh-call.c mesh-collective.c
# NCCL's API (nccl.h) over libmesh's page table; every function also under its pncclX profiling name, as NCCL exports
# both
NCCL_API = ncclMemAlloc ncclMemFree ncclGetVersion ncclGetUniqueId ncclCommInitRank ncclCommDestroy ncclCommAbort \
  ncclCommSplit ncclCommShrink ncclGetErrorString ncclGetLastError ncclCommGetAsyncError ncclCommCount \
  ncclCommUserRank ncclReduce ncclBroadcast ncclAllReduce ncclReduceScatter ncclAllGather ncclAlltoAll ncclGather \
  ncclScatter ncclGroupStart ncclGroupEnd ncclCommRevoke ncclCommGetUniqueId ncclCommGrow
.build/libnccl-mesh.dylib: nccl-mesh.c nccl.h $(MESH_H) .build/libmesh.dylib Makefile
	cc -O2 $(WARN) -dynamiclib -Wl,-install_name,@rpath/libnccl-mesh.dylib -Wl,-rpath,@loader_path \
	  $(foreach f,$(NCCL_API),-Wl,-alias,_$(f),_p$(f)) -o $@ nccl-mesh.c -L.build -lmesh
.build/libmesh-coreml.dylib: mesh-coreml.m Makefile
	mkdir -p .build
	cc -fobjc-arc -O2 $(WARN) -dynamiclib -Wl,-install_name,@rpath/libmesh-coreml.dylib -o $@ mesh-coreml.m \
	  -framework CoreML -framework Foundation
.build/libmesh-phase.dylib: mesh-phase.mm $(MESH_H) .build/libmesh.dylib Makefile
	mkdir -p .build
	clang++ -O2 -std=c++17 -fobjc-arc -shared -fPIC -o $@ mesh-phase.mm -L.build -lmesh -Wl,-rpath,@loader_path \
	  -framework Metal -framework MetalPerformanceShaders -framework Foundation
# design/transport.md: each built image whose bytes differ from the installed one's copied beside it and renamed into
# place, in IMAGES' order, and the bridge's binary anew wherever libmesh changed; a running process keeps the files it
# mapped, and the bridge runs a new binary once stopped by SIGTERM (bin/mesh-user.sh; design/transport.md section 9)
install: $(IMAGES:%=.build/%)
	@moved=; for f in $(IMAGES); do \
	  if cmp -s .build/$$f $$f; then [ $$f = mesh-flow ] && [ -n "$$moved" ] || continue; \
	  else [ $$f != libmesh.dylib ] || moved=1; fi; \
	  cp .build/$$f $$f.install && mv -f $$f.install $$f || exit 1; echo "rdma/$$f installed"; \
	done
# the built images; never an installed image (the launchd job's binary and the libraries clients map)
clean:
	rm -rf $(IMAGES:%=.build/%) .build/mesh-install-state .build/route-forward.o
.PHONY: all install clean
# _mesh_c: the headers as one Python module (mesh_c_build.py), for the interpreter that imports it, bound to the
# installed libraries it loads (make install puts them in rdma/; this target installs nothing, so it never stops the
# bridge: design/transport.md)
PYTHON ?= $(HOME)/.venv-mesh-uv/bin/python
mesh-c: mesh_c_build.py mesh-view.h mesh-call.h nccl.h
	$(PYTHON) mesh_c_build.py
.PHONY: mesh-c
