# design/transport.md: the node's images, built in .build by `all`, which installs nothing; `install` renames
# each that changed into rdma/, in this order: the libraries first, the bridge's binary last, then libnccl-mesh with its
# kernels (install-nccl, below)
IMAGES = libmesh.dylib libnccl-mesh.dylib libmesh-coreml.dylib libmesh-statistics.dylib mesh-stat mesh-heartbeat mesh-flow
.DEFAULT_GOAL := all
METAL_CONFIGURATION ?= ../etc/compute/metal.mk
include $(METAL_CONFIGURATION)
include isl.mk
include cpu.mk
METAL_DEFINES = -DMESH_METAL_LANGUAGE=$(MESH_METAL_LANGUAGE_ENUM)
RDMA_CONFIGURATION ?= ../etc/bridge/thunderbolt.mk
include $(RDMA_CONFIGURATION)
MESH_CODESIGN_IDENTITY ?= Mesh Bridge
# every warning an error, but a deprecation in the installed binaries, which each node builds with its own SDK (a newer
# one's deprecation must not leave a node's bridge unbuilt)
WARN = -Wall -Wextra -Werror -Wno-error=deprecated-declarations
$(IMAGES:%=.build/%): $(METAL_CONFIGURATION)
$(IMAGES:%=.build/%): metal-capabilities.h ../etc/compute/metal.def ../etc/bridge/layout.h ../etc/bridge/thunderbolt.h
all: $(IMAGES:%=.build/%) .build/libsleef.a
# the bridge and mesh-stat read the region through libmesh: beside them once installed, and at ../rdma from $MESH_ROOT/bin
LIBMESH = -L.build -lmesh -Wl,-rpath,@loader_path -Wl,-rpath,@loader_path/../rdma
# design/transport.md: the bridge, mesh-link.c and mesh-records.c over mesh-bridge.h
.build/mesh-flow: mesh-link.c mesh-records.c mesh-bridge.h mesh-verbs.h mesh.h mesh-plan.h mesh-fabric.h mesh-dataflow.h mesh-call.h .build/libmesh.dylib Makefile $(RDMA_CONFIGURATION) $(METAL_CONFIGURATION)
	cc $(CPPFLAGS) -DMESH_PROVIDER_RESERVED_QPS=$(MESH_PROVIDER_RESERVED_QPS) -fblocks -O2 $(WARN) -o $@ mesh-link.c mesh-records.c $(LIBMESH) -lrdma -framework CoreFoundation -framework IOKit
	codesign --force --sign "$(MESH_CODESIGN_IDENTITY)" --identifier io.mesh.bridge $@ || codesign --force --sign - --identifier io.mesh.bridge $@
# design/transport.md section 2: the heartbeat, the job launchd keeps alive, which opens and closes the bridge
.build/mesh-heartbeat: mesh-heartbeat.c mesh.h mesh-plan.h mesh-fabric.h mesh-dataflow.h mesh-observe.h Makefile
	mkdir -p .build
	cc -O2 $(WARN) -o $@ mesh-heartbeat.c
	codesign --force --sign "$(MESH_CODESIGN_IDENTITY)" --identifier io.mesh.bridge $@ || codesign --force --sign - --identifier io.mesh.bridge $@
.build/mesh-stat: mesh-stat.c mesh.h mesh-plan.h mesh-fabric.h .build/libmesh.dylib Makefile
	cc -O2 $(WARN) -o $@ mesh-stat.c $(LIBMESH)
# design/transport.md: each built image but libnccl-mesh.dylib whose bytes differ from the installed one's copied beside
# it and renamed into place, in IMAGES' order, and the bridge's binary anew wherever libmesh changed; a running process
# keeps the files it mapped, and the bridge runs a new binary once stopped by SIGTERM (bin/mesh-user.sh;
# design/transport.md section 9); then libnccl-mesh (install-nccl), whose failure leaves these installed
install: install-images install-nccl
OTHERS = $(filter-out libnccl-mesh.dylib,$(IMAGES)) libsleef.a
install-images: $(OTHERS:%=.build/%)
	@moved=; for f in $(OTHERS); do \
	  if cmp -s .build/$$f $$f; then [ $$f = mesh-flow ] && [ -n "$$moved" ] || continue; \
	  else [ $$f != libmesh.dylib ] || moved=1; fi; \
	  cp .build/$$f $$f.install && mv -f $$f.install $$f || exit 1; echo "rdma/$$f installed"; \
	done
# libnccl-mesh.dylib, after install-images, whose libmesh it links
install-nccl: install-images .build/libnccl-mesh.dylib
	@cmp -s .build/libnccl-mesh.dylib libnccl-mesh.dylib || { cp .build/libnccl-mesh.dylib libnccl-mesh.dylib.install && \
	  mv -f libnccl-mesh.dylib.install libnccl-mesh.dylib && echo "rdma/libnccl-mesh.dylib installed"; }
# NCCL's API (nccl.h) on the prepared transfers and the collective planner of libmesh.dylib; every function also
# under its pncclX profiling name, as NCCL exports both
NCCL_API = ncclMemAlloc ncclMemFree ncclGetVersion ncclGetUniqueId ncclCommInitRankConfig ncclCommInitRank \
  ncclCommInitAll ncclCommFinalize ncclCommDestroy ncclCommAbort ncclGetErrorString \
  ncclGetLastError ncclCommGetAsyncError ncclCommCount ncclCommCuDevice ncclCommUserRank ncclCommRegister \
  ncclCommDeregister ncclRedOpCreatePreMulSum ncclRedOpDestroy ncclCommRevoke ncclCommGetUniqueId
# A changed OS or SDK invalidates native products; an unchanged install retains them.
.build/metal-environment: FORCE
	@mkdir -p .build; environment="$$(sw_vers -buildVersion) $$(xcrun --show-sdk-build-version)"; \
	  [ "$$(cat $@ 2>/dev/null)" = "$$environment" ] || printf '%s\n' "$$environment" > $@
$(IMAGES:%=.build/%): .build/metal-environment
.PHONY: FORCE install-images install-nccl
.build/mesh-cpu-source.o: mesh-cpu-source.m mesh-cpu-source.h mesh-cpu-kernels.h mesh-host.h Makefile .build/metal-environment
	mkdir -p .build
	cc -fobjc-arc -fblocks -O2 $(WARN) -c mesh-cpu-source.m -o $@
.build/libnccl-mesh.dylib: .build/mesh-cpu-source.o nccl-mesh.c mesh-statistics.c mesh-statistics.h metal.m metal.h mesh-page-sites.h mesh-host.h mesh-span-plan.h mesh-landing.h mesh-lease-retire.h mesh-publication.h nccl.h mesh-graph.h mesh-write-compiler.h \
  mesh-call.h mesh-dataflow.h mesh.h mesh-plan.h mesh-fabric.h .build/libmesh.dylib Makefile $(METAL_CONFIGURATION)
	cc -fblocks -O2 $(WARN) $(METAL_DEFINES) -dynamiclib -I .build \
	  -Wl,-install_name,@rpath/libnccl-mesh.dylib -Wl,-rpath,@loader_path \
	  $(foreach f,$(NCCL_API),-Wl,-alias,_$(f),_p$(f)) -o $@ .build/mesh-cpu-source.o nccl-mesh.c mesh-statistics.c metal.m -L.build -lmesh \
	  -framework Metal -framework Foundation -framework Accelerate
.build/libmesh-coreml.dylib: mesh-coreml.m mesh-coreml.h mesh-cpu-source.h mesh-host.h mesh-cpu-kernels.h mesh-ane-entry.h mesh-ane-entry.inc mesh-tensor-layout.h .build/libnccl-mesh.dylib Makefile
	mkdir -p .build
	cc -fobjc-arc -O2 $(WARN) -dynamiclib -Wl,-install_name,@rpath/libmesh-coreml.dylib -o $@ mesh-coreml.m \
	  -Wl,-rpath,@loader_path -L.build -lnccl-mesh -framework CoreML -framework Foundation -framework Accelerate -framework IOSurface
.build/libmesh-statistics.dylib: mesh-statistics.c mesh-statistics.h Makefile
	mkdir -p .build
	cc -O2 $(WARN) -dynamiclib -Wl,-install_name,@rpath/libmesh-statistics.dylib -o $@ mesh-statistics.c
# The reporting host needs only this ordinary library dependency, not a bridge.
statistics: .build/libmesh-statistics.dylib
	cp $< libmesh-statistics.dylib
.PHONY: statistics
.build/libmesh.dylib: mesh-graph.c mesh-graph.h mesh-page-map.c mesh-page-map.h mesh-dataflow.c mesh-dataflow.h mesh-call.c mesh-call.h mesh-collective.c mesh-fabric.c mesh-routed.c mesh-diagnostics.c mesh-diagnostics.h mesh.h mesh-plan.h mesh-fabric.h Makefile
	mkdir -p .build
	cc -fblocks -O2 $(WARN) -dynamiclib -Wl,-install_name,@rpath/libmesh.dylib -o $@ mesh-graph.c mesh-page-map.c mesh-dataflow.c mesh-call.c mesh-collective.c mesh-fabric.c mesh-routed.c mesh-diagnostics.c -framework IOKit -framework CoreFoundation
# the built images; never an installed image (the launchd job's binary and the libraries clients map)
clean:
	rm -rf $(IMAGES:%=.build/%) \
	  .build/libnccl-mesh.air $(KERNEL_FILES:%=.build/libnccl-mesh.%)
.PHONY: all install clean

# _mesh_c: the headers as one Python module (mesh_c_build.py), for the interpreter that imports it, bound to the
# installed libraries it loads (make install puts them in rdma/; this target installs nothing, so it never stops the
# bridge: design/transport.md)
PYTHON ?= $(HOME)/.venv-mesh-uv/bin/python
# the interpreter's extension suffix, read once a make: a module is a file, so an up-to-date one is an mtime read and
# the preparation that finds nothing to build costs milliseconds, not each builder's interpreter start (0.75 s and 0.55
# s a node, every deploy and every hour, before 2026-11-01)
EXT_SUFFIX := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))' 2>/dev/null)
MESH_C = _mesh_c$(EXT_SUFFIX)
$(MESH_C): mesh_c_build.py ../etc/bridge/layout.h mesh-plan.h mesh-fabric.h nccl.h mesh-write-compiler.h mesh-statistics.h
	$(PYTHON) mesh_c_build.py && touch $@
mesh-c: $(MESH_C)
# torch_mesh's extension: its sources, every header of this directory and the configurations it includes (setup.py's
# include_dirs), and setup.py; the libraries it links are found by install name at load, so their rebuild needs none.
# make decides; build_ext is forced, since its own check reads the two sources alone and would keep an extension
# built against changed headers
STREAM = torch-mesh/torch_mesh/_stream$(EXT_SUFFIX)
$(STREAM): torch-mesh/setup.py torch-mesh/torch_mesh/_stream.mm torch-mesh/torch_mesh/allocator.mm \
  $(wildcard torch-mesh/torch_mesh/*.h) $(wildcard *.h) $(wildcard ../etc/compute/*) $(wildcard ../etc/bridge/*.h)
	cd torch-mesh && $(PYTHON) setup.py -q build_ext --inplace --force --build-temp ../.build/torch-mesh && touch ../$@
# the Python modules (mesh_nodes node prepare, after `make all install`)
modules: $(MESH_C) $(STREAM)
.PHONY: mesh-c modules
