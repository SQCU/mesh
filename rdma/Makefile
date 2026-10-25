# design/transport.md: the node's images, built in .build by `all`, which installs nothing; `install` renames
# each that changed into rdma/, in this order: the libraries first, the bridge's binary last, then libnccl-mesh with its
# kernels (install-nccl, below)
IMAGES = libmesh.dylib libnccl-mesh.dylib libmesh-coreml.dylib mesh-stat mesh-heartbeat mesh-flow
MESH_CODESIGN_IDENTITY ?= Mesh Bridge
# every warning an error, but a deprecation in the installed binaries, which each node builds with its own SDK (a newer
# one's deprecation must not leave a node's bridge unbuilt)
WARN = -Wall -Wextra -Werror -Wno-error=deprecated-declarations
all: $(IMAGES:%=.build/%)
# the bridge and mesh-stat read the region through libmesh: beside them once installed, and at ../rdma from $MESH_ROOT/bin
LIBMESH = -L.build -lmesh -Wl,-rpath,@loader_path -Wl,-rpath,@loader_path/../rdma
# design/transport.md: the bridge, mesh-link.c and mesh-records.c over mesh-bridge.h
.build/mesh-flow: mesh-link.c mesh-records.c mesh-bridge.h mesh-verbs.h mesh.h mesh-plan.h mesh-dataflow.h mesh-call.h .build/libmesh.dylib Makefile
	cc $(CPPFLAGS) -fblocks -O2 $(WARN) -o $@ mesh-link.c mesh-records.c $(LIBMESH) -lrdma -framework CoreFoundation -framework IOKit
	codesign --force --sign "$(MESH_CODESIGN_IDENTITY)" --identifier io.mesh.bridge $@ || codesign --force --sign - --identifier io.mesh.bridge $@
# design/transport.md section 2: the heartbeat, the job launchd keeps alive, which opens and closes the bridge
.build/mesh-heartbeat: mesh-heartbeat.c mesh.h mesh-plan.h mesh-dataflow.h mesh-observe.h Makefile
	mkdir -p .build
	cc -O2 $(WARN) -o $@ mesh-heartbeat.c
	codesign --force --sign "$(MESH_CODESIGN_IDENTITY)" --identifier io.mesh.bridge $@ || codesign --force --sign - --identifier io.mesh.bridge $@
.build/mesh-stat: mesh-stat.c mesh.h mesh-plan.h .build/libmesh.dylib Makefile
	cc -O2 $(WARN) -o $@ mesh-stat.c $(LIBMESH)
# design/transport.md: each built image but libnccl-mesh.dylib whose bytes differ from the installed one's copied beside
# it and renamed into place, in IMAGES' order, and the bridge's binary anew wherever libmesh changed; a running process
# keeps the files it mapped, and the bridge runs a new binary once stopped by SIGTERM (bin/mesh-user.sh;
# design/transport.md section 9); then libnccl-mesh (install-nccl), whose failure leaves these installed
install: install-images install-nccl
OTHERS = $(filter-out libnccl-mesh.dylib,$(IMAGES))
install-images: $(OTHERS:%=.build/%)
	@moved=; for f in $(OTHERS); do \
	  if cmp -s .build/$$f $$f; then [ $$f = mesh-flow ] && [ -n "$$moved" ] || continue; \
	  else [ $$f != libmesh.dylib ] || moved=1; fi; \
	  cp .build/$$f $$f.install && mv -f $$f.install $$f || exit 1; echo "rdma/$$f installed"; \
	done
# libnccl-mesh's kernel files under the kernels' SHA-256 (the AIR checked or built: kernels; this node's binary
# archive made anew: below), then libnccl-mesh.dylib, so a new library finds its own; after install-images, whose
# libmesh it links.  Earlier kernels' files stay, for a process that mapped an earlier library and has not yet loaded
# its kernels (clean-kernels removes them)
install-nccl: install-images .build/libnccl-mesh.dylib .build/libnccl-mesh.metallib .build/libnccl-mesh.binarchive
	@h=$$(cat .build/libnccl-mesh.sha256); for f in $(KERNEL_FILES); do \
	  cmp -s .build/libnccl-mesh.$$f libnccl-mesh.$$h.$$f && continue; \
	  cp .build/libnccl-mesh.$$f libnccl-mesh.$$h.$$f.install && mv -f libnccl-mesh.$$h.$$f.install libnccl-mesh.$$h.$$f || exit 1; \
	  echo "rdma/libnccl-mesh.$$h.$$f installed"; \
	done
	@cmp -s .build/libnccl-mesh.dylib libnccl-mesh.dylib || { cp .build/libnccl-mesh.dylib libnccl-mesh.dylib.install && \
	  mv -f libnccl-mesh.dylib.install libnccl-mesh.dylib && echo "rdma/libnccl-mesh.dylib installed"; }
# every other kernels' files in rdma/, which the installed libnccl-mesh.dylib does not load: run by hand where no
# process of an earlier library still runs (one that has not loaded its kernels would fail its first collective)
clean-kernels: .build/libnccl-mesh.sha256
	@h=$$(cat .build/libnccl-mesh.sha256); for f in $(KERNEL_FILES:%=libnccl-mesh.*.%); do \
	  [ -e "$$f" ] || continue; case $$f in libnccl-mesh.$$h.*) ;; *) rm -f "$$f" && echo "rdma/$$f removed";; esac; \
	done
# NCCL's API (nccl.h) on the prepared transfers and the collective planner of libmesh.dylib; every function also
# under its pncclX profiling name, as NCCL exports both
NCCL_API = ncclMemAlloc ncclMemFree ncclGetVersion ncclGetUniqueId ncclCommInitRankConfig ncclCommInitRank \
  ncclCommInitAll ncclCommFinalize ncclCommDestroy ncclCommAbort ncclCommSplit ncclCommShrink ncclGetErrorString \
  ncclGetLastError ncclCommGetAsyncError ncclCommCount ncclCommCuDevice ncclCommUserRank ncclCommRegister \
  ncclCommDeregister ncclRedOpCreatePreMulSum ncclRedOpDestroy ncclReduce ncclBcast ncclBroadcast ncclAllReduce \
  ncclReduceScatter ncclAllGather ncclAlltoAll ncclGather ncclScatter ncclAllReduceConfig ncclBroadcastConfig \
  ncclReduceConfig ncclAllGatherConfig ncclReduceScatterConfig ncclAlltoAllConfig ncclGatherConfig ncclScatterConfig \
  ncclGroupStart ncclGroupEnd ncclCommRevoke ncclCommGetUniqueId ncclCommGrow
# rdma/mesh.metal, the GPU transport primitives, is libnccl-mesh's kernels' one copy of them: its text embedded in
# the build tool's source (nccl-mesh-kernels.m) as the header mesh_metal.h
.build/mesh_metal.h: mesh.metal
	mkdir -p .build
	xxd -i mesh.metal > $@
# libnccl-mesh's kernels, compiled when the library is built and never in a process it serves (rdma/NCCL.md "The
# library's kernels"; nccl-mesh-kernels.h): KERNEL_FILES, the two files a libnccl-mesh.dylib loads beside itself, each
# named libnccl-mesh.<SHA-256>.<kind> by the kernels' SHA-256, of their compile line (METAL) and their text, which is
# compiled into the library that loads them (KERNELS_SHA256; nccl-mesh-metal.m kernels_of)
KERNEL_FILES = metallib binarchive
# the build tool (nccl-mesh-kernels.m), the kernels' text it writes (rdma/mesh.metal's primitives, then
# nccl-mesh-kernels.h's) and the kernels' SHA-256
.build/nccl-mesh-kernels: nccl-mesh-kernels.m nccl-mesh-kernels.h nccl-mesh-metal.h .build/mesh_metal.h Makefile
	cc -O2 $(WARN) -I .build -o $@ nccl-mesh-kernels.m -framework Metal -framework Foundation
.build/libnccl-mesh.metal: .build/nccl-mesh-kernels
	.build/nccl-mesh-kernels $@
.build/libnccl-mesh.sha256: .build/libnccl-mesh.metal Makefile
	h=$$({ echo '$(METAL)'; cat $<; } | shasum -a 256) && echo "$${h%% *}" > $@
# kernels: the kernels' AIR, which every Apple GPU's driver takes: vendor/libnccl-mesh.metallib, committed with
# vendor/libnccl-mesh.sha256 (the kernels' SHA-256 it was compiled from, then the AIR's own; vendor/PROVENANCE.md), so
# that a node with no Metal toolchain still installs.  The Metal toolchain is an Xcode with its Metal toolchain
# component (XCODE): DEVELOPER_DIR's, or else the last /Applications/Xcode*.app, which bin/mesh-devtools-init.sh readies
# on every node.  The two commands below alone run under it and its own SDK (SDKROOT=macosx; xcrun otherwise takes a
# newer Command Line Tools' SDK, whose version the AIR records), and every other command here keeps the active developer
# directory (the Command Line Tools: verbs.h, librdma).  Metal 4.0; floating point as MTLMathModeSafe; macOS 26.0's
# AIR, the floor the library's Metal 4 calls need; every warning an error but an unused function (mesh.metal is every
# consumer's header of primitives, and these kernels call some of them).  Where the record is these kernels' and that
# AIR's, the committed AIR is used; where it is not, the kernels are compiled here into .build/ (a node's checkout stays
# its branch's: a pull is never blocked by a rewritten vendor/), `kernels` puts them in vendor/ for a commit, and a
# machine that cannot compile fails naming that build: never an AIR of other kernels.  Not in `all`, so that no image
# waits on it (install-nccl takes it)
METAL = xcrun metal -std=metal4.0 -fmetal-math-mode=safe -mmacosx-version-min=26.0 -Wall -Werror -Wno-unused-function \
  -Wno-error=deprecated-declarations
XCODE = DEVELOPER_DIR=$(or $(DEVELOPER_DIR),$(lastword $(wildcard /Applications/Xcode*.app/Contents/Developer))) \
  SDKROOT=macosx
VENDOR = ../vendor
kernels: .build/libnccl-mesh.metallib
	@a=$$(shasum -a 256 < $<) && { cat .build/libnccl-mesh.sha256; echo "$${a%% *}"; } | cmp -s - $(VENDOR)/libnccl-mesh.sha256 || { \
	  cp $< $(VENDOR)/libnccl-mesh.metallib && { cat .build/libnccl-mesh.sha256; echo "$${a%% *}"; } > $(VENDOR)/libnccl-mesh.sha256 && \
	  echo "rdma/Makefile: vendor/libnccl-mesh.metallib and vendor/libnccl-mesh.sha256 compiled anew: commit them" >&2; }
.build/libnccl-mesh.metallib: .build/libnccl-mesh.sha256 $(wildcard $(VENDOR)/libnccl-mesh.sha256 $(VENDOR)/libnccl-mesh.metallib)
	@[ -f $(VENDOR)/libnccl-mesh.metallib ] && a=$$(shasum -a 256 < $(VENDOR)/libnccl-mesh.metallib) && \
	  { cat $<; echo "$${a%% *}"; } | cmp -s - $(VENDOR)/libnccl-mesh.sha256 && cp $(VENDOR)/libnccl-mesh.metallib $@ || { \
	  $(XCODE) $(METAL) -c -o .build/libnccl-mesh.air .build/libnccl-mesh.metal && \
	  $(XCODE) xcrun metallib -o $@ .build/libnccl-mesh.air || { echo \
	    "rdma/Makefile: vendor/libnccl-mesh.metallib is not these kernels' AIR (vendor/libnccl-mesh.sha256 against" \
	    "$<), and compiling them here ($(XCODE)) failed (above): where an Xcode has its Metal toolchain" \
	    "(bin/mesh-devtools-init.sh), make -C rdma kernels, then commit vendor/libnccl-mesh.metallib and" \
	    "vendor/libnccl-mesh.sha256 (rdma/NCCL.md, The library's kernels)" >&2; exit 1; }; \
	  echo "rdma/Makefile: vendor/libnccl-mesh.metallib is not these kernels' AIR: compiled here into $@;" \
	    "make -C rdma kernels puts it in vendor/ for a commit" >&2; }
# this node's binary archive: every kind's pipeline compiled from the metallib for this machine's GPU by the build
# tool, made at every install (the system's GPU compiler may change under an unchanged metallib: an OS update), by the
# person or agent who runs it; a process looks each pipeline up in it and compiles none
.build/libnccl-mesh.binarchive: .build/nccl-mesh-kernels .build/libnccl-mesh.metallib
	rm -f $@ && .build/nccl-mesh-kernels .build/libnccl-mesh.metallib $@
.PHONY: .build/libnccl-mesh.binarchive kernels install-images install-nccl clean-kernels
.build/libnccl-mesh.dylib: nccl-mesh.c nccl-mesh-trace.h mesh-write-order.h mesh-write-routes.h mesh-span-plan.h mesh-row-span-plan.h mesh-lease-retire.h mesh-byte-account.h nccl-mesh-metal.m nccl-mesh-metal.h nccl-mesh-kernels.h .build/libnccl-mesh.sha256 .build/mesh_metal.h nccl.h \
  mesh-call.h mesh-dataflow.h mesh.h mesh-plan.h .build/libmesh.dylib Makefile
	cc -fblocks -O2 $(WARN) -dynamiclib -I .build -DKERNELS_SHA256=\"$$(cat .build/libnccl-mesh.sha256)\" \
	  -Wl,-install_name,@rpath/libnccl-mesh.dylib -Wl,-rpath,@loader_path \
	  $(foreach f,$(NCCL_API),-Wl,-alias,_$(f),_p$(f)) -o $@ nccl-mesh.c nccl-mesh-metal.m -L.build -lmesh \
	  -framework Metal -framework Foundation
.build/libmesh-coreml.dylib: mesh-coreml.m Makefile
	mkdir -p .build
	cc -fobjc-arc -O2 $(WARN) -dynamiclib -Wl,-install_name,@rpath/libmesh-coreml.dylib -o $@ mesh-coreml.m \
	  -framework CoreML -framework Foundation
.build/libmesh.dylib: mesh-dataflow.c mesh-dataflow.h mesh-call.c mesh-call.h mesh-collective.c mesh-fabric.c mesh-routed.c mesh-diagnostics.c mesh-diagnostics.h mesh.h mesh-plan.h Makefile
	mkdir -p .build
	cc -fblocks -O2 $(WARN) -dynamiclib -Wl,-install_name,@rpath/libmesh.dylib -o $@ mesh-dataflow.c mesh-call.c mesh-collective.c mesh-fabric.c mesh-routed.c mesh-diagnostics.c
# the built images; never an installed image (the launchd job's binary and the libraries clients map)
clean:
	rm -rf $(IMAGES:%=.build/%) .build/nccl-mesh-kernels .build/libnccl-mesh.metal .build/libnccl-mesh.sha256 \
	  .build/libnccl-mesh.air $(KERNEL_FILES:%=.build/libnccl-mesh.%)
.PHONY: all install clean

# _mesh_c: the headers as one Python module (mesh_c_build.py), for the interpreter that imports it, bound to the
# installed libraries it loads (make install puts them in rdma/; this target installs nothing, so it never stops the
# bridge: design/transport.md)
PYTHON ?= $(HOME)/.venv-mesh-uv/bin/python
mesh-c: mesh_c_build.py mesh-plan.h nccl.h
	$(PYTHON) mesh_c_build.py
.PHONY: mesh-c
