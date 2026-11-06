BINS = mesh-flow mesh-heartbeat mesh-stat libmesh.dylib
MESH_CODESIGN_IDENTITY ?= Mesh Bridge
MESH_PREFIX ?= /usr/local/mesh
# the provider's own queue pairs on each device, which a link's budget leaves to it (tag pre-return-20261103
# etc/bridge/thunderbolt.mk; 76be3217, design/HETEROGENEOUS-LINKS.md bd85613e): max_qp - this, both ends' smaller
MESH_PROVIDER_RESERVED_QPS ?= 1
CPPFLAGS += -DMESH_PROVIDER_RESERVED_QPS=$(MESH_PROVIDER_RESERVED_QPS)
# no fused multiply-add anywhere the plan's solve or the rule is computed: every node must reach the same trees and the
# same (cable, queue pair) per page from the same map (mesh-graph.h: "IEEE-exact operations alone ... so that every node
# computes the same trees"; mesh.h, both ends the same hops in the same order).  Apple clang at -O2 contracts a*b+c in
# mesh-graph.c's exp, log, flow and pack (review R1, round 2); the tag had the same hole.  Every node builds with its own
# SDK, so the contraction is refused here for every translation unit, not left to one file's pragma
CFLAGS_EXACT = -ffp-contract=off
# HiGHS (design/plan-to-bridge-change-request.md R5, Code: "rdma/Makefile builds HiGHS from its source at a pinned
# release and links it statically.  The release is tag v1.15.1, commit 04024d701f79feb8e2f18bc3df0dffc04ef05088, fetched
# as https://github.com/ERGO-Code/HiGHS/archive/refs/tags/v1.15.1.tar.gz and checked against SHA-256 a840d269...").
# Fetched and checked as metal-microbench's Makefile fetches its pinned sources (E2B_REFERENCE_REVISION with
# E2B_REFERENCE_SHA256: curl --fail, shasum -a 256 --check, then the move).  Built with cc and c++ alone (R5: "It is
# built with the tools every member has, cc and c++"), as the source's own BUILD.bazel builds its cc_library "highs":
# its srcs globs (HIGHS_SOURCES), its copts (-fexceptions -Wno-unused-variable -Wno-unused-but-set-variable), its
# defines (CUPDLP_CPU), its includes (extern, extern/catch, highs, and the root where its copy_file puts HConfig.h, a
# copy of highs/HConfig.h.bazel.in) and its linkopts (-lpthread).  R5: "It takes the same -ffp-contract=off as
# mesh-graph.c and no fast-math"; -O2 as mesh-graph.c.  -std=c++17 and -DNDEBUG, which BUILD.bazel leaves to the
# toolchain, are design/compilation-coordination.md, "The revisions of f3f7cf46", HB1.  Linked statically as its
# objects (HB2), with libc++ for them, wherever mesh-graph.c is linked; mesh-graph.c includes the C API, "the C API is
# highs/interfaces/highs_c_api.h of that source" (R5), from the source's highs directory as a system include, since
# highs_c_api.h declares a static function it never defines ("The revisions of 46ae8fdf", LP8), and HConfig.h from
# the root.  Nothing is installed by hand, and nothing comes from Homebrew (R5)
HIGHS_TAG = v1.15.1
HIGHS_SHA256 = a840d269dff2fafb371dd247df13ad5e026d7ce3b35ad3dc1eedd59bf0c2fb16
HIGHS_TARBALL = .build/highs/$(HIGHS_TAG).tar.gz
HIGHS = .build/highs/HiGHS-1.15.1
HIGHS_BUILT = .build/highs/built
HIGHS_INPUTS = .build/highs/inputs
HIGHS_INPUT_TEXT = $(HIGHS_TAG) $(HIGHS_FLAGS)
HIGHS_SOURCES = highs/interfaces/highs_c_api.cpp highs/HighsExternalApi.cpp highs/HighsExternalDeps.cpp highs/io/*.cpp \
  highs/io/filereaderlp/*.cpp highs/ipm/*.cpp highs/ipm/ipx/*.cc highs/ipm/basiclu/*.c highs/ipm/hipo/ipm/*.cpp \
  highs/ipm/hipo/factorhighs/*.cpp highs/ipm/hipo/auxiliary/*.cpp highs/lp_data/*.cpp highs/mip/*.cpp highs/model/*.cpp \
  highs/parallel/*.cpp highs/pdlp/*.cpp highs/pdlp/cupdlp/*.c highs/pdlp/hipdlp/*.cc highs/presolve/*.cpp \
  highs/qpsolver/*.cpp highs/simplex/*.cpp highs/test_kkt/*.cpp highs/util/*.cpp
HIGHS_FLAGS = -O2 -DNDEBUG $(CFLAGS_EXACT) -fexceptions -Wno-unused-variable -Wno-unused-but-set-variable -DCUPDLP_CPU \
  -I. -Iextern -Iextern/catch -Ihighs
# its translation units compiled at once, one per hardware thread, where a shell loop compiled them one after another
# (2026-11-05: `make -B -C rdma` took 113 s, 99.5 s of it user time in that loop, on 18 threads).  Why: the operator,
# 2026-10-31: "nothing you're running should take more than around 8000ms in any situation, no matter how complicated it
# might sound. remember that you have over 20x4 hardware cpu cores".  xargs -P, since make 3.81, the members' make,
# leaves -j to its caller.  What must stay: each unit its own compiler process, every one's failure the recipe's
HIGHS_JOBS := $(shell sysctl -n hw.logicalcpu)
HIGHS_CFLAGS = -isystem $(HIGHS)/highs -isystem $(HIGHS)
HIGHS_LIBS = .build/highs/obj/*.o -lc++ -lpthread
HEADERS = mesh.h mesh-map.h mesh-verbs.h mesh-bridge.h mesh-holders.h mesh-graph.h mesh-dataflow.h mesh-call.h mesh-collective.h \
  mesh-section.h
# the resident bridge (design/page-queue-change-request.md): its main and event loop (mesh-link.c), its ports' children
# from the parent's side (mesh-children.c) and their own (mesh-child.c), discovery and pairing (mesh-ports.c), the
# kernel's loss and change events (mesh-events.c), the child's verbs jobs (mesh-lanes.c), the map (mesh-records.c) and
# the data path over the page lists (mesh-flow.c); one image, its children `mesh-flow --child ...`
BRIDGE = mesh-link.c mesh-children.c mesh-child.c mesh-ports.c mesh-events.c mesh-lanes.c mesh-records.c mesh-flow.c \
  mesh-dataflow.c mesh-call.c mesh-graph.c
all: $(BINS) $(HIGHS_INPUTS)
# HiGHS's fetch and build (above), after `all`, the default goal
$(HIGHS_TARBALL):
	mkdir -p .build/highs
	curl --fail --location --output $@.part https://github.com/ERGO-Code/HiGHS/archive/refs/tags/$(HIGHS_TAG).tar.gz
	echo '$(HIGHS_SHA256)  $@.part' | shasum -a 256 --check
	mv $@.part $@
# 2026-11-05: HiGHS's objects remade only where their own inputs change, its tarball or HIGHS_INPUTS (its tag and
# flags), where any edit of this Makefile remade them.  HIGHS_INPUTS is rewritten only where its text differs, as git's
# Makefile keeps GIT-CFLAGS; where it is missing beside built objects it takes their stamp's time, since until now the
# stamp followed this whole file, so those objects are of these inputs.  Why: the operator, 2026-11-05: "stop
# arbitrarily rebuilding unchanged dependencies as part of linking or assembling whatever runtime our compiler uses".
# What must stay: the objects remade at a change of the tarball, the tag or the flags, and at no other
$(HIGHS_BUILT): $(HIGHS_TARBALL) $(wildcard $(HIGHS_INPUTS))
	tar -xzf $(HIGHS_TARBALL) -C .build/highs
	cp $(HIGHS)/highs/HConfig.h.bazel.in $(HIGHS)/HConfig.h
	mkdir -p .build/highs/obj
	cd $(HIGHS) && printf '%s\n' $(HIGHS_SOURCES) | xargs -P $(HIGHS_JOBS) -n 1 sh -c 'f=$$1; o=../obj/$$(echo $$f | tr / _).o; \
	  case $$f in *.c) exec cc $(HIGHS_FLAGS) -c $$f -o $$o;; *) exec c++ -std=c++17 $(HIGHS_FLAGS) -c $$f -o $$o;; esac' sh
	echo '$(HIGHS_INPUT_TEXT)' | cmp -s - $(HIGHS_INPUTS) || echo '$(HIGHS_INPUT_TEXT)' > $(HIGHS_INPUTS)
	touch $@
$(HIGHS_INPUTS): FORCE
	mkdir -p .build/highs
	if test ! -f $@ && test -f $(HIGHS_BUILT); then echo '$(HIGHS_INPUT_TEXT)' > $@ && touch -r $(HIGHS_BUILT) $@; \
	else echo '$(HIGHS_INPUT_TEXT)' | cmp -s - $@ || echo '$(HIGHS_INPUT_TEXT)' > $@; fi
FORCE:
native-audit: .build/mesh-flow.s .build/mesh-call.s .build/mesh-dataflow.s
.build/%.s: %.c $(HEADERS) Makefile
	mkdir -p .build
	cc $(CPPFLAGS) $(CFLAGS_EXACT) -fblocks -O2 -Wall -Wextra -Werror -S -o $@ $<
mesh-flow: $(BRIDGE) $(HEADERS) $(HIGHS_BUILT) Makefile
	mkdir -p .build
	cc $(CPPFLAGS) $(CFLAGS_EXACT) $(HIGHS_CFLAGS) -fblocks -O2 -Wall -Wextra -Werror -o .build/$@ $(BRIDGE) -lrdma $(HIGHS_LIBS) -framework CoreFoundation -framework IOKit
	codesign --force --sign "$(MESH_CODESIGN_IDENTITY)" --identifier io.mesh.bridge .build/$@ || codesign --force --sign - --identifier io.mesh.bridge .build/$@
	mv -f .build/$@ $@
# the bridge's heartbeat (tag pre-return-20261103 rdma/Makefile): no device, no verbs, its own translation unit
mesh-heartbeat: mesh-heartbeat.c mesh.h mesh-map.h mesh-holders.h Makefile
	cc $(CPPFLAGS) -O2 -Wall -Wextra -Werror -o $@ mesh-heartbeat.c
# 2026-11-04 (design/compilation-coordination.md, B2's report): mesh-stat walks the claim heads (mesh-dataflow.h
# mesh_head_after), so it links mesh-dataflow.c, as the bridge and libmesh do
mesh-stat: mesh-stat.c mesh-dataflow.c mesh.h mesh-map.h mesh-holders.h mesh-dataflow.h mesh-call.h mesh-section.h
	cc -O2 -o $@ mesh-stat.c mesh-dataflow.c
libmesh.dylib: mesh-dataflow.c mesh-call.c mesh-collective.c mesh-graph.c $(HEADERS) $(HIGHS_BUILT) Makefile
	cc $(CPPFLAGS) $(CFLAGS_EXACT) $(HIGHS_CFLAGS) -fblocks -O2 -dynamiclib -Wl,-install_name,@rpath/libmesh.dylib -o $@ mesh-dataflow.c mesh-call.c mesh-collective.c mesh-graph.c -lobjc $(HIGHS_LIBS)
clean:
	rm -f $(BINS)
.PHONY: all clean native-audit FORCE
install-bridge: mesh-flow mesh-heartbeat mesh-stat
	install -d $(MESH_PREFIX)/bin
	install -m 755 mesh-flow mesh-heartbeat mesh-stat $(MESH_PREFIX)/bin/
	codesign -dv $(MESH_PREFIX)/bin/mesh-flow 2>&1 | grep -E "^Identifier=io.mesh.bridge$$"
