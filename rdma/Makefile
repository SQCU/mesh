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
HEADERS = mesh.h mesh-map.h mesh-verbs.h mesh-bridge.h mesh-holders.h mesh-graph.h mesh-dataflow.h mesh-call.h mesh-collective.h
# the resident bridge (design/page-queue-change-request.md): its main and event loop (mesh-link.c), its ports' children
# from the parent's side (mesh-children.c) and their own (mesh-child.c), discovery and pairing (mesh-ports.c), the
# kernel's loss and change events (mesh-events.c), the child's verbs jobs (mesh-lanes.c), the map (mesh-records.c) and
# the data path over the page lists (mesh-flow.c); one image, its children `mesh-flow --child ...`
BRIDGE = mesh-link.c mesh-children.c mesh-child.c mesh-ports.c mesh-events.c mesh-lanes.c mesh-records.c mesh-flow.c \
  mesh-dataflow.c mesh-call.c mesh-graph.c
all: $(BINS)
native-audit: .build/mesh-flow.s .build/mesh-call.s .build/mesh-dataflow.s
.build/%.s: %.c $(HEADERS) Makefile
	mkdir -p .build
	cc $(CPPFLAGS) $(CFLAGS_EXACT) -fblocks -O2 -Wall -Wextra -Werror -S -o $@ $<
mesh-flow: $(BRIDGE) $(HEADERS) Makefile
	mkdir -p .build
	cc $(CPPFLAGS) $(CFLAGS_EXACT) -fblocks -O2 -Wall -Wextra -Werror -o .build/$@ $(BRIDGE) -lrdma -framework CoreFoundation -framework IOKit
	codesign --force --sign "$(MESH_CODESIGN_IDENTITY)" --identifier io.mesh.bridge .build/$@ || codesign --force --sign - --identifier io.mesh.bridge .build/$@
	mv -f .build/$@ $@
# the bridge's heartbeat (tag pre-return-20261103 rdma/Makefile): no device, no verbs, its own translation unit
mesh-heartbeat: mesh-heartbeat.c mesh.h mesh-holders.h Makefile
	cc $(CPPFLAGS) -O2 -Wall -Wextra -Werror -o $@ mesh-heartbeat.c
mesh-stat: mesh-stat.c mesh.h mesh-map.h mesh-holders.h
	cc -O2 -o $@ mesh-stat.c
libmesh.dylib: mesh-dataflow.c mesh-call.c mesh-collective.c mesh-graph.c $(HEADERS) Makefile
	cc $(CPPFLAGS) $(CFLAGS_EXACT) -fblocks -O2 -dynamiclib -Wl,-install_name,@rpath/libmesh.dylib -o $@ mesh-dataflow.c mesh-call.c mesh-collective.c mesh-graph.c -lobjc
clean:
	rm -f $(BINS)
.PHONY: all clean native-audit
install-bridge: mesh-flow mesh-heartbeat mesh-stat
	install -d $(MESH_PREFIX)/bin
	install -m 755 mesh-flow mesh-heartbeat mesh-stat $(MESH_PREFIX)/bin/
	codesign -dv $(MESH_PREFIX)/bin/mesh-flow 2>&1 | grep -E "^Identifier=io.mesh.bridge$$"
