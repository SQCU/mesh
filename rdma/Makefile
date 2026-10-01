BINS = mesh-flow mesh-stat libmesh.dylib
MESH_CODESIGN_IDENTITY ?= Mesh Bridge
all: $(BINS)
native-audit: .build/mesh-flow.s .build/mesh-call.s .build/mesh-dataflow.s
.build/%.s: %.c mesh.h mesh-verbs.h mesh-dataflow.h mesh-call.h Makefile
	mkdir -p .build
	cc $(CPPFLAGS) -fblocks -O2 -Wall -Wextra -Werror -S -o $@ $<
mesh-flow: mesh-flow.c mesh-verbs.h mesh.h mesh-dataflow.h mesh-call.h mesh-dataflow.c Makefile
	mkdir -p .build
	cc $(CPPFLAGS) -fblocks -O2 -o .build/$@ mesh-flow.c mesh-dataflow.c -lrdma
	codesign --force --sign "$(MESH_CODESIGN_IDENTITY)" --identifier io.mesh.bridge .build/$@ || codesign --force --sign - --identifier io.mesh.bridge .build/$@
	mv -f .build/$@ $@
mesh-stat: mesh-stat.c mesh.h
	cc -O2 -o $@ mesh-stat.c
libmesh.dylib: mesh-dataflow.c mesh-dataflow.h mesh-call.c mesh-call.h mesh-collective.c mesh-collective.h mesh.h Makefile
	cc -fblocks -O2 -dynamiclib -Wl,-install_name,@rpath/libmesh.dylib -o $@ mesh-dataflow.c mesh-call.c mesh-collective.c -lobjc
clean:
	rm -f $(BINS)
.PHONY: all clean native-audit
