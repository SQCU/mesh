BINS = mesh-flow mesh-stat libmesh.dylib
all: $(BINS) .build/mesh_abi.py
mesh-flow: mesh-flow.c mesh-verbs.h mesh.h mesh-dataflow.h Makefile
	mkdir -p .build
	cc $(CPPFLAGS) -O2 -o .build/$@ mesh-flow.c -lrdma
	codesign --force --sign - .build/$@
	mv -f .build/$@ $@
mesh-stat: mesh-stat.c mesh.h
	cc -O2 -o $@ mesh-stat.c
mesh-page-stream: mesh-page-stream.c mesh-client.c mesh-dataflow.c mesh-dataflow.h mesh.h
	mkdir -p .build
	cc -O2 -o .build/$@ mesh-page-stream.c mesh-client.c mesh-dataflow.c
mesh-verbs-stream: mesh-verbs-stream.c mesh-verbs.h mesh.h Makefile
	mkdir -p .build
	cc $(CPPFLAGS) -O2 -o .build/$@ mesh-verbs-stream.c -lrdma
	codesign --force --sign - .build/$@
libmesh.dylib: mesh-client.c mesh-dataflow.c mesh-dataflow.h mesh.h
	cc -O2 -dynamiclib -Wl,-install_name,@rpath/libmesh.dylib -o $@ mesh-client.c mesh-dataflow.c
clean:
	rm -f $(BINS)
.PHONY: all clean mesh-page-stream mesh-verbs-stream

.build/mesh_abi.py: mesh-abi.py mesh.h mesh-dataflow.h
	python3 mesh-abi.py mesh-dataflow.h $@
