BINS = mesh-flow mesh-stat libmesh.dylib libmesh-tensor.dylib
all: $(BINS)
mesh-flow: mesh-flow.c mesh-verbs.h mesh.h mesh-dataflow.h mesh-flight.h mesh-links.h Makefile
	mkdir -p .build
	cc $(CPPFLAGS) -O2 -o .build/$@ mesh-flow.c -lrdma
	codesign --force --sign - .build/$@
	mv -f .build/$@ $@
mesh-stat: mesh-stat.c mesh.h
	cc -O2 -o $@ mesh-stat.c
mesh-page-stream: mesh-page-stream.c mesh-client.c mesh-dataflow.c mesh-dataflow.h mesh.h
	mkdir -p .build
	cc -O2 -o .build/$@ mesh-page-stream.c mesh-client.c mesh-dataflow.c
mesh-verbs-stream: mesh-verbs-stream.c mesh-verbs.h mesh.h mesh-flight.h Makefile
	mkdir -p .build
	cc $(CPPFLAGS) -O2 -o .build/$@ mesh-verbs-stream.c -lrdma
	codesign --force --sign - .build/$@
libmesh.dylib: mesh-client.c mesh-dataflow.c mesh-dataflow.h mesh.h
	cc -O2 -dynamiclib -Wl,-install_name,@rpath/libmesh.dylib -o $@ mesh-client.c mesh-dataflow.c
libmesh-tensor.dylib: mesh-tensor.m mesh-tensor.h mesh-metal.m mesh-metal.h libmesh.dylib
	cc -O2 -fobjc-arc -dynamiclib -framework Foundation -framework Metal -Wl,-install_name,@rpath/libmesh-tensor.dylib -o $@ mesh-tensor.m mesh-metal.m -L. -lmesh -Wl,-rpath,@loader_path
clean:
	rm -f $(BINS)
.PHONY: all clean mesh-page-stream mesh-verbs-stream
