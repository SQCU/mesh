BINS = mesh-flow mesh-stat libmesh.dylib libmesh-tensor.dylib
all: $(BINS)
mesh-flow: mesh-flow.c mesh.h mesh-flight.h mesh-wire.h mesh-links.h Makefile
	mkdir -p .build
	cc -O2 -o .build/$@ mesh-flow.c -lrdma
	codesign --force --sign - .build/$@
	mv -f .build/$@ $@
mesh-stat: mesh-stat.c mesh.h
	cc -O2 -o $@ mesh-stat.c
libmesh.dylib: mesh-client.c mesh-functions.c mesh-functions.h mesh-reduce.c mesh-reduce.h mesh-pages.c mesh-pages.h mesh-dataflow.c mesh-dataflow.h mesh.h mesh-wire.h
	cc -O2 -dynamiclib -Wl,-install_name,@rpath/libmesh.dylib -o $@ mesh-client.c mesh-functions.c mesh-reduce.c mesh-pages.c mesh-dataflow.c
libmesh-tensor.dylib: mesh-tensor.m mesh-tensor.h mesh-metal.m mesh-metal.h mesh-metal-executor.m mesh-metal-executor.h libmesh.dylib
	cc -O2 -fobjc-arc -dynamiclib -framework Foundation -framework Metal -Wl,-install_name,@rpath/libmesh-tensor.dylib -o $@ mesh-tensor.m mesh-metal.m mesh-metal-executor.m -L. -lmesh -Wl,-rpath,@loader_path
mesh-stream: mesh-stream.c mesh-client.c mesh.h
	cc -O2 -o $@ mesh-stream.c mesh-client.c
clean:
	rm -f $(BINS) mesh-stream
.PHONY: all clean
