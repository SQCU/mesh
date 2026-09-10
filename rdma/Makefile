BINS = mesh-flow mesh-stat libmesh.dylib libmesh-tensor.dylib libmesh-transport.dylib
all: $(BINS)
mesh-flow: mesh-flow.c mesh.h mesh-dataflow.h mesh-flight.h mesh-links.h mesh-transport.c mesh-transport.h Makefile
	mkdir -p .build
	cc -O2 -o .build/$@ mesh-flow.c mesh-transport.c -lrdma
	codesign --force --sign - .build/$@
	mv -f .build/$@ $@
mesh-stat: mesh-stat.c mesh.h
	cc -O2 -o $@ mesh-stat.c
libmesh-transport.dylib: mesh-transport.c mesh-transport.h
	cc -O2 -dynamiclib -Wl,-install_name,@rpath/libmesh-transport.dylib -o $@ mesh-transport.c -lrdma
libmesh.dylib: mesh-client.c mesh-dataflow.c mesh-dataflow.h mesh.h
	cc -O2 -dynamiclib -Wl,-install_name,@rpath/libmesh.dylib -o $@ mesh-client.c mesh-dataflow.c
libmesh-tensor.dylib: mesh-tensor.m mesh-tensor.h mesh-metal.m mesh-metal.h libmesh.dylib
	cc -O2 -fobjc-arc -dynamiclib -framework Foundation -framework Metal -Wl,-install_name,@rpath/libmesh-tensor.dylib -o $@ mesh-tensor.m mesh-metal.m -L. -lmesh -Wl,-rpath,@loader_path
clean:
	rm -f $(BINS)
.PHONY: all clean
