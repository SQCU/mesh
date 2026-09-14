BINS = mesh-flow mesh-stat libmesh.dylib libmesh-algebra.dylib
all: $(BINS)
mesh-flow: mesh-memory.h mesh-flow.c mesh-verbs.h mesh.h mesh-dataflow.h mesh-dataflow.c Makefile
	mkdir -p .build
	cc $(CPPFLAGS) -fblocks -O2 -o .build/$@ mesh-flow.c mesh-dataflow.c -lrdma
	codesign --force --sign - .build/$@
	mv -f .build/$@ $@
mesh-stat: mesh-stat.c mesh.h
	cc -O2 -o $@ mesh-stat.c
libmesh.dylib: mesh-memory.h mesh-dataflow.c mesh-dataflow.h mesh.h
	cc -fblocks -O2 -dynamiclib -Wl,-install_name,@rpath/libmesh.dylib -o $@ mesh-dataflow.c
clean:
	rm -f $(BINS)
.PHONY: all clean

libmesh-algebra.dylib: mesh-algebra.m mesh-algebra.h mesh-kernel.h mesh-dataflow.h libmesh.dylib
	clang -O2 -fobjc-arc -dynamiclib -Wl,-install_name,@rpath/libmesh-algebra.dylib -Wl,-rpath,@loader_path -o $@ mesh-algebra.m -L. -lmesh -framework Accelerate -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework CoreML
