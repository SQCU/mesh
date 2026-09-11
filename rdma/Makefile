BINS = mesh-flow mesh-stat libmesh.dylib
all: $(BINS)
mesh-flow: mesh-flow.c mesh-verbs.h mesh.h mesh-dataflow.h mesh-dataflow.c Makefile
	mkdir -p .build
	cc $(CPPFLAGS) -O2 -o .build/$@ mesh-flow.c mesh-dataflow.c -lrdma
	codesign --force --sign - .build/$@
	mv -f .build/$@ $@
mesh-stat: mesh-stat.c mesh.h
	cc -O2 -o $@ mesh-stat.c
mesh-page-stream: mesh-page-stream.c mesh-dataflow.c mesh-dataflow.h mesh.h
	mkdir -p .build
	cc -O2 -o .build/$@ mesh-page-stream.c mesh-dataflow.c
mesh-verbs-stream: mesh-verbs-stream.c mesh-verbs.h mesh.h Makefile
	mkdir -p .build
	cc $(CPPFLAGS) -O2 -o .build/$@ mesh-verbs-stream.c -lrdma
	codesign --force --sign - .build/$@
libmesh.dylib: mesh-dataflow.c mesh-dataflow.h mesh-stream.c mesh-stream.h mesh.h
	cc -O2 -dynamiclib -Wl,-install_name,@rpath/libmesh.dylib -o $@ mesh-dataflow.c mesh-stream.c
mesh-stream-profile: mesh-stream-profile.c mesh-stream.c mesh-stream.h
	mkdir -p .build
	cc -O2 -o .build/$@ mesh-stream-profile.c mesh-stream.c && .build/$@
clean:
	rm -f $(BINS)
.PHONY: all clean mesh-page-stream mesh-verbs-stream mesh-stream-profile
