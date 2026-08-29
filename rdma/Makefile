BINS = mesh-flow mesh-stat libmesh.dylib
all: $(BINS)
mesh-flow: mesh-flow.c mesh.h
	cc -O2 -o $@ mesh-flow.c -lrdma
mesh-stat: mesh-stat.c mesh.h
	cc -O2 -o $@ mesh-stat.c
libmesh.dylib: mesh-client.c mesh.h
	cc -O2 -dynamiclib -o $@ mesh-client.c
clean:
	rm -f $(BINS)
.PHONY: all clean
