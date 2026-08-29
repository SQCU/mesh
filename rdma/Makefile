BINS = mesh-flow mesh-stat mesh-app libmesh.dylib

all: $(BINS)

mesh-flow: mesh-flow.c mesh-mem.c mesh-mem.h mesh.h
	cc -O2 -o $@ mesh-flow.c mesh-mem.c -lrdma

mesh-stat: mesh-stat.c mesh.h
	cc -O2 -o $@ mesh-stat.c

mesh-app: mesh-app.c mesh-client.c mesh.h
	cc -O2 -o $@ mesh-app.c mesh-client.c

libmesh.dylib: mesh-client.c mesh.h
	cc -O2 -dynamiclib -o $@ mesh-client.c

clean:
	rm -f $(BINS)

.PHONY: all clean
