BINS = mesh-flow mesh-stat
all: $(BINS)
mesh-flow: mesh-flow.c mesh.h mesh-flight.h mesh-wire.h mesh-links.h
	cc -O2 -o $@ mesh-flow.c -lrdma
	codesign --force --sign "Mesh Bridge" --identifier io.mesh.bridge $@ || codesign --force --sign - --identifier io.mesh.bridge $@
mesh-stat: mesh-stat.c mesh.h
	cc -O2 -o $@ mesh-stat.c
mesh-stream: mesh-stream.c mesh-client.c mesh.h
	cc -O2 -o $@ mesh-stream.c mesh-client.c
clean:
	rm -f $(BINS) mesh-stream
.PHONY: all clean
