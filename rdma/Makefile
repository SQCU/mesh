mesh-flow: mesh-flow.c mesh-mem.c mesh-mem.h mesh.h
	cc -O2 -o $@ mesh-flow.c mesh-mem.c -lrdma
clean:
	rm -f mesh-flow qcaps qfill

mesh-app: mesh-app.c mesh-client.c mesh.h
	cc -O2 -o $@ mesh-app.c mesh-client.c

libmesh.dylib: mesh-client.c mesh.h
	cc -O2 -dynamiclib -o $@ mesh-client.c

mesh-stat: mesh-stat.c mesh.h
	cc -O2 -o $@ mesh-stat.c
