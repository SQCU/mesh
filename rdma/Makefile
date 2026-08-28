mesh-flow: mesh-flow.c mesh-mem.c mesh-f.h mesh-mem.h mesh.h
	cc -O2 -o $@ mesh-flow.c mesh-mem.c -lrdma
clean:
	rm -f mesh-flow qcaps qfill

mesh-app: mesh-app.c mesh-client.c mesh.h
	cc -O2 -o $@ mesh-app.c mesh-client.c
