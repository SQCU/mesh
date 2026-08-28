F ?= f-identity.c
mesh-flow: mesh-flow.c mesh-mem.c $(F) mesh-f.h mesh-mem.h
	cc -O2 -o $@ mesh-flow.c mesh-mem.c $(F) -lrdma
clean:
	rm -f mesh-flow qcaps qfill
