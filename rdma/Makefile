F ?= f-identity.c
mesh-flow: mesh-flow.c $(F) mesh-f.h
	cc -O2 -o $@ mesh-flow.c $(F) -lrdma
clean:
	rm -f mesh-flow qcaps qfill
