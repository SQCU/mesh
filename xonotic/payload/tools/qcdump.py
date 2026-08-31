#!/usr/bin/env python3
"""Dump the FUNCTION and GLOBAL name tables of a QuakeC progs.dat.

The server is a VM and progs.dat is the program: statements, globals, a string
table and a field layout. So two .dat are two programs, and the way to compare
them is to read their symbol tables -- not their sizes or mtimes.
"""
import struct, sys

def load(path):
    b = open(path, 'rb').read()
    ver, crc = struct.unpack_from('<ii', b, 0)
    off = {}
    names = ['statements', 'globaldefs', 'fielddefs', 'functions', 'strings', 'globals']
    p = 8
    for n in names:
        o, c = struct.unpack_from('<ii', b, p); p += 8
        off[n] = (o, c)
    so, sc = off['strings']
    def s(i):
        e = b.index(b'\0', so + i)
        return b[so + i:e].decode('latin-1')
    fo, fc = off['functions']
    funcs = []
    for i in range(fc):
        first, pstart, nloc, prof, sname, sfile, nparm = struct.unpack_from('<7i', b, fo + i * 36)
        funcs.append(s(sname))
    go, gc = off['globaldefs']
    globs = []
    for i in range(gc):
        t, ofs, sname = struct.unpack_from('<HHi', b, go + i * 8)
        globs.append(s(sname))
    return crc, funcs, globs

if __name__ == '__main__':
    crc, f, g = load(sys.argv[1])
    print('crc=%d functions=%d globaldefs=%d' % (crc, len(f), len(g)))
    if len(sys.argv) > 2:
        open(sys.argv[2], 'w').write('\n'.join(sorted(set(f))))
        open(sys.argv[2] + '.globals', 'w').write('\n'.join(sorted(set(g))))
