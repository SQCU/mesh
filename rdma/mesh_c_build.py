#!/usr/bin/env python3
"""The mesh's C headers as one Python module, _mesh_c (ffi, lib): the declarations are the headers' own text and the
module is compiled against the same headers, so no Python layout or prototype is written by hand.  `make mesh-c`
runs it with the interpreter that will import it (torch-mesh's venv by default); the module lands beside the headers.
Text the binder cannot take as written is left to the compiler: a C++-only block is skipped, an #include and a
function-like #define are dropped, an object-like #define of an integer expression and an enum value written as an
expression are `...` and one of a floating literal a static const double (the compiled module reads them from the
header), and one of anything else (a string, NULL, an initializer) is dropped."""
import os
import re
import sys

import cffi

RDMA = os.path.dirname(os.path.abspath(__file__))
HEADERS = ('../etc/bridge/layout.h', 'mesh-fabric.h', 'mesh-plan.h', 'nccl.h', 'mesh-write-compiler.h', 'mesh-statistics.h')
LITERAL = r'-?(?:0x[0-9a-fA-F]+|\d+)[uUlL]*'
FLOATING = r'-?\d+\.\d*(?:[eE][-+]?\d+)?'


def definitions_dropped(text):
    """The text without its static inline function definitions: cdef takes declarations, and no Python caller calls a
    header's inline helper (mesh-plan.h mesh_step_bytes)."""
    out, at = [], 0
    for m in re.finditer(r'\bstatic\s+inline\b', text):
        brace, semi = text.find('{', m.end()), text.find(';', m.end())
        if m.start() < at or brace < 0 or 0 <= semi < brace:
            continue
        depth, k = 0, brace
        while k < len(text):
            depth += {'{': 1, '}': -1}.get(text[k], 0)
            k += 1
            if not depth:
                break
        out.append(text[at:m.start()])
        at = k
    return ''.join(out) + text[at:]


def declarations(text):
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    text = re.sub(r'//[^\n]*', '', text)
    text = definitions_dropped(text)
    out, continued, skipping, depth = [], False, 0, 0
    for line in text.splitlines():
        s = line.strip()
        if continued:
            continued = s.endswith('\\')
            continue
        if s.startswith('#'):
            continued = s.endswith('\\')
            directive = s[1:].strip()
            if re.match(r'if(n?def)?\b', directive):
                depth += 1
                if not skipping and re.match(r'ifdef\s+__cplusplus\b', directive):
                    skipping = depth
            elif re.match(r'endif\b', directive):
                skipping = 0 if skipping == depth else skipping
                depth -= 1
            elif not skipping:
                defined = re.match(r'define\s+([A-Za-z_]\w*)\s+(.+?)\s*$', directive)
                if defined and re.fullmatch(LITERAL, defined.group(2)):
                    out.append(f'#define {defined.group(1)} {defined.group(2)}')
                elif defined and re.fullmatch(FLOATING, defined.group(2)):
                    out.append(f'static const double {defined.group(1)};')
                elif defined and re.fullmatch(r'[\w\s()+\-*/<>|&~^]+', defined.group(2)) and defined.group(2) != 'NULL':
                    out.append(f'#define {defined.group(1)} ...')
            continue
        if not skipping:
            out.append(line)
    text = '\n'.join(out)

    def enum(match):
        return match.group(1) + re.sub(rf'=\s*(?!{LITERAL}\s*[,}}])[^,}}]+', '= ...', match.group(2)) + match.group(3)
    return re.sub(r'(enum\s*\w*\s*\{)([^}]*)(\})', enum, text, flags=re.S)


def build(output_directory=None, library_directory=None):
    output_directory = output_directory or RDMA
    library_directory = library_directory or RDMA
    ffi = cffi.FFI()
    for header in HEADERS:
        with open(os.path.join(RDMA, header)) as file:
            ffi.cdef(declarations(file.read()))
    ffi.set_source('_mesh_c', ''.join(f'#include "{h}"\n' for h in HEADERS), include_dirs=[RDMA], library_dirs=[library_directory],
                   libraries=['nccl-mesh', 'mesh'], extra_link_args=[f'-Wl,-rpath,{library_directory}'])
    return ffi.compile(tmpdir=output_directory, verbose=False)


if __name__ == '__main__':
    print(build())
    sys.exit(0)
