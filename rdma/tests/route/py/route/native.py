"""The route headers as one cffi FFI (ABI mode): the declarations are the headers' own text, through the repo's binder
(rdma/mesh_c_build.py `declarations`), with route.h's static inline bodies (the features, which only C and Metal run)
left out. numpy views of the C types come from the same FFI: `dtype("struct route_fabric")` is a structured dtype at
the compiler's offsets, so an array of it is the C array (ffi.from_buffer passes it where it lies)."""
import os
import re
import sys

import cffi
import numpy as np

ROUTE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # rdma/tests/route
RDMA = os.path.dirname(os.path.dirname(ROUTE))
sys.path.insert(0, RDMA)
from mesh_c_build import declarations  # noqa: E402


def _without_inline(text):
    """`static inline` definitions dropped, braces matched"""
    out, at = [], 0
    for m in re.finditer(r'static\s+inline\b', text):
        if m.start() < at:
            continue
        brace = text.index('{', m.end())
        depth, k = 0, brace
        while True:
            depth += {'{': 1, '}': -1}.get(text[k], 0)
            k += 1
            if depth == 0:
                break
        out.append(text[at:m.start()]); at = k
    return ''.join(out) + text[at:]


def _header(name):
    with open(os.path.join(ROUTE, name)) as f:
        text = f.read()
    text = re.sub(r'#include\s+"[^"]+"', '', text)
    text = re.sub(r'#ifndef __METAL_VERSION__|#endif|#ifndef ROUTE_\w+_H|#define ROUTE_\w+_H', '', text)
    text = re.sub(r'static const char \*const (\w+)\[[^\]]*\] = \{.*?\};', '', text, flags=re.S)   # telemetry.h's names: names()
    decl = declarations(_without_inline(text))
    decl = re.sub(r'#define \w+ \.\.\.', '', decl)   # aliases of enum values (RT_FIRST_GAUGE): the header's text, names()
    floats = re.findall(r"static const double (\w+);", decl)
    decl = re.sub(r'static const double \w+;', '', decl)
    return decl, floats


ffi = cffi.FFI()
CONSTANTS = {}
for _h in ('split.h', 'telemetry.h', 'route.h', 'kernels.h'):
    _d, _ = _header(_h)
    ffi.cdef(_d, override=True)
for _name, _value in re.findall(r'#define\s+(ROUTE_\w+)\s+(-?\d+\.\d*)', open(os.path.join(ROUTE, 'route.h')).read()):
    CONSTANTS[_name] = float(_value)


def dtype(ctype):
    """the numpy structured dtype of a C struct, at cffi's (the compiler's) offsets, arrays as subarrays"""
    t = ffi.typeof(ctype) if isinstance(ctype, str) else ctype
    names, formats, offsets = [], [], []
    for name, field in t.fields:
        ft = field.type
        shape = []
        while ft.kind == 'array':
            shape.append(ft.length); ft = ft.item
        base = dtype(ft) if ft.kind == 'struct' else np.dtype({'int': 'i4', 'unsigned int': 'u4', 'uint32_t': 'u4', 'uint64_t': 'u8',
                                                               'double': 'f8', 'float': 'f4', 'int64_t': 'i8'}[ft.cname])
        names.append(name); formats.append((base, tuple(shape)) if shape else base); offsets.append(field.offset)
    return np.dtype({'names': names, 'formats': formats, 'offsets': offsets, 'itemsize': ffi.sizeof(t)})


def const(name):
    """a route header's constant: an enum value, or a floating #define (ROUTE_PAGE)"""
    return CONSTANTS[name] if name in CONSTANTS else getattr(_constants, name)


_constants = ffi.dlopen(None)   # the cdef's enum values (no library's symbols read)


def names(header, array):
    """a header's static string table (telemetry.h's route_tel_names), as written"""
    text = open(os.path.join(ROUTE, header)).read()
    body = re.search(rf'{array}\[[^\]]*\] = \{{(.*?)\}};', text, flags=re.S).group(1)
    return re.findall(r'"([^"]*)"', body)


def alias(header, name):
    """an object-like #define naming an enum value (RT_FIRST_GAUGE RT_WORKERS), resolved"""
    text = open(os.path.join(ROUTE, header)).read()
    return const(re.search(rf'#define\s+{name}\s+(\w+)', text).group(1))


def load(path=None):
    """libroute.dylib (build.sh): the simulator, the deployed forward, the trainer"""
    return ffi.dlopen(path or os.path.join(ROUTE, 'libroute.dylib'))


def records(ctype, array, writable=False):
    """a record array as the C array `ctype[]`, where it lies; its records must be the C type's size (numpy's
    promotion of a structured dtype can drop a struct's trailing padding)"""
    if array.dtype.itemsize != ffi.sizeof(ctype) or not array.flags['C_CONTIGUOUS']:
        raise TypeError(f'{ctype}: records of {array.dtype.itemsize} bytes (contiguous {array.flags["C_CONTIGUOUS"]}), the C type {ffi.sizeof(ctype)}')
    return ffi.from_buffer(f'{ctype}[]', array, require_writable=writable)
