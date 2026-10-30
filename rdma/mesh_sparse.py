"""Preparation binding for the shared sparse contraction schedule compiler.

Domains partition output rectangles and declare retained reduction intervals.
Python and native graph frontends consume the same C compiler and record ABI.
"""
import ctypes as C
from functools import cache
import os
from pathlib import Path


class _Domain(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ('row', 'rows', 'column', 'columns')] + [
        (name, C.c_size_t) for name in ('first_interval', 'interval_count')]


class _Product(C.Structure):
    _fields_ = [(name, C.POINTER(C.c_uint32)) for name in ('records', 'pieces')] + [
        (name, C.c_size_t) for name in ('tiles', 'piece_count')] + [
        (name, C.c_uint32) for name in ('groups', 'full_rows', 'full_columns', 'full_inner')]


@cache
def _library():
    directory = Path(os.environ.get('MESH_LIBDIR', Path(__file__).parent))
    library = C.CDLL(str(directory / 'libnccl-mesh.dylib'), use_errno=True)
    library.mesh_sparse_product_compile.argtypes = (
        C.POINTER(C.c_uint32), C.POINTER(_Domain), C.c_size_t,
        C.POINTER(C.c_uint32 * 2), C.POINTER(_Product))
    library.mesh_sparse_product_compile.restype = C.c_int
    library.mesh_sparse_product_free.argtypes = (C.POINTER(_Product),)
    library.mesh_sparse_product_free.restype = None
    return library


def product_schedule(domains, geometry):
    """Compile (row, rows, column, columns, reduction intervals) domains.

    Geometry is row/column/inner tile width and SIMD groups per threadgroup.
    Return two uint4 records per tile, uint2 inner pieces, and dispatch groups.
    Repeated supports share pieces; empty supports write zero in this product.
    """
    declarations, intervals = [], []
    for row, rows, column, columns, support in domains:
        first = len(intervals)
        intervals.extend(support)
        declarations.append(_Domain(row, rows, column, columns, first, len(intervals) - first))
    declaration = (_Domain * len(declarations))(*declarations)
    retained = ((C.c_uint32 * 2) * len(intervals))(*(tuple(map(int, pair)) for pair in intervals))
    result = _Product()
    library = _library()
    if library.mesh_sparse_product_compile((C.c_uint32 * 4)(*geometry), declaration,
                                           len(declarations), retained, C.byref(result)):
        error = C.get_errno()
        raise OSError(error, os.strerror(error))
    try:
        records = tuple(tuple(result.records[i:i + 4]) for i in range(0, result.tiles * 8, 4))
        pieces = tuple(tuple(result.pieces[i:i + 2]) for i in range(0, result.piece_count * 2, 2))
        return records, pieces, result.groups
    finally:
        library.mesh_sparse_product_free(C.byref(result))
