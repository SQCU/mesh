"""Finite index relations for graph importers, during preparation only.

This binds the same ISL archive as MetalTensorLayout. Its objects and integers
use mesh's page allocator. No relation object is an execution operand.
"""
import ctypes as C
import os
from pathlib import Path


def coordinates(shape, stem):
    return [f'{stem}{i}' for i in range(len(shape))]


def bounds(shape, names):
    return [f'0 <= {name} < {int(n)}' for name, n in zip(names, shape)]


def flat(shape, names):
    value = '0'
    for n, name in zip(shape, names):
        value = f'({value}) * {int(n)} + {name}'
    return value


class Algebra:
    """One preparation context; the library location is an explicit build operand."""
    def __init__(self, library=None):
        directory = Path(os.environ.get('MESH_LIBDIR', Path(__file__).parent))
        self.lib = C.CDLL(str(library or directory / 'libnccl-mesh.dylib'))
        specs = {
            'ctx_alloc': (C.c_void_p, ()), 'ctx_free': (None, (C.c_void_p,)),
            'ctx_last_error_msg': (C.c_char_p, (C.c_void_p,)),
            'val_get_num_si': (C.c_long, (C.c_void_p,)), 'val_free': (C.c_void_p, (C.c_void_p,)),
        }
        for kind in ('map', 'set'):
            specs.update({f'{kind}_read_from_str': (C.c_void_p, (C.c_void_p, C.c_char_p)),
                          f'{kind}_copy': (C.c_void_p, (C.c_void_p,)),
                          f'{kind}_free': (C.c_void_p, (C.c_void_p,)),
                          f'{kind}_to_str': (C.c_void_p, (C.c_void_p,)),
                          f'{kind}_is_equal': (C.c_int, (C.c_void_p, C.c_void_p)),
                          f'{kind}_is_empty': (C.c_int, (C.c_void_p,)),
                          f'{kind}_union': (C.c_void_p, (C.c_void_p, C.c_void_p))})
        for name in ('map_apply_range', 'map_intersect_range', 'map_intersect', 'map_subtract_domain',
                     'set_intersect', 'set_subtract', 'set_flat_product'):
            specs[name] = C.c_void_p, (C.c_void_p, C.c_void_p)
        specs['map_domain'] = C.c_void_p, (C.c_void_p,)
        specs['set_project_out'] = C.c_void_p, (C.c_void_p, C.c_int, C.c_uint, C.c_uint)
        for name in ('set_dim_min_val', 'set_dim_max_val'):
            specs[name] = C.c_void_p, (C.c_void_p, C.c_int)
        for name, (result, arguments) in specs.items():
            function = getattr(self.lib, 'isl_' + name)
            function.restype, function.argtypes = result, arguments
            setattr(self, name, function)
        self.lib.mesh_isl_free.argtypes, self.lib.mesh_isl_free.restype = (C.c_void_p,), None
        self.lib.MeshIndexSetRuns.argtypes = C.c_void_p, C.POINTER(C.c_size_t)
        self.lib.MeshIndexSetRuns.restype = C.POINTER(C.c_int64)
        self.ctx = self.ctx_alloc()
        self.references = 1

    def __del__(self):
        if getattr(self, 'ctx', None):
            self.release()

    def release(self):
        # Cyclic Python graph declarations may finalize the context owner before
        # its relations. Keep ISL alive until their native objects are freed.
        self.references -= 1
        if not self.references:
            self.ctx_free(self.ctx)
            self.ctx = None

    def checked(self, value):
        if not value:
            message = self.ctx_last_error_msg(self.ctx)
            raise ValueError(message.decode() if message else 'integer relation could not be resolved')
        return value

    def boolean(self, value):
        if value < 0:
            self.checked(None)
        return bool(value)

    def set(self, text, rank):
        return IndexSet(self, self.set_read_from_str(self.ctx, text.encode()), rank)

    def ownership(self, shape, axis, unit, extent):
        names = coordinates(shape, 'i')
        equation = f'p = floor({names[axis]} / {unit}) mod {extent}' if all(shape) else 'false'
        text = '{ [' + ','.join(names) + '] -> [p] : ' + ' and '.join([*bounds(shape, names), equation]) + ' }'
        return IndexRelation(self, self.map_read_from_str(self.ctx, text.encode()), shape)


class _Object:
    def __init__(self, algebra, pointer):
        self.algebra = algebra
        self.pointer = algebra.checked(pointer)
        algebra.references += 1

    def __del__(self):
        if getattr(self, 'pointer', None):
            getattr(self.algebra, self.kind + '_free')(self.pointer)
            self.algebra.release()

    def copy(self):
        return getattr(self.algebra, self.kind + '_copy')(self.pointer)

    def __str__(self):
        text = self.algebra.checked(getattr(self.algebra, self.kind + '_to_str')(self.pointer))
        try:
            return C.string_at(text).decode()
        finally:
            self.algebra.lib.mesh_isl_free(text)

    def equal(self, other):
        return self.algebra.boolean(getattr(self.algebra, self.kind + '_is_equal')(self.pointer, other.pointer))

    def empty(self):
        return self.algebra.boolean(getattr(self.algebra, self.kind + '_is_empty')(self.pointer))


class IndexSet(_Object):
    kind = 'set'

    def __init__(self, algebra, pointer, rank):
        super().__init__(algebra, pointer)
        self.rank = rank

    def binary(self, other, operation, rank=None):
        return IndexSet(self.algebra, getattr(self.algebra, 'set_' + operation)(self.copy(), other.copy()),
                        self.rank if rank is None else rank)

    def project(self, axis):
        # isl_dim_set is the public enum's output/set dimension (3).
        result = self.copy()
        for first, count in ((axis + 1, self.rank - axis - 1), (0, axis)):
            result = self.algebra.set_project_out(result, 3, first, count)
        return IndexSet(self.algebra, result, 1)

    def edge(self, maximum=False):
        value = self.algebra.checked((self.algebra.set_dim_max_val if maximum else self.algebra.set_dim_min_val)(self.copy(), 0))
        try:
            return self.algebra.val_get_num_si(value)
        finally:
            self.algebra.val_free(value)

    def spans(self):
        """The shared native set-to-run lowering; repetitions stay compact."""
        count = C.c_size_t()
        runs = self.algebra.checked(self.algebra.lib.MeshIndexSetRuns(self.pointer, C.byref(count)))
        try:
            return tuple(tuple(runs[k * 4 + j] for j in range(4)) for k in range(count.value))
        finally:
            self.algebra.lib.mesh_isl_free(runs)

    def cartesian(self):
        """Axis projections if their product is exactly this domain, else None."""
        axes = [self.project(axis) for axis in range(self.rank)]
        product = self.algebra.set('{ [] }', 0)
        for axis in axes:
            product = product.binary(axis, 'flat_product', product.rank + 1)
        return [axis.spans() for axis in axes] if product.equal(self) else None


class IndexRelation(_Object):
    kind = 'map'

    def __init__(self, algebra, pointer, shape):
        super().__init__(algebra, pointer)
        self.shape = tuple(map(int, shape))

    def pullback(self, shape, equations):
        """Compose target coordinates -> source coordinates -> owner labels."""
        x, y = coordinates(shape, 'x'), coordinates(self.shape, 'y')
        conditions = [*bounds(shape, x), *bounds(self.shape, y), *equations(x, y)]
        text = '{ [' + ','.join(x) + '] -> [' + ','.join(y) + '] : ' + (' and '.join(conditions) or 'true') + ' }'
        mapping = self.algebra.checked(self.algebra.map_read_from_str(self.algebra.ctx, text.encode()))
        return IndexRelation(self.algebra, self.algebra.map_apply_range(mapping, self.copy()), shape)

    def reshape(self, shape):
        return self.pullback(shape, lambda x, y: [f'{flat(shape, x)} = {flat(self.shape, y)}'])

    def union(self, other):
        return IndexRelation(self.algebra, self.algebra.map_union(self.copy(), other.copy()), self.shape)

    def intersect(self, other):
        return IndexRelation(self.algebra, self.algebra.map_intersect(self.copy(), other.copy()), self.shape)

    def complete(self, extent):
        """An operation imposes no owner constraint outside the values it uses."""
        names = coordinates(self.shape, 'i')
        text = '{ [' + ','.join(names) + '] -> [p] : ' + ' and '.join([
            *bounds(self.shape, names), f'0 <= p < {int(extent)}']) + ' }'
        universe = self.algebra.checked(self.algebra.map_read_from_str(self.algebra.ctx, text.encode()))
        outside = self.algebra.map_subtract_domain(universe, self.algebra.map_domain(self.copy()))
        return IndexRelation(self.algebra, self.algebra.map_union(outside, self.copy()), self.shape)

    def axes(self, shape, pairs):
        return self.pullback(shape, lambda x, y: [f'{x[b]} = {y[a]}' for a, b in pairs])

    def depends(self, axis):
        projected = self.axes(self.shape, [(d, d) for d in range(len(self.shape)) if d != axis])
        return not projected.equal(self)

    def selected(self, first, count):
        owners = self.algebra.set(f'{{ [p] : {int(first)} <= p < {int(first + count)} }}', 1)
        selected = self.algebra.map_intersect_range(self.copy(), owners.copy())
        return IndexSet(self.algebra, self.algebra.map_domain(selected), len(self.shape))
