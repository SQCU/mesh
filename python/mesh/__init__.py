import ctypes as C
import errno
import itertools
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np

from ._native import Native, Shape, View, CopyRegion

__all__ = ['Program', 'Tensor', 'Ref', 'BlockSpec', 'ShapeDtypeStruct', 'Result']
_PROGRAMS = set()
_DTYPES = tuple(map(np.dtype, ('float16', 'float32', 'int32', 'uint32', 'int64', 'uint64', 'uint8', 'bool')))


# design/algorithm-sources.md#indexed-library-functions
def check(code):
    if code:
        raise OSError(code, os.strerror(code))


@dataclass(frozen=True)
class Partial:
    required: frozenset
    terms: frozenset

    @classmethod
    # design/algorithm-sources.md#partial
    def merge(cls, refs):
        values = tuple(ref.partial for ref in refs if ref.partial is not None)
        if not values:
            return None
        required = frozenset.union(*(value.required for value in values))
        terms = frozenset.union(*(value.terms for value in values))
        if sum(len(value.terms) for value in values) != len(terms):
            raise ValueError('A partial contribution occurs twice in the same reduction')
        return None if required == terms else cls(required, terms)


class Ref:
    partial = None
    # design/algorithm-sources.md#indexed-library-functions
    def __init__(self, program, view, dtype):
        self.program, self.view, self.dtype = program, view, np.dtype(dtype)
        self.shape = (view.rows, view.columns)
        self._writer = C.c_void_p()
        self._writer_error = program.native.algebra_writer(program.handle, view, C.byref(self._writer))
        address = program.native.tensor_data(view.tensor, view.extent)
        length = view.offset + (view.rows - 1) * view.row_stride + (view.columns - 1) * view.column_stride + 1
        buffer = (C.c_ubyte * (length * self.dtype.itemsize)).from_address(address)
        self.array = np.ndarray(self.shape, self.dtype, buffer,
            offset=view.offset * self.dtype.itemsize,
            strides=(view.row_stride * self.dtype.itemsize, view.column_stride * self.dtype.itemsize))

    # design/algorithm-sources.md#indexed-library-functions
    def whole(self):
        full = self.program.native.tensor_view(self.view.tensor, self.view.extent)
        dense = (self.view.column_stride == 1 and self.view.row_stride == self.view.columns) or (self.view.row_stride == 1 and self.view.column_stride == self.view.rows)
        if not dense or self.view.offset or self.view.rows * self.view.columns != full.rows * full.columns:
            raise ValueError('Publication and transfer require a whole block; allocate smaller blocks for finer progress')
        return self

    @property
    # design/algorithm-sources.md#indexed-library-functions
    def T(self):
        result = Ref(self.program, self.program.native.view_transpose(self.view), self.dtype)
        result.partial = self.partial
        return result

    # design/algorithm-sources.md#indexed-library-functions
    def slice(self, row, column, rows, columns):
        if min(row, column, rows, columns) < 0 or row + rows > self.shape[0] or column + columns > self.shape[1]:
            raise ValueError('Slice is outside the reference')
        if rows == 0 or columns == 0:
            return self.program.tensor((rows, columns), dtype=self.dtype)
        if row == column == 0 and (rows, columns) == self.shape:
            return self
        view = self.program.native.view_slice(self.view, row, column, rows, columns)
        if not view.tensor:
            raise ValueError('Slice is outside the reference')
        result = Ref(self.program, view, self.dtype)
        result.partial = self.partial
        return result

    # design/algorithm-sources.md#indexed-library-functions
    def broadcast(self, rows, columns):
        if min(rows, columns) < 0 or any(source != target and source != 1 for source, target in zip(self.shape, (rows, columns))):
            raise ValueError('Incompatible broadcast shape')
        if rows == 0 or columns == 0:
            return self.program.tensor((rows, columns), dtype=self.dtype)
        view = self.program.native.view_broadcast(self.view, rows, columns)
        if not view.tensor:
            raise ValueError('Incompatible broadcast shape')
        result = Ref(self.program, view, self.dtype)
        result.partial = self.partial
        return result

    @property
    # design/algorithm-sources.md#view-scoped-host-production
    def writable(self):
        check(self._writer_error)
        return bool(self.program.native.writer_writable(self._writer))

    # design/algorithm-sources.md#indexed-library-functions
    def on(self, peer):
        return (self, peer)


class Tensor:
    # design/algorithm-sources.md#collective
    def _with_blocks(self, blocks):
        result = object.__new__(Tensor)
        result.__dict__ = self.__dict__ | {'blocks': blocks, 'handle': None, '_span': None}
        return result

    # design/algorithm-sources.md#indexed-range-generation
    def __init__(self, program, shape, block_shape, dtype, transferable, contiguous=False):
        import operator
        self.program, self.shape, self.block_shape = program, tuple(map(operator.index, shape)), tuple(map(operator.index, block_shape))
        self.dtype = np.dtype(dtype)
        if len(self.shape) != 2 or len(self.block_shape) != 2 or min(self.shape) < 0 or min(self.block_shape) <= 0:
            raise ValueError('Tensor shapes require two nonnegative dimensions and positive block dimensions')
        scalar = _DTYPES.index(self.dtype)
        self.grid = tuple((s + b - 1) // b for s, b in zip(self.shape, self.block_shape))
        coordinates = tuple(itertools.product(*(range(n) for n in self.grid))) if all(self.grid) else ()
        shapes = [Shape(*(min(b, s - i*b) for s, b, i in zip(self.shape, self.block_shape, coord)),
                        scalar) for coord in coordinates]
        if contiguous and shapes:
            shapes = [Shape(*self.shape, scalar)]
        self.handle = program.native.tensor_create(program.handle, (Shape * len(shapes))(*shapes), len(shapes), transferable, contiguous) if shapes else None
        if shapes and not self.handle:
            check(C.get_errno() or errno.ENOMEM)
        self._span = Ref(program, program.native.tensor_view(self.handle, 0), self.dtype) if contiguous and shapes else None
        self.blocks = {coord: self._span.slice(*(i*b for i, b in zip(coord, self.block_shape)),
                           *(min(b, s-i*b) for s, b, i in zip(self.shape, self.block_shape, coord)))
                       if self._span is not None else Ref(program, program.native.tensor_view(self.handle, i), self.dtype)
                       for i, coord in enumerate(coordinates)}

    # design/algorithm-sources.md#pallas-call-ergonomics
    def region(self, row, column, rows, columns):
        if min(row, column, rows, columns) < 0 or row + rows > self.shape[0] or column + columns > self.shape[1]:
            raise ValueError('Region is outside the tensor')
        if rows == 0 or columns == 0:
            return self.program.tensor((rows, columns), dtype=self.dtype)
        if getattr(self, "_span", None) is not None:
            return self._span.slice(row, column, rows, columns)
        i, j = row // self.block_shape[0], column // self.block_shape[1]
        ref = self[i, j]
        r, c = row % self.block_shape[0], column % self.block_shape[1]
        if r + rows > ref.shape[0] or c + columns > ref.shape[1]:
            raise ValueError('Region crosses backing blocks; realize storage with blocks containing the requested region')
        return ref.slice(r, c, rows, columns)

    # design/algorithm-sources.md#indexed-library-functions
    def __getitem__(self, coordinate):
        return self.blocks[coordinate]

    # design/algorithm-sources.md#indexed-library-functions
    def on(self, peer):
        return (self, peer)

    @property
    # design/algorithm-sources.md#indexed-library-functions
    def T(self):
        result = object.__new__(Tensor)
        result.program, result.dtype, result.handle = self.program, self.dtype, self.handle
        result.shape, result.block_shape, result.grid = self.shape[::-1], self.block_shape[::-1], self.grid[::-1]
        result._span = self._span.T if getattr(self, "_span", None) is not None else None
        result.blocks = {(j, i): ref.T for (i, j), ref in self.blocks.items()}
        return result

    # design/algorithm-sources.md#streamed-normalization-and-embedding
    def broadcast_to(self, shape):
        shape = tuple(shape)
        if len(shape) != 2 or min(shape) < 0 or any(source != target and source != 1 for source, target in zip(self.shape, shape)):
            raise ValueError('Incompatible tensor broadcast shape')
        if 0 in shape:
            return self.program.tensor(shape, tuple(max(1, target) if source == 1 else block
                for source, target, block in zip(self.shape, shape, self.block_shape)), self.dtype)
        result = object.__new__(Tensor)
        result.program, result.dtype, result.handle = self.program, self.dtype, self.handle
        result.shape, result.grid = shape, self.grid
        result._span = self._span.broadcast(*shape) if getattr(self, "_span", None) is not None else None
        result.block_shape = tuple(target if source == 1 else block
            for source, target, block in zip(self.shape, shape, self.block_shape))
        result.blocks = {coordinate: ref.broadcast(*(target if source == 1 else size
            for source, target, size in zip(self.shape, shape, ref.shape)))
            for coordinate, ref in self.blocks.items()}
        return result

@dataclass(frozen=True)
class ShapeDtypeStruct:
    shape: tuple
    dtype: object = np.float32


@dataclass(frozen=True)
class BlockSpec:
    block_shape: object
    index_map: object = None
    _tensor: object = None

    # design/algorithm-sources.md#pallas-call-ergonomics
    def _bind(self, tensor):
        return replace(self, _tensor=tensor)

    # design/algorithm-sources.md#pallas-call-ergonomics
    def resolve(self, coordinate):
        if self._tensor is None:
            raise ValueError('BlockSpec is bound by kernel_call')
        if self.block_shape is None:
            return self._tensor
        index = tuple(self.index_map(*coordinate))
        shape = tuple(self.block_shape)
        if len(shape) != 2 or len(index) != 2 or min(shape) <= 0:
            raise ValueError('BlockSpec requires two positive block dimensions and two indices')
        row, column = (i * b for i, b in zip(index, shape))
        if isinstance(self._tensor, Ref):
            return self._tensor.slice(row, column,
                min(shape[0], self._tensor.shape[0] - row),
                min(shape[1], self._tensor.shape[1] - column))
        return self._tensor.region(row, column,
            min(shape[0], self._tensor.shape[0] - row),
            min(shape[1], self._tensor.shape[1] - column))


class Result:
    # design/algorithm-sources.md#view-scoped-consumption
    def __init__(self, ref):
        self.ref = ref
        self._array = ref.array.view()
        self._array.flags.writeable = False
        first, count = C.c_size_t(), C.c_size_t()
        check(ref.program.native.algebra_export(ref.program.handle, ref.view, C.byref(first), C.byref(count)))
        self.indices = tuple(range(first.value, first.value + count.value))

    @property
    # design/algorithm-sources.md#view-scoped-consumption
    def ready(self):
        return all(self.ref.program.native.algebra_available(self.ref.program.handle, index) for index in self.indices)

    @property
    # design/algorithm-sources.md#indexed-library-functions
    def array(self):
        if not self.ready:
            raise BlockingIOError(errno.EAGAIN, 'Result region is not published')
        return self._array

    # design/algorithm-sources.md#view-scoped-consumption
    def consume(self):
        for index in self.indices:
            self.ref.program.native.algebra_consume(self.ref.program.handle, index)


class Program:
    # design/algorithm-sources.md#indexed-library-functions
    def __init__(self, backend='cpu', region=None, coreml=None, *, functions=None):
        self.native = Native()
        self.context = self.native.context()
        self._functions = dict(functions or {})
        self._constant_extents = set()
        self._replicated_extents = {}
        create = {'cpu': self.native.algebra_create_cpu, 'metal': self.native.algebra_create}[backend]
        check(self.native.attach(self.context, os.fsencode(region) if region else None))
        self.handle = create(self.context)
        if not self.handle:
            code = C.get_errno() or errno.ENOMEM
            if not _PROGRAMS:
                self.native.detach(self.context)
            check(code)
        _PROGRAMS.add(self.handle)
        self.node = self.native.algebra_node(self.handle)
        if coreml:
            check(self.native.algebra_coreml(self.handle, *(os.fsencode(p) for p in coreml)))

    # design/algorithm-sources.md#indexed-range-generation
    def tensor(self, shape, block_shape=None, dtype=np.float32, transferable=True, *, contiguous=False):
        shape = tuple(shape)
        return Tensor(self, shape, tuple(max(1, size) for size in shape) if block_shape is None else block_shape, dtype, transferable, contiguous)

    # design/algorithm-sources.md#literal-contiguous-materialization
    def contiguous(self, source, *, transferable=True):
        if source.program is not self:
            raise ValueError('Source belongs to another program')
        result = self.tensor(source.shape, dtype=source.dtype, transferable=transferable, contiguous=True)
        if not result.blocks:
            return result
        regions = [CopyRegion(ref.view, i*source.block_shape[0], j*source.block_shape[1])
                   for (i, j), ref in source.blocks.items()] if isinstance(source, Tensor) else [CopyRegion(source.view, 0, 0)]
        check(self.native.algebra_materialize(self.handle, (CopyRegion * len(regions))(*regions),
                                             len(regions), result[0, 0].view))
        return result

    # design/algorithm-sources.md#pallas-call-ergonomics
    def kernel_call(self, kernel, *, out_shape, grid, in_specs, out_specs, peer=None):
        single = isinstance(out_shape, ShapeDtypeStruct)
        shapes = (out_shape,) if single else tuple(out_shape)
        specs = (out_specs,) if single else tuple(out_specs)
        inputs = tuple(in_specs)
        if not shapes or len(shapes) != len(specs):
            raise ValueError('Each output requires one shape and one BlockSpec')

        # design/algorithm-sources.md#pallas-call-ergonomics
        def configure(*operands):
            from . import kernels
            if len(operands) != len(inputs):
                raise ValueError('Each input requires one BlockSpec')
            partial = next((ref for operand in operands
                            for ref in (operand.blocks.values() if isinstance(operand, Tensor) else (operand,))
                            if ref.partial is not None), None)
            if partial is not None and kernel is not kernels.add:
                raise TypeError(f'Partial input {partial!r} requires addition or reduce_scatter before this kernel')
            outputs = tuple(self.tensor(shape.shape, block_shape=spec.block_shape,
                dtype=shape.dtype) for shape, spec in zip(shapes, specs))
            if peer is None or peer == self.node:
                self._call(kernel, grid=grid,
                    inputs=tuple(spec._bind(tensor) for spec, tensor in zip(inputs, operands)),
                    outputs=tuple(spec._bind(tensor) for spec, tensor in zip(specs, outputs)))
            return outputs[0] if single else outputs
        return configure

    # design/algorithm-sources.md#indexed-library-functions
    def _call(self, kernel, *, grid, inputs=(), outputs=()):
        from .kernels import _ExpressionKernel
        import itertools
        binding = self._functions.get(kernel, kernel)
        check(self.native.algebra_kernel(self.handle))
        if isinstance(binding, _ExpressionKernel):
            binding.bind_grid(self, tuple(grid), inputs, outputs)
        else:
            for coordinate in itertools.product(*(range(size) for size in grid)):
                sources = tuple(spec.resolve(coordinate) for spec in inputs)
                targets = tuple(spec.resolve(coordinate) for spec in outputs)
                binding(self, sources, targets)
                for target in targets:
                    target.partial = Partial.merge(sources)

    # design/algorithm-sources.md#indexed-library-functions
    def copy(self, source, destination, *, queue=0):
        src, sender = source
        dst, receiver = destination
        if src.program is not self or dst.program is not self:
            raise ValueError('References belong to another program')
        if isinstance(src, Tensor) and isinstance(dst, Tensor):
            if src.shape != dst.shape:
                raise ValueError('Transfer shapes must match')
            if sender == receiver:
                if sender != self.node:
                    return
                targets = (((0, 0), dst._span),) if getattr(dst, '_span', None) is not None else tuple(dst.blocks.items())
                if any(np.shares_memory(ref.array, target.array) for ref in src.blocks.values() for _, target in targets):
                    raise ValueError('Copy sources overlap destination storage')
                for (i, j), target in targets:
                    row, column = i * dst.block_shape[0], j * dst.block_shape[1]
                    end_row, end_column = row + target.shape[0], column + target.shape[1]
                    regions = []
                    for si in range(row // src.block_shape[0], (end_row - 1) // src.block_shape[0] + 1):
                        for sj in range(column // src.block_shape[1], (end_column - 1) // src.block_shape[1] + 1):
                            r, c = si * src.block_shape[0], sj * src.block_shape[1]
                            ref = src[si, sj]
                            lo_r, lo_c = max(row, r), max(column, c)
                            hi_r, hi_c = min(end_row, r + ref.shape[0]), min(end_column, c + ref.shape[1])
                            part = ref.slice(lo_r - r, lo_c - c, hi_r - lo_r, hi_c - lo_c)
                            regions.append(CopyRegion(part.view, lo_r - row, lo_c - column))
                    check(self.native.algebra_materialize(self.handle,
                        (CopyRegion * len(regions))(*regions), len(regions), target.view))
                return
            cuts = []
            for axis, size in enumerate(src.shape):
                boundaries = {0, size}
                for tensor in (src, dst):
                    if getattr(tensor, '_span', None) is None:
                        boundaries.update(range(0, size, tensor.block_shape[axis]))
                cuts.append(sorted(boundaries))
            for row, end_row in zip(cuts[0], cuts[0][1:]):
                for column, end_column in zip(cuts[1], cuts[1][1:]):
                    shape = end_row - row, end_column - column
                    self.copy(src.region(row, column, *shape).on(sender),
                              dst.region(row, column, *shape).on(receiver), queue=queue)
            return
        check(self.native.algebra_copy(self.handle,
            src.view, sender, dst.view, receiver, queue))
        dst.partial = src.partial

    # design/algorithm-sources.md#canonical-view-replication
    def replicate(self, source, peer):
        tensor, sender = source
        if not isinstance(tensor, Tensor) or tensor.program is not self:
            raise ValueError('Replication requires a tensor belonging to this program')
        if sender == peer:
            return tensor
        result = object.__new__(Tensor)
        result.program, result.dtype, result.handle = self, tensor.dtype, None
        result.shape, result.block_shape, result.grid = tensor.shape, tensor.block_shape, tensor.grid
        result.blocks = {}
        for coordinate, ref in tensor.blocks.items():
            key = (sender, peer, ref.view.tensor, ref.view.extent)
            if key not in self._replicated_extents:
                original = Ref(self, self.native.tensor_view(ref.view.tensor, ref.view.extent), ref.dtype)
                backing = self.tensor(original.shape, dtype=original.dtype)[0, 0]
                self.copy(original.on(sender), backing.on(peer))
                self._replicated_extents[key] = backing
            backing = self._replicated_extents[key]
            view = View(backing.view.tensor, backing.view.extent, ref.view.offset,
                        ref.view.rows, ref.view.columns, ref.view.row_stride, ref.view.column_stride)
            result.blocks[coordinate] = Ref(self, view, ref.dtype)
        return result

    # design/algorithm-sources.md#indexed-library-functions
    def export(self, ref):
        return Result(ref)

    @contextmanager
    # design/algorithm-sources.md#view-scoped-host-production
    def write(self, ref):
        if ref.program is not self:
            raise ValueError("Reference belongs to another program")
        check(ref._writer_error)
        if not self.native.writer_issue(ref._writer):
            raise BlockingIOError(errno.EAGAIN, 'Producer region still has readers')
        published = False
        try:
            yield ref.array
            published = True
        finally:
            self.native.writer_complete(ref._writer, published)

    # design/algorithm-sources.md#indexed-library-functions
    def constant(self, ref, value):
        ref.whole()
        ref.array[...] = value
        check(self.native.tensor_constant(ref.view.tensor, ref.view.extent))
        self._constant_extents.add((ref.view.tensor, ref.view.extent))

    # design/algorithm-sources.md#indexed-library-functions
    def realize(self):
        check(self.native.algebra_realize(self.handle))
        return self

    @property
    # design/algorithm-sources.md#indexed-library-functions
    def report(self):
        return self.native.algebra_report(self.handle)

    # design/algorithm-sources.md#indexed-library-functions
    def close(self):
        if self.handle:
            self.native.algebra_destroy(self.handle)
            _PROGRAMS.discard(self.handle)
            self.handle = None
            self._constant_extents.clear()
            self._replicated_extents.clear()
            if not _PROGRAMS:
                check(self.native.detach(self.context))

    # design/algorithm-sources.md#indexed-library-functions
    def __enter__(self):
        return self

    # design/algorithm-sources.md#indexed-library-functions
    def __exit__(self, *error):
        self.close()
