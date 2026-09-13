import ctypes as C
import errno
import itertools
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np

from ._native import Native, Shape, View, Endpoint, Submission

__all__ = ['Program', 'Tensor', 'Ref', 'BlockSpec', 'ShapeDtypeStruct', 'Result']
_DTYPES = tuple(map(np.dtype, ('float16', 'float32', 'int32', 'uint32')))


# design/algorithm-sources.md#indexed-library-functions
def check(code):
    if code:
        raise OSError(code, os.strerror(code))


class Ref:
    # design/algorithm-sources.md#indexed-library-functions
    def __init__(self, program, view, dtype):
        self.program, self.view, self.dtype = program, view, np.dtype(dtype)
        self.shape = (view.rows, view.columns)
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
        return Ref(self.program, self.program.native.view_transpose(self.view), self.dtype)

    # design/algorithm-sources.md#indexed-library-functions
    def slice(self, row, column, rows, columns):
        view = self.program.native.view_slice(self.view, row, column, rows, columns)
        if not view.tensor:
            raise ValueError('Slice is outside the reference')
        return Ref(self.program, view, self.dtype)

    # design/algorithm-sources.md#indexed-library-functions
    def broadcast(self, rows, columns):
        view = self.program.native.view_broadcast(self.view, rows, columns)
        if not view.tensor:
            raise ValueError('Incompatible broadcast shape')
        return Ref(self.program, view, self.dtype)

    @property
    # design/algorithm-sources.md#streaming-overlap-measurement
    def present(self):
        self.whole()
        return bool(self.program.native.tensor_present(self.view.tensor, self.view.extent))

    # design/algorithm-sources.md#indexed-library-functions
    def on(self, peer):
        return (self, peer)


class Tensor:
    # design/algorithm-sources.md#indexed-library-functions
    def __init__(self, program, shape, block_shape, dtype, transferable):
        self.program, self.shape, self.block_shape = program, tuple(shape), tuple(block_shape)
        self.dtype = np.dtype(dtype)
        if len(self.shape) != 2 or len(self.block_shape) != 2 or min(*self.shape, *self.block_shape) <= 0:
            raise ValueError('Tensor and block shapes must have two positive dimensions')
        self.grid = tuple((s + b - 1) // b for s, b in zip(self.shape, self.block_shape))
        coordinates = tuple(itertools.product(*(range(n) for n in self.grid)))
        shapes = [Shape(*(min(b, s - i*b) for s, b, i in zip(self.shape, self.block_shape, coord)),
                        _DTYPES.index(self.dtype)) for coord in coordinates]
        self.handle = program.native.tensor_create(program.handle, (Shape * len(shapes))(*shapes), len(shapes), transferable)
        if not self.handle:
            check(C.get_errno() or errno.ENOMEM)
        self.blocks = {coord: Ref(program, program.native.tensor_view(self.handle, i), self.dtype)
                       for i, coord in enumerate(coordinates)}

    # design/algorithm-sources.md#pallas-call-ergonomics
    def region(self, row, column, rows, columns):
        if min(row, column) < 0 or min(rows, columns) <= 0 or row + rows > self.shape[0] or column + columns > self.shape[1]:
            raise ValueError('Region is outside the tensor')
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
        result.blocks = {(j, i): ref.T for (i, j), ref in self.blocks.items()}
        return result

@dataclass(frozen=True)
class ShapeDtypeStruct:
    shape: tuple
    dtype: object = np.float32


@dataclass(frozen=True)
class BlockSpec:
    block_shape: object
    index_map: object
    _tensor: object = None

    # design/algorithm-sources.md#pallas-call-ergonomics
    def _bind(self, tensor):
        return replace(self, _tensor=tensor)

    # design/algorithm-sources.md#pallas-call-ergonomics
    def resolve(self, coordinate):
        index = tuple(self.index_map(*coordinate))
        shape = tuple(self.block_shape)
        if len(shape) != 2 or len(index) != 2 or min(shape) <= 0:
            raise ValueError('BlockSpec requires two positive block dimensions and two indices')
        if self._tensor is None:
            raise ValueError('BlockSpec is bound by kernel_call')
        row, column = (i * b for i, b in zip(index, shape))
        return self._tensor.region(row, column,
            min(shape[0], self._tensor.shape[0] - row),
            min(shape[1], self._tensor.shape[1] - column))


class Result:
    # design/algorithm-sources.md#indexed-library-functions
    def __init__(self, ref):
        self.ref = ref.whole()
        self._array = ref.array.view()
        self._array.flags.writeable = False
        index = C.c_size_t()
        check(ref.program.native.algebra_export(ref.program.handle, ref.view.tensor, ref.view.extent, C.byref(index)))
        self.index = index.value

    @property
    # design/algorithm-sources.md#indexed-library-functions
    def ready(self):
        return bool(self.ref.program.native.algebra_available(self.ref.program.handle, self.index))

    @property
    # design/algorithm-sources.md#indexed-library-functions
    def array(self):
        if not self.ready:
            raise BlockingIOError(errno.EAGAIN, 'Result region is not published')
        return self._array

    # design/algorithm-sources.md#indexed-library-functions
    def consume(self):
        self.ref.program.native.algebra_consume(self.ref.program.handle, self.index)


class Program:
    # design/algorithm-sources.md#indexed-library-functions
    def __init__(self, backend='cpu', region=None, coreml=None):
        self.native = Native()
        self.context = self.native.context()
        self.callbacks, self.errors = [], []
        create = {'cpu': self.native.algebra_create_cpu, 'metal': self.native.algebra_create}[backend]
        check(self.native.attach(self.context, os.fsencode(region) if region else None))
        self.handle = create(self.context)
        if not self.handle:
            code = C.get_errno() or errno.ENOMEM
            self.native.detach(self.context)
            check(code)
        if coreml:
            check(self.native.algebra_coreml(self.handle, *(os.fsencode(p) for p in coreml)))

    # design/algorithm-sources.md#indexed-library-functions
    def tensor(self, shape, block_shape=None, dtype=np.float32, transferable=True):
        return Tensor(self, shape, block_shape or shape, dtype, transferable)

    # design/algorithm-sources.md#pallas-call-ergonomics
    def kernel_call(self, kernel, *, out_shape, grid, in_specs, out_specs):
        single = isinstance(out_shape, ShapeDtypeStruct)
        shapes = (out_shape,) if single else tuple(out_shape)
        specs = (out_specs,) if single else tuple(out_specs)
        inputs = tuple(in_specs)
        if not shapes or len(shapes) != len(specs):
            raise ValueError('Each output requires one shape and one BlockSpec')

        # design/algorithm-sources.md#pallas-call-ergonomics
        def configure(*operands):
            if len(operands) != len(inputs):
                raise ValueError('Each input requires one BlockSpec')
            outputs = tuple(self.tensor(shape.shape,
                block_shape=spec.block_shape if spec.block_shape[1] != shape.shape[1] else None,
                dtype=shape.dtype) for shape, spec in zip(shapes, specs))
            self._call(kernel, grid=grid,
                inputs=tuple(spec._bind(tensor) for spec, tensor in zip(inputs, operands)),
                outputs=tuple(spec._bind(tensor) for spec, tensor in zip(specs, outputs)))
            return outputs[0] if single else outputs
        return configure

    # design/algorithm-sources.md#indexed-library-functions
    def _bind_native(self, submission, inputs, outputs, binding=None):
        inputs, outputs = tuple(inputs), tuple(outputs)
        if any(r.program is not self for r in inputs + outputs):
            raise ValueError('References belong to another program')
        submit = submission if isinstance(submission, Submission) else Submission(submission)
        check(self.native.algebra_function(self.handle,
            (View * len(inputs))(*(r.view for r in inputs)), len(inputs),
            (View * len(outputs))(*(r.view for r in outputs)), len(outputs), submit, binding))
        self.callbacks.append((submit, inputs, outputs, binding))

    # design/algorithm-sources.md#indexed-library-functions
    def _call(self, kernel, *, grid, inputs=(), outputs=()):
        for coordinate in itertools.product(*(range(n) for n in grid)):
            reads = tuple(spec.resolve(coordinate) for spec in inputs)
            writes = tuple(spec.resolve(coordinate) for spec in outputs)
            from .kernels import _Operation
            if isinstance(kernel, _Operation):
                if len(reads) != kernel.arity or len(writes) != 1:
                    raise ValueError('Kernel operand count does not match its specifications')
                check(self.native.algebra_bind(self.handle, kernel.op, reads[0].view,
                    reads[1].view if len(reads) == 2 else View(), writes[0].view,
                    kernel.alpha, kernel.beta))
                continue
            arrays = tuple(r.array.view() for r in reads) + tuple(r.array for r in writes)
            for array in arrays[:len(reads)]:
                array.flags.writeable = False

            # design/algorithm-sources.md#indexed-library-functions
            def submit(binding, complete, context, arrays=arrays):
                try:
                    kernel(*arrays)
                except BaseException as error:
                    self.errors.append(error)
                    complete(context, errno.EIO)
                else:
                    complete(context, 0)
            self._bind_native(submit, reads, writes)

    # design/algorithm-sources.md#indexed-library-functions
    def copy(self, source, destination, *, queue=0):
        src, sender = source
        dst, receiver = destination
        if isinstance(src, Tensor) and isinstance(dst, Tensor):
            if src.shape != dst.shape or src.block_shape != dst.block_shape:
                raise ValueError('Transfers must share an indexed partition')
            for coordinate in src.blocks:
                self.copy(src[coordinate].on(sender), dst[coordinate].on(receiver), queue=queue)
            return
        src.whole()
        dst.whole()
        if src.program is not self or dst.program is not self:
            raise ValueError('References belong to another program')
        if src.shape != dst.shape or src.array.strides != dst.array.strides:
            raise ValueError('Transfer views must have identical layouts')
        check(self.native.algebra_copy(self.handle,
            Endpoint(src.view.tensor, sender, src.view.extent, 1),
            Endpoint(dst.view.tensor, receiver, dst.view.extent, 1), 1, queue))

    # design/algorithm-sources.md#indexed-library-functions
    def export(self, ref):
        return Result(ref)

    @contextmanager
    # design/algorithm-sources.md#indexed-library-functions
    def write(self, ref):
        ref.whole()
        if not self.native.tensor_issue(ref.view.tensor, ref.view.extent):
            raise BlockingIOError(errno.EAGAIN, 'Producer region still has readers')
        yield ref.array
        self.native.tensor_complete(ref.view.tensor, ref.view.extent)

    # design/algorithm-sources.md#indexed-library-functions
    def constant(self, ref, value):
        ref.whole()
        ref.array[...] = value
        check(self.native.tensor_constant(ref.view.tensor, ref.view.extent))

    # design/algorithm-sources.md#indexed-library-functions
    def realize(self):
        check(self.native.algebra_realize(self.handle))
        return self

    # design/algorithm-sources.md#indexed-library-functions
    def scan(self):
        self.native.algebra_scan(self.handle)
        if self.errors:
            raise self.errors.pop(0)
        check(self.report.code)

    @property
    # design/algorithm-sources.md#indexed-library-functions
    def report(self):
        return self.native.algebra_report(self.handle)

    # design/algorithm-sources.md#indexed-library-functions
    def close(self):
        if self.handle:
            report = self.report
            if report.submitted != report.completed:
                raise BlockingIOError(errno.EBUSY, 'Numerical submissions are still running')
            self.native.algebra_destroy(self.handle)
            self.handle = None
            self.callbacks.clear()
            check(self.native.detach(self.context))

    # design/algorithm-sources.md#indexed-library-functions
    def __enter__(self):
        return self

    # design/algorithm-sources.md#indexed-library-functions
    def __exit__(self, *error):
        self.close()
