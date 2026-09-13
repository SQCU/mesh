import ctypes as C
import errno
import itertools
import os
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from ._native import Native, Shape, View, Endpoint, Submission, Completion

__all__ = ['Program', 'Tensor', 'Ref', 'BlockSpec', 'Result', 'Submission', 'Completion']
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

    # design/algorithm-sources.md#indexed-library-functions
    def elementwise(self, operation, other=None, *, alpha=1, beta=0, dtype=None):
        if other is not None and (other.program is not self.program or other.grid != self.grid or other.shape != self.shape or other.block_shape != self.block_shape):
            raise ValueError('Elementwise operands must share an indexed partition')
        result = self.program.tensor(self.shape, self.block_shape, dtype or self.dtype)
        for coordinate, ref in self.blocks.items():
            self.program.bind(operation, ref, result[coordinate],
                other[coordinate] if other is not None else None, alpha=alpha, beta=beta)
        return result

    # design/algorithm-sources.md#indexed-library-functions
    def __add__(self, other):
        return self.elementwise('add', other, beta=1) if isinstance(other, Tensor) else self.elementwise('affine', beta=other)

    __radd__ = __add__

    # design/algorithm-sources.md#indexed-library-functions
    def __mul__(self, other):
        return self.elementwise('multiply', other) if isinstance(other, Tensor) else self.elementwise('affine', alpha=other)

    __rmul__ = __mul__

    # design/algorithm-sources.md#indexed-library-functions
    def __matmul__(self, other):
        if other.program is not self.program or self.shape[1] != other.shape[0] or self.block_shape[1] != other.block_shape[0]:
            raise ValueError('Contraction operands must share the contracted partition')
        result = self.program.tensor((self.shape[0], other.shape[1]),
            (self.block_shape[0], other.block_shape[1]), np.float32)
        result.contributions = {}
        for i, j in result.blocks:
            result.contributions[i, j] = self.program.contract(
                (self[i, q] for q in range(self.grid[1])),
                (other[q, j] for q in range(other.grid[0])), result[i, j])
        return result

    # design/algorithm-sources.md#indexed-library-functions
    def astype(self, dtype):
        return self.elementwise('affine', dtype=dtype)

    # design/algorithm-sources.md#indexed-library-functions
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__' or kwargs:
            return NotImplemented
        unary = {np.tanh: 'tanh', np.exp: 'exp'}
        if ufunc in unary and len(inputs) == 1:
            return self.elementwise(unary[ufunc])
        if ufunc is np.add:
            return self + (inputs[1] if inputs[0] is self else inputs[0])
        if ufunc is np.multiply:
            return self * (inputs[1] if inputs[0] is self else inputs[0])
        if ufunc is np.matmul:
            return inputs[0] @ inputs[1]
        return NotImplemented

    # design/algorithm-sources.md#indexed-library-functions
    def sum(self, axis=1):
        if axis == 0:
            return self.T.sum(1).T
        if axis != 1:
            raise ValueError('Use axis 0 or 1 for a tensor reduction')
        result = self.program.tensor((self.shape[0], 1), (self.block_shape[0], 1), np.float32)
        result.contributions = {}
        for i in range(self.grid[0]):
            terms = []
            for q in range(self.grid[1]):
                term = self.program.tensor(result[i, 0].shape, dtype=np.float32)[0, 0]
                self.program.bind('sum', self[i, q], term)
                terms.append(term)
            result.contributions[i, 0] = tuple(terms)
            self.program.reduce_sum(terms, result[i, 0])
        return result


@dataclass(frozen=True)
class BlockSpec:
    tensor: Tensor
    index_map: object

    # design/algorithm-sources.md#indexed-library-functions
    def resolve(self, coordinate):
        return self.tensor[tuple(self.index_map(*coordinate))]


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

    # design/algorithm-sources.md#indexed-library-functions
    def bind_native(self, submission, inputs, outputs, binding=None):
        inputs, outputs = tuple(inputs), tuple(outputs)
        if any(r.program is not self for r in inputs + outputs):
            raise ValueError('References belong to another program')
        submit = submission if isinstance(submission, Submission) else Submission(submission)
        check(self.native.algebra_function(self.handle,
            (View * len(inputs))(*(r.view for r in inputs)), len(inputs),
            (View * len(outputs))(*(r.view for r in outputs)), len(outputs), submit, binding))
        self.callbacks.append((submit, inputs, outputs, binding))

    # design/algorithm-sources.md#indexed-library-functions
    def call(self, kernel, *, grid, inputs=(), outputs=()):
        for coordinate in itertools.product(*(range(n) for n in grid)):
            reads = tuple(spec.resolve(coordinate) for spec in inputs)
            writes = tuple(spec.resolve(coordinate) for spec in outputs)
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
            self.bind_native(submit, reads, writes)

    # design/algorithm-sources.md#indexed-library-functions
    def call_native(self, prepare, *, grid, inputs=(), outputs=()):
        for coordinate in itertools.product(*(range(n) for n in grid)):
            reads = tuple(spec.resolve(coordinate) for spec in inputs)
            writes = tuple(spec.resolve(coordinate) for spec in outputs)
            submission, binding = prepare(coordinate, reads, writes)
            self.bind_native(submission, reads, writes, binding)

    # design/algorithm-sources.md#indexed-library-functions
    def bind(self, operation, a, out, b=None, *, alpha=1, beta=None):
        beta = (1 if operation == 'add' else 0) if beta is None else beta
        ops = ('affine', 'add', 'multiply', 'tanh', 'exp', 'sum', 'contract', 'rsqrt')
        check(self.native.algebra_bind(self.handle, ops.index(operation), a.view,
              b.view if b else View(), out.view, alpha, beta))
        return out

    # design/algorithm-sources.md#indexed-library-functions
    def contract(self, left, right, out, *, alpha=1):
        left, right = tuple(left), tuple(right)
        if len(left) != len(right) or not left:
            raise ValueError('Contraction requires paired nonempty partitions')
        handle = self.native.algebra_contract(self.handle,
            (View * len(left))(*(r.view for r in left)),
            (View * len(right))(*(r.view for r in right)), len(left), out.view, alpha)
        if not handle:
            check(C.get_errno() or errno.EINVAL)
        return tuple(Ref(self, self.native.tensor_view(handle, i), np.float32) for i in range(len(left)))

    # design/algorithm-sources.md#indexed-library-functions
    def reduce_sum(self, refs, out):
        level = tuple(refs)
        if not level:
            raise ValueError('Reduction requires at least one region')
        while len(level) > 1:
            following = []
            for i in range(0, len(level), 2):
                if i + 1 == len(level):
                    following.append(level[i])
                else:
                    target = out if len(level) == 2 else self.tensor(out.shape, dtype=out.dtype)[0, 0]
                    self.bind('add', level[i], target, level[i + 1], beta=1)
                    following.append(target)
            level = tuple(following)
        if level[0] is not out:
            self.bind('affine', level[0], out)
        return out

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
