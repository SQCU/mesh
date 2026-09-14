import ctypes as C
import errno
import itertools
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np

from ._native import Native, Shape, View, Endpoint, Submission, MetalDispatch, MetalConstant

__all__ = ['Program', 'Tensor', 'Ref', 'BlockSpec', 'ShapeDtypeStruct', 'Result']
_PROGRAMS = set()
_DTYPES = tuple(map(np.dtype, ('float16', 'float32', 'int32', 'uint32', 'int64', 'uint64', 'uint8', 'bool')))


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
        if min(row, column, rows, columns) < 0 or row + rows > self.shape[0] or column + columns > self.shape[1]:
            raise ValueError('Slice is outside the reference')
        if rows == 0 or columns == 0:
            return self.program.tensor((rows, columns), dtype=self.dtype)
        view = self.program.native.view_slice(self.view, row, column, rows, columns)
        if not view.tensor:
            raise ValueError('Slice is outside the reference')
        return Ref(self.program, view, self.dtype)

    # design/algorithm-sources.md#indexed-library-functions
    def broadcast(self, rows, columns):
        if min(rows, columns) < 0 or any(source != target and source != 1 for source, target in zip(self.shape, (rows, columns))):
            raise ValueError('Incompatible broadcast shape')
        if rows == 0 or columns == 0:
            return self.program.tensor((rows, columns), dtype=self.dtype)
        view = self.program.native.view_broadcast(self.view, rows, columns)
        if not view.tensor:
            raise ValueError('Incompatible broadcast shape')
        return Ref(self.program, view, self.dtype)

    @property
    # design/algorithm-sources.md#xonotic-frame-migration
    def writable(self):
        self.whole()
        return bool(self.program.native.tensor_writable(self.view.tensor, self.view.extent))

    @property
    # design/algorithm-sources.md#streaming-overlap-measurement
    def present(self):
        self.whole()
        return bool(self.program.native.tensor_present(self.view.tensor, self.view.extent))

    # design/algorithm-sources.md#indexed-library-functions
    def on(self, peer):
        return (self, peer)


class Tensor:
    # design/algorithm-sources.md#indexed-range-generation
    def __init__(self, program, shape, block_shape, dtype, transferable):
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
        self.handle = program.native.tensor_create(program.handle, (Shape * len(shapes))(*shapes), len(shapes), transferable) if shapes else None
        if shapes and not self.handle:
            check(C.get_errno() or errno.ENOMEM)
        self.blocks = {coord: Ref(program, program.native.tensor_view(self.handle, i), self.dtype)
                       for i, coord in enumerate(coordinates)}

    # design/algorithm-sources.md#pallas-call-ergonomics
    def region(self, row, column, rows, columns):
        if min(row, column, rows, columns) < 0 or row + rows > self.shape[0] or column + columns > self.shape[1]:
            raise ValueError('Region is outside the tensor')
        if rows == 0 or columns == 0:
            return self.program.tensor((rows, columns), dtype=self.dtype)
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
    def tensor(self, shape, block_shape=None, dtype=np.float32, transferable=True):
        shape = tuple(shape)
        return Tensor(self, shape, tuple(max(1, size) for size in shape) if block_shape is None else block_shape, dtype, transferable)

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
            if len(operands) != len(inputs):
                raise ValueError('Each input requires one BlockSpec')
            outputs = tuple(self.tensor(shape.shape, block_shape=spec.block_shape,
                dtype=shape.dtype) for shape, spec in zip(shapes, specs))
            if peer is None or peer == self.node:
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
        from .kernels import _Operation, Metal, _ExpressionKernel
        grid = tuple(grid)
        if 0 in grid or outputs and all(not spec._tensor.blocks for spec in outputs):
            return
        if isinstance(kernel, _ExpressionKernel):
            kernel.bind_grid(self, grid, inputs, outputs)
            return
        for coordinate in itertools.product(*(range(n) for n in grid)):
            reads = tuple(spec.resolve(coordinate) for spec in inputs)
            writes = tuple(spec.resolve(coordinate) for spec in outputs)
            if isinstance(kernel, Metal):
                dispatches = (MetalDispatch * len(kernel.dispatches))(*(
                    MetalDispatch(d.name.encode(), (C.c_size_t * 3)(*d.grid),
                        (C.c_size_t * 3)(*d.group), d.argument_buffer, d.argument_offset) for d in kernel.dispatches))
                buffers = tuple(C.create_string_buffer(bytes(value)) for value in kernel.constants)
                constants = (MetalConstant * len(buffers))(*(
                    MetalConstant(C.cast(value, C.c_void_p), len(value) - 1) for value in buffers))
                check(self.native.algebra_metal(self.handle, kernel.source.encode(),
                    dispatches, len(dispatches), constants, len(constants),
                    (View * len(reads))(*(r.view for r in reads)), len(reads),
                    (View * len(writes))(*(r.view for r in writes)), len(writes)))
                continue
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
            if src.shape != dst.shape or src.blocks and src.block_shape != dst.block_shape:
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
        self._constant_extents.add((ref.view.tensor, ref.view.extent))

    # design/algorithm-sources.md#indexed-library-functions
    def realize(self):
        check(self.native.algebra_realize(self.handle))
        return self

    @property
    # design/algorithm-sources.md#region-expression-fusion
    def trace(self):
        result = []
        for index in range(self.native.algebra_trace_count(self.handle)):
            event = self.native.algebra_trace(self.handle, index)
            item = {name: getattr(event, name) for name, _ in event._fields_}
            # design/algorithm-sources.md#function-cost-profiles
            profile = self.native.algebra_profile(self.handle, index)
            backends = ('external', 'cpu_sgemm', 'cpu_neon_contract', 'cpu_builtin', 'cpu_compiled',
                        'metal_compiled', 'metal_mps', 'metal_builtin', 'coreml', 'selected_mixed')
            item['profile'] = dict(successful=profile.successful, failed=profile.failed, backend=backends[profile.backend])
            for domain in ('dispatch', 'execution', 'gpu'):
                count = profile.gpu_samples if domain == 'gpu' else profile.successful
                item['profile'][domain + '_ns'] = dict(count=count,
                    mean=getattr(profile, domain + '_mean_ns') if count else None,
                    sample_variance=getattr(profile, domain + '_m2_ns2') / (count-1) if count > 1 else None)
            item['plans'] = []
            for plan_index in range(self.native.algebra_plan_count(self.handle, index)):
                plan = self.native.algebra_plan(self.handle, index, plan_index)
                descriptor = dict(backend=backends[plan.backend],
                    operation=('affine', 'add', 'multiply', 'tanh', 'exp', 'sum', 'contract', 'rsqrt', 'swish')[plan.operation],
                    first=plan.first, count=plan.count, alpha=plan.alpha, beta=plan.beta, rectangles=plan.rectangles)
                for name in ('left', 'right', 'output'):
                    view = getattr(plan, name)
                    descriptor[name] = {field: getattr(view, field) for field, _ in view._fields_}
                    descriptor[name]['dtype'] = ('float16', 'float32', 'int32', 'uint32', 'int64', 'uint64', 'uint8', 'bool')[getattr(plan, name + '_scalar')]
                item['plans'].append(descriptor)
            active = self.native.algebra_trace_active(self.handle, index)
            if active.function != 0xffffffffffffffff:
                item['active'] = {name: getattr(active, name) for name, _ in active._fields_}
                item['active']['count_regions'] = []
                for map_index in range(active.count_maps):
                    region = self.native.algebra_trace_active_count(self.handle, index, map_index)
                    members = []
                    for row in range(region.count):
                        reader = self.native.algebra_trace_active_reader(self.handle, index, map_index, row)
                        if reader.member != 0xffffffff:
                            members.append({name: getattr(reader, name) for name, _ in reader._fields_})
                    item['active']['count_regions'].append(dict(first=region.first, count=region.count, reader_groups=members))
            for label, count, read in (('inputs', event.input_maps, self.native.algebra_trace_input),
                                       ('outputs', event.output_maps, self.native.algebra_trace_output)):
                regions = (read(self.handle, index, i) for i in range(count))
                item[label] = tuple(dict(first=region.first, count=region.count) for region in regions)
            indexed = (self.native.algebra_trace_indexed(self.handle, index, i)
                for i in range(self.native.algebra_trace_indexed_count(self.handle, index)))
            item['indexed'] = tuple({name: getattr(entry, name) for name, _ in entry._fields_}
                for entry in indexed)
            item['reader_groups'] = []
            for label, regions, read in (('inputs', item['inputs'], self.native.algebra_trace_input_reader),
                                         ('indexed', item['indexed'], self.native.algebra_trace_indexed_reader)):
                for map_index, region in enumerate(regions):
                    for row in range(region['count']):
                        reader = read(self.handle, index, map_index, row)
                        if reader.member != 0xffffffff:
                            item['reader_groups'].append(dict(binding=label, map=map_index,
                                **{name: getattr(reader, name) for name, _ in reader._fields_}))
            result.append(item)
        return tuple(result)

    @property
    # design/algorithm-sources.md#compiled-specialization-identities
    def code_trace(self):
        import json
        bindings, sources = [], {}
        backends = ('external', 'cpu_sgemm', 'cpu_neon_contract', 'cpu_builtin', 'cpu_compiled',
                    'metal_compiled', 'metal_mps', 'metal_builtin', 'coreml', 'selected_mixed')
        for index in range(self.native.algebra_trace_count(self.handle)):
            encoded = self.native.algebra_specialization(self.handle, index)
            if encoded is None:
                continue
            binding = dict(json.loads(encoded), function=index)
            binding['backend'] = backends[binding['backend']]
            for view in (*binding['inputs'], *binding['outputs']):
                view['dtype'] = str(_DTYPES[view.pop('scalar')])
            for language, name in enumerate(('cpu', 'metal')):
                identity = binding['sources'].get(name)
                if identity is None:
                    continue
                if identity not in sources:
                    sources[identity] = dict(text=self.native.algebra_source_text(self.handle, index, language).decode(), languages=[])
                if name not in sources[identity]['languages']:
                    sources[identity]['languages'].append(name)
            bindings.append(binding)
        return dict(bindings=bindings, sources=sources)

    @property
    # design/algorithm-sources.md#shared-sparse-routing-lowering
    def route_trace(self):
        result = []
        for index in range(self.native.algebra_trace_route_count(self.handle)):
            event = self.native.algebra_trace_route(self.handle, index)
            item = {name: getattr(event, name) for name, _ in event._fields_}
            producer = self.native.algebra_trace_route_producer(self.handle, index)
            if producer.function != 0xffffffffffffffff:
                item['producer'] = {name: getattr(producer, name) for name, _ in producer._fields_}
            item['reader_groups'] = []
            for row in range(event.count):
                reader = self.native.algebra_trace_route_reader(self.handle, index, row)
                if reader.member != 0xffffffff:
                    item['reader_groups'].append({name: getattr(reader, name) for name, _ in reader._fields_})
            result.append(item)
        return tuple(result)

    @property
    # design/algorithm-sources.md#publication-work-lists
    def transfer_trace(self):
        result = []
        for index in range(self.native.transfer_trace_count(self.context)):
            event = self.native.transfer_trace(self.context, index)
            item = {name: getattr(event, name) for name, _ in event._fields_ if name != 'transfer'}
            item['transfer'] = {name: getattr(event.transfer, name) for name, _ in event.transfer._fields_}
            result.append(item)
        return tuple(result)

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
            self.callbacks.clear()
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
