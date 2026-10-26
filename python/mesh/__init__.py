"""Partial tensors as AOT graph declarations.

Every rank records the same block operations. ``realize`` resolves ownership,
dependencies, storage and crossings, and compiles numerical spans with the common
FX frontend. ``run`` is the resulting fixed call graph. Input writes do not start
workers, and results do not hold reader tokens or issue acknowledgments.

The package also exposes the bridge module in rdma/mesh.py.
"""
import importlib.util
import sys
import types
from pathlib import Path

_RDMA = Path(__file__).resolve().parents[2] / 'rdma'
_SUBSTRATE, _MISSING = None, None
try:
    _spec = importlib.util.spec_from_file_location('mesh._substrate', _RDMA / 'mesh.py')
    _SUBSTRATE = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _SUBSTRATE
    _spec.loader.exec_module(_SUBSTRATE)
    globals().update({name: value for name, value in vars(_SUBSTRATE).items()
                      if not name.startswith('_') and (name == 'lib' or not isinstance(value, types.ModuleType))})
except (ImportError, OSError) as _missing:
    sys.modules.pop('mesh._substrate', None)
    _SUBSTRATE, _MISSING = None, _missing

import heapq  # noqa: E402
import itertools  # noqa: E402
from collections import defaultdict  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from . import _native  # noqa: E402

__all__ = ['Program', 'Tensor', 'Ref', 'Partial', 'BlockSpec', 'ShapeDtypeStruct', 'Result']


# design/algorithm-sources.md#meshobserve
def __getattr__(name):
    if _SUBSTRATE is None and _MISSING is not None:
        raise ImportError(f'mesh.{name}: rdma/mesh.py did not load: {_MISSING}') from _MISSING
    raise AttributeError(f'module mesh has no attribute {name}')


@dataclass(frozen=True)
class Partial:
    required: frozenset
    terms: frozenset

    @classmethod
    # design/algorithm-sources.md#kernelsdot
    def merge(cls, refs):
        values = tuple(ref.partial for ref in refs if ref.partial is not None)
        if not values:
            return None
        required = frozenset.union(*(value.required for value in values))
        terms = frozenset.union(*(value.terms for value in values))
        if sum(len(value.terms) for value in values) != len(terms):
            raise ValueError('A partial contribution occurs twice in the same reduction')
        return None if required == terms else cls(required, terms)


class _Backing:
    """One allocation of a program, by its record index: its numel and scalar; its storage made at realize where a
    member's task touches it, else at its first host write or constant; in the node's bridge's region where a crossing
    of this member reads or writes it."""

    def __init__(self, program, numel, scalar, transferable):
        self.program, self.numel, self.scalar, self.transferable = program, numel, scalar, transferable
        self.index = len(program._backings)
        self.storage, self.in_region = None, False
        program._backings.append(self)

    # design/algorithm-sources.md#programtensor
    def flat(self):
        if self.storage is None:
            program = self.program
            self.storage = _native.allocate(self.numel, self.scalar, program.device,
                                            self.in_region and self.transferable and program.size > 1)
        return self.storage


class Ref:
    """A section's view: offset, rows, columns and strides into its backing; `origin` its corner in the logical
    tensor it was cut from; `partial` its pending contribution set."""
    partial = None

    # design/algorithm-sources.md#program
    def __init__(self, program, backing, offset, rows, columns, row_stride, column_stride, origin=(0, 0)):
        self.program, self.backing = program, backing
        self.offset, self.rows, self.columns = offset, rows, columns
        self.row_stride, self.column_stride, self.origin = row_stride, column_stride, origin
        self.shape, self.dtype = (rows, columns), backing.scalar

    @property
    # design/algorithm-sources.md#programtensor
    def key(self):
        return (self.backing.index, self.offset, self.rows, self.columns, self.row_stride, self.column_stride)

    @property
    # design/algorithm-sources.md#programtensor
    def tensor(self):
        return torch.as_strided(self.backing.flat(), self.shape, (self.row_stride, self.column_stride), self.offset)

    @property
    # design/algorithm-sources.md#programtensor
    def array(self):
        value = self.tensor
        return value.numpy() if value.device.type == 'cpu' else value.cpu().numpy()

    # design/algorithm-sources.md#program
    def whole(self):
        if (self.offset or self.rows * self.columns != self.backing.numel
                or not ((self.column_stride == 1 and self.row_stride == self.columns)
                        or (self.row_stride == 1 and self.column_stride == self.rows))):
            raise ValueError('Publication and transfer require a whole block; allocate smaller blocks for finer progress')
        return self

    # design/algorithm-sources.md#program
    def _derived(self, offset, rows, columns, row_stride, column_stride, origin):
        result = Ref(self.program, self.backing, offset, rows, columns, row_stride, column_stride, origin)
        result.partial = self.partial
        return result

    @property
    # design/algorithm-sources.md#program
    def T(self):
        return self._derived(self.offset, self.columns, self.rows, self.column_stride, self.row_stride, self.origin[::-1])

    # design/algorithm-sources.md#program
    def slice(self, row, column, rows, columns):
        if min(row, column, rows, columns) < 0 or row + rows > self.rows or column + columns > self.columns:
            raise ValueError('Slice is outside the reference')
        if rows == 0 or columns == 0:
            return self.program.tensor((rows, columns), dtype=self.dtype)
        if row == column == 0 and (rows, columns) == self.shape:
            return self
        return self._derived(self.offset + row * self.row_stride + column * self.column_stride, rows, columns,
                             self.row_stride, self.column_stride, (self.origin[0] + row, self.origin[1] + column))

    # design/algorithm-sources.md#program
    def broadcast(self, rows, columns):
        if min(rows, columns) < 0 or any(source != target and source != 1
                                         for source, target in zip(self.shape, (rows, columns))):
            raise ValueError('Incompatible broadcast shape')
        if rows == 0 or columns == 0:
            return self.program.tensor((rows, columns), dtype=self.dtype)
        return self._derived(self.offset, rows, columns, 0 if self.rows == 1 else self.row_stride,
                             0 if self.columns == 1 else self.column_stride, self.origin)

    # design/algorithm-sources.md#program
    def on(self, peer):
        return (self, peer)

    # design/algorithm-sources.md#programcopy
    def _overlaps(self, other):
        a, b = self, other
        if a.backing is not b.backing or not a.rows or not a.columns or not b.rows or not b.columns:
            return False
        last = lambda r: r.offset + (r.rows - 1) * r.row_stride + (r.columns - 1) * r.column_stride  # noqa: E731
        if last(a) < b.offset or last(b) < a.offset:
            return False
        if a.row_stride < a.column_stride:
            a = a.T
        if b.row_stride < b.column_stride:
            b = b.T
        widths = (a.columns - 1) * a.column_stride + 1, (b.columns - 1) * b.column_stride + 1
        i = j = 0
        while i < a.rows and j < b.rows:
            left, right = a.offset + i * a.row_stride, b.offset + j * b.row_stride
            if left < right + widths[1] and right < left + widths[0]:
                return True
            if left + widths[0] <= right:
                i += 1
            else:
                j += 1
        return False


class Tensor:
    # design/algorithm-sources.md#collectivereduce_scatter
    def _with_blocks(self, blocks):
        result = object.__new__(Tensor)
        result.__dict__ = self.__dict__ | {'blocks': blocks, '_span': None}
        return result

    # design/algorithm-sources.md#programtensor
    def __init__(self, program, shape, block_shape, dtype, transferable, contiguous=False):
        import operator
        self.program = program
        self.shape, self.block_shape = tuple(map(operator.index, shape)), tuple(map(operator.index, block_shape))
        self.dtype = _native.dtype(dtype)
        if len(self.shape) != 2 or len(self.block_shape) != 2 or min(self.shape) < 0 or min(self.block_shape) <= 0:
            raise ValueError('Tensor shapes require two nonnegative dimensions and positive block dimensions')
        self.grid = tuple((s + b - 1) // b for s, b in zip(self.shape, self.block_shape))
        coordinates = tuple(itertools.product(*(range(n) for n in self.grid))) if all(self.grid) else ()
        self._span = None
        if contiguous and coordinates:
            span = _Backing(program, self.shape[0] * self.shape[1], self.dtype, transferable)
            self._span = Ref(program, span, 0, self.shape[0], self.shape[1], self.shape[1], 1)
        self.blocks = {}
        for coordinate in coordinates:
            row, column = (i * b for i, b in zip(coordinate, self.block_shape))
            rows, columns = (min(b, s - i * b) for s, b, i in zip(self.shape, self.block_shape, coordinate))
            if self._span is not None:
                ref = self._span.slice(row, column, rows, columns)
            else:
                ref = Ref(program, _Backing(program, rows * columns, self.dtype, transferable), 0, rows, columns,
                          columns, 1, (row, column))
            self.blocks[coordinate] = ref
            program._blocks[ref.backing.index].append(ref)

    # design/algorithm-sources.md#programkernel_call
    def region(self, row, column, rows, columns):
        if min(row, column, rows, columns) < 0 or row + rows > self.shape[0] or column + columns > self.shape[1]:
            raise ValueError('Region is outside the tensor')
        if rows == 0 or columns == 0:
            return self.program.tensor((rows, columns), dtype=self.dtype)
        if getattr(self, '_span', None) is not None:
            return self._span.slice(row, column, rows, columns)
        i, j = row // self.block_shape[0], column // self.block_shape[1]
        ref = self[i, j]
        r, c = row % self.block_shape[0], column % self.block_shape[1]
        if r + rows > ref.shape[0] or c + columns > ref.shape[1]:
            raise ValueError('Region crosses backing blocks; realize storage with blocks containing the requested region')
        return ref.slice(r, c, rows, columns)

    # design/algorithm-sources.md#program
    def __getitem__(self, coordinate):
        return self.blocks[coordinate]

    # design/algorithm-sources.md#program
    def on(self, peer):
        return (self, peer)

    # design/algorithm-sources.md#programtensor
    def dense(self):
        if getattr(self, '_span', None) is not None:
            return self._span.tensor
        if len(self.blocks) == 1:
            return next(iter(self.blocks.values())).tensor
        return torch.cat([torch.cat([self.blocks[i, j].tensor for j in range(self.grid[1])], dim=1)
                          for i in range(self.grid[0])], dim=0)

    @property
    # design/algorithm-sources.md#program
    def T(self):
        result = object.__new__(Tensor)
        result.program, result.dtype = self.program, self.dtype
        result.shape, result.block_shape, result.grid = self.shape[::-1], self.block_shape[::-1], self.grid[::-1]
        result._span = self._span.T if getattr(self, '_span', None) is not None else None
        result.blocks = {(j, i): ref.T for (i, j), ref in self.blocks.items()}
        return result

    # design/algorithm-sources.md#nnembedding
    def broadcast_to(self, shape):
        shape = tuple(shape)
        if len(shape) != 2 or min(shape) < 0 or any(source != target and source != 1
                                                    for source, target in zip(self.shape, shape)):
            raise ValueError('Incompatible tensor broadcast shape')
        if 0 in shape:
            return self.program.tensor(shape, tuple(max(1, target) if source == 1 else block
                                                    for source, target, block in zip(self.shape, shape, self.block_shape)), self.dtype)
        result = object.__new__(Tensor)
        result.program, result.dtype = self.program, self.dtype
        result.shape, result.grid = shape, self.grid
        result._span = self._span.broadcast(*shape) if getattr(self, '_span', None) is not None else None
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
    region_map: object = None
    _tensor: object = None

    # design/algorithm-sources.md#programkernel_call
    def _bind(self, tensor):
        return replace(self, _tensor=tensor)

    # design/algorithm-sources.md#programkernel_call
    def resolve(self, coordinate):
        if self._tensor is None:
            raise ValueError('BlockSpec is bound by kernel_call')
        if self.block_shape is None:
            ref = self._tensor
        else:
            index = tuple(self.index_map(*coordinate))
            shape = tuple(self.block_shape)
            if len(shape) != 2 or len(index) != 2 or min(shape) <= 0:
                raise ValueError('BlockSpec requires two positive block dimensions and two indices')
            row, column = (i * b for i, b in zip(index, shape))
            extent = (min(shape[0], self._tensor.shape[0] - row), min(shape[1], self._tensor.shape[1] - column))
            ref = (self._tensor.slice if isinstance(self._tensor, Ref) else self._tensor.region)(row, column, *extent)
        return self.region_map(ref, *coordinate) if self.region_map is not None else ref


def _refs(value):
    return tuple(value.blocks.values()) if isinstance(value, Tensor) else (value,)


@dataclass(eq=False)
class _Task:
    index: int
    kind: str
    owners: tuple
    row: int
    roles: object
    run: object
    queue: int = 0
    crossing: bool = False
    members: tuple = ()
    prepared: object = None
    participants: tuple = ()

    def __lt__(self, other):
        return (self.row, self.index) < (other.row, other.index)


class Result:
    """A declared output view; completion belongs to the whole submitted call."""
    def __init__(self, ref):
        self.ref = ref

    @property
    def tensor(self):
        return self.ref.tensor


class Program:
    # design/algorithm-sources.md#program
    def __init__(self, backend='cpu', *, functions=None, group=None):
        self.device = _native.DEVICES[backend]
        self._functions = dict(functions or {})
        self._world = _native.World(self.device, group)
        self.node = self.rank = self._world.rank
        self.size = self._world.size
        self._backings, self._blocks, self._constants = [], defaultdict(list), set()
        self._replicated = {}
        self._tasks, self._exports = [], []
        self._realized = False
        self.graph, self.run = None, None

    @property
    # design/algorithm-sources.md#independent-native-queues
    def group(self):
        return self._world.group

    # design/algorithm-sources.md#programtensor
    def tensor(self, shape, block_shape=None, dtype=np.float32, transferable=True, *, contiguous=False):
        shape = tuple(shape)
        return Tensor(self, shape, tuple(max(1, size) for size in shape) if block_shape is None else block_shape,
                      dtype, transferable, contiguous)

    # design/algorithm-sources.md#programkernel_call
    def _record(self, kind, owners, row, roles, run, *, queue=0, crossing=False, members=None, participants=None):
        if self._realized:
            raise RuntimeError('A realized program takes no more calls')
        owners = tuple(sorted(set(owners)))
        task = _Task(len(self._tasks), kind, owners, row, roles, run, queue, crossing,
                     tuple(sorted(set(members))) if members is not None else owners,
                     participants=tuple(participants) if participants is not None else owners)
        self._tasks.append(task)
        return task

    # design/algorithm-sources.md#programkernel_call
    def kernel_call(self, kernel, *, out_shape, grid, in_specs, out_specs, peer=None):
        single = isinstance(out_shape, ShapeDtypeStruct)
        shapes = (out_shape,) if single else tuple(out_shape)
        specs = (out_specs,) if single else tuple(out_specs)
        inputs = tuple(in_specs)
        if not shapes or len(shapes) != len(specs):
            raise ValueError('Each output requires one shape and one BlockSpec')

        # design/algorithm-sources.md#programkernel_call
        def configure(*operands):
            if len(operands) != len(inputs):
                raise ValueError('Each input requires one BlockSpec')
            partial = next((ref for operand in operands for ref in _refs(operand) if ref.partial is not None), None)
            if partial is not None and not getattr(kernel, 'combine', False):
                raise TypeError(f'Partial input {partial!r} requires a combine (a kernel declaring combine) or a reduction first')
            outputs = tuple(self.tensor(shape.shape, block_shape=spec.block_shape, dtype=shape.dtype)
                            for shape, spec in zip(shapes, specs))
            self._call(kernel, grid=grid, peer=peer,
                       inputs=tuple(spec._bind(tensor) for spec, tensor in zip(inputs, operands)),
                       outputs=tuple(spec._bind(tensor) for spec, tensor in zip(specs, outputs)))
            return outputs[0] if single else outputs
        return configure

    # design/algorithm-sources.md#programkernel_call
    def _call(self, kernel, *, grid, inputs=(), outputs=(), peer=None):
        owners = tuple(range(self.size)) if peer is None else (peer,)
        for coordinate in itertools.product(*(range(size) for size in grid)):
            sources = tuple(spec.resolve(coordinate) for spec in inputs)
            targets = tuple(spec.resolve(coordinate) for spec in outputs)
            reads = tuple(ref for source in sources for ref in _refs(source))
            defs = tuple(ref for target in targets for ref in _refs(target))
            for target in defs:
                target.partial = Partial.merge(reads)
            self._record('kernel', owners, defs[0].origin[0] if defs else 0, (reads, defs, ()),
                         (kernel, sources, targets, coordinate))

    # design/algorithm-sources.md#program
    def copy(self, source, destination, *, queue=0):
        src, sender = source
        dst, receiver = destination
        if src.program is not self or dst.program is not self:
            raise ValueError('References belong to another program')
        if isinstance(src, Tensor) and isinstance(dst, Tensor):
            if src.shape != dst.shape:
                raise ValueError('Transfer shapes must match')
            if sender == receiver:
                targets = (((0, 0), dst._span),) if getattr(dst, '_span', None) is not None else tuple(dst.blocks.items())
                if any(ref._overlaps(target) for ref in src.blocks.values() for _, target in targets):
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
                            regions.append((ref.slice(lo_r - r, lo_c - c, hi_r - lo_r, hi_c - lo_c), lo_r - row, lo_c - column))
                    self._record('copy', (sender,), target.origin[0], (tuple(r for r, _, _ in regions), (target,), ()),
                                 ('materialize', tuple(regions), target))
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
                    self.copy(src.region(row, column, *shape).on(sender), dst.region(row, column, *shape).on(receiver),
                              queue=queue)
            return
        if sender == receiver:
            if src is not dst and src.key != dst.key:
                self._record('copy', (sender,), dst.origin[0], ((src,), (dst,), ()), ('materialize', ((src, 0, 0),), dst))
        else:
            from .collective import _exchange
            if src.shape != dst.shape:
                raise ValueError('Transfer shapes must match')
            _exchange(self, [[(src, sender, dst, receiver)]], queue)
        dst.partial = src.partial

    send = recv = copy

    # design/algorithm-sources.md#programcopy
    def replicate(self, source, peer):
        tensor, sender = source
        if not isinstance(tensor, Tensor) or tensor.program is not self:
            raise ValueError('Replication requires a tensor belonging to this program')
        if sender == peer:
            return tensor
        result = object.__new__(Tensor)
        result.program, result.dtype, result._span = self, tensor.dtype, None
        result.shape, result.block_shape, result.grid = tensor.shape, tensor.block_shape, tensor.grid
        result.blocks = {}
        for coordinate, ref in tensor.blocks.items():
            key = (sender, peer, ref.backing.index)
            if key not in self._replicated:
                whole = ref.backing
                backing = _Backing(self, whole.numel, whole.scalar, True)
                copy = Ref(self, backing, 0, 1, whole.numel, whole.numel, 1)
                self._blocks[backing.index].append(copy)
                self.copy(Ref(self, whole, 0, 1, whole.numel, whole.numel, 1).on(sender), copy.on(peer))
                self._replicated[key] = backing
            backing = self._replicated[key]
            result.blocks[coordinate] = Ref(self, backing, ref.offset, ref.rows, ref.columns, ref.row_stride,
                                            ref.column_stride, ref.origin)
        return result

    # design/algorithm-sources.md#programexport
    def export(self, ref):
        if ref.program is not self:
            raise ValueError('Reference belongs to another program')
        if self._realized:
            raise RuntimeError('A realized program takes no more calls')
        result = Result(ref)
        self._exports.append(result)
        return result

    # design/algorithm-sources.md#programwrite
    def _block_key(self, ref):
        if not any(ref.key == block.key for block in self._blocks[ref.backing.index]):
            raise ValueError('Publication requires a whole block of a tensor; allocate smaller blocks for finer progress')
        return ref.key

    @contextmanager
    def write(self, ref):
        """The input's existing storage, with no publication or staging buffer.

        The caller fills inputs before submitting the prepared call. Changes to
        inputs between calls follow the enclosing function's page schedule.
        """
        if ref.program is not self:
            raise ValueError('Reference belongs to another program')
        yield ref.tensor

    # design/algorithm-sources.md#program
    def constant(self, ref, value=None):
        key = self._block_key(ref.whole())
        if value is not None:
            value = torch.as_tensor(np.asarray(value)).to(ref.dtype)
            ref.tensor.copy_(value.reshape(ref.shape) if value.numel() == ref.rows * ref.columns else value)
        self._constants.add(key)

    # design/algorithm-sources.md#kernelsdot
    def is_constant(self, ref):
        return any(block.key in self._constants and block._overlaps(ref) for block in self._blocks[ref.backing.index])

    # design/algorithm-sources.md#programkernel_call
    def _roles(self, task, rank):
        if rank not in task.owners:
            return (), (), ()
        return task.roles.get(rank, ((), (), ())) if isinstance(task.roles, dict) else task.roles

    def _graph(self, rank):
        """Resolve overlapping reads and writes once, in declaration order."""
        edges, prior = set(), []
        for task in self._tasks:
            reads, definitions, updates = self._roles(task, rank)
            reads, writes = reads + updates, definitions + updates
            for earlier, used, changed in prior:
                if (any(a._overlaps(b) for a in changed for b in reads + writes)
                        or any(a._overlaps(b) for a in used for b in writes)):
                    edges.add((earlier.index, task.index))
            prior.append((task, reads, writes))
        return edges

    def _numerical(self, tasks):
        """Compile a maximal local numerical span, binding its storage once."""
        from torch_mesh.declarations import Function
        backings = tuple(dict.fromkeys(ref.backing for task in tasks
                                       for refs in self._roles(task, self.rank) for ref in refs))
        stored = tuple(backing.flat() for backing in backings)

        def operations(*operands):
            # This callable is used only by fake tracing. The compiled graph
            # receives the original backings as explicit, fixed arguments.
            for backing, operand in zip(backings, operands):
                backing.storage = operand
            try:
                for task in tasks:
                    if task.kind == 'kernel':
                        kernel, sources, targets, coordinate = task.run
                        self._functions.get(kernel, kernel)(self, sources, targets, coordinate)
                    else:
                        _, regions, target = task.run
                        out = target.tensor
                        for ref, row, column in regions:
                            out[row:row + ref.rows, column:column + ref.columns].copy_(ref.tensor)
                return operands
            finally:
                for backing, value in zip(backings, stored):
                    backing.storage = value

        function = Function(operations, *stored)
        function.declare(self.group)
        if any(function.groups.values()):
            raise ValueError('partial kernels declare communication through block collective operands')
        return function

    def realize(self, repetitions=1):
        """Compile the block graph to numerical spans and bound native calls.

        No payload is executed here. There is no alternate eager executor if
        tracing, compilation or native preparation cannot realize the graph.
        """
        if self._realized:
            return self
        if self.device == 'cpu' and any(task.crossing for task in self._tasks):
            raise NotImplementedError('distributed partial graphs require the prepared Metal transport')
        edges = set().union(*(self._graph(rank) for rank in range(self.size)))
        after, incoming = defaultdict(set), defaultdict(int)
        for a, b in edges:
            after[a].add(b)
            incoming[b] += 1
        heap = [task for task in self._tasks if not incoming[task.index]]
        heapq.heapify(heap)
        order = []
        while heap:
            task = heapq.heappop(heap)
            order.append(task)
            for b in after[task.index]:
                incoming[b] -= 1
                if not incoming[b]:
                    heapq.heappush(heap, self._tasks[b])
        if len(order) != len(self._tasks):
            raise ValueError('The calls depend on one another in a cycle')
        local = [task for task in order if self.rank in task.owners]
        for task in local:
            if task.crossing:
                for ref in itertools.chain(*self._roles(task, self.rank)):
                    ref.backing.in_region = True
        for task in local:
            for ref in itertools.chain(*self._roles(task, self.rank)):
                ref.backing.flat()
        for result in self._exports:
            result.ref.backing.flat()
        self._world.prepare((task.queue, task.members) for task in order if task.crossing)
        for task in local:
            if task.crossing:
                task.prepared = task.run[0](self, task)
        self._world.declare(order)
        from torch_mesh.declarations import execution_sequence
        steps, numerical = [], []
        for task in local:
            if task.crossing:
                if numerical:
                    steps.append(self._numerical(tuple(numerical)))
                    numerical.clear()
                steps.append(task.prepared[0])
            else:
                numerical.append(task)
        if numerical:
            steps.append(self._numerical(tuple(numerical)))
        self.graph = execution_sequence(steps)
        self.graph.realize(repetitions)
        self.completion = self.graph.completion
        self.run = self.graph.forward
        self._realized = True
        return self

    start = realize

    def close(self):
        self._world.close()

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()
