"""The partial-tensor library's Python surface (design/deliverables.md section 2; design/algorithm-sources.md#program):
higher-order functions over partial tensors, the partial tensors' collectives, zero-copy and asynchronous once the
call graph is known.  The same contract as swift/Mesh.swift, spelled as Pallas spells it: Program.kernel_call over a
grid with BlockSpecs, Refs to sections, Partial contribution sets that refuse a nonlinear function before their sum
completes, the collective verbs (mesh.collective) and the nn compositions (mesh.nn) of K-split partials published
section by section.

A program is SPMD: every member records the same calls; `peer=` places a kernel, a copy's two ends and a collective's
members place its crossings.  Recording builds the graph; realize() (start(), Swift's name) prepares it; then the
host writes its inputs (write), each publication firing what it completes (a pending count reaching zero, the old
executor's rule, design/deliverables.md L3), and reads its results (export, Result.ready, consume).  Kernels run
on torch on one compute thread; crossings and collectives are libnccl-mesh's (mesh._native), issued in one order every
member computes from the graph (each queue its own communicator), so their positions pair (rdma/NCCL.md "Rules a
program keeps").  Every wait is bounded by the program's `timeout`; a failure is a value (Program.report, MeshError).

The package is `mesh`, the import name rdma/mesh.py also takes: importing it loads that module from this checkout's
rdma/ and carries its names (ffi, lib, lease, current, ...), so torch_mesh and mesh_mpi find what they import."""
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
                      if not name.startswith('_') and not isinstance(value, types.ModuleType)})
except (ImportError, OSError) as _missing:
    sys.modules.pop('mesh._substrate', None)
    _SUBSTRATE, _MISSING = None, _missing

import errno  # noqa: E402
import heapq  # noqa: E402
import itertools  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import dataclass, field, replace  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from . import _native  # noqa: E402

__all__ = ['Program', 'Tensor', 'Ref', 'Partial', 'BlockSpec', 'ShapeDtypeStruct', 'Result', 'MeshError', 'Report']


# design/algorithm-sources.md#meshobserve
def __getattr__(name):
    if _SUBSTRATE is None and _MISSING is not None:
        raise ImportError(f'mesh.{name}: rdma/mesh.py did not load: {_MISSING}') from _MISSING
    raise AttributeError(f'module mesh has no attribute {name}')


# design/algorithm-sources.md#program
def check(code):
    if code:
        raise OSError(code, os.strerror(code))


# design/algorithm-sources.md#mesherror
class MeshError(Exception):
    """swift/Mesh.swift's MeshError as an exception: `kind` busy, link (`peer` the member across it) or function
    (`call` the task's record index); `code` an errno."""

    def __init__(self, kind, code, *, peer=None, call=None, detail=''):
        super().__init__(f'{kind}: {os.strerror(code) if code > 0 else code}'
                         + (f' (peer {peer})' if peer is not None else '') + (f' (call {call})' if call is not None else '')
                         + (f': {detail}' if detail else ''))
        self.kind, self.code, self.peer, self.call = kind, code, peer, call


@dataclass
class Report:
    code: int = 0
    kind: str = ''
    peer: object = None
    call: object = None
    detail: str = ''
    kernels: int = 0
    crossings: int = 0


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
    member's task touches it, else at its first host write or constant (the window's where the program crosses)."""

    def __init__(self, program, numel, scalar, transferable):
        self.program, self.numel, self.scalar, self.transferable = program, numel, scalar, transferable
        self.index = len(program._backings)
        self.storage = None
        program._backings.append(self)

    # design/algorithm-sources.md#programtensor
    def flat(self):
        if self.storage is None:
            program = self.program
            self.storage = _native.allocate(self.numel, self.scalar, program.device,
                                            self.transferable and program.size > 1)
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

    @property
    # design/algorithm-sources.md#programwrite
    def writable(self):
        return self.program._writable(self)

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
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    guards: list = field(default_factory=list)
    guarding: list = field(default_factory=list)
    cyclic: bool = True
    running: bool = False
    fired: bool = False
    done: bool = False
    event: object = None

    # design/algorithm-sources.md#programkernel_call
    def __lt__(self, other):
        return (self.row, self.index) < (other.row, other.index)


class _Edge:
    __slots__ = ('source', 'target', 'tokens', 'sticky', 'works', 'waited')

    def __init__(self, source, target, sticky=False):
        self.source, self.target = source, target
        self.tokens, self.sticky, self.works, self.waited = 0, sticky, (), True


class Result:
    # design/algorithm-sources.md#programexport
    def __init__(self, ref, task):
        self.ref, self._task = ref, task

    @property
    # design/algorithm-sources.md#programexport
    def ready(self):
        program = self.ref.program
        with program._cond:
            program._raise()
            if not self._task.done:
                return False
            event = self._task.event
        return event is None or event.query()

    @property
    # design/algorithm-sources.md#program
    def tensor(self):
        if not self.ready:
            raise BlockingIOError(errno.EAGAIN, 'Result region is not published')
        return self.ref.tensor

    @property
    # design/algorithm-sources.md#program
    def array(self):
        result = self.tensor
        result = result.numpy() if result.device.type == 'cpu' else result.cpu().numpy()
        result.flags.writeable = False
        return result

    # design/algorithm-sources.md#programexport
    def consume(self):
        program = self.ref.program
        with program._cond:
            task = self._task
            if task.done:
                for edge in task.inputs:
                    if not edge.sticky:
                        edge.tokens -= 1
                    program._drained(edge.source)
                task.done, task.event = False, None
                program._touch(task)
                program._cond.notify_all()


class _Order:
    """A queue's crossings in the order every member computed: the first round all of them, each later round those
    that a host write can feed again (a crossing whose inputs are all constants runs once)."""

    def __init__(self, tasks):
        self.first, self.later = list(tasks), [task for task in tasks if task.cyclic]
        self.sequence, self.position = self.first, 0

    # design/algorithm-sources.md#independent-native-queues
    def head(self):
        return self.sequence[self.position] if self.position < len(self.sequence) else None

    # design/algorithm-sources.md#independent-native-queues
    def advance(self):
        self.position += 1
        if self.position == len(self.sequence) and self.later:
            self.sequence, self.position = self.later, 0


class Program:
    # design/algorithm-sources.md#program
    def __init__(self, backend='cpu', *, functions=None, engines=None, group=None, timeout=10.0, trace=False):
        if timeout is None or not timeout > 0:
            raise ValueError('A program bounds every wait: timeout is a positive number of seconds')
        self.device = _native.DEVICES[backend]
        if engines is not None and self.device != 'mps':
            raise ValueError("torch_mesh's engines share an MPS program's contractions; this program is on the CPU")
        self.engines, self.timeout = engines, float(timeout)
        self._functions = dict(functions or {})
        self._world = _native.World(self.device, group)
        self.node = self.rank = self._world.rank
        self.size = self._world.size
        self._backings, self._blocks, self._constants = [], defaultdict(list), set()
        self._replicated, self._operations = {}, {}
        self._tasks, self._exports = [], []
        self._cond = threading.Condition()
        self._threads, self._published, self._sources = [], set(), defaultdict(list)
        self._candidates, self._heap = set(), []
        self.report = Report()
        self.trace = [] if trace else None
        self._realized = self._closed = False

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
    def _record(self, kind, owners, row, roles, run, *, queue=0, crossing=False, members=None):
        if self._realized:
            raise RuntimeError('A realized program takes no more calls')
        owners = tuple(sorted(set(owners)))
        task = _Task(len(self._tasks), kind, owners, row, roles, run, queue, crossing,
                     tuple(sorted(set(members))) if members is not None else owners)
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
            from . import kernels
            if len(operands) != len(inputs):
                raise ValueError('Each input requires one BlockSpec')
            partial = next((ref for operand in operands for ref in _refs(operand) if ref.partial is not None), None)
            if partial is not None and kernel is not kernels.add:
                raise TypeError(f'Partial input {partial!r} requires addition or reduce_scatter before this kernel')
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
            roles = {sender: ((src,), (), ()), receiver: ((), (dst,), ())}
            self._record('copy', (sender, receiver), src.origin[0], roles, (Program._issue_copy, src, sender, dst, receiver),
                         queue=queue, crossing=True, members=range(self.size))
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
        task = _Task(-1 - len(self._exports), 'export', (self.rank,), ref.origin[0], ((ref,), (), ()), ('export', ref))
        result = Result(ref, task)
        self._exports.append(result)
        return result

    # design/algorithm-sources.md#programwrite
    def _block_key(self, ref):
        if not any(ref.key == block.key for block in self._blocks[ref.backing.index]):
            raise ValueError('Publication requires a whole block of a tensor; allocate smaller blocks for finer progress')
        return ref.key

    # design/algorithm-sources.md#programwrite
    def _writable(self, ref):
        key = self._block_key(ref)
        with self._cond:
            self._raise()
            return not self._realized or all(edge.tokens == 0 for edge in self._sources.get(key, ()))

    @contextmanager
    # design/algorithm-sources.md#programwrite
    def write(self, ref):
        if ref.program is not self:
            raise ValueError('Reference belongs to another program')
        key = self._block_key(ref)
        with self._cond:
            self._raise()
            if self._realized and any(edge.tokens for edge in self._sources.get(key, ())):
                raise BlockingIOError(errno.EAGAIN, 'Producer region still has readers')
        if ref.tensor.device.type == 'cpu':
            yield ref.tensor.numpy()
        else:
            staged = np.empty(ref.shape, _native.numpy_dtype(ref.dtype))
            yield staged
            ref.tensor.copy_(torch.from_numpy(staged))
        with self._cond:
            if self._realized:
                for edge in self._sources.get(key, ()):
                    edge.tokens = 1
                    self._touch(edge.target)
                self._cond.notify_all()
            else:
                self._published.add(key)

    # design/algorithm-sources.md#program
    def constant(self, ref, value=None):
        key = self._block_key(ref.whole())
        if value is not None:
            value = torch.as_tensor(np.asarray(value)).to(ref.dtype)
            ref.tensor.copy_(value.reshape(ref.shape) if value.numel() == ref.rows * ref.columns else value)
        self._constants.add(key)

    # design/algorithm-sources.md#kernelsdot
    def _is_constant(self, ref):
        return any(block.key in self._constants and block._overlaps(ref) for block in self._blocks[ref.backing.index])

    # design/algorithm-sources.md#kernelsdot
    def _contract(self, left, right, key=None):
        return _native.contract(self.engines, self._operations, left, right, key)

    # design/algorithm-sources.md#programkernel_call
    def _roles(self, task, rank):
        if rank not in task.owners:
            return (), (), ()
        return task.roles[rank] if isinstance(task.roles, dict) else task.roles

    # design/algorithm-sources.md#programkernel_call
    def _graph(self, rank, tasks):
        """The dependencies on member `rank`: a region's definition, then the collectives that update it in place in
        record order, its readers taking the last; host-written blocks (no definition on `rank`, not constants) before
        their readers; and each writer's guards, the later updaters of what it writes, whose readers it must not
        overwrite."""
        writers = defaultdict(list)
        for task in tasks:
            _, defs, updates = self._roles(task, rank)
            for ref in defs:
                writers[ref.backing.index].append((task, ref, False))
            for ref in updates:
                writers[ref.backing.index].append((task, ref, True))
        for found in writers.values():
            definitions = [w for w in found if not w[2]]
            for (a, x, _), (b, y, _) in itertools.combinations(definitions, 2):
                if a is not b and x._overlaps(y):
                    raise ValueError(f'calls {a.index} and {b.index} both define one region on member {rank}')
        edges, sources, guards = set(), set(), defaultdict(set)
        for task in tasks:
            reads, _, updates = self._roles(task, rank)
            for ref in reads + updates:
                found = [w for w in writers[ref.backing.index] if w[0] is not task and w[1]._overlaps(ref)]
                updating = any(ref is u for u in updates)
                for writer, region, _ in found:
                    if updating and writer.index > task.index:
                        continue
                    if not any(update and u.index > writer.index and other._overlaps(region)
                               and (not updating or u.index < task.index) for u, other, update in found):
                        edges.add((writer.index, task.index))
                for block in self._blocks[ref.backing.index]:
                    if (block.key not in self._constants and block._overlaps(ref)
                            and not any(not update and region._overlaps(block)
                                        for _, region, update in writers[ref.backing.index])):
                        sources.add((block.key, task.index))
        for found in writers.values():
            for writer, region, _ in found:
                for updater, other, update in found:
                    if update and updater.index > writer.index and other._overlaps(region):
                        guards[writer.index].add(updater.index)
        return edges, sources, guards

    # design/algorithm-sources.md#programkernel_call
    def realize(self):
        """Every member's dependencies, from the calls all members recorded: the order of the crossings (a
        topological order of the calls, the earliest row first, then the earliest record), which calls a host write
        feeds again (cyclic), and this member's edges, pending counts and storage; then the threads start."""
        if self._realized:
            return self
        graphs = {rank: self._graph(rank, self._tasks) for rank in range(self.size)}
        edges = set().union(*(graph[0] for graph in graphs.values()))
        fed = {index for graph in graphs.values() for _, index in graph[1]}
        after, before, incoming = defaultdict(set), defaultdict(set), defaultdict(int)
        for a, b in edges:
            after[a].add(b)
            before[b].add(a)
        for b, sources in before.items():
            incoming[b] = len(sources)
        heap = [task for task in self._tasks if not incoming[task.index]]
        heapq.heapify(heap)
        order = []
        while heap:
            task = heapq.heappop(heap)
            order.append(task)
            task.cyclic = task.index in fed or any(self._tasks[a].cyclic for a in before[task.index])
            for b in after[task.index]:
                incoming[b] -= 1
                if not incoming[b]:
                    heapq.heappush(heap, self._tasks[b])
        if len(order) != len(self._tasks):
            raise ValueError('The calls depend on one another in a cycle')
        exports = [result._task for result in self._exports]
        local = self._tasks + exports
        mine, sources, guards = self._graph(self.rank, local)
        named = {task.index: task for task in local}
        for task in exports:
            task.cyclic = True
        for a, b in sorted(mine):
            edge = _Edge(named[a], named[b], sticky=not named[a].cyclic)
            named[a].outputs.append(edge)
            named[b].inputs.append(edge)
        for key, b in sorted(sources):
            edge = _Edge(None, named[b])
            edge.tokens = 1 if key in self._published else 0
            self._sources[key].append(edge)
            named[b].inputs.append(edge)
        for a, updaters in guards.items():
            named[a].guards = [named[u] for u in sorted(updaters)]
            for u in updaters:
                named[u].guarding.append(named[a])
        self._local = [task for task in local if self.rank in task.owners]
        for task in self._local:
            for ref in itertools.chain(*self._roles(task, self.rank)):
                ref.backing.flat()
        self._world.prepare((task.queue, task.members) for task in self._tasks if task.crossing)
        self._compute = [task for task in self._local if not task.crossing]
        queues = defaultdict(list)
        for task in order:
            if task.crossing and self.rank in task.owners:
                queues[task.queue].append(task)
        self._orders = [_Order(queues[queue]) for queue in sorted(queues)]
        self._candidates = set(self._compute)
        from .kernels import _ExpressionKernel
        self._expression = _ExpressionKernel
        self._realized = True
        if self.device == 'mps':
            self._threads = [threading.Thread(target=self._serve, args=(self._orders,), daemon=True)]
        else:
            self._threads = [threading.Thread(target=self._serve, args=((),), daemon=True)]
            self._threads += [threading.Thread(target=self._stream, args=(order,), daemon=True) for order in self._orders]
        for thread in self._threads:
            thread.start()
        return self

    start = realize

    # design/algorithm-sources.md#mesherror
    def _raise(self):
        if self.report.code:
            raise MeshError(self.report.kind, self.report.code, peer=self.report.peer, call=self.report.call,
                            detail=self.report.detail)

    # design/algorithm-sources.md#mesherror
    def _fail(self, task, error):
        with self._cond:
            if not self.report.code:
                self.report.code = errno.EIO
                self.report.kind = 'link' if task.crossing else 'function'
                self.report.peer = tuple(r for r in task.owners if r != self.rank) if task.crossing else None
                self.report.call, self.report.detail = task.index, f'{type(error).__name__}: {error}'
            self._cond.notify_all()

    # design/algorithm-sources.md#programkernel_call
    def _touch(self, task):
        if task is not None and not task.crossing and self._realized:
            self._candidates.add(task)

    # design/algorithm-sources.md#programkernel_call
    def _drained(self, task):
        if task is not None:
            self._touch(task)
            for other in task.guarding:
                self._touch(other)

    # design/algorithm-sources.md#programkernel_call
    def _ready(self, task):
        if task.running or task.done or (task.fired and not task.cyclic):
            return False
        if any(edge.tokens < 1 for edge in task.inputs) or any(edge.tokens > 0 for edge in task.outputs):
            return False
        return not any(other.running or any(edge.tokens > 0 for edge in other.outputs) for other in task.guards)

    # design/algorithm-sources.md#programkernel_call
    def _complete(self, task, works=()):
        task.running, task.fired = False, True
        if task.kind == 'export':
            task.done = True
        else:
            for edge in task.inputs:
                if not edge.sticky:
                    edge.tokens -= 1
                self._drained(edge.source)
            for edge in task.outputs:
                edge.tokens, edge.works, edge.waited = 1, works, not works
                self._touch(edge.target)
        self._drained(task)
        self._cond.notify_all()

    # design/algorithm-sources.md#programkernel_call
    def _next(self):
        for task in self._candidates:
            if self._ready(task):
                heapq.heappush(self._heap, (any(not edge.waited for edge in task.inputs), task.row, task.index, task))
        self._candidates.clear()
        while self._heap:
            *_, task = heapq.heappop(self._heap)
            if self._ready(task):
                return task
        return None

    # design/algorithm-sources.md#programkernel_call
    def _execute(self, task):
        began = time.perf_counter_ns() if self.trace is not None else 0
        for edge in task.inputs:
            if not edge.waited:
                for work in edge.works:
                    work.wait()
                edge.waited = True
        what, works = task.run, ()
        if task.kind == 'kernel':
            kernel, sources, targets, coordinate = what
            binding = self._functions.get(kernel, kernel)
            if isinstance(binding, self._expression):
                binding(self, sources, targets, coordinate)
            else:
                binding(self, sources, targets)
            self.report.kernels += 1
        elif what[0] == 'materialize':
            _, regions, target = what
            out = target.tensor
            for ref, row, column in regions:
                out[row:row + ref.rows, column:column + ref.columns].copy_(ref.tensor)
        elif what[0] == 'export':
            if self.device == 'mps':
                task.event = torch.mps.Event()
                task.event.record()
        else:
            works = tuple(work for work in what[0](self, task) if work is not None)
            works = works if self.device == 'mps' else ()
            self.report.crossings += 1
        if self.trace is not None:
            self.trace.append((task.kind, task.index, task.row, began, time.perf_counter_ns(), threading.get_ident()))
        return works

    # design/algorithm-sources.md#programkernel_call
    def _serve(self, orders):
        """The compute thread: the ready call earliest by (waits on an issued receive, row, record); on MPS also each
        queue's next crossing in its order, issued once ready (there its calls are encoded and never block)."""
        while True:
            with self._cond:
                while True:
                    if self._closed or self.report.code:
                        return
                    chosen = next(((order, order.head()) for order in orders
                                   if order.head() is not None and self._ready(order.head())), None)
                    if chosen is None:
                        task = self._next()
                        chosen = (None, task) if task is not None else None
                    if chosen is not None:
                        chosen[1].running = True
                        break
                    self._cond.wait(0.05)
            order, task = chosen
            try:
                works = self._execute(task)
            except Exception as error:
                self._fail(task, error)
                return
            with self._cond:
                self._complete(task, works)
                if order is not None:
                    order.advance()

    # design/algorithm-sources.md#independent-native-queues
    def _stream(self, order):
        """A queue's crossings on the CPU, one thread: in the order every member computed, round after round; each
        waits for its local inputs, then its host-path call completes where it returns."""
        while True:
            with self._cond:
                while True:
                    task = order.head()
                    if self._closed or self.report.code or task is None:
                        return
                    if self._ready(task):
                        task.running = True
                        break
                    self._cond.wait(0.05)
            try:
                works = self._execute(task)
            except Exception as error:
                self._fail(task, error)
                return
            with self._cond:
                self._complete(task, works)
                order.advance()

    # design/algorithm-sources.md#programcopy
    def _issue_copy(self, task):
        _, src, sender, dst, receiver = task.run
        group = self._world.of(task.queue, task.members)
        if self.rank == sender:
            work = _native.send(src.tensor, task.members.index(receiver), group)
            work.wait()
            return ()
        return (_native.recv(dst.tensor, task.members.index(sender), group),)

    # design/algorithm-sources.md#collectivesync_on_remote_fill
    def _wait_results(self, results, timeout=None):
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while not all(result.ready for result in results):
            with self._cond:
                self._raise()
                if time.monotonic() > deadline:
                    raise TimeoutError(errno.ETIMEDOUT, 'sync_on_remote_fill: a result was not filled within the bound')
                self._cond.wait(0.0005)

    # design/algorithm-sources.md#program
    def close(self):
        if self._closed:
            return
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        for thread in self._threads:
            thread.join(self.timeout)
        self._world.close()
        self._operations.clear()
        self._replicated.clear()

    # design/algorithm-sources.md#program
    def __enter__(self):
        return self

    # design/algorithm-sources.md#program
    def __exit__(self, *error):
        self.close()
