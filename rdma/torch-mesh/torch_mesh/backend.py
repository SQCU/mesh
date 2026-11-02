"""PyTorch declarations for the shared page-based collective compiler.

A ProcessGroup owns the physical communicator and preparation configuration.
Call bindings, page views, routes and numerical source are declared before
execution. The common source-effect pass composes numerical work with the
collective's page dependencies; the native recorder then binds that result.
Execution consumes those prepared commands through the shared schedule.
"""
import collections
import contextlib
import math

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp

from . import _stream  # _stream's kernels for torch's functional collectives on MPS register as it loads

import mesh  # noqa: E402  (rdma/mesh.py, on the path the mesh's install gives the environment: its _mesh_c)
from routing import Fabric

ffi, LIB, check = mesh.ffi, mesh.lib, mesh.check
OPS = ((ReduceOp.SUM, LIB.ncclSum), (ReduceOp.PRODUCT, LIB.ncclProd), (ReduceOp.MAX, LIB.ncclMax), (ReduceOp.MIN, LIB.ncclMin),
       (ReduceOp.AVG, LIB.ncclAvg))
# Frontend call kinds map to the shared declaration ABI below.
ALLREDUCE, REDUCE, BROADCAST, ALLGATHER, REDUCE_SCATTER, ALLTOALL = range(6)
WRITE = 8
# a declared call (ProcessGroupMesh.plan): its kind, its elements, its dtype, its root or peer, its operator and, for an
# all-gather or a reduce-scatter, each rank's elements (an all-to-all's to each rank, then from each)
# element_shape declares one indivisible numerical element in those scalar units.
# outer, for an all-gather or a reduce-scatter along a later axis, its frames: the product of the dimensions before the
# axis. Its positions are its operand's own order, frame by frame, each rank's block in every frame
# (collective_views._pieces); counts are each rank's in all of them.
# bindings, where given, are the (send, receive) tensors the call is issued with (nccl.h ncclMeshCall bindings).
# design/eager-execution.md#what-follows
Call = collections.namedtuple('Call', 'kind count dtype root op counts destination contributors matrix recipients tile_bytes production element_shape bindings views frames max_packet source_ready ready_value ready_layout sources route_scope coverage outer',
                              defaults=(0, ReduceOp.SUM, None, 0, None, None, None, 0, None, (), None, None, 1, 0, None, 0, None, (), None, (), 1))
WHAT = {ALLREDUCE: LIB.MESH_ALLREDUCE, REDUCE: LIB.MESH_REDUCE, BROADCAST: LIB.MESH_BROADCAST, ALLGATHER: LIB.MESH_ALLGATHER,
        REDUCE_SCATTER: LIB.MESH_REDUCE_SCATTER, ALLTOALL: LIB.ncclMeshCallAlltoAllV, WRITE: LIB.ncclMeshCallWrite}
TYPES = {torch.int8: LIB.ncclInt8, torch.uint8: LIB.ncclUint8, torch.bool: LIB.ncclUint8, torch.int32: LIB.ncclInt32,
         torch.uint32: LIB.ncclUint32, torch.int64: LIB.ncclInt64, torch.uint64: LIB.ncclUint64, torch.float16: LIB.ncclFloat16,
         torch.float32: LIB.ncclFloat32, torch.float64: LIB.ncclFloat64, torch.bfloat16: LIB.ncclBfloat16,
         torch.float8_e4m3fn: LIB.ncclFloat8e4m3, torch.float8_e5m2: LIB.ncclFloat8e5m2}


def bind_source(call, *, source, occurrences, work, route_scope, occurrence_extent=1):
    """Attach declared source identity and ordered occurrence/byte-work operands."""
    from .operand import layout_configuration
    def configured(value):
        return layout_configuration(value) if isinstance(value, _stream.TensorLayout) else value
    record = dict(source=source, occurrences=configured(occurrences), work=configured(work),
                  occurrence_extent=occurrence_extent)
    return call._replace(sources=(*call.sources, record), route_scope=route_scope)


def source_work(call, order):
    """Carry a physical fragment's byte order through its original source work."""
    from .operand import layout_configuration, relation_layout
    import json
    records = []
    for record in call.sources:
        declaration = record['work']
        if 'steps' in declaration:
            from mesh_relation import Algebra
            relation = Algebra().selection_relation(declaration)
            original = relation_layout(relation, relation.shape, math.prod(declaration['shape']))
        else:
            original = _stream.TensorLayout(declaration['shape'], declaration['strides'], declaration['offset'],
                {int(axis): values for axis, values in declaration['maps'].items()}, None,
                declaration['index_shape'], json.dumps(declaration['index_coordinates']))
        selected = order.rebase(_stream.TensorLayout((original.count,), (1,)), original.reshape((original.count,)))
        records.append(dict(record, work=layout_configuration(selected)))
    coverage = call.coverage
    if coverage and call.kind == WRITE:
        # A WRITE's transaction is its one declared byte selection. Restrict
        # masks through exactly the same order as the payload/source relation;
        # routing will retain transaction 0 across every physical hop.
        from .compilation import relations
        algebra = relations()
        selected = algebra.layout(layout_configuration(order))
        projected = []
        for row in coverage:
            domain = algebra.set('{ [b] : ' + str(row['first']) + ' <= b < ' +
                                 str(row['first'] + row['bytes']) + ' }', 1)
            for first, count, stride, repeat in selected.preimage(domain).spans():
                if stride == count:
                    projected.append(dict(row, first=first, bytes=count * repeat))
                else:
                    projected.extend(dict(row, first=first + index * stride, bytes=count)
                                     for index in range(repeat))
        coverage = tuple(projected)
    return call._replace(sources=tuple(records), coverage=coverage)


def op(reduce_op):
    for known, code in OPS:
        if reduce_op == known:
            return code
    raise NotImplementedError(f'the mesh backend reduces by sum, product, max, min and avg, not {reduce_op}')


def typed(dtype, reduces):
    """Declared scalar type and scalar units per logical collective element."""
    if reduces and dtype.is_complex:
        return TYPES[dtype.to_real()], 2
    return (TYPES[dtype], 1) if dtype in TYPES else (LIB.ncclUint8, torch.empty(0, dtype=dtype).element_size())


def _binding(tensor):
    """Resolve a declared tensor binding during preparation."""
    if tensor is None:
        return (ffi.NULL, 0)
    if tensor.is_mps:
        return (ffi.cast('void *', tensor.untyped_storage().data_ptr()), tensor.storage_offset() * tensor.element_size())
    return (ffi.NULL, tensor.data_ptr())


# each value space's handle in the library (nccl.h ncclMeshSpaceCreate), by its serial: created once, bound per
# occurrence by the lowering (lowering.bind_spaces), never by a chooser
_SPACES = {}


def space_handle(space):
    """type: a ValueSpace -> its ncclMeshSpace_t, created on first use under the same serial (design/page-maps-2026-11-02.md)."""
    handle = _SPACES.get(space.serial)
    if handle is None:
        made = ffi.new('ncclMeshSpace_t *')
        check(LIB.ncclMeshSpaceCreate(space.serial, space.pages * space.page_bytes, made))
        handle = _SPACES[space.serial] = made[0]
    return handle


def _slots(value):
    """One ordinary page binding and its explicit occurrence coordinates. A value names its space and the byte offset
    of its first element there (design/page-maps-2026-11-02.md), never a buffer."""
    from .operand import TensorValue
    if isinstance(value, TensorValue):
        first = value.layout.index(0) * value.dtype.itemsize if value.layout.count else 0
        return {'buffer': (ffi.NULL, first), 'instances': 1, 'stride': 0, 'invocation': 0,
                'space': space_handle(value.space)}
    record = value if isinstance(value, dict) else {'tensor': value}
    return {'buffer': _binding(record.get('tensor')), 'instances': record.get('instances', 1),
            'stride': record.get('stride', 0), 'invocation': record.get('invocation', 0)}


class ProcessGroupMesh(dist.ProcessGroup):
    supports_coalescing = True
    supports_splitting = False

    def _get_backend(self, device):
        """The group itself, its one backend on every device it runs (torch asks a group's backend by device:
        init_process_group's device_id reads it)."""
        return self

    def __init__(self, rank, size, store):
        super().__init__(rank, size)
        made, unique = ffi.new('ncclComm_t *'), ffi.new('ncclUniqueId *')
        if rank == 0:
            check(LIB.ncclGetUniqueId(unique))
            store.set('mesh/unique', bytes(ffi.buffer(unique)))
        else:
            ffi.memmove(unique, store.get('mesh/unique'), ffi.sizeof('ncclUniqueId'))
        check(LIB.ncclCommInitRank(made, size, unique[0], rank))
        self._held(rank, size, made[0])

    def _held(self, rank, size, comm):
        self._rank, self._size = rank, size
        # the completions submitted under this group (mesh_statistics.Observations), emitted by its shutdown at the latest
        self.observations = []
        # Completed cohorts and still-missing peers are preparation operands.
        # Completion callbacks never exchange or install configurations.
        self.configuration_observations = []
        self._declaration = {'groups': [], 'order': [], 'production_seconds': None, 'routing': None}
        self.comm = comm
        self.handle = int(ffi.cast('uintptr_t', self.comm))
        # Native realization installs default routes over every physical path.
        # Keep this communicator's topology snapshot for configuration lowering;
        # migrations neither rediscover it nor guess route state from an address.
        try:
            import json
            from .compilation import implementations, implementation_assignment
            from .cpu import compilation
            implementation = implementations()[implementation_assignment()['collective_cpu']]
            check(LIB.ncclMeshCommCompiler(comm,
                json.dumps(compilation(implementation), separators=(',', ':')).encode(),
                json.dumps(implementation['resources'], separators=(',', ':')).encode()))
            self.fabric = Fabric.of(comm)
        except Exception:
            LIB.ncclCommDestroy(comm)
            raise
        _stream.attach(self, self.handle)

    @property
    def page(self):
        """The bytes of a page of this communicator's region, on which each segment of a V collective starts (nccl.h
        ncclMeshCommPage)."""
        held = ffi.new('size_t *')
        check(LIB.ncclMeshCommPage(self.comm, held))
        return held[0]

    @classmethod
    def _of(cls, comm):
        """A group over a communicator libnccl-mesh made (a shrink's or a grow's), with no bootstrap of its own."""
        count, rank = ffi.new('int *'), ffi.new('int *')
        check(LIB.ncclCommCount(comm, count))
        check(LIB.ncclCommUserRank(comm, rank))
        made = cls.__new__(cls)
        dist.ProcessGroup.__init__(made, rank[0], count[0])
        made._held(rank[0], count[0], comm)
        return made

    def shrink(self, exclude, abort=False):
        """This group without the ranks `exclude`, the kept ranks in their order (ncclCommShrink; `abort` revokes this
        group's work in flight first); None on a rank among those excluded."""
        made, ranks = ffi.new('ncclComm_t *'), ffi.new('int[]', list(exclude) or [0])
        check(LIB.ncclCommShrink(self.comm, ranks, len(exclude), made, ffi.NULL, 1 if abort else 0))
        return ProcessGroupMesh._of(made[0]) if made[0] != ffi.NULL else None

    def grow(self, size, unique=None, rank=-1):
        """This group's ranks and `size` - its size more (ncclCommGrow), met at the id this group's root gives
        (ncclCommGetUniqueId, `unique` None on its ranks); a joining process passes the root's id and its `rank` on a
        None group: ProcessGroupMesh.grow(None, size, unique, rank)."""
        made, held = ffi.new('ncclComm_t *'), ffi.new('ncclUniqueId *')
        if self is not None and self._rank == 0:
            check(LIB.ncclCommGetUniqueId(self.comm, held))
        elif unique is not None:
            ffi.memmove(held, unique, ffi.sizeof('ncclUniqueId'))
        check(LIB.ncclCommGrow(ffi.NULL if self is None else self.comm, size, held, rank, made, ffi.NULL))
        return ProcessGroupMesh._of(made[0]), bytes(ffi.buffer(held))

    def revoke(self):
        """This group's work in flight is revoked and none is issued after it (ncclCommRevoke): its bridge sends what
        it published and nothing else, and its own GPU waits end at their landings or their bounds."""
        check(LIB.ncclCommRevoke(self.comm, 0))

    def shutdown(self):
        """Observe ready reports once; prepared Functions own their page lifetimes.

        Applications finish pending observations before group shutdown.
        """
        for observations in self.observations:
            observations.close()
        super().shutdown()

    def size(self):
        return self._size

    def getBackendName(self):
        return 'mesh'

    @property
    def group_name(self):
        """The name torch registered this group under (a creator's ProcessGroup is never given its name)."""
        return dist.distributed_c10d._world.pg_names[self]

    pg_name = group_name

    def plan(self, groups, *, order=None, production_seconds=None, routing=None):
        """This group's collectives declared once, in their order (nccl.h ncclMeshCommPlan): `groups` the groups its
        calls issue, each a list of Calls (one torch collective is one group), or (that list, repeat) issued repeat
        times in a row, the order repeated from this group's next collective on.  Each is compiled now, so none is
        planned where it is issued, and none is compared with its declaration.  Returns each
        declared group's (send entries, landing entries) on this rank's lists.
        An explicit order supplies ncclMeshPlanRun dictionaries to repeat whole
        blocks of group definitions without expanding their calls."""
        groups = list(groups)
        import assignment
        routes, route_held = assignment.routing_bindings(routing)
        rows = [(g, *(group if isinstance(group, tuple) else (group, 1))) for g, group in enumerate(groups)]
        if any(not calls for _, calls, _ in rows):
            raise ValueError('a declared collective group must contain calls')
        order = ([{'first': g, 'count': 1, 'repeat': repeat} for g, calls, repeat in rows]
                 if order is None else list(order))
        if production_seconds is not None and len(production_seconds) != self._size:
            raise ValueError('producer durations require one operand per physical rank')
        producer_groups = [0] * self._size
        for block in order:
            for _, calls, _ in rows[block['first']:block['first'] + block['count']]:
                for source in {c.root for c in calls if c.production is not None}:
                    producer_groups[source] += block['repeat']
        rows = [(g, repeat, call) for g, calls, repeat in rows for call in calls]
        made, kept = [], list(route_held)
        for g, repeat, c in rows:
            reduces = c.kind in (ALLREDUCE, REDUCE, REDUCE_SCATTER)
            nccl, per = typed(c.dtype, reduces)
            code = (c.op if isinstance(c.op, int) else op(c.op)) if reduces else LIB.ncclSum
            if c.dtype == torch.bool and reduces:
                code = LIB.ncclMax if code == LIB.ncclSum else LIB.ncclMin if code == LIB.ncclProd else code
            kept.append(ffi.new('size_t[]', [n * per for n in c.counts]) if c.counts is not None else ffi.NULL)
            counts = kept[-1]
            if c.matrix is not None and len(c.matrix) != self._size * self._size:
                raise ValueError('counted collective matrix must cover every ordered physical rank pair')
            kept.append(ffi.new('size_t[]', [n * per for n in c.matrix]) if c.matrix is not None else ffi.NULL)
            matrix = kept[-1]
            masks = []
            for participants in (c.contributors, c.recipients):
                bits = [0] * ((self._size + 63) // 64)
                if participants is not None:
                    for rank in participants:
                        if rank < 0 or rank >= self._size:
                            raise ValueError('collective participant is outside the physical communicator')
                        bits[rank // 64] |= 1 << (rank % 64)
                kept.append(ffi.new('uint64_t[]', bits) if participants is not None else ffi.NULL)
                masks.append(kept[-1])
            if c.production is not None:
                tiles = -(-(c.count * c.dtype.itemsize) // c.tile_bytes) if c.tile_bytes else 1
                if len(c.production) != tiles:
                    raise ValueError('producer records must cover every logical publication tile')
            kept.append(ffi.new('ncclMeshProduction[]', c.production) if c.production is not None else ffi.NULL)
            production = kept[-1]
            if c.production is not None and production_seconds is not None:
                check(LIB.ncclMeshProductionTimes(production, len(c.production),
                      production_seconds[c.root] / producer_groups[c.root]))
            local = list(c.bindings) if c.bindings is not None else [None, None]
            geometries = []
            if c.views is not None:
                for side, view in enumerate(c.views):
                    if view is None:
                        geometries.append({})
                        continue
                    storage, bits, shape, strides, offset, mapped, runs, *frame = view
                    if storage.is_meta:
                        geometries.append({})
                        continue
                    local[side] = {'tensor': storage, **(local[side] if isinstance(local[side], dict) else {})}
                    arrays = [ffi.new('uint64_t[]', shape or [0]), ffi.NULL if mapped else ffi.new('int64_t[]', strides or [0]),
                              ffi.new('uint64_t[]', runs or [0])]
                    kept.extend(arrays)
                    geometries.append(dict(element_bits=bits, rank=len(shape), shape=arrays[0], strides=arrays[1],
                        offset=offset, runs=arrays[2] if mapped else ffi.NULL, nruns=len(runs) // 2,
                        frame_stride=frame[0] if frame else 0))
            bindings = [_slots(value) for value in local]
            slots = ffi.new('ncclMeshBufferSlots[2]', bindings) if c.bindings is not None or c.views is not None else ffi.NULL
            views = ffi.new('ncclMeshLayout[2]', geometries) if c.views is not None else ffi.NULL
            kept.extend((slots, views))
            sources, source_held = assignment.source_bindings(c.sources)
            kept.extend(source_held)
            coverage = [dict({name: row[name] for name in
                ('transaction', 'frame', 'first', 'bytes', 'cut', 'cuts', 'before')},
                active=_slots(row['active']), expected=_slots(row['expected'])) for row in c.coverage]
            held_coverage = ffi.new('ncclMeshCoverage[]', coverage) if coverage else ffi.NULL
            kept.append(held_coverage)
            made.append({'what': WHAT[c.kind], 'peer': c.root, 'count': c.count * per, 'type': nccl, 'op': code,
                         'group': g, 'destination': c.destination, 'tile_bytes': c.tile_bytes,
                         'production': production, 'bindings': slots, 'views': views, 'frames': c.frames,
                         'max_packet': c.max_packet, 'source_ready': _binding(c.source_ready),
                         'ready_value': c.ready_value,
                         'ready_layout': c.ready_layout or {},
                         'counts': counts, 'matrix': matrix, 'contributors': masks[0], 'recipients': masks[1],
                         'sources': sources, 'source_count': len(c.sources),
                         'coverage': held_coverage, 'coverage_count': len(coverage),
                         'route_scope': assignment.json_operand(c.route_scope, kept) if c.route_scope is not None else ffi.NULL,
                         'routing': routes if not made else ffi.NULL,
                         'routing_count': len(routing or ()) if not made else 0})
        view = ffi.new('ncclMeshPlan **')
        check(LIB.ncclMeshCommPlan(self.comm, ffi.new('ncclMeshCall[]', made or 1), len(made),
                                   ffi.new('ncclMeshPlanRun[]', order or 1), len(order), view))
        self._declaration = {'groups': groups, 'order': order, 'production_seconds': production_seconds,
                             'routing': assignment.routing_records(view[0])}
        self._plan = view[0]
        return [] if view[0] == ffi.NULL else [tuple(view[0].entries[g]) for g in range(view[0].groups)]

    def count_matrices(self, calls):
        """Each all-to-all of `calls`, declared with this rank's counts alone, its complete source-major count matrix
        as the library makes it from every rank's (nccl.h ncclMeshCommPlan's count exchange, ncclMeshPlan matrices),
        in the call's elements: the calls declared for this as a configuration (configuration), each its own group."""
        n = self._size
        with self.configuration([[call] for call in calls]):
            return [[value // typed(call.dtype, False)[1] for value in ffi.unpack(self._plan.matrices[k], n * n)]
                    for k, call in enumerate(calls)]

    @contextlib.contextmanager
    def configuration(self, groups):
        """Execute a declared configuration transfer, then restore the function graph.

        Repartition/scatter invokes this outside the numerical function. Its
        metadata is compiled once for that configuration transition, never by
        each data-path invocation of a variable-count exchange.
        """
        previous = self._declaration
        self.plan(groups)
        try:
            yield
        finally:
            self.plan(**previous)

    def plan_demands(self):
        """Compiler-derived useful deliveries for the entire declared graph.

        This configuration query includes repeats and every collective's actual
        lowered destinations; it never counts forwarding bytes as useful work.
        An undeclared or unsupported native graph raises its native error.
        """
        n = self.size()
        values = ffi.new('uint64_t[]', n * n)
        check(LIB.ncclMeshCommPlanDemands(self.comm, values, n * n))
        return {(a, b): int(values[a * n + b]) for a in range(n) for b in range(n)
                if a != b and values[a * n + b]}

    def prepare_writes(self, deliveries, *, declarations=None, frame_strides=None):
        """Bind local page geometry and explicit frame strides into ordinary Calls.

        Frame strides are in the selected tensor's scalar units. The Call's
        frame count and binding-instance operands determine their finite use.
        """
        from .operand import TensorView
        bindings, calls, views = [], [], []
        strides = [0] * len(deliveries) if frame_strides is None else frame_strides
        for (value, source, destination), frame_stride in zip(deliveries, strides, strict=True):
            view = TensorView.of(value)
            views.append((view, view))
            geometry = view.write_layout()
            bindings.append((*geometry, int(source), int(destination)))
            sides = tuple((*geometry, frame_stride) if self.rank() == rank else None for rank in (source, destination))
            calls.append(Call(WRITE, view.numel() * view.element_size(), torch.uint8,
                              root=source, destination=destination, views=sides))
        if declarations is not None:
            calls = [declared._replace(views=physical.views)
                     for declared, physical in zip(declarations, calls, strict=True)]
        return dict(transfer='write', bindings=tuple(bindings), views=tuple(views), calls=tuple(calls), group=self,
                    before=(), after=()), calls

    def exchange_configuration(self, value):
        """Every member's preparation operand through the shared native channel."""
        return mesh.exchange_configuration(self.comm, value)

    def broadcast_configuration(self, value, root=0):
        """The root's preparation operand through the shared native channel."""
        return mesh.broadcast_configuration(self.comm, value, root)

    def union_cuts(self, cuts):
        """Agree view boundaries during configuration, before the function runs."""
        cuts = sorted(set(cuts))
        rows = self.exchange_configuration(cuts)
        if any(row[0] != 0 or row[-1] != cuts[-1] for row in rows):
            raise ValueError('collective views have different logical extents across ranks')
        return sorted(set(cut for row in rows for cut in row))


def prepare_fragments(bindings, group):
    """The common AOT binding for partial blocks and frontend graph crossings."""
    bindings = list(bindings)
    if bindings and all(call.kind == WRITE for call, _, _ in bindings):
        from .operand import TensorView
        deliveries = [(TensorView.of(source if group.rank() == call.root else destination
                       if group.rank() == call.destination else source), call.root, call.destination)
                      for call, source, destination in bindings]
        return group.prepare_writes(deliveries, declarations=[call for call, _, _ in bindings])
    from .collective_views import lower
    bindings, declarations, before, after = lower(bindings, group)
    # each declaration names the tensors its issue passes, so CommPlan lays and places it for them.
    # design/eager-execution.md#what-follows
    declarations = [d._replace(bindings=(source, destination)) for d, source, destination in bindings]
    from .operand import TensorValue, TensorView
    views, operands = [], []
    for declaration, source, destination in bindings:
        views.append(tuple(None if value is None else value if isinstance(value, TensorValue) else TensorView.of(value)
                           for value in (source, destination)))
        operands.extend(t for t in (source, destination) if t is not None)
    return dict(transfer='collective', views=tuple(views), calls=tuple(declarations),
                operands=tuple(operands), group=group, before=tuple(before), after=tuple(after)), declarations


def partial(out, lse):
    """An attention's output over some keys (shape x dim) and each row's logsumexp (shape) as the library's partials,
    float32 rows (shape x (2 + dim)) of [m, l, O]: m the logsumexp, l 1, O the output; a row that saw no key (its
    logsumexp -inf) l 0, so it counts for none."""
    seen = torch.isfinite(lse)[..., None]
    return torch.cat([torch.where(seen, lse[..., None], 0), seen.float(), torch.where(seen, out, 0).float()], -1)


def attention(merged):
    """Merged partials (rows of [M, L, O]) as the attention over every member's keys, O / L (0 where L is 0), and each
    row's logsumexp, M + log L (-inf where L is 0)."""
    m, l, o = merged[..., 0], merged[..., 1], merged[..., 2:]
    seen = l > 0
    l = torch.where(seen, l, 1)
    return torch.where(seen[..., None], o / l[..., None], 0), torch.where(seen, m + l.log(), -math.inf)


def _mesh(group_name, what):
    group = dist.distributed_c10d._resolve_process_group(group_name)
    if not isinstance(group, ProcessGroupMesh):
        raise TypeError(f'{what} runs on a group of the mesh backend, not {group_name}\'s')
    return group


def merge_partials(values: torch.Tensor, root: int, group_name: str, rows: list[int]) -> torch.Tensor:
    """The supplied summary combine enters the same source/autograd graph."""
    from .reduction import partials
    group = _mesh(group_name, 'merge_partials')
    return partials(values.dtype, values.shape[-1])(values, group, root=root, counts=rows)


@torch.library.custom_op('torch_mesh::all_reduce', mutates_args=())
def all_reduce(input: torch.Tensor, reduce_op: str, group_name: str, repeats: int = 1) -> torch.Tensor:
    """Declare a reduction of any supported scalar type on the named mesh.

    The same whole-function crossing handles all reduction operators, devices
    and membership sizes; a declaration never delegates numerical execution.
    """
    _mesh(group_name, 'all_reduce')
    raise RuntimeError('all_reduce requires whole-function preparation')


@all_reduce.register_fake
def _(input, reduce_op, group_name, repeats=1):
    return torch.empty_like(input)


@torch.library.register_fake('torch_mesh::linears')
def _(x, w, bias, act, x1):
    return [a.new_empty(a.shape[0], b.shape[0]) for a, b in zip(x, w)]


@torch.library.register_fake('torch_mesh::linear')
def _(x, w, bias=None, act=0, x1=None):
    return x.new_empty(x.shape[0], w.shape[0])


def create(opts, options):
    """The group (torch's extended creator: `opts` its store, rank and size; `options`, the caller's pg_options, not
    read: a group's map is its bridges')."""
    return ProcessGroupMesh(opts.group_rank, opts.group_size, opts.store)


dist.Backend.register_backend('mesh', create, devices=['cpu', 'mps'], extended_api=True)
dist.Backend.default_device_backend_map['mps'] = 'mesh'  # init_process_group() without a backend: MPS's default is mesh
