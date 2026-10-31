"""torch.distributed's "mesh" backend: a ProcessGroup (torch's Python form of one, as
torch/testing/_internal/distributed/multi_threaded_pg.py's) whose every collective is one NCCL group of
../../libnccl-mesh.dylib (nccl.h on the bridges' prepared transfers), issued by one path, _stream.mm's `issue`: a list
of calls, each a kind, its tensors, counts, operator and root.  MPS tensors take the library's Metal path: the group is
encoded on torch's MPS stream, each tensor passed as its storage's MTLBuffer and byte offset (ncclMeshBuffer), so the
work is ordered with torch's own and the call returns at once (an async op issued and not completed: a Work whose wait
completes it on torch's stream).  CPU tensors take the host path, complete where the call returns.  A non-contiguous
tensor fails as a value (_stream.mm Group): nothing is copied to make it one.

One form a collective, the library's V call where it has one (the library takes equal segments as its equal
program): an all-gather and a reduce-scatter are ncclMeshAllGatherV and ncclMeshReduceScatterV; all_to_all,
all_to_all_single, gather and scatter are ncclMeshAlltoAllV, MPI_Alltoallv with its counts on the host (a gather and a
scatter of any sizes the rooted V, MPI's Gatherv and Scatterv: every count 0 but toward or from the root, as NCCL's
gather and scatter are its forms); a list supplies separate original/final storage bindings to that same collective compiler. Directed transfers are declared writes or sparse counted exchanges on the full physical communicator.
The partials merge (merge_partials) is an all-reduce,
reduce or V reduce-scatter by the library's partials operator.

A group's communicator is NCCL's bootstrap's (ncclCommInitRank, rank 0's ncclGetUniqueId passed through the group's
store, as ProcessGroupNCCL does, the store's one use), its sessions each on the map the bridges keep when it opens; its
pg_options are not read.  What its ranks agree before a function runs (exchange_configuration) crosses the
communicator's configuration channel (ncclMeshCommConfigGather).
The silent-link bound is the library's (nccl.h)."""
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
# a call's kind (_stream.mm Kind)
ALLREDUCE, REDUCE, BROADCAST, ALLGATHER, REDUCE_SCATTER, ALLTOALL = range(6)
WRITE = 8
# a declared call (ProcessGroupMesh.plan): its kind, its elements, its dtype, its root or peer, its operator and, for an
# all-gather or a reduce-scatter, each rank's elements (an all-to-all's to each rank, then from each)
# element_shape declares one indivisible numerical element in those scalar units.
# bindings, where given, are the (send, receive) tensors the call is issued with (nccl.h ncclMeshCall bindings).
# design/eager-execution.md#what-follows
Call = collections.namedtuple('Call', 'kind count dtype root op counts destination fused contributors matrix recipients tile_bytes production element_shape bindings views frames max_packet source_ready landings ready_value ready_layout land_layout land_capacity sources route_scope',
                              defaults=(0, ReduceOp.SUM, None, 0, False, None, None, None, 0, None, (), None, None, 1, 0, None, None, 0, None, None, 0, (), None))
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
    return call._replace(sources=tuple(records))


def op(reduce_op):
    for known, code in OPS:
        if reduce_op == known:
            return code
    raise NotImplementedError(f'the mesh backend reduces by sum, product, max, min and avg, not {reduce_op}')


def typed(dtype, reduces):
    """A dtype as _stream.mm's Group passes a tensor of it (moving, reducing): its NCCL type and that type's elements an
    element (a complex reduction's two; a type NCCL lacks moved as its bytes)."""
    if reduces and dtype.is_complex:
        return TYPES[dtype.to_real()], 2
    return (TYPES[dtype], 1) if dtype in TYPES else (LIB.ncclUint8, torch.empty(0, dtype=dtype).element_size())


def _binding(tensor):
    """A tensor as an issue passes it (_stream.mm Group::where; nccl.h ncclMeshCall bindings): an MPS one its storage's
    MTLBuffer and byte offset, a CPU one {NULL, its address}, None {NULL, 0}."""
    if tensor is None:
        return (ffi.NULL, 0)
    if tensor.is_mps:
        return (ffi.cast('void *', tensor.untyped_storage().data_ptr()), tensor.storage_offset() * tensor.element_size())
    return (ffi.NULL, tensor.data_ptr())


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
        self._rank, self._size, self._operators = rank, size, {}
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
        """torch's end of this group (destroy_process_group): the fused crossings' forms on its communicator released,
        their leases back to the region once the stream has passed them (_stream.mm fused_release), its completions'
        observations emitted first."""
        for observations in self.observations:
            observations.close()
        _stream.fused_release(self.handle)
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
            bindings = []
            for value in local:
                record = value if isinstance(value, dict) else {'tensor': value}
                bindings.append({'buffer': _binding(record.get('tensor')), 'instances': record.get('instances', 1),
                                 'stride': record.get('stride', 0), 'invocation': record.get('invocation', 0)})
            slots = ffi.new('ncclMeshBufferSlots[2]', bindings) if c.bindings is not None or c.views is not None else ffi.NULL
            views = ffi.new('ncclMeshLayout[2]', geometries) if c.views is not None else ffi.NULL
            kept.extend((slots, views))
            sources, source_held = assignment.source_bindings(c.sources)
            kept.extend(source_held)
            made.append({'what': WHAT[c.kind], 'peer': c.root, 'count': c.count * per, 'type': nccl, 'op': code,
                         'group': g, 'destination': c.destination, 'fused': int(c.fused), 'tile_bytes': c.tile_bytes,
                         'production': production, 'bindings': slots, 'views': views, 'frames': c.frames,
                         'max_packet': c.max_packet, 'source_ready': _binding(c.source_ready),
                         'landings': _binding(c.landings), 'ready_value': c.ready_value,
                         'ready_layout': c.ready_layout or {}, 'land_layout': c.land_layout or {}, 'land_capacity': c.land_capacity,
                         'counts': counts, 'matrix': matrix, 'contributors': masks[0], 'recipients': masks[1],
                         'sources': sources, 'source_count': len(c.sources),
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

    def prepare_writes(self, deliveries, *, declarations=None):
        """Bind local source/result geometry into ordinary communicator Calls."""
        from .operand import TensorView
        bindings, calls = [], []
        for value, source, destination in deliveries:
            view = TensorView.of(value)
            geometry = view.write_layout()
            bindings.append((*geometry, int(source), int(destination)))
            sides = tuple(geometry if self.rank() == rank else None for rank in (source, destination))
            calls.append(Call(WRITE, view.numel() * view.element_size(), torch.uint8,
                              root=source, destination=destination, views=sides))
        if declarations is not None:
            calls = [declared._replace(views=physical.views)
                     for declared, physical in zip(declarations, calls, strict=True)]
        return _stream.ViewWrites(bindings, self.handle), calls

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

    def partials_operator(self, width):
        """Realize a partial-row reduction operator before declaring a function."""
        if width not in self._operators:
            made = ffi.new('ncclRedOp_t *')
            check(LIB.ncclMeshRedOpCreatePartials(made, width - 2, LIB.ncclFloat32, self.comm))
            self._operators[width] = made[0]
        return self._operators[width]

def prepare_fragments(bindings, group):
    """The common AOT binding for partial blocks and frontend graph crossings."""
    from .collective_views import lower
    bindings, declarations, before, after = lower(bindings, group)
    # each declaration names the tensors its issue passes, so CommPlan lays and places it for them.
    # design/eager-execution.md#what-follows
    declarations = [d._replace(bindings=(source, destination)) for d, source, destination in bindings]
    calls, operands = [], []
    for declaration, source, destination in bindings:
        code = declaration.op if isinstance(declaration.op, int) else op(declaration.op)
        counts = [declaration.count] if declaration.kind in (ALLREDUCE, REDUCE, BROADCAST) else declaration.counts
        calls.append((declaration.kind, [source], [destination], counts or [], code, declaration.root))
        operands.extend(t for t in (source, destination) if t is not None)
    return _stream.Collectives(calls, group.handle, operands, before, after), declarations


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


@torch.library.custom_op('torch_mesh::merge_partials', mutates_args=())
def merge_partials(partials: torch.Tensor, root: int, group_name: str, rows: list[int]) -> torch.Tensor:
    """The partials merge on the mesh group named `group_name` as a torch operator (torch_mesh.merge_partials):
    float32 rows of [m, l, O] (partial) merged by their log-sum-exp, on every member (`root` -1) or at `root`."""
    _mesh(group_name, 'merge_partials')
    raise RuntimeError('torch_mesh::merge_partials runs inside a prepared function (declarations.Function), whose '
                       'preparation declares it; an eager collective is not executed (design/eager-execution.md)')


@merge_partials.register_fake
def _(partials, root, group_name, rows):
    shape = (rows[_mesh(group_name, 'merge_partials').rank()], partials.shape[-1]) if rows else partials.shape
    return partials.new_empty(shape)


@torch.library.custom_op('torch_mesh::all_reduce', mutates_args=())
def all_reduce(input: torch.Tensor, reduce_op: str, group_name: str, repeats: int = 1) -> torch.Tensor:
    """torch's functional all_reduce as an op a compiled graph calls (torch_mesh.compiler).  A sum of MPS tensors on a
    mesh group is a compiler-indexed crossing, as allgatherv's: this rank's part published from a held lease by
    a kernel of partition's own, every peer's awaited and all of them added in rank order by another, the host issuing
    the crossing (once for `repeats` of them: the graph's, torch_mesh.compiler) and two encodes; any other its
    functional kernel's, its work waited where the graph waits on it (wait_tensor)."""
    group = dist.distributed_c10d._resolve_process_group(group_name)
    if (reduce_op == 'sum' and isinstance(group, ProcessGroupMesh) and input.device.type == 'mps'
            and input.dtype in (torch.float16, torch.float32)):
        if group.size() == 1:
            return input.clone()
        raise RuntimeError('mesh sum must be lowered to an indexed crossing before execution')
    return torch.ops._c10d_functional.all_reduce.default(input, reduce_op, group_name)


@all_reduce.register_fake
def _(input, reduce_op, group_name, repeats=1):
    return torch.empty_like(input)


@torch.library.custom_op('torch_mesh::linear_sum', mutates_args=())
def linear_sum(x: torch.Tensor, w: torch.Tensor, group_name: str, repeats: int = 1) -> torch.Tensor:
    """x w^T (torch_mesh::linear's product, x [R, K], w [N, K]) summed over the mesh group named `group_name` as one
    compiler-indexed crossing, issued once for `repeats` of them (torch_mesh.compiler): linear.metal's
    linear_publish computes this rank's product into its segment and publishes it through the collective's steps,
    sum_land adds every rank's in rank order (torch_mesh.compiler)."""
    if dist.distributed_c10d._resolve_process_group(group_name).size() == 1:
        return torch.ops.torch_mesh.linear(x, w)
    raise RuntimeError('linear_sum must be lowered to an indexed crossing before execution')


@linear_sum.register_fake
def _(x, w, group_name, repeats=1):
    return x.new_empty(x.shape[0], w.shape[0])


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
