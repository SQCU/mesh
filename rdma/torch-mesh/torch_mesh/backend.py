"""torch.distributed's "mesh" backend: a ProcessGroup (torch's Python form of one, as
torch/testing/_internal/distributed/multi_threaded_pg.py's) whose every collective is one NCCL group of
../../libnccl-mesh.dylib (nccl.h on the bridges' prepared transfers), called through _mesh_c (the headers' own
declarations, compiled against them: ../../mesh_c_build.py).  MPS tensors take the library's Metal path: the group
is encoded on torch's MPS stream (_stream.mm) and committed, each tensor passed as its storage's MTLBuffer and byte
offset (ncclMeshBuffer), so the work is ordered with torch's own and the call returns at once.  CPU tensors take the
host path, a non-contiguous one staged through a contiguous copy.

A group's communicator is over its topology, an operand (Options, torch's pg_options of init_process_group and
new_group): its ranks' link map (mesh.LinkMap, mesh-plan.h's: a kind, the links, each one's cost), each rank's bridge
node and this rank's bridge region.  Without one, the topology is what the bridges report (observe)."""
import json
import math
import os
import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp, _create_work_from_future
from torch.futures import Future

_stream = None
if torch.backends.mps.is_available():
    from . import _stream  # its kernels for torch's functional collectives on MPS register as it loads

RDMA = os.environ.get('MESH_RDMA') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if RDMA not in sys.path:
    sys.path.insert(0, RDMA)
import mesh  # noqa: E402  (rdma/mesh.py: the planner's values, and _mesh_c)

ffi, LIB = mesh.ffi, mesh.lib
TYPES = {torch.int8: LIB.ncclInt8, torch.uint8: LIB.ncclUint8, torch.bool: LIB.ncclUint8, torch.int32: LIB.ncclInt32,
         torch.uint32: LIB.ncclUint32, torch.int64: LIB.ncclInt64, torch.uint64: LIB.ncclUint64, torch.float16: LIB.ncclFloat16,
         torch.float32: LIB.ncclFloat32, torch.float64: LIB.ncclFloat64, torch.bfloat16: LIB.ncclBfloat16,
         torch.float8_e4m3fn: LIB.ncclFloat8e4m3, torch.float8_e5m2: LIB.ncclFloat8e5m2}
OPS = ((ReduceOp.SUM, LIB.ncclSum), (ReduceOp.PRODUCT, LIB.ncclProd), (ReduceOp.MAX, LIB.ncclMax), (ReduceOp.MIN, LIB.ncclMin),
       (ReduceOp.AVG, LIB.ncclAvg))
# an observed link's alpha: the TB5 crossing's one-way latency through the bridges (metal-microbench
# docs/measurement.md: the 3 KB host round trip 10.29 us); its beta is its port's bandwidth's
ALPHA_US = 5.0
# the longest a link may stay silent before the library cancels it, whatever a group's timeout (seconds)
WAIT_BOUND = 60


@dataclass
class Options:
    """A group's topology, torch's pg_options: `topology` its ranks' link map (mesh.link_map; None: observed),
    `paths` the trees a message between unlinked ranks takes (mesh.trees(); None: the topology's shortest-path
    trees), `programs` the compiled collectives the group runs ([(mesh.compile(), below bytes)]; None: the library's
    declared default, compiled once at the group's creation), `node` each rank's bridge node (None: observed, or rank r
    on node r with a topology given), `region` this rank's bridge region (None: MESH_REGION, else /mesh0)."""
    topology: mesh.LinkMap | None = None
    paths: object = None
    programs: list | None = None
    node: list | None = None
    region: str | None = None


def link_map(kind, ranks, pairs, cost=None):
    """A topology's link map over `ranks` (mesh.link_map: kind mesh, ring, tree or graph; each link's (alpha us,
    beta ns a byte) in `cost`)."""
    return mesh.link_map(kind, ranks, pairs, cost)


def observe(store, rank, size, region):
    """The group's topology: each rank's node as its bridge reports it (mesh_observe), exchanged through the group's
    store; its links and their costs from the link map MESH_LINK_MAP names, whose node ids are the bridges' (the
    launcher's: metal-microbench tools/mesh/grid.py ranks_run), else as the bridges report them, each link at ALPHA_US
    and its port's bandwidth (no costs at all while a port's is unknown: the library's defaults); every pair linked a
    mesh, else a graph, whose links may leave a subgroup's ranks apart (the library joins them along the world's
    paths)."""
    views, node = ffi.new('struct mesh_link_view[]', 64), ffi.new('uint32_t *')
    count = LIB.mesh_observe(region.encode(), views, 64, node)
    if count < 0:
        raise RuntimeError(f'libmesh: the bridge of region {region}: {os.strerror(-count)}')
    store.set(f'mesh/observed/{rank}', json.dumps({'node': node[0], 'links': [[views[i].peer, views[i].bandwidth] for i in range(min(count, 64))]}))
    reports = [json.loads(store.get(f'mesh/observed/{r}')) for r in range(size)]
    nodes = [report['node'] for report in reports]
    if len(set(nodes)) < size:
        raise RuntimeError(f'ranks share a node ({nodes}): one rank a node is what the session carries yet')
    place = {n: r for r, n in enumerate(nodes)}
    costs = {}
    if os.environ.get('MESH_LINK_MAP'):
        given = mesh.read_link_map(os.environ['MESH_LINK_MAP'])
        listed = dict(zip(map(frozenset, given.pairs), given.cost or [None] * len(given.pairs)))
        every = [frozenset((a, b)) for a in range(given.nodes) for b in range(a + 1, given.nodes)]
        for key in (every if given.kind == 'mesh' else listed):
            if key <= set(place):
                costs[frozenset(place[n] for n in key)] = listed.get(key)
    else:
        for report in reports:
            for peer, bits in report['links']:
                if peer in place:
                    key = frozenset((place[report['node']], place[peer]))
                    known = costs.get(key)
                    costs[key] = (ALPHA_US, 8e9 / bits) if bits and (known is None or 8e9 / bits < known[1]) else known
    pairs = sorted(tuple(sorted(key)) for key in costs)
    cost = [costs[frozenset(p)] for p in pairs]
    every = len(pairs) == size * (size - 1) // 2
    return mesh.link_map('mesh' if every else 'graph', size, pairs, None if None in cost else cost, apart=True), nodes


def check(result):
    if result:
        raise RuntimeError(f'libnccl-mesh: {ffi.string(LIB.ncclGetErrorString(result)).decode()}: '
                           f'{ffi.string(LIB.ncclGetLastError(ffi.NULL)).decode()}')


def op(reduce_op):
    for known, code in OPS:
        if reduce_op == known:
            return code
    raise NotImplementedError(f'the mesh backend reduces by sum, product, max, min and avg, not {reduce_op}')


def kind(tensor):
    if tensor.dtype not in TYPES:
        raise NotImplementedError(f'the mesh backend moves {sorted(map(str, TYPES))}, not {tensor.dtype}')
    return TYPES[tensor.dtype]


class Host:
    """A tensor's host image: the tensor itself where it is a contiguous CPU tensor, else a contiguous CPU copy
    (`read` its contents, else uninitialized) that `back` writes into it."""

    def __init__(self, tensor, read=True):
        self.tensor = tensor
        self.direct = tensor.device.type == 'cpu' and tensor.is_contiguous()
        self.host = tensor if self.direct else (tensor.detach().to('cpu').contiguous() if read
                                                else torch.empty(tensor.shape, dtype=tensor.dtype))

    def at(self, elements=0):
        return ffi.cast('void *', self.host.data_ptr() + elements * self.host.element_size())

    def back(self):
        if not self.direct:
            self.tensor.copy_(self.host)


class Device:
    """An MPS tensor where it lies: its storage's MTLBuffer and byte offset, `at` an ncclMeshBuffer kept alive
    until the group ends; a non-contiguous tensor through a contiguous copy (`read`) that `back` writes back."""

    def __init__(self, tensor, read=True):
        self.tensor = tensor
        self.direct = tensor.is_contiguous()
        self.host = tensor if self.direct else (tensor.contiguous() if read else torch.empty(tensor.shape, dtype=tensor.dtype,
                                                                                              device=tensor.device))
        self.made = []

    def at(self, elements=0):
        made = ffi.new('ncclMeshBuffer *')
        made.buffer = ffi.cast('void *', self.host.untyped_storage().data_ptr())
        made.offset = (self.host.storage_offset() + elements) * self.host.element_size()
        self.made.append(made)
        return made

    def back(self):
        if not self.direct:
            self.tensor.copy_(self.host)


def operand(tensor, read=True):
    return Device(tensor, read) if tensor.device.type == 'mps' else Host(tensor, read)


def on_mps(value):
    if isinstance(value, torch.Tensor):
        return value.device.type == 'mps'
    return isinstance(value, (list, tuple)) and any(on_mps(v) for v in value)


class After:
    """A write-back that is a function: run where the staged outputs are written back."""

    def __init__(self, back):
        self.back = back


def issued(works, result):
    """The last of a call's groups issued and not completed (completing it completes the groups before it: _stream.mm
    MeshWork), else its result done."""
    last = None
    for work in works:
        last = work if work is not None else last
    return last if last is not None else done(result)


def asynchronous(opts):
    """An async op (torch sets the option's asyncOp from async_op): issued now, completed where it is waited."""
    return bool(getattr(opts, 'asyncOp', False))


def done(result):
    future = Future()
    future.set_result(result)
    return _create_work_from_future(future)


class ProcessGroupMesh(dist.ProcessGroup):
    _streamed = False
    supports_coalescing = True
    supports_splitting = False

    def _get_backend(self, device):
        """The group itself, its one backend on every device it runs (torch asks a group's backend by device:
        batch_isend_irecv coalesces through it, init_process_group's device_id reads it)."""
        return self

    def __init__(self, rank, size, store=None, options=None):
        super().__init__(rank, size)
        options = options or Options()
        self._rank, self._size, self._pending = rank, size, None
        self._stream = ffi.NULL
        region = options.region or os.environ.get('MESH_REGION') or '/mesh0'
        topology, node = options.topology, options.node
        if topology is None and size > 1:
            topology, node = observe(store, rank, size, region)
        elif topology is None:
            topology = mesh.link_map('mesh', 1, [])
        if topology.nodes != size:
            raise ValueError(f'a topology of {topology.nodes} ranks for a group of {size}')
        made, nodes = ffi.new('ncclComm_t *'), ffi.new('int[]', list(node)) if node is not None else ffi.NULL
        paths = options.paths.c if options.paths is not None else ffi.NULL
        table = (ffi.new('ncclMeshProgram[]', [{'program': p.c, 'below': below} for p, below in options.programs])
                 if options.programs else ffi.NULL)
        self._programs = (options.paths, options.programs, table)
        check(LIB.ncclMeshCommInitRank(made, rank, topology.c, paths, table, len(options.programs or []), nodes, region.encode()))
        self.comm, self.topology, self.node = made[0], topology, node
        self.handle = int(ffi.cast('uintptr_t', self.comm))
        if _stream is not None:
            _stream.attach(self, self.handle)

    def size(self):
        return self._size

    def getBackendName(self):
        return 'mesh'

    @property
    def group_name(self):
        """The name torch registered this group under (a creator's ProcessGroup is never given its name)."""
        return dist.distributed_c10d._world.pg_names[self]

    pg_name = group_name

    def _group(self, issue, outputs, result=None, inputs=None):
        """One NCCL group: `issue` makes its calls (their stream self._stream: torch's MPS stream, its open encoder
        and allocator, where the operands, `result` and `inputs`, are MPS tensors, else none), then every staged
        output is written back.
        An MPS group is committed at once, so its publications reach the peer while it computes on; a host group
        after MPS groups first completes every group issued and not completed and waits for the stream, as the
        session's positions run in its order.  Between
        start_coalescing and end_coalescing the calls join the coalesced group instead."""
        if self._pending is not None:
            self._pending.append((issue, outputs, result, inputs))
            return done(result)
        mps = on_mps(result) or on_mps(inputs)
        if mps:
            self._stream = ffi.cast('void *', _stream.begin())
            ProcessGroupMesh._streamed = True
        else:
            self._stream = ffi.NULL
            if ProcessGroupMesh._streamed:
                _stream.complete_all()
                torch.mps.synchronize()
                ProcessGroupMesh._streamed = False
        check(LIB.ncclGroupStart())
        try:
            issue()
        finally:
            ended = LIB.ncclGroupEnd()
            if mps:
                _stream.commit()
            check(ended)
        for output in outputs:
            output.back()
        return done(result)

    def _direct(self, *tensors):
        """torch's MPS stream, through the one-turn C++ calls (_stream.mm), where every tensor is a contiguous MPS
        tensor and no coalesced group is open; else None."""
        if self._pending is not None or _stream is None or not all(t.device.type == 'mps' and t.is_contiguous() for t in tensors):
            return None
        ProcessGroupMesh._streamed = True
        return _stream

    def start_coalescing(self, device):
        self._pending = []

    def end_coalescing(self, device):
        pending, self._pending = self._pending, None
        return self._group(lambda: [issue() for issue, _, _, _ in pending], [o for _, outputs, _, _ in pending for o in outputs],
                           [r for _, _, r, _ in pending], [i for _, _, _, i in pending])

    def allreduce(self, tensors, opts=dist.AllreduceOptions()):
        if (s := self._direct(*tensors)) is not None:
            return issued([s.allreduce(t, op(opts.reduceOp), self.handle, asynchronous(opts)) for t in tensors], tensors)
        hosts = [operand(t) for t in tensors]
        return self._group(lambda: [check(LIB.ncclAllReduce(h.at(), h.at(), h.host.numel(), kind(h.host), op(opts.reduceOp),
                                                            self.comm, self._stream)) for h in hosts], hosts, tensors)

    def allreduce_coalesced(self, tensors, opts=None):
        return self.allreduce(tensors, opts or dist.AllreduceOptions())

    def reduce(self, tensors, opts=dist.ReduceOptions()):
        if (s := self._direct(*tensors)) is not None:
            return issued([s.reduce(t, op(opts.reduceOp), opts.rootRank, self.handle, asynchronous(opts)) for t in tensors], tensors)
        hosts = [operand(t) for t in tensors]
        return self._group(lambda: [check(LIB.ncclReduce(h.at(), h.at(), h.host.numel(), kind(h.host), op(opts.reduceOp),
                                                         opts.rootRank, self.comm, self._stream)) for h in hosts], hosts, tensors)

    def broadcast(self, tensors, opts=dist.BroadcastOptions()):
        if (s := self._direct(*tensors)) is not None:
            return issued([s.broadcast(t, opts.rootRank, self.handle, asynchronous(opts)) for t in tensors], tensors)
        hosts = [operand(t) for t in tensors]
        return self._group(lambda: [check(LIB.ncclBroadcast(h.at(), h.at(), h.host.numel(), kind(h.host), opts.rootRank,
                                                            self.comm, self._stream)) for h in hosts], hosts, tensors)

    def barrier(self, opts=None):
        return self.allreduce([torch.ones(1)])

    def all_gather_single(self, output, input, opts=None):
        if (s := self._direct(output, input)) is not None:
            return issued([s.allgather(output, input, self.handle, asynchronous(opts))], output)
        source, target = operand(input), operand(output, read=False)
        return self._group(lambda: check(LIB.ncclAllGather(source.at(), target.at(), source.host.numel(), kind(source.host),
                                                           self.comm, self._stream)), [target], output)

    def all_gather_single_coalesced(self, outputs, inputs, opts=None):
        pairs = [(operand(i), operand(o, read=False)) for o, i in zip(outputs, inputs)]
        return self._group(lambda: [check(LIB.ncclAllGather(s.at(), t.at(), s.host.numel(), kind(s.host), self.comm, self._stream))
                                    for s, t in pairs], [t for _, t in pairs], outputs)

    allgather_into_tensor_coalesced = all_gather_single_coalesced

    def allgather(self, outputs, inputs, opts=None):
        """Each input gathered into its list of outputs; outputs of unequal sizes (an Allgatherv in torch's
        spelling) as ProcessGroupNCCL takes them: one broadcast rooted at each rank, in one group."""
        if all(len({t.numel() for t in out}) == 1 for out in outputs):
            sources = [operand(i) for i in inputs]
            flats = [operand(torch.empty(self._size * s.host.numel(), dtype=s.host.dtype, device=s.host.device), read=False)
                     for s in sources]

            def back():
                for out, flat in zip(outputs, flats):
                    for r, piece in enumerate(flat.host.chunk(self._size)):
                        out[r].copy_(piece.view(out[r].shape))
            return self._group(lambda: [check(LIB.ncclAllGather(s.at(), f.at(), s.host.numel(), kind(s.host), self.comm, self._stream))
                                        for s, f in zip(sources, flats)], [After(back)], outputs)
        sources, targets = [operand(i) for i in inputs], [[operand(t, read=False) for t in out] for out in outputs]

        def issue():
            for source, out in zip(sources, targets):
                for r, target in enumerate(out):
                    sent = source if r == self._rank else target
                    check(LIB.ncclBroadcast(sent.at(), target.at(), target.host.numel(), kind(target.host), r, self.comm, self._stream))
        return self._group(issue, [t for out in targets for t in out], outputs)

    def reduce_scatter_single(self, output, input, opts=dist.ReduceScatterOptions()):
        if (s := self._direct(output, input)) is not None:
            return issued([s.reduce_scatter(output, input, op(opts.reduceOp), self.handle, asynchronous(opts))], output)
        source, target = operand(input), operand(output, read=False)
        return self._group(lambda: check(LIB.ncclReduceScatter(source.at(), target.at(), target.host.numel(), kind(source.host),
                                                               op(opts.reduceOp), self.comm, self._stream)), [target], output)

    def reduce_scatter_single_coalesced(self, outputs, inputs, opts=dist.ReduceScatterOptions()):
        pairs = [(operand(i), operand(o, read=False)) for o, i in zip(outputs, inputs)]
        return self._group(lambda: [check(LIB.ncclReduceScatter(s.at(), t.at(), t.host.numel(), kind(s.host), op(opts.reduceOp),
                                                                 self.comm, self._stream)) for s, t in pairs], [t for _, t in pairs], outputs)

    reduce_scatter_tensor_coalesced = reduce_scatter_single_coalesced

    def reduce_scatter(self, outputs, inputs, opts=dist.ReduceScatterOptions()):
        """Each list of inputs reduced, rank r's block left at rank r; blocks of unequal sizes (a
        Reduce_scatter of a list in torch's spelling) as ProcessGroupNCCL takes them: one reduce rooted at
        each rank, in one group."""
        if all(len({t.numel() for t in parts}) == 1 for parts in inputs):
            flats = [operand(torch.cat([x.detach().reshape(-1) for x in parts])) for parts in inputs]
            targets = [operand(o, read=False) for o in outputs]
            return self._group(lambda: [check(LIB.ncclReduceScatter(f.at(), t.at(), t.host.numel(), kind(t.host),
                                                                    op(opts.reduceOp), self.comm, self._stream))
                                        for f, t in zip(flats, targets)], targets, outputs)
        sources, targets = [[operand(x) for x in parts] for parts in inputs], [operand(o, read=False) for o in outputs]

        def issue():
            for parts, target in zip(sources, targets):
                for r, source in enumerate(parts):
                    received = target if r == self._rank else source
                    check(LIB.ncclReduce(source.at(), received.at(), source.host.numel(), kind(source.host), op(opts.reduceOp),
                                         r, self.comm, self._stream))
        return self._group(issue, targets, outputs)

    def alltoall(self, outputs, inputs, opts=None):
        sources, targets = [operand(i) for i in inputs], [operand(o, read=False) for o in outputs]

        def issue():
            for r in range(self._size):
                check(LIB.ncclSend(sources[r].at(), sources[r].host.numel(), kind(sources[r].host), r, self.comm, self._stream))
                check(LIB.ncclRecv(targets[r].at(), targets[r].host.numel(), kind(targets[r].host), r, self.comm, self._stream))
        return self._group(issue, targets, outputs)

    def all_to_all_single(self, output, input, output_split_sizes, input_split_sizes, opts=None):
        if (s := self._direct(output, input)) is not None:
            width = math.prod(input.shape[1:])
            rows_in = list(input_split_sizes) or [input.shape[0] // self._size] * self._size
            rows_out = list(output_split_sizes) or [output.shape[0] // self._size] * self._size
            return issued([s.alltoall(output, input, [r * width for r in rows_in], [r * width for r in rows_out], self.handle,
                                      asynchronous(opts))], output)
        source, target = operand(input), operand(output, read=False)
        if not output_split_sizes and not input_split_sizes:
            return self._group(lambda: check(LIB.ncclAlltoAll(source.at(), target.at(), source.host.numel() // self._size,
                                                              kind(source.host), self.comm, self._stream)), [target], output)
        rows_in = list(input_split_sizes) or [input.shape[0] // self._size] * self._size
        rows_out = list(output_split_sizes) or [output.shape[0] // self._size] * self._size
        width = math.prod(input.shape[1:])

        def issue():
            at_in = at_out = 0
            for r in range(self._size):
                check(LIB.ncclSend(source.at(at_in * width), rows_in[r] * width, kind(source.host), r, self.comm, self._stream))
                check(LIB.ncclRecv(target.at(at_out * width), rows_out[r] * width, kind(target.host), r, self.comm, self._stream))
                at_in, at_out = at_in + rows_in[r], at_out + rows_out[r]
        return self._group(issue, [target], output)

    def gather(self, outputs, inputs, opts=dist.GatherOptions()):
        """Every rank's input to the root's list of outputs, any sizes: grouped sends and receives, as
        ProcessGroupNCCL's gather."""
        source = operand(inputs[0])
        targets = [operand(t, read=False) for t in outputs[0]] if self._rank == opts.rootRank else []

        def issue():
            check(LIB.ncclSend(source.at(), source.host.numel(), kind(source.host), opts.rootRank, self.comm, self._stream))
            for r, target in enumerate(targets):
                check(LIB.ncclRecv(target.at(), target.host.numel(), kind(target.host), r, self.comm, self._stream))
        return self._group(issue, targets, outputs, inputs)

    def scatter(self, outputs, inputs, opts=dist.ScatterOptions()):
        """The root's list of inputs, one to each rank, any sizes (a Scatterv in torch's spelling): grouped sends
        and receives, as ProcessGroupNCCL's scatter."""
        target = operand(outputs[0], read=False)
        sources = [operand(t) for t in inputs[0]] if self._rank == opts.rootRank else []

        def issue():
            for r, source in enumerate(sources):
                check(LIB.ncclSend(source.at(), source.host.numel(), kind(source.host), r, self.comm, self._stream))
            check(LIB.ncclRecv(target.at(), target.host.numel(), kind(target.host), opts.rootRank, self.comm, self._stream))
        return self._group(issue, [target], outputs)

    def send(self, tensors, dst, tag):
        if (s := self._direct(*tensors)) is not None:
            return issued([s.send(t, dst, self.handle, True) for t in tensors], tensors)
        hosts = [operand(t) for t in tensors]
        return self._group(lambda: [check(LIB.ncclSend(h.at(), h.host.numel(), kind(h.host), dst, self.comm, self._stream))
                                    for h in hosts], [], tensors)

    def recv(self, tensors, src, tag):
        if (s := self._direct(*tensors)) is not None:
            return issued([s.recv(t, src, self.handle, True) for t in tensors], tensors)
        hosts = [operand(t, read=False) for t in tensors]
        return self._group(lambda: [check(LIB.ncclRecv(h.at(), h.host.numel(), kind(h.host), src, self.comm, self._stream))
                                    for h in hosts], hosts, tensors)


_installed = False


def create(opts, options):
    """The group (torch's extended creator: `opts` its store, rank and size, `options` the caller's pg_options, an
    Options); the first one also completes what PyTorch's parallelism APIs need of MPS tensors (_mps.py) and
    installs the partition header (partition.py)."""
    global _installed
    if not _installed:
        from . import _mps, partition  # noqa: F401 (partition installs itself)
        _mps.register()
        _installed = True
    if options is not None and not isinstance(options, Options):
        raise TypeError(f'the mesh backend takes torch_mesh.Options as pg_options, not {type(options).__name__}')
    if not os.environ.get('MESH_REMOTE_BOUND') and opts.timeout:
        # the group's timeout (init_process_group's, torch's contract for a collective's wait), at most WAIT_BOUND, is how
        # long a link may stay silent with positions pending before the library cancels it: a wait on the Metal path is a
        # kernel polling on the GPU, which no peer's silence may hold for torch's default half hour (MESH_REMOTE_BOUND
        # sets any other bound)
        os.environ['MESH_REMOTE_BOUND'] = str(max(1, min(WAIT_BOUND, int(opts.timeout.total_seconds()))))
    return ProcessGroupMesh(opts.group_rank, opts.group_size, opts.store, options)


dist.Backend.register_backend('mesh', create, devices=['cpu', 'mps'], extended_api=True)
dist.Backend.default_device_backend_map['mps'] = 'mesh'  # init_process_group() without a backend: MPS tensors on the mesh
