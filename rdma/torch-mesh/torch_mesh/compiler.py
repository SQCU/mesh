"""torch_mesh's compiler (design/page-maps-2026-11-02.md): a stepper captured, lowered, solved and run.

capture  torch's FX tracer (make_fx) over fake tensors, the caller's tensors (state, weights, inputs) lifted as
         constants by identity.
lower    a view is a view of its operand's root; a pointwise op, a matrix product, an indexed copy or a sum is a command
         over views (kernels.py), a chain of single-consumer pointwise ops of one shape fused into one; an all-gather is
         a declared group of calls (nccl.h ncclMeshCommDeclare), one a contiguous frame of its gathered axis.
solve    each command's reads and writes and each collective step's SEND, completion, post and landing are the run's
         events and accesses (mesh-page-map.h mesh_run_solve); a caller's storage is a pinned bin, the arena the last.
run      a command a dispatch, a link and a release a word's store, a landing and a SEND's completion a presence, and the
         solve's edges between them (rdma/metal.h); an event with no node (a relay's SEND, the bridge's) passes its edges
         through. A repeated chunk is captured, lowered and solved once and instanced in the run, its collective
         occurrences each lowered anew (their channels' absolute positions move).
"""
import itertools
import math
import operator

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from . import _stream, kernels

aten = torch.ops.aten
VIEWS = {aten.select.int, aten.t.default, aten.view.default, aten._unsafe_view.default, aten.unsqueeze.default,
         aten.squeeze.dim, aten.slice.Tensor, aten.narrow.default, aten.permute.default, aten.transpose.int,
         aten.expand.default, aten.alias.default, aten.as_strided.default, aten.unbind.int}
UNARY = {aten.sigmoid.default: '(1.0f / (1.0f + exp(-({0}))))', aten.tanh.default: 'tanh({0})',
         aten.exp.default: 'exp({0})', aten.neg.default: '(-({0}))', aten.relu.default: 'max({0}, 0.0f)',
         aten._to_copy.default: '({0})', aten.clone.default: '({0})'}
BINARY = {aten.mul.Tensor: '({0}) * ({1})', aten.div.Tensor: '({0}) / ({1})', aten.add.Tensor: '({0}) + {2} * ({1})',
          aten.sub.Tensor: '({0}) - {2} * ({1})', aten.add_.Tensor: '({0}) + {2} * ({1})',
          aten.mul_.Tensor: '({0}) * ({1})'}
INPLACE = {aten.copy_.default, aten.add_.Tensor, aten.mul_.Tensor, aten.index_copy_.default}
_EVENTS = itertools.count(1)
READ, WRITE = 1, 2


class Root:
    """a storage values lie in: a caller tensor's (pinned at its own storage) or one of the run's (placed in the
    arena); coherent where a collective step reads or writes it (its kernels' loads and stores system-coherent). A
    version of a caller's storage (`base` the caller's root, `prev` the version before) is one of the run's: a write
    after another node's read of the storage starts one, so a landing never waits for the readers of the bytes it
    replaces (design/page-maps-2026-11-02.md: renaming; docs/facts.md: the LSTM's link racing a peer's SEND)."""
    def __init__(self, nbytes, tensor=None, base=None, prev=None):
        self.nbytes, self.tensor, self.coherent = nbytes, tensor, False
        self.base, self.prev, self.covered, self.readers = base, prev, [], set()


class Value:
    """a strided view of a root: element type, shape, strides and element offset (torch's)"""
    def __init__(self, root, dtype, shape, stride, offset):
        self.root, self.dtype, self.shape, self.stride, self.offset = root, dtype, tuple(shape), tuple(stride), offset

    def span(self):
        """the bytes [first, end) of the root this value reaches"""
        width = self.dtype.itemsize
        if any(n == 0 for n in self.shape):
            return self.offset * width, self.offset * width
        last = self.offset + sum((n - 1) * s for n, s in zip(self.shape, self.stride) if s > 0)
        first = self.offset + sum((n - 1) * s for n, s in zip(self.shape, self.stride) if s < 0)
        return first * width, (last + 1) * width

    def view(self, index):
        return kernels.View(index, self.dtype, self.shape, self.stride, self.offset, self.root.coherent)

    def runs(self):
        """the byte runs [first, end) of its root this value covers exactly, merged"""
        width, out = self.dtype.itemsize, []
        dims = [(n, s) for n, s in zip(self.shape, self.stride) if n != 1]
        if any(n == 0 for n, _ in dims):
            return []
        inner = dims[-1] if dims and dims[-1][1] == 1 else None
        outer = dims[:-1] if inner else dims
        length = (inner[0] if inner else 1) * width
        for at in itertools.product(*(range(n) for n, _ in outer)):
            first = (self.offset + sum(c * s for c, (_, s) in zip(at, outer))) * width
            out.append((first, first + length))
        out.sort()
        merged = []
        for a, b in out:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        return merged

    def on(self, root):
        return Value(root, self.dtype, self.shape, self.stride, self.offset)


class Command:
    """a kernel over values: its writes and reads (for the solve) and how its text is made (`make(views)`)"""
    def __init__(self, writes, reads, make):
        self.writes, self.reads, self.make = writes, reads, make


class Collective:
    """a declared all-gather: its calls' frames (each a contiguous run of the gathered axis in `out`'s root), each a
    rank's segment sizes; its steps once declared"""
    def __init__(self, out, frames, segments, dtype, group):
        self.out, self.frames, self.segments, self.dtype, self.group = out, frames, segments, dtype, group


def _caller(roots, current, tensor):
    """`tensor`'s value on its storage's current version (the caller's own root before any write renames it)"""
    storage = tensor.untyped_storage()
    root = roots.setdefault(storage.data_ptr(), Root(storage.nbytes(), tensor))
    return Value(current.get(id(root), root), tensor.dtype, tensor.shape, tensor.stride(), tensor.storage_offset())


def _meta(root, val):
    return Value(root, val.dtype, val.shape, val.stride(), val.storage_offset())


def _broadcast(value, shape):
    """`value`'s view over `shape` (torch's broadcast: leading axes added, size-1 axes stride 0)"""
    extra = len(shape) - len(value.shape)
    stride = (0,) * extra + tuple(0 if n == 1 and m != 1 else s for n, m, s in zip(value.shape, shape[extra:], value.stride))
    return Value(value.root, value.dtype, shape, stride, value.offset)


class Template:
    """a stepper captured, lowered and solved once"""

    def __init__(self, fn, args):
        self.args = args
        # traced through a variadic wrapper: make_fx counts a callable's parameters, defaults among them
        graph = make_fx(lambda *a: fn(*a), tracing_mode='fake', _allow_non_fake_inputs=True)(*args)
        self.roots, self.commands, self.collectives, self.order, self.current, self.node = {}, [], [], [], {}, None
        self._lower(graph)
        self._declare()
        self._solve()

    # ---- lower ----
    def _lower(self, graph):
        from mesh_graph import absorbed
        values, expression = {}, {}
        placeholders = iter(self.args)
        pointwise = set(UNARY) | set(BINARY) | {aten.copy_.default, aten.clamp.default}
        same = lambda n, u: n.meta.get('val') is not None and u.meta.get('val') is not None and \
            tuple(n.meta['val'].shape) == tuple(u.meta['val'].shape)
        # a pointwise node its one pointwise consumer of its shape takes in (mesh_graph.absorbed)
        users = absorbed([n for n in graph.graph.nodes if n.target in pointwise and n.target not in INPLACE and
                          n.target is not aten.copy_.default], lambda n: list(n.users),
                         lambda n, u: u.target in pointwise and same(n, u))
        for node in graph.graph.nodes:
            if node.target not in pointwise:
                # an expression this node consumes but cannot inline is materialized first
                for arg in node.all_input_nodes:
                    if arg in expression:
                        self._emit(*expression.pop(arg))
            self.node = node
            if node.op == 'placeholder':
                values[node] = _caller(self.roots, self.current, next(placeholders))
            elif node.op == 'get_attr':
                values[node] = _caller(self.roots, self.current, getattr(graph, node.target))
            elif node.op == 'output':
                pass
            elif node.target is operator.getitem:
                values[node] = values[node.args[0]][node.args[1]]
            elif node.target in VIEWS:
                base, val = values[node.args[0]], node.meta['val']
                values[node] = [_meta(base.root, v) for v in val] if isinstance(val, (list, tuple)) else _meta(base.root, val)
            elif node.target in pointwise:
                self._pointwise(node, values, expression, users)
            elif node.target in (aten.addmm.default, aten.mm.default, aten.linear.default):
                self._product(node, values)
            elif node.target in (aten.cat.default, aten.stack.default):
                self._joined(node, values)
            elif node.target in (aten.index_select.default, aten.index_copy_.default):
                self._indexed(node, values)
            elif node.target in (aten.sum.dim_IntList,):
                self._summed(node, values)
            elif str(node.target).startswith('torch_mesh.allgatherv_out'):
                self._gathered(node, values)
            else:
                raise NotImplementedError(f'no lowering of {node.target} (torch_mesh/compiler.py)')
        # an expression no consumer took is its command
        for node, (tree, out) in expression.items():
            self._emit(tree, out)
        # each renamed storage's last version copied back to the caller's own bytes, so a chunk ends where it began
        self.node = None
        for base in list(self.roots.values()):
            last = self.current.get(id(base))
            if last is not None and last is not base:
                self._complete(last)
                whole = Value(last, torch.uint8, (base.nbytes,), (1,), 0)
                last.readers.add(None)
                command = Command([whole.on(base)], [whole], None)
                command.make = lambda name, c=command: kernels.pointwise(
                    name, c.writes[0].view(0), [c.reads[0].view(1)], 'x0')
                self.commands.append(command)
                self.current[id(base)] = base

    def _complete(self, root):
        """a version's bytes no write covered, copied from the version before (before its first foreign read)"""
        if root.base is None or root.prev is None or getattr(root, 'whole', False):
            return
        root.whole = True
        at, gaps = 0, []
        for a, b in sorted(root.covered):
            if a > at:
                gaps.append((at, a))
            at = max(at, b)
        if at < root.nbytes:
            gaps.append((at, root.nbytes))
        for a, b in gaps:
            part = Value(root.prev, torch.uint8, (b - a,), (1,), a)
            self._copy(part, part.on(root))

    def _copy(self, source, into):
        self._command([into], [source], lambda name, c: kernels.pointwise(
            name, c.writes[0].view(0), [c.reads[0].view(1)], 'x0'))

    def _command(self, writes, reads, make, collective=None):
        """a command's reads and writes on their storages' versions: a read takes the current version (completed
        first); a write after another node's read of its storage starts a version of its own"""
        node = self.node
        # writes first: a version this command writes is covered by its writes before its own reads complete it
        bound_writes = []
        for value in writes:
            root = value.root
            if root.base is not None or root.tensor is not None:
                base = root.base if root.base is not None else root
                current = self.current.get(id(base), base)
                if current.readers - {node}:
                    current = Root(base.nbytes, base=base, prev=current)
                    current.coherent = base.coherent
                    self.current[id(base)] = current
                current.covered.extend(value.runs())
                value = value.on(current)
            bound_writes.append(value)
        bound_reads = []
        for value in reads:
            # a read is of the version its value was made on (the current one when its node ran), completed first
            root = value.root
            if root.base is not None and node not in root.readers:
                self._complete(root)
            root.readers.add(node)
            bound_reads.append(value)
        if collective is not None:
            collective.mine, collective.out = bound_reads[0], bound_writes[0]
            self.commands.append(collective)
            return bound_writes
        command = Command(bound_writes, bound_reads, None)
        command.make = lambda name, c=command: make(name, c)
        self.commands.append(command)
        return bound_writes

    def _new(self, node):
        val = node.meta['val']
        return _meta(Root(val.untyped_storage().nbytes()), val)

    def _pointwise(self, node, values, expression, users):
        """a pointwise node as an expression tree; one an equal-shaped, unaliased, single consumer takes is inlined"""
        def operand(arg, shape):
            if isinstance(arg, torch.fx.Node):
                if arg in expression and arg in users and expression[arg][1].shape == tuple(shape):
                    tree, _ = expression.pop(arg)
                    return tree
                if arg in expression:
                    self._emit(*expression.pop(arg))
                return ('load', _broadcast(values[arg], tuple(shape)))
            return ('const', float(arg))
        target, args = node.target, node.args
        if target is aten.copy_.default:
            out = values[args[0]]
            tree = operand(args[1], out.shape)
        else:
            inplace = target in INPLACE
            out = values[args[0]] if inplace else self._new(node)
            shape = out.shape
            if target in UNARY:
                tree = (UNARY[target], operand(args[0], shape))
            elif target is aten.clamp.default:
                low = args[1] if len(args) > 1 and args[1] is not None else node.kwargs.get('min')
                high = args[2] if len(args) > 2 else node.kwargs.get('max')
                tree = ('clamp', operand(args[0], shape), low, high)
            else:
                alpha = node.kwargs.get('alpha', 1)
                tree = (BINARY[target], operand(args[0], shape), operand(args[1], shape), ('const', float(alpha)))
        if target in INPLACE or target is aten.copy_.default:
            values[node] = self._emit(tree, out)
        else:
            values[node] = out
            expression[node] = (tree, out)

    def _emit(self, tree, out):
        loads = []

        def render(t):
            if t[0] == 'load':
                loads.append(t[1])
                return f'x{len(loads) - 1}'
            if t[0] == 'const':
                return f'{t[1]}f'
            if t[0] == 'clamp':
                inner = render(t[1])
                low = f'{float(t[2])}f' if t[2] is not None else '-INFINITY'
                high = f'{float(t[3])}f' if t[3] is not None else 'INFINITY'
                return f'clamp({inner}, {low}, {high})'
            return t[0].format(*(render(a) for a in t[1:]))
        text = render(tree)
        return self._command([out], list(loads), lambda name, c, x=text: kernels.pointwise(
            name, c.writes[0].view(0), [v.view(k + 1) for k, v in enumerate(c.reads)], x))[0]

    def _product(self, node, values):
        out = self._new(node)
        if node.target is aten.addmm.default:
            bias, a, b = (values[x] for x in node.args[:3])
            beta, alpha = node.kwargs.get('beta', 1), node.kwargs.get('alpha', 1)
            reads = [a, b, _broadcast(bias, out.shape)]
        elif node.target is aten.linear.default:
            # x W^T (+ bias): W [N, K] read as its transpose [K, N]
            a, w = values[node.args[0]], values[node.args[1]]
            if len(a.shape) != 2:
                raise NotImplementedError('a linear of an input of rank other than 2')
            b = Value(w.root, w.dtype, (w.shape[1], w.shape[0]), (w.stride[1], w.stride[0]), w.offset)
            bias = node.args[2] if len(node.args) > 2 else node.kwargs.get('bias')
            beta, alpha = (1 if bias is not None else 0), 1
            reads = [a, b] + ([_broadcast(values[bias], out.shape)] if bias is not None else [])
        else:
            a, b = values[node.args[0]], values[node.args[1]]
            beta, alpha, reads = 0, 1, [a, b]
        values[node] = self._command([out], reads, lambda name, c: kernels.product(
            name, c.writes[0].view(0), c.reads[0].view(1), c.reads[1].view(2), c.reads[2].view(3) if len(c.reads) > 2 else None,
            float(alpha), float(beta)))[0]

    def _joined(self, node, values):
        """cat and stack: each operand copied into its part of a new value"""
        out = self._new(node)
        values[node] = out
        parts, axis = node.args[0], node.args[1] if len(node.args) > 1 else 0
        axis %= len(out.shape)
        at = 0
        for part in parts:
            value = values[part]
            if node.target is aten.stack.default:
                shape = out.shape[:axis] + out.shape[axis + 1:]
                stride = out.stride[:axis] + out.stride[axis + 1:]
                into = Value(out.root, out.dtype, shape, stride, out.offset + at * out.stride[axis])
                at += 1
            else:
                shape = out.shape[:axis] + (value.shape[axis],) + out.shape[axis + 1:]
                into = Value(out.root, out.dtype, shape, out.stride, out.offset + at * out.stride[axis])
                at += value.shape[axis]
            self._emit(('load', value), into)

    def _indexed(self, node, values):
        if node.target is aten.index_select.default:
            source, axis, index = values[node.args[0]], node.args[1], values[node.args[2]]
            out = self._new(node)
            values[node] = self._command([out], [source, index], lambda name, c: kernels.indexed(
                name, c.writes[0].view(0), c.reads[0].view(1), c.reads[1].view(2), axis % len(c.writes[0].shape), False))[0]
        else:
            out, axis, index, source = values[node.args[0]], node.args[1], values[node.args[2]], values[node.args[3]]
            values[node] = self._command([out], [source, index], lambda name, c: kernels.indexed(
                name, c.writes[0].view(0), c.reads[0].view(1), c.reads[1].view(2), axis % len(c.writes[0].shape), True))[0]

    def _summed(self, node, values):
        source, axes = values[node.args[0]], [a % len(values[node.args[0]].shape) for a in node.args[1]]
        out = self._new(node)
        values[node] = self._command([out], [source], lambda name, c: kernels.summed(
            name, c.writes[0].view(0), c.reads[0].view(1), axes))[0]

    def _gathered(self, node, values):
        """allgatherv_out(mine, out, axis, sizes, group): a call a frame of out's gathered axis (the axes before it),
        each a contiguous run of its root (out's later axes row-major)"""
        mine, out, axis, sizes, group = values[node.args[0]], values[node.args[1]], node.args[2], node.args[3], node.args[4]
        axis %= len(out.shape)
        inner = math.prod(out.shape[axis + 1:])
        dense = 1
        for d in reversed(range(axis, len(out.shape))):
            if out.shape[d] != 1 and out.stride[d] != dense:
                raise NotImplementedError('an all-gather whose gathered axis and later ones are not row-major')
            dense *= out.shape[d]
        frames = [out.offset + sum(c * s for c, s in zip(at, out.stride[:axis]))
                  for at in itertools.product(*(range(n) for n in out.shape[:axis]))]
        collective = Collective(out, frames, [s * inner for s in sizes], out.dtype, group)
        self._command([out], [mine], None, collective)
        for root in (collective.out.root, collective.mine.root):
            root.coherent = True
            if root.base is not None:
                root.base.coherent = True
        self.collectives.append(collective)
        values[node] = None

    # ---- declare ----
    def _declare(self):
        import mesh
        from .backend import TYPES
        ffi, lib = mesh.ffi, mesh.lib
        for c in self.collectives:
            c.comm = ffi.cast('ncclComm_t', _stream.communicator_of(c.group))
            calls = ffi.new('ncclMeshCollective[]', len(c.frames))
            segments = ffi.new('uint64_t[]', c.segments)
            for k in range(len(c.frames)):
                calls[k].what, calls[k].type, calls[k].op, calls[k].root = lib.MESH_ALLGATHER, TYPES[c.dtype], 0, 0
                calls[k].elements, calls[k].segments = sum(c.segments), segments
            definition = ffi.new('uint64_t *')
            mesh.check(lib.ncclMeshCommDeclare(c.comm, calls, len(c.frames), definition))
            c.definition = definition[0]
            count = ffi.new('size_t *')
            mesh.check(lib.ncclMeshCommSteps(c.comm, c.definition, ffi.NULL, 0, count))
            steps = ffi.new('ncclMeshTransfer[]', max(1, count[0]))
            mesh.check(lib.ncclMeshCommSteps(c.comm, c.definition, steps, count[0], count))
            width = c.dtype.itemsize
            c.steps = [dict(call=s.call, channel=s.channel, op=s.op, round=s.round, buffer=s.buffer, first=s.first,
                            bytes=s.bytes) for s in steps[0:count[0]]]
            # a step's bytes: buffer 0 its frame's in out's root; a scratch buffer a run root of its own per call
            c.scratch = {}
            for s in c.steps:
                if s['buffer']:
                    key = (s['call'], s['buffer'])
                    c.scratch[key] = max(c.scratch.get(key, 0), s['first'] + s['bytes'])
            c.scratch = {key: Root(n) for key, n in c.scratch.items()}
            for root in c.scratch.values():
                root.coherent = True
            for s in c.steps:
                if s['buffer']:
                    s['root'], s['at'] = c.scratch[s['call'], s['buffer']], s['first']
                else:
                    s['root'], s['at'] = c.out.root, c.frames[s['call']] * width + s['first']
                s['kind'] = {lib.MESH_STEP_SEND: 'send', lib.MESH_STEP_COPY: 'copy', lib.MESH_STEP_REDUCE: 'reduce'}[s['op']]
                if s['kind'] == 'reduce':
                    raise NotImplementedError('a REDUCE step (an all-reduce or reduce-scatter): its fold is next')

    # ---- solve ----
    def _solve(self):
        """the run's events, in the declared order, and their accesses; the solve's edges and places"""
        events, accesses, links = [], [], []

        def event(kind, *key):
            events.append((kind, *key))
            return len(events) - 1
        items, item_of = [], {}

        def item(root):
            if id(root) not in item_of:
                item_of[id(root)] = len(items)
                items.append(root)
            return item_of[id(root)]

        def access(begin, end, mode, root, first, end_byte):
            if end_byte > first:
                accesses.append((begin, end, mode, item(root), first, end_byte - first))
        # every occurrence's post first: its landings go into versions of their own, so nothing before orders them
        # (docs/facts.md: a link after its tick's readers raced the peers' SENDs, which TN3205's credit then held)
        for c, command in enumerate(self.commands):
            if isinstance(command, Collective):
                command.post = event('post', c)
        for c, command in enumerate(self.commands):
            if isinstance(command, Collective):
                post = command.post
                for k, s in enumerate(command.steps):
                    if s['kind'] == 'send':
                        s['event'] = event('send', c, k)
                for k, s in enumerate(command.steps):
                    if s['kind'] == 'copy':
                        s['event'] = event('landed', c, k)
                        access(post, s['event'], WRITE, s['root'], s['at'], s['at'] + s['bytes'])
                        links.append((post, s['event']))
                for k, s in enumerate(command.steps):
                    if s['kind'] == 'send':
                        s['completion'] = event('sent', c, k)
                        access(s['event'], s['completion'], READ, s['root'], s['at'], s['at'] + s['bytes'])
                        links.append((s['event'], s['completion']))
                continue
            e = event('command', c)
            for value in command.reads:
                access(e, e, READ, value.root, *value.span())
            for value in command.writes:
                access(e, e, WRITE, value.root, *value.span())
        page = _stream.region_page_bytes()
        pinned = [root for root in items if root.tensor is not None]
        bins, first = [], 0
        pins = {}
        for root in pinned:
            pages = max(1, -(-root.nbytes // page))
            bins.append((first, pages))
            pins[id(root)] = first
            first += pages
        own = sum(max(1, -(-root.nbytes // page)) for root in items if root.tensor is None)
        bins.append((first, max(1, own)))
        rows = [(max(1, -(-root.nbytes // page)), pins.get(id(root), _stream.RUN_FREE),
                 (_stream.RUN_BEFORE | _stream.RUN_AFTER) if root.tensor is not None else 0, 1) for root in items]
        solved = _stream.run_solve(page, len(events), accesses, links, rows, bins)
        self.events, self.items, self.page = events, items, page
        self.edges = sorted({(f, t) for f, t, *_ in solved['edges'] if f != t})
        # each occurrence's forwards: a SEND reading bytes a COPY of the same occurrence landed (its RAW edge)
        from mesh_graph import joined
        self.forwards = joined(self.edges, lambda e: (events[e] + (None, None))[:3], 'landed', 'send')
        self.order = solved['order']
        arena_first = bins[-1][0]
        self.arena_pages = solved['arena']
        for root, runs in zip(items, solved['runs']):
            if root.tensor is None:
                (place, _), = runs
                root.at = (place - arena_first) * page
        # each command's kernel, named by nothing but its text, so a tick's kernels compile once for every tick
        for command in self.commands:
            if isinstance(command, Collective):
                continue
            source, grid, group = command.make('mesh_command')
            command.grid, command.group = grid, group
            command.pipeline = _pipeline(source, 'mesh_command')


_PIPELINES = {}


def _pipeline(source, name):
    if (source, name) not in _PIPELINES:
        _PIPELINES[source, name] = _stream.pipeline(source, name)
    return _PIPELINES[source, name]


def _whole(tensor):
    """a uint8 tensor over the whole storage of `tensor`, from its first byte"""
    storage = tensor.untyped_storage()
    return torch.empty((0,), dtype=torch.uint8, device=tensor.device).set_(storage, 0, (storage.nbytes(),))


class Program:
    """steppers run in order as runs: each `(fn, args)` a chunk, one Template a distinct fn object; prepare(n) makes n
    runs, each enqueued once by forward()"""

    def __init__(self, chunks):
        from . import empty
        self.chunks, templates = chunks, {}
        for fn, args in chunks:
            if id(fn) not in templates:
                templates[id(fn)] = Template(fn, args)
        self.templates = templates
        page = next(iter(templates.values())).page if templates else _stream.region_page_bytes()
        self.arena = empty((max(1, max((t.arena_pages for t in templates.values()), default=1)) * page,),
                           dtype=torch.uint8)
        self.runs, self.done = [], []

    def _binding(self, root):
        return (_whole(root.tensor), 0) if root.tensor is not None else (self.arena, root.at)

    def _address(self, root, at):
        tensor, base = self._binding(root)
        start = _stream.address(tensor)
        if not start:
            raise ValueError(f'a collective step\'s bytes lie outside the region: {"a caller tensor" if root.tensor is not None else "the arena"}'
                             f' of {root.nbytes} bytes ({tuple(tensor.shape)} {tensor.dtype} on {tensor.device})')
        return start + base + at

    def prepare(self, count=1):
        for _ in range(count):
            self.runs.append(self._run())

    def _run(self):
        import mesh
        from . import empty
        ffi, lib = mesh.ffi, mesh.lib
        run = _stream.Run()
        store = _pipeline(kernels.store(), kernels.STORE)
        # pass 1: each chunk's occurrences lowered onto their bytes (their channels' positions move each time), and
        # each word store's value (a link's table offset, a release's count) given its slot
        instances, stores = [], []
        for fn, _ in self.chunks:
            t = self.templates[id(fn)]
            lowered = {}
            for c, command in enumerate(t.commands):
                if not isinstance(command, Collective):
                    continue
                addresses = ffi.new('void *[]', max(1, len(command.steps)))
                for k, s in enumerate(command.steps):
                    addresses[k] = ffi.cast('void *', self._address(s['root'], s['at']))
                room = 4 * len(command.steps) + 64
                nodes, count = ffi.new('ncclMeshNode[]', room), ffi.new('size_t *')
                pairs = t.forwards.get(c, [])
                forwards = ffi.new('uint32_t[][2]', max(1, len(pairs)))
                for p, (send, copy) in enumerate(pairs):
                    forwards[p][0], forwards[p][1] = send, copy
                mesh.check(lib.ncclMeshCommLowerOccurrence(command.comm, command.definition, addresses, ffi.NULL, 0,
                                                           forwards, len(pairs), nodes, room, count))
                made = []
                for n in nodes[0:count[0]]:
                    slot = None
                    if n.kind in (lib.ncclMeshNodeLink, lib.ncclMeshNodeRelease):
                        slot = len(stores)
                        stores.append(n.value)
                    made.append((n.kind, n.transfer, int(ffi.cast('uintptr_t', n.word)), n.value, slot))
                lowered[c] = made
                run.communicator(int(ffi.cast('uintptr_t', command.comm)))
            instances.append((t, lowered))
        values = empty((max(1, len(stores)),), dtype=torch.int64)
        values.copy_(torch.tensor(stores or [0], dtype=torch.int64).to(values.device))
        # pass 2: the run's nodes, the caller's work so far (torch's stream, the store values among it) its input
        start = next(_EVENTS)
        run.event(start)
        run.inputs()
        edges, previous_sinks = [], [start]

        def stored(slot, word):
            run.dispatch(store, [(0, word, 0), (1, values, 8 * slot)], (1, 1, 1), (1, 1, 1), True)
        for t, lowered in instances:
            ids = {e: next(_EVENTS) for e in range(len(t.events))}
            has_node = set()
            # a step's send, sent and landed events are its channel's node's: one id a (occurrence, kind, channel)
            # group, its node appended at the group's last event in the order
            group = {}
            for e in t.order:
                kind, c, *rest = t.events[e]
                if kind in ('send', 'sent', 'landed'):
                    group.setdefault((kind, c, t.commands[c].steps[rest[0]]['channel']), []).append(e)
            last = {}
            for key, members in group.items():
                for e in members:
                    ids[e] = ids[members[0]]
                last[members[-1]] = key
            for e in t.order:
                kind, c, *rest = t.events[e]
                command = t.commands[c]
                if kind == 'command':
                    run.event(ids[e])
                    operands = []
                    for index, value in enumerate([*command.writes[:1], *command.reads]):
                        tensor, base = self._binding(value.root)
                        operands.append((index, tensor, base))
                    run.dispatch(command.pipeline, operands, command.grid, command.group, True)
                    has_node.add(e)
                elif kind == 'post':
                    links = [n for n in lowered[c] if n[0] == lib.ncclMeshNodeLink]
                    if links:
                        run.event(ids[e])
                        for _, _, word, _, slot in links:
                            stored(slot, word)
                        has_node.add(e)
                elif e in last:
                    kind, c, channel = last[e]
                    wanted = {'send': lib.ncclMeshNodeRelease, 'sent': lib.ncclMeshNodeSent,
                              'landed': lib.ncclMeshNodeLanding}[kind]
                    mine = [n for n in lowered[c] if n[0] == wanted and n[1] == channel]
                    if mine:
                        run.event(ids[e])
                        for _, _, word, value, slot in mine:
                            if kind == 'send':
                                stored(slot, word)
                            else:
                                run.landed(word, value)
                        has_node.update(group[kind, c, channel])
            # the solve's edges between nodes, through events with none; the instance after the one before
            from mesh_graph import sinks, sources, through
            kept = [e for e in t.order if e in has_node]
            own = [(a, b) for a, b in through(t.edges, set(kept)) if ids[a] != ids[b]]
            edges += [(ids[a], ids[b]) for a, b in own] + [(p, ids[s]) for p in previous_sinks for s in sources(kept, own)]
            previous_sinks = [ids[e] for e in sinks(kept, own)] or previous_sinks
        run.edges(edges)
        run.finish()
        return run

    def forward(self):
        run = self.runs.pop(0)
        run.inputs_now()  # the caller's work so far on torch's stream (a repeat's reset among it) precedes the run
        run.enqueue()
        self.done.append(run)
        return run

    def completion(self):
        """the latest run's end (mesh_statistics_now's clock) once it has completed, else None"""
        return self.done[-1].end() if self.done else None
