# design/aot-compilation-change-request.md A1 to A4, A8 to A10 and appendix item 7;
# design/plan-to-bridge-change-request.md R1; Y. Xu et al., GSPMD, arXiv 2105.04663, 2021
import collections
import ctypes as C
import errno
import json
import math
import operator
import os
import subprocess
import time
import weakref

import torch
import torch.utils._pytree as pytree
from torch._export.passes.remove_runtime_assertions import _RemoveRuntimeAssertionsPass
from torch._library._out_variant import _is_functional, get_out_arg_names, to_out_variant
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.node import Node, map_arg

import mesh
import mesh_graph

# torch 2.14.0 torch/export/__init__.py, export; torch/_ops.py, OpOverload
A = torch.ops.aten
# Y. Xu et al., GSPMD, arXiv 2105.04663, 2021, section 3.5
ELEMENTWISE = {A.add, A.sub, A.mul, A.div, A.neg, A.relu, A.tanh, A.sigmoid, A.exp, A.gelu, A.silu, A.copy_}
# GSPMD sections 3.2 and 3.5
DOT = {A.matmul, A.mm, A.bmm}
# GSPMD section 4.3
CONVOLUTION = {A.conv1d: 1, A.conv2d: 2, A.conv3d: 3}
TRANSPOSED = {A.conv_transpose1d: 1, A.conv_transpose2d: 2, A.conv_transpose3d: 3}
MAXIMUM = {A.max_pool1d: 1, A.max_pool2d: 2, A.max_pool3d: 3}
AVERAGE = {A.avg_pool1d: 1, A.avg_pool2d: 2, A.avg_pool3d: 3}
PAD = {A.pad, A.constant_pad_nd}
# design/aot-compilation-change-request.md A2; GSPMD section 3.1; L. Zheng et al., Alpa, OSDI 2022, section 4.1;
# mesh-dataflow.h mesh_nodes_read
Placement = collections.namedtuple('Placement', 'mesh nodes splits')
# GSPMD sections 2.1, 3.2, 4.2 and 4.5
Collective = collections.namedtuple('Collective', 'kind node dims dim position')
# design/aot-compilation-change-request.md A10, "Its output": "the order constraints between calls, which the program
# keeps when it submits"; M. C. Rinard and P. C. Diniz, ACM TOPLAS 19(6), 1997, section 2; mesh-collective.h mesh_plan:
# each function's outstanding calls, `norders` pairs (a, b) of call indexes in `orders`, and `chains[j]` the first call
# of call j's commuting chain
Calls = collections.namedtuple('Calls', 'outstanding orders causes norders chains')
# GSPMD section 4.1, "Static operator configurations"
Local = collections.namedtuple('Local', 'op target args kwargs')
# GSPMD sections 3.2, 3.5, 4 and 4.6
Rule = collections.namedtuple('Rule', 'kind level operands labels out propagate contracting windows fused merged '
                                      'configure')
# GSPMD sections 4.1 and 4.3, Appendix A.2
Window = collections.namedtuple('Window', 'label operand dim extent stride low transposed fill')


# design/aot-compilation-change-request.md A2 to A4, A8, A10 and appendix item 7;
# design/plan-to-bridge-change-request.md R3 and R4; M. Cowan et al., MSCCLang, ASPLOS 2023, sections 4 and 5; GSPMD
# section 3.1; NCCL User Guide, Collective Operations
class IR:
    def __init__(self, gm, placement, nodes):
        self.gm, self.placement, self.weights = gm, placement, None
        self.order = list(gm.graph.nodes)
        self.shape, self.coords, self.map = _mesh(placement, nodes.count)
        here = [d for d, index in enumerate(self.map) if index == nodes.self]
        self.device, self.index, self.serial, self.identity = here[0] if here else None, nodes.self, nodes.serial, 0
        self.sharding, self.program, self.roles, self.events = {}, [], {}, []
        self.declared, self.dependences, self.values, self.layouts = [], [], [], []
        self.made, self.locals, self.shapes = set(), {}, {}
        self.metas, self.forms, self.over, self.producer, self.places = {}, {}, {}, {}, {}
        self.homes, self.at, self.inside, self.added, self.owned = {}, {}, {}, {}, {}
        self.publishes, self.inputs, self.declares, self.waits = {}, {}, {}, {}
        self.attributes, self.loads, self.kinds, self.page, self.hull = {}, set(), {}, None, None
        self.taken, self.results, self.sent, self.contiguous = {}, {}, set(), set()
        self.given, self.landed = None, {}

    # GSPMD section 3.1
    def local(self, node, sharding):
        return tuple(s // self.shape[m] if m >= 0 else s for s, m in zip(_value(node).shape, sharding))

    # mesh-collective.h struct mesh_declared; design/plan-to-bridge-change-request.md R2 and R4;
    # design/aot-compilation-change-request.md, appendix item 5; design/compilation-composition-change-request.md item
    # 6, unit 6D: a destination is a device of this function's map, or with `at` a map index outside it (`at[d]`)
    def transfer(self, source, destinations, kind, part, landings, at=None):
        self.identity += 1
        transfers = {}
        for d in destinations:
            transfers[d] = len(self.declared)
            self.declared.append(mesh.Declared(self.map[source], self.map[d] if at is None else at[d], self.identity,
                                               kind, mesh.ABSENT, 0, part, landings[d]))
        return transfers

    # mesh-collective.h struct mesh_value; design/aot-compilation-change-request.md, appendix item 7; torch 2.14.0
    # include/c10/core/TensorImpl.h, `Storage storage_`
    def value(self, dtype, numel, device, wire, begin, end, kind=mesh.EXECUTION, lag=0):
        self.values.append(mesh.Value(section=mesh.Section(bytes=dtype.itemsize * numel), kind=kind, wire=wire,
                                      begin=begin, end=end, lag=lag, run=1))
        self.layouts.append((dtype, numel, device))
        return len(self.values) - 1


# J. Ansel et al., PyTorch 2, ASPLOS 2024, sections 3.5 and 5.1; design/aot-compilation-change-request.md A1 and
# appendix item 7; torch 2.14.0 torch/fx/experimental/proxy_tensor.py set_meta; torch/_subclasses/fake_tensor.py
# FakeTensor
def _value(node):
    value = node.meta.get('val')
    if not isinstance(value, torch.Tensor):
        raise NotImplementedError('%s (%s): no tensor value (torch.export: its meta[\'val\'])'
                                  % (node.name, node.target))
    if not all(isinstance(size, int) for size in value.shape):
        raise ValueError('%s: symbolic sizes %s; A1 compiles shapes, so nothing at run time changes one'
                         % (node.name, tuple(value.shape)))
    if value.device.type not in ('cpu', 'mps'):
        raise NotImplementedError('%s on %s: no binding of its tensors to the placed pages (a DLPack tensor of kDLCPU '
                                  'or kDLMetal alone: K1)' % (node.name, value.device))
    return value


# GSPMD section 3.1
def _shape(ir, key):
    return ir.shapes[key] if key in ir.shapes else ir.local(key[0], key[1])


def _layout(ir, key):
    if key in ir.metas:
        return ir.metas[key].dtype, tuple(ir.metas[key].shape)
    return _value(key[0]).dtype, tuple(_shape(ir, key))


def _device(ir, key):
    return (key[0].meta['val'][key[3]] if key in ir.results else _value(key[0])).device.type


# GSPMD section 3.1; Alpa section 4.1; design/aot-compilation-change-request.md A10, "Its graph operations are the graph
# module's"
def _mesh(placement, count):
    def item(at):
        value = placement.mesh
        for i in at:
            value = value[i]
        return value

    def sequence(at):
        return isinstance(item(at), (list, tuple))
    first = mesh_graph.walk([()], lambda at: [at + (0,)] if sequence(at) and item(at) else [])
    if sequence(first[-1]):
        raise ValueError('the placement operand\'s mesh holds an empty sequence (A2)')
    shape, coords = [len(item(at)) for at in first[:-1]], {}
    for at in mesh_graph.walk([()], lambda at: [at + (i,) for i in range(len(item(at)))]
                              if len(at) < len(shape) and sequence(at) else []):
        if len(at) == len(shape):
            if not isinstance(item(at), int) or item(at) in coords:
                raise ValueError('the placement operand\'s mesh at %s: %r is no new device id (A2)' % (at, item(at)))
            coords[item(at)] = at
        elif not sequence(at) or len(item(at)) != shape[len(at)]:
            raise ValueError('the placement operand\'s mesh at %s is not of shape %s (A2)' % (at, shape))
    nodes = list(placement.nodes)
    if sorted(coords) != list(range(len(coords))) or len(nodes) != len(coords) or len(set(nodes)) != len(nodes) \
            or not all(isinstance(n, int) and 0 <= n < count for n in nodes):
        raise ValueError('the placement operand: devices %s onto map indices %s of the map\'s %d nodes; each device '
                         '0 .. D - 1 once, each onto its own index (A2)' % (sorted(coords), nodes, count))
    return tuple(shape), coords, nodes


# GSPMD section 3.5, "Merging compatible shardings"
def _merge(a, b):
    merged = tuple(x if y < 0 else y for x, y in zip(a, b))
    used = [m for m in merged if m >= 0]
    if any(0 <= x != y >= 0 for x, y in zip(a, b)) or len(used) != len(set(used)):
        return None
    return merged


# GSPMD section 3.5, "Preserved dimensions in operators"
def _carry(sharding, source, target):
    return tuple(sharding[source.index(label)] if label in source else -1 for label in target)


# torch 2.14.0 torch/_C/__init__.pyi, FunctionSchema `arguments`, Argument `name`
def _names(local):
    return tuple(argument.name for argument in local.target._schema.arguments)


def _ints(value, k):
    value = tuple(value) if isinstance(value, (list, tuple)) else (value,)
    return value * k if len(value) == 1 else value


def _given(local, names):
    return {**dict(zip(names, local.args)), **local.kwargs}


def _with(local, names, changes):
    args, kwargs = list(local.args), dict(local.kwargs)
    for name, value in changes.items():
        if names.index(name) < len(args):
            args[names.index(name)] = value
        else:
            kwargs[name] = value
    return local._replace(args=tuple(args), kwargs=kwargs)


def _named(local, names):
    found = []
    for name, value in list(zip(names + (None,) * len(local.args), local.args)) + list(local.kwargs.items()):
        if isinstance(value, Node):
            found.append(name)
        else:
            map_arg(value, lambda n: found.append(None))
    return found


# GSPMD sections 3.2 and 3.5
def _matmul(operands, shapes, out):
    batch = len(out) - (len(shapes[0]) > 1) - (len(shapes[1]) > 1)

    def side(s, k, matrix):
        rank = len(s) - len(matrix)
        return tuple(batch - rank + i if s[i] == out[batch - rank + i] else ('broadcast', k, i)
                     for i in range(rank)) + matrix
    labels = [side(shapes[0], 0, ('n', 'k') if len(shapes[0]) > 1 else ('k',)),
              side(shapes[1], 1, ('k', 'm') if len(shapes[1]) > 1 else ('k',))]
    through = tuple(k for k, own in enumerate(labels) if not any(isinstance(x, tuple) for x in own))
    return Rule('dot', 1, operands, labels, tuple(range(batch)) + ('n',) * (len(shapes[0]) > 1)
                + ('m',) * (len(shapes[1]) > 1), through, ('k',), (), False, None, lambda sharded: ())


# GSPMD section 3.2
def _einsum(local, operands, shapes, out):
    equation = local.args[0] if local.args and isinstance(local.args[0], str) else '.'
    left, arrow, right = equation.replace(' ', '').partition('->')
    terms = left.split(',')
    letters = ''.join(terms)
    right = right if arrow else ''.join(sorted(c for c in set(letters) if letters.count(c) == 1))
    sizes = {}
    for term, shape in zip(terms, shapes):
        for c, size in zip(term, shape):
            sizes.setdefault(c, set()).add(size)
    if '.' in equation or len(terms) != len(operands) or len(right) != len(out) or len(set(right)) != len(right) \
            or not set(right) <= set(letters) or any(len(v) != 1 for v in sizes.values()) \
            or any(len(t) != len(s) or len(set(t)) != len(t) for t, s in zip(terms, shapes)):
        return None
    contracting = tuple(sorted(set.intersection(*map(set, terms)) - set(right))) if len(terms) > 1 else ()
    return Rule('dot', 1, operands, [tuple(t) for t in terms], tuple(right), tuple(range(len(operands))), contracting,
                (), False, None, lambda sharded: ())


# GSPMD sections 4.1, 4.3 and 4.4, Appendix A.2
def _convolution(node, local, operands, shapes):
    packet = local.target.overloadpacket
    transposed = packet in TRANSPOSED
    k, names = (TRANSPOSED if transposed else CONVOLUTION)[packet], _names(local)
    named = _named(local, names)
    if None in named or len(set(named)) != len(named) or not {'input', 'weight'} <= set(named):
        return None
    given, x, w = _given(local, names), shapes[named.index('input')], shapes[named.index('weight')]
    lead, groups = len(x) - k - 1, given.get('groups', 1)
    if lead not in (0, 1) or len(w) != k + 2:
        return None
    stride, dilation = _ints(given.get('stride', 1), k), _ints(given.get('dilation', 1), k)
    reach = [d * (n - 1) for d, n in zip(dilation, w[2:])]
    padding = given.get('padding', 0)
    lows = (0,) * k if padding == 'valid' else tuple(r // 2 for r in reach) if padding == 'same' else _ints(padding, k)
    windowed = padding != 'same' or all(r % 2 == 0 for r in reach)
    ci, co = ('ci', 'co') if groups == 1 else (('channel', 'in'), ('channel', 'out'))
    weight = ((ci, co) if transposed else (co, ci)) if groups == 1 else (('weight', 0), ('weight', 1))
    labels = {'input': ('n',) * lead + (ci,) + tuple(('s', i) if windowed else ('x', i) for i in range(k)),
              'weight': weight + tuple(('kernel', i) for i in range(k)), 'bias': (co,) if groups == 1 else ('bias',)}
    out = ('n',) * lead + (co,) + tuple(('s', i) if windowed else ('y', i) for i in range(k))
    at = named.index('input')
    windows = tuple(Window(('s', i), at, lead + 1 + i, reach[i] + 1, stride[i], None if transposed else lows[i],
                           (lows[i], reach[i]) if transposed else None, 0) for i in range(k)) if windowed else ()
    extra = _ints(given.get('output_padding', 0), k)

    def configure(sharded):
        if not sharded:
            return ()
        changes = {'padding': tuple(0 if ('s', i) in sharded else lows[i] for i in range(k))}
        if transposed:
            changes['output_padding'] = tuple(0 if ('s', i) in sharded else extra[i] for i in range(k))
        return ((node, _with(local, names, changes)),)
    return Rule('window', 1, operands, [labels[name] for name in named], out, tuple(range(len(operands))),
                ('ci',) if groups == 1 else (), windows, 'bias' in named and groups == 1, None, configure)


# GSPMD sections 4.1 and 4.3; native_functions.yaml
def _pooling(node, local, operands, shapes):
    packet = local.target.overloadpacket
    maximum = packet in MAXIMUM
    k, names = (MAXIMUM if maximum else AVERAGE)[packet], _names(local)
    given, lead = _given(local, names), len(shapes[0]) - k
    if _named(local, names) != ['self'] or lead not in (1, 2):
        return None
    kernel, lows = _ints(given['kernel_size'], k), _ints(given.get('padding', 0), k)
    stride, dilation = _ints(given.get('stride') or given['kernel_size'], k), _ints(given.get('dilation', 1), k)
    dtype = _value(node).dtype
    fill = (-math.inf if dtype.is_floating_point else torch.iinfo(dtype).min) if maximum else 0
    windowed = [maximum or given.get('divisor_override') is not None
                or not given.get('ceil_mode', False) and (given.get('count_include_pad', True) or lows[i] == 0)
                for i in range(k)]
    windows = tuple(Window(('s', i), 0, lead + i, dilation[i] * (kernel[i] - 1) + 1 if maximum else kernel[i],
                           stride[i], lows[i], None, fill) for i in range(k) if windowed[i])

    def configure(sharded):
        changes = {'padding': tuple(0 if ('s', i) in sharded else lows[i] for i in range(k))}
        return ((node, _with(local, names, changes)),) if sharded else ()
    return Rule('window', 1, operands, [tuple(range(lead)) + tuple(('s', i) if windowed[i] else ('x', i)
                                                                   for i in range(k))],
                tuple(range(lead)) + tuple(('s', i) if windowed[i] else ('y', i) for i in range(k)), (0,), (), windows,
                False, None, configure)


# GSPMD section 4.3, Figure 5b; native_functions.yaml; Python documentation, slice.indices
def _pad(node, local, operands, shapes):
    names = _names(local)
    given, r = _given(local, names), len(shapes[0])
    pads = list(given['pad'])
    if _named(local, names) != ['self'] or len(pads) % 2 or len(pads) > 2 * r \
            or not all(type(p) is int for p in pads):
        return None
    padded = {r - 1 - j: (pads[2 * j], pads[2 * j + 1]) for j in range(len(pads) // 2)
              if pads[2 * j: 2 * j + 2] != [0, 0]}
    constant, value = given.get('mode', 'constant') == 'constant', given.get('value') or 0

    def configure(sharded):
        changes = {'pad': tuple(0 if r - 1 - j // 2 in sharded else p for j, p in enumerate(pads))}
        return ((node, _with(local, names, changes)),) if sharded else ()
    return Rule('pad' if constant else 'format', 1, operands,
                [tuple(d if constant or d not in padded else ('x', d) for d in range(r))],
                tuple(d if constant or d not in padded else ('y', d) for d in range(r)), (0,), (),
                tuple(Window(d, 0, d, 1, 1, low, None, value) for d, (low, _) in padded.items()) if constant else (),
                False, None, configure)


def _slice(node, local, operands, shapes):
    packet, names, x = local.target.overloadpacket, _names(local), shapes[0]
    given, r = _given(local, names), len(x)
    dim = given.get('dim', 0)
    if _named(local, names) != ['self'] or type(dim) is not int:
        return None
    if packet is A.unsqueeze:
        d = dim % (r + 1)
        return Rule('slice', 1, operands, [tuple(range(r))], tuple(range(d)) + (('new', d),) + tuple(range(d, r)),
                    (0,), (), (), False, None, lambda sharded: ())
    dim %= r
    if packet is A.select:
        if type(given.get('index')) is not int:
            return None
        return Rule('slice', 1, operands, [tuple(('x', d) if d == dim else d for d in range(r))],
                    tuple(d for d in range(r) if d != dim), (0,), (), (), False, None, lambda sharded: ())
    if not all(v is None or type(v) is int for v in (given.get('start'), given.get('end'), given.get('step', 1))):
        return None
    start, stop, step = slice(given.get('start'), given.get('end'), given.get('step', 1)).indices(x[dim])
    windows = () if (start, stop, step) == (0, x[dim], 1) else (Window(dim, 0, dim, 1, step, -start, None, 0),)

    def configure(sharded):
        return ((node, _with(local, names, {'start': None, 'end': None})),) if dim in sharded else ()
    return Rule('slice', 1, operands, [tuple(range(r))], tuple(range(r)), (0,), (), windows, False, None, configure)


# design/aot-compilation-change-request.md A2 and A4; GSPMD sections 3.1, 3.2, 3.5, 4.3 to 4.6 and Appendix A.2; torch
# 2.14.0 torch/_export/verifier.py
def _operator(node, local):
    operands = []
    map_arg((local.args, local.kwargs), operands.append)
    general = Rule('general', None, operands, None, None, (), (), (), False, None, lambda sharded: ())
    if not isinstance(local.target, torch._ops.OpOverload) or not isinstance(node.meta.get('val'), torch.Tensor) \
            or not all(isinstance(o.meta.get('val'), torch.Tensor) for o in operands):
        return general
    out, shapes = tuple(_value(node).shape), [tuple(_value(o).shape) for o in operands]
    packet, rule = local.target.overloadpacket, None
    if packet in ELEMENTWISE and all(len(s) <= len(out) for s in shapes):
        rule = Rule('elementwise', 0, operands,
                    [tuple(len(out) - len(s) + i if size == out[len(out) - len(s) + i] else ('broadcast', k, i)
                           for i, size in enumerate(s)) for k, s in enumerate(shapes)],
                    tuple(range(len(out))), tuple(k for k, s in enumerate(shapes) if s == out), (), (), False, None,
                    general.configure)
    elif packet is A.linear and len(operands) in (2, 3) and shapes[0] and len(shapes[1]) == 2 \
            and _named(local, _names(local)) == ['input', 'weight', 'bias'][:len(operands)]:
        r = len(shapes[0])
        rule = Rule('dot', 1, operands, [tuple(range(r - 1)) + ('k',), ('m', 'k'), ('m',)][:len(operands)],
                    tuple(range(r - 1)) + ('m',), tuple(range(len(operands))), ('k',), (), len(operands) == 3, None,
                    general.configure)
    elif packet in DOT and len(operands) == 2 and shapes[0] and shapes[1]:
        rule = _matmul(operands, shapes, out)
    elif packet is A.einsum:
        rule = _einsum(local, operands, shapes, out)
    elif packet in CONVOLUTION or packet in TRANSPOSED:
        rule = _convolution(node, local, operands, shapes)
    elif packet in MAXIMUM or packet in AVERAGE:
        rule = _pooling(node, local, operands, shapes)
    elif packet in PAD:
        rule = _pad(node, local, operands, shapes)
    elif packet in (A.slice, A.select, A.unsqueeze):
        rule = _slice(node, local, operands, shapes)
    return rule or general


# GSPMD sections 3.3 and 4.6, "Pre-processing"
def _merged(pad, padding, slicing):
    pads, fill = {w.label: w for w in padding.windows}, padding.windows[0].fill if padding.windows else 0
    windows = [w._replace(low=w.low + (pads.pop(w.label).low if w.label in pads else 0), fill=fill)
               for w in slicing.windows]
    windows += [w for label, w in pads.items() if label in slicing.out]

    def configure(sharded):
        return tuple(padding.configure(sharded)) + tuple(slicing.configure(sharded))
    return slicing._replace(operands=padding.operands, windows=tuple(windows), merged=pad, configure=configure)


# design/joint-compilation.md §3, row 1
def _complete(ir, operators, annotated):
    def refine(node, candidate):
        candidate = tuple(m if m < 0 or size and size % ir.shape[m] == 0 else -1
                          for m, size in zip(candidate, _value(node).shape))
        merged = _merge(ir.sharding[node], candidate)
        if node in annotated or merged is None or merged == ir.sharding[node]:
            return False
        ir.sharding[node] = merged
        return True

    def forward(node, k):
        rule = operators[node]
        return refine(node, _carry(ir.sharding[rule.operands[k]], rule.labels[k], rule.out))

    def backward(node, k):
        rule = operators[node]
        return refine(rule.operands[k], _carry(ir.sharding[node], rule.out, rule.labels[k]))
    for level in (0, 1):
        mesh_graph.propagate(ir.order, lambda node: operators[node].propagate if operators.get(node) is not None
                             and operators[node].level is not None and operators[node].level <= level else (),
                             forward, backward)


# GSPMD section 4.5; MPICH allgather_intra_ring.c and alltoall_intra_pairwise.c, MPIR_Localcopy;
# design/aot-compilation-change-request.md, appendix item 7: a mesh dimension the result does not shard is gathered;
# one it shards along another dimension moves there by an AllToAll once that dimension is free, a cycle of such moves
# broken by gathering one of its dimensions, which the final slices shard again
def _reshard(ir, node, have, want):
    def step(item):
        if item[2] not in ir.made:
            ir.made.add(item[2])
            ir.program.append(item)

    def gather(have, m):
        after, d, shape = tuple(-1 if s == m else s for s in have), have.index(m), ir.local(node, have)
        own = (node, after, 'own')
        ir.shapes[own] = shape
        ir.inside[own] = ((node, after), d, {e: ir.coords[e][m] * shape[d] for e in ir.coords})
        step(('copy', ((node, have),), own, None))
        step(('collective', (own,), (node, after), Collective('all-gather', node, (m,), d, len(ir.program))))
        return after

    def exchange(have, m):
        i, j = have.index(m), want.index(m)
        after, shape = tuple(-1 if k == i else m if k == j else s for k, s in enumerate(have)), ir.local(node, have)
        own = (node, after, 'own')
        ir.shapes[own] = tuple(n // ir.shape[m] if k == j else n for k, n in enumerate(shape))
        ir.inside[own] = ((node, after), i, {e: ir.coords[e][m] * shape[i] for e in ir.coords})
        step(('copy', ((node, have),), own, (j, m)))
        step(('collective', ((node, have),), (node, after),
              Collective('all-to-all', node, (m,), (i, j), len(ir.program))))
        return after
    if node.op in ('placeholder', 'get_attr'):
        have = (-1,) * len(have)
    for m in range(len(ir.shape)):
        if m in have and m not in want:
            have = gather(have, m)
    while True:
        moving = [m for m in range(len(ir.shape)) if m in have and m in want and have.index(m) != want.index(m)]
        if not moving:
            break
        free = [m for m in moving if have[want.index(m)] < 0]
        have = exchange(have, free[0]) if free else gather(have, have[want.index(moving[0])])
    for m in range(len(ir.shape)):
        if m in want and m not in have:
            after = tuple(m if k == want.index(m) else s for k, s in enumerate(have))
            step(('slice', ((node, have),), (node, after), (want.index(m), m)))
            have = after
    return node, have


# GSPMD Appendix A.2 and Figure 9b
def _extent(window, parts, n, o):
    if window.transposed is None:
        starts = [q * o * window.stride - window.low for q in range(parts)]
        length, after = (o - 1) * window.stride + window.extent, [0] * parts
    else:
        (padding, reach), s = window.transposed, window.stride
        starts = [(q * o + padding - reach) // s for q in range(parts)]
        after = [q * o + padding - a * s for q, a in enumerate(starts)]
        length = max([((q + 1) * o - 1 + padding) // s - a + 1 for q, a in enumerate(starts)]
                     + [-(-(f + o - 1 - reach) // s) + 1 for f in after] + [1])
    return starts, length, after, max([q * n - a for q, a in enumerate(starts)] + [0]), \
        max([a + length - (q + 1) * n for q, a in enumerate(starts)] + [0])


# GSPMD sections 4.1, 4.3 and 4.5, Appendix A.2, Figures 5a and 9b; design/aot-compilation-change-request.md, appendix
# item 7; design/plan-to-bridge-change-request.md R1; MPICH allgather_intra_ring.c, MPIR_Localcopy
def _region(ir, consumer, key, window, m, o, whole):
    node, shape = key[0], _shape(ir, key)
    dim, parts, size = window.dim, ir.shape[m], _value(key[0]).shape[window.dim]
    n = shape[dim]
    starts, length, after, left, right = _extent(window, parts, n, o)
    if whole:
        left, right = max([-a for a in starts] + [0]), max([a + length - size for a in starts] + [0])
    region, middle, first = (node, key[1], 'region', consumer, dim), (node, key[1], 'middle', consumer, dim), \
        len(ir.program)
    fill = window.fill if any(a < 0 or a + length > size for a in starts) else None
    ir.shapes[region] = tuple(length if i == dim else s for i, s in enumerate(shape))
    offsets = tuple(a - (0 if whole else q * n) + left for q, a in enumerate(starts))
    reads = [key]
    if left or right:
        ir.shapes[middle], reads = shape, [middle]
        ir.program.append(('copy', (key,), middle, None))
    for side, width in () if whole else (('left', left), ('right', right)):
        pieces = []
        for j in range(1, -(-width // n) + 1):
            part = min(n, width - (j - 1) * n)
            piece, arrival = (node, key[1], 'halo', consumer, dim, side, j), (node, key[1], 'arrival', consumer, dim,
                                                                               side, j)
            ir.shapes[piece] = ir.shapes[arrival] = tuple(part if i == dim else s for i, s in enumerate(shape))
            ir.program.append(('narrow', (middle,), piece, (dim, n - part if side == 'left' else 0, part)))
            pairs = tuple((p, p + j) if side == 'left' else (p + j, p) for p in range(parts - j))
            ir.program.append(('collective', (piece,), arrival,
                               Collective('collective-permute', node, (m,), pairs, len(ir.program))))
            pieces.append(arrival)
        reads = pieces[::-1] + reads if side == 'left' else reads + pieces
    if left or right:
        landed, at = region + ('landed',), left if whole else 0
        ir.shapes[landed] = tuple(left + n + right if i == dim else s for i, s in enumerate(shape))
        ir.added[landed] = (first, len(ir.program))
        for k in reads:
            ir.inside[k] = (landed, dim, dict.fromkeys(ir.coords, at))
            at += ir.shapes[k][dim]
        ir.inside[region] = (landed, dim, {e: offsets[ir.coords[e][m]] for e in ir.coords})
    ir.program.append(('window', tuple(reads), region, (dim, m, offsets, length, tuple(starts), size, fill)))
    return region, after


# GSPMD sections 2.1, 3, 3.2, 4, 4.2, 4.5 and 5.1, Figures 6 and 7; design/aot-compilation-change-request.md A4 and
# appendix item 7; XLA spmd_partitioner.cc, PartitionedHlo::ReshardAsWindowedInput
def _partition(ir, operators):
    inputs = 0
    for node in ir.order:
        if node.op in ('placeholder', 'get_attr'):
            sharding = ir.sharding[node]
            whole = None if sharding is None else (-1,) * len(sharding)
            ir.program.append(('input', (), (node, whole), inputs) if node.op == 'placeholder'
                              else ('constant', (), (node, whole), node.target))
            ir.made.add((node, whole))
            if sharding is not None:
                _reshard(ir, node, whole, sharding)
            inputs += node.op == 'placeholder'
        elif node.op == 'output':
            keys = {value: (value, None) if ir.sharding[value] is None
                    else _reshard(ir, value, ir.sharding[value], (-1,) * len(ir.sharding[value]))
                    for value in node.all_input_nodes}
            ir.program.append(('output', tuple(keys.values()), None, (node, keys)))
        elif operators.get(node) is not None:
            rule, sharding = operators[node], ir.sharding[node]
            if rule.labels is None:
                keys = [(o, None) if ir.sharding[o] is None
                        else _reshard(ir, o, ir.sharding[o], (-1,) * len(ir.sharding[o])) for o in rule.operands]
                whole = None if sharding is None else (-1,) * len(sharding)
                ir.program.append(('call', tuple(keys), (node, whole), node))
                if sharding is not None:
                    _reshard(ir, node, whole, sharding)
                continue
            out, labels = rule.out, rule.labels
            free = {label for label in out if not any(label in own for own in labels)}
            target = tuple(-1 if label in free else s for s, label in zip(sharding, out))
            used, partial, contracted = {m for m in target if m >= 0}, [], {}
            for label in rule.contracting:
                held = {ir.sharding[o][own.index(label)] for o, own in zip(rule.operands, labels) if label in own}
                (m,) = held if len(held) == 1 and not rule.fused and not partial else (-1,)
                if m >= 0:
                    partial.append(m)
                contracted[label] = m
            local = tuple(-1 if s in partial else s for s in target)
            wants = [list(_carry(local, out, own)) for own in labels]
            for label, m in contracted.items():
                for want, own in zip(wants, labels):
                    if label in own:
                        want[own.index(label)] = m
            whole = {w.label for w in rule.windows if local[out.index(w.label)] >= 0
                     and (rule.operands[w.operand].op in ('placeholder', 'get_attr')
                          or local[out.index(w.label)] not in ir.sharding[rule.operands[w.operand]])}
            for window in rule.windows:
                if window.label in whole:
                    wants[window.operand][window.dim] = -1
            keys = [_reshard(ir, o, ir.sharding[o], tuple(w)) for o, w in zip(rule.operands, wants)]
            sharded, cuts, dims = {}, [], {w.dim: w for w in rule.windows}
            for dim in range(len(_value(rule.operands[rule.windows[0].operand]).shape) if dims else 0):
                window = dims.get(dim)
                at = None if window is None else out.index(window.label)
                if window is not None and local[at] >= 0:
                    sharded[window.label] = ir.local(node, local)[at]
                    keys[window.operand], after = _region(ir, node, keys[window.operand], window, local[at],
                                                          sharded[window.label], window.label in whole)
                    if window.transposed is not None:
                        cuts.append((at, local[at], tuple(after), sharded[window.label]))
            for n, configured in rule.configure(sharded):
                ir.locals[n] = configured
            reads = tuple(keys)
            if rule.merged is not None:
                ir.program.append(('call', reads, (rule.merged, 'merged', node), rule.merged))
                reads = ((rule.merged, 'merged', node),)
            key = (node, local, 'partial') if partial else (node, local)
            made = [key + ('cut', j) for j in range(len(cuts))] + [key]
            ir.program.append(('call', reads, made[0], node))
            for j, (at, m, after, o) in enumerate(cuts):
                ir.program.append(('window', (made[j],), made[j + 1], (at, m, after, o, None, None, None)))
            if any(m in used for m in partial):
                (m,) = partial
                ir.program.append(('collective', (key,), (node, target),
                                   Collective('reduce-scatter', node, (m,), target.index(m), len(ir.program))))
            elif partial:
                ir.program.append(('collective', (key,), (node, target),
                                   Collective('all-reduce', node, tuple(partial), None, len(ir.program))))
            if target != sharding:
                _reshard(ir, node, target, sharding)


# design/aot-compilation-change-request.md A2 and A4; GSPMD sections 3, 3.1 and 4.6
def partition(gm, placement, nodes, page):
    if not isinstance(placement, Placement):
        raise TypeError('the placement function returned %r, not a Placement (A2)' % (placement,))
    ir, operators, annotated = IR(gm, placement, nodes), {}, set()
    ir.page = page
    for node in ir.order:
        if node.op == 'call_function':
            ir.locals[node] = Local(node.op, node.target, node.args, {k: v for k, v in node.kwargs.items() if k != 'device'}
                                    if getattr(node.target, 'overloadpacket', None) is A._to_copy else node.kwargs)
            operators[node] = _operator(node, ir.locals[node])
        if node.op != 'output':
            ir.sharding[node] = (-1,) * _value(node).dim() if isinstance(node.meta.get('val'), torch.Tensor) else None
    for node, rule in list(operators.items()):
        pad = rule.operands[0] if rule is not None and rule.kind == 'slice' else None
        if pad is not None and operators.get(pad) is not None and operators[pad].kind == 'pad' and len(pad.users) == 1:
            operators[node], operators[pad] = _merged(pad, operators[pad], rule), None
    names = {node.name: node for node in ir.order if node.op == 'placeholder'}
    for name, mapping in placement.splits.items():
        node, mapping = names.get(name), tuple(mapping)
        if node is None or ir.sharding[node] is None or len(mapping) != len(ir.sharding[node]) \
                or not all(isinstance(m, int) and -1 <= m < len(ir.shape) for m in mapping) \
                or len([m for m in mapping if m >= 0]) != len({m for m in mapping if m >= 0}):
            raise ValueError('splits[%r] = %s: no placeholder\'s dims_mapping over a mesh of %d dimensions, each at '
                             'most once (GSPMD section 3.1)' % (name, mapping, len(ir.shape)))
        ir.sharding[node] = tuple(m if m >= 0 and ir.shape[m] > 1 else -1 for m in mapping)
        annotated.add(node)
    # GSPMD sections 3.1, 3.5 and 4.5; torch 2.14.0 torch/export/exported_program.py, ExportedProgram;
    # design/problems-of-14b9a757-change-request.md, "Sources for the third stops", 8
    replicated = mesh_graph.forward(ir.order, lambda node: node.all_input_nodes,
                                    lambda node, operands: node.op == 'get_attr' or node.op == 'call_function'
                                    and bool(operands) and all(operands) and _aliases(ir.locals[node]))
    annotated |= {node for node, whole in replicated.items() if whole and ir.sharding[node] is not None}
    _complete(ir, operators, annotated)
    for node, sharding in ir.sharding.items():
        for size, m in zip(_value(node).shape if sharding is not None else (), sharding or ()):
            if m >= 0 and size % ir.shape[m]:
                raise NotImplementedError('%s: a dimension of %d over a mesh dimension of %d (GSPMD section 4.1 pads '
                                          'it, which the contract does not give)' % (node.name, size, ir.shape[m]))
    _partition(ir, operators)
    ir.producer = {item[2]: g for g, item in enumerate(ir.program) if item[0] != 'output'}
    items = {item[2]: item for item in ir.program if item[0] != 'output'}
    ir.sent = mesh_graph.reach([item[1][0] for item in ir.program if item[0] == 'collective'],
                               lambda key: items[key][1][:1] if _sliced(items[key]) else ())
    _forms(ir)
    # GSPMD section 4.5
    bases = mesh_graph.forward(list(items), lambda key: items[key][1],
                               lambda key, reads: _bases(ir, items[key], reads))
    ir.attributes = {key: items[key] for key, base in bases.items() if base is not None}
    read = {k for _, reads, key, _ in ir.program if key not in ir.attributes for k in reads if k in ir.attributes}
    ir.loads = {k for k in read if _sliced(ir.attributes[k])} \
        | {b for k in read if not _sliced(ir.attributes[k]) for b in bases[k]}
    _inside(ir)
    ir.places = {device: _places(ir, device) for device in ir.coords}
    _over(ir)
    _kinds(ir)
    return ir


# torch 2.14.0 torch/_export/verifier.py
def _call(local, args, kwargs):
    return local.target(*args, **kwargs)


# design/aot-compilation-change-request.md, appendix item 7; ExecuTorch, "Concepts"; PyTorch, "Meta device"; torch
# 2.14.0 torch/_library/_out_variant.py, to_out_variant, _is_functional, get_out_arg_names and
# _has_valid_out_variant_returns; torch/_inductor/ir.py; torchgen/native_function_generation.py; native_functions.yaml
UNKNOWN = object()


def _forms(ir):
    items = {item[2]: item for item in ir.program if item[0] != 'output'}
    mesh_graph.forward(list(items), lambda key: items[key][1], lambda key, operands: _formed(ir, items[key], operands))


def _meta(ir, key):
    if key in ir.shapes or len(key) == 2 and isinstance(key[1], tuple):
        return torch.empty(_shape(ir, key), dtype=_value(key[0]).dtype, device='meta')
    if len(key) == 2 and key[1] is None:
        return pytree.tree_map(lambda v: torch.empty_strided(v.shape, v.stride(), dtype=v.dtype, device='meta')
                               if isinstance(v, torch.Tensor) else v, key[0].meta.get('val'))
    return UNKNOWN


def _formed(ir, item, operands):
    kind, reads, key, operand = item
    if kind != 'call':
        return _meta(ir, key)
    local = ir.locals[operand]
    if local.target is operator.getitem and isinstance(ir.forms.get(reads[0]), tuple) \
            and len(ir.forms[reads[0]][1]) > 1:
        ir.taken[key] = ir.forms[reads[0]][1][local.args[1]][1]
    if any(o is UNKNOWN for o in operands):
        return _meta(ir, key)
    given = iter(operands)
    args, kwargs = map_arg((local.args, local.kwargs), lambda n: next(given))
    try:
        result = _call(local, args, kwargs)
    except (NotImplementedError, RuntimeError):
        return _meta(ir, key)
    if isinstance(result, torch.Tensor):
        ir.metas[key] = result
    schema = local.target._schema if isinstance(local.target, torch._ops.OpOverload) else None
    if schema is None or not _is_functional(schema) or any(r.alias_info is not None for r in schema.returns):
        return result
    results = tuple(result) if isinstance(result, (tuple, list)) else (result,)
    out = to_out_variant(local.target) if len(results) == len(schema.returns) \
        and all(isinstance(r, torch.Tensor) and r.numel() for r in results) else None
    names = get_out_arg_names(out) if out is not None else ()
    if len(names) != len(results):
        ir.forms[key] = 'functional'
        return result
    made = (key,) if len(results) == 1 else tuple(key + ('out', j) for j in range(len(results)))
    if len(results) > 1:
        for k, r in zip(made, results):
            ir.metas[k], ir.added[k], ir.results[k] = r, (ir.producer[key],) * 2, key
    ir.forms[key] = (out, tuple(zip(names, made)))
    return result


# design/aot-compilation-change-request.md, appendix item 7
def _viewed(item):
    return item[0] == 'window' and len(item[1]) == 1 and item[3][6] is None


def _sliced(item):
    return item[0] in ('slice', 'narrow') or _viewed(item)


# GSPMD section 4.5; torch 2.14.0 torch/export/exported_program.py, ExportedProgram
def _bases(ir, item, reads):
    kind, _, key, operand = item
    if kind == 'constant':
        return frozenset((key,)) if isinstance(key[0].meta.get('val'), torch.Tensor) else None
    if reads and None not in reads and (_sliced(item) or kind == 'call' and key in ir.metas
                                        and _aliases(ir.locals[operand])):
        return frozenset().union(*reads)
    return None


# design/aot-compilation-change-request.md, appendix item 7
def _over(ir):
    items, places = {item[2]: item for item in ir.program if item[0] != 'output'}, ir.places[min(ir.coords)]

    def rule(key, sets):
        kind, reads, _, operand = items[key]
        joined = frozenset().union(*sets)
        if key in ir.inside:
            return frozenset((ir.inside[key][0],)) | (joined if kind == 'window' else frozenset())
        if places[key] is not None and places[key].owner == key:
            return frozenset((key,))
        if _sliced(items[key]):
            return joined
        if kind == 'call' and isinstance(ir.forms.get(key), tuple):
            return frozenset(ir.inside[result][0] if result in ir.inside else result for _, result in ir.forms[key][1])
        return joined if kind == 'call' and ir.forms.get(key) is None else frozenset()
    ir.over = mesh_graph.forward(list(items), lambda key: items[key][1], rule)


# design/torch-source-lowering-change-request.md item 1; design/plan-to-bridge-change-request.md R4;
# design/aot-compilation-change-request.md A8; J. Gibbons, The third homomorphism theorem, J. Functional Programming,
# 1996; L. Meertens, Algorithmics, CWI Monographs 1, 1986; G. Graefe, Query evaluation techniques for large databases,
# ACM Comput. Surv. 25(2), 1993; PyTorch 2.14 documentation, Broadcasting semantics
COPIES = {A.clone, A._to_copy, A._unsafe_view}


def _whole(ir, places, items, key):
    place = places.get(key)
    if place is None:
        return items[key][0] in ('input', 'call') and len(key) == 2 \
            and isinstance(key[0].meta.get('val'), torch.Tensor) and key[0].meta['val'].is_contiguous()
    shape = _layout(ir, place.owner)[1]
    return place.offset == 0 and place.sizes == shape and place.strides == _contiguous(shape)


def _paged(ir, places, items, key, reads, local):
    packet = getattr(local.target, 'overloadpacket', None)
    if not reads or not all(isinstance(k[0].meta.get('val'), torch.Tensor) for k in (key, *reads)) \
            or places.get(key) is not None and places[key].owner != key \
            or places.get(key) is None and packet is not A.scaled_dot_product_attention:
        return False
    dtype, shape = _layout(ir, key)
    if packet in REDUCTIONS:
        return _whole(ir, places, items, reads[0])
    if packet in SOFTMAX:
        return _layout(ir, reads[0]) == (dtype, shape) and _whole(ir, places, items, reads[0])
    if packet is A.scaled_dot_product_attention:
        (kd, ks), (vd, vs) = _layout(ir, reads[1]), _layout(ir, reads[2])
        nk, nv = ir.page // kd.itemsize, ir.page // vd.itemsize
        return nk % ks[-1] == 0 and nv % vs[-1] == 0 and nk // ks[-1] == nv // vs[-1] \
            and _whole(ir, places, items, reads[1]) and _whole(ir, places, items, reads[2])
    if packet not in POINTWISE and packet is not A.copy and packet not in COPIES:
        return False
    for operand, size in (_layout(ir, k) for k in reads):
        if operand.itemsize != dtype.itemsize or (math.prod(size) != math.prod(shape) if packet in COPIES
                                                  else size != shape):
            return False
    return all(_whole(ir, places, items, k) for k in (key, *reads))


def _kinds(ir):
    places, items = ir.places[min(ir.coords)], {item[2]: item for item in ir.program if item[0] != 'output'}
    for kind, reads, key, operand in ir.program:
        if kind == 'call':
            ir.kinds[key] = mesh.PAGES if _paged(ir, places, items, key, reads, ir.locals[operand]) else mesh.JOIN


# GSPMD sections 3.1 and 4.4
def _groups(ir, dims):
    groups = {}
    for device in sorted(ir.coords, key=ir.coords.get):
        groups.setdefault(tuple(c for m, c in enumerate(ir.coords[device]) if m not in dims), []).append(device)
    return list(groups.values())


# design/aot-compilation-change-request.md, appendix item 7
def _readers(ir, roots, default):
    spans = [(3 * g, 3 * g + (item[0] == 'collective' and item[3].kind != 'all-reduce'))
             for g, item in enumerate(ir.program) if any(ir.over.get(k, frozenset()) & roots for k in item[1])]
    return (min(s for s, _ in spans), max(e for _, e in spans)) if spans else (default, default)


# design/plan-to-bridge-change-request.md R4 and R6; design/aot-compilation-change-request.md A8 and appendix item 5;
# design/torch-source-lowering-change-request.md item 1
def _consumer(ir, roots):
    def read(whole):
        def reads(g):
            if not isinstance(g, int):
                return []
            _, keys, key, operand = ir.program[g]
            if whole and ir.kinds.get(key) == mesh.PAGES:
                paged = _paging(ir.locals[operand].target)[0]
                keys = [] if paged is None else [k for j, k in enumerate(keys) if j not in paged]
            return [owner for k in keys for owner in ir.over.get(k, ())]
        return reads
    return mesh.PAGES if roots & mesh_graph.reach(range(len(ir.program)), read(False)) \
        and not roots & mesh_graph.reach(range(len(ir.program)), read(True)) else mesh.JOIN


# design/plan-to-bridge-change-request.md R1; design/aot-compilation-change-request.md, appendix item 7;
# mesh-collective.h struct mesh_strided; torch 2.14.0 torch/_prims/__init__.py, collapse_view and _collapse_view_helper
Place = collections.namedtuple('Place', 'owner offset sizes strides writer')


def _contiguous(shape):
    return tuple(math.prod(shape[d + 1:]) for d in range(len(shape)))


def _own(ir, key):
    shape = _layout(ir, key)[1]
    return Place(key, 0, shape, _contiguous(shape), ir.results.get(key, key))


def _narrowed(place, dim, start, length):
    return place._replace(offset=place.offset + start * place.strides[dim],
                          sizes=place.sizes[:dim] + (length,) + place.sizes[dim + 1:])


def _part(ir, key, j):
    shape = _layout(ir, key)[1][1:]
    return Place(key, j * math.prod(shape), shape, _contiguous(shape), key)


# GSPMD section 4.5
def _cut(ir, kind, operand, sizes, device):
    if kind == 'slice':
        d, m = operand
        n = sizes[d] // ir.shape[m]
        return d, ir.coords[device][m] * n, n
    if kind == 'narrow':
        return operand
    return operand[0], operand[2][ir.coords[device][operand[1]]], operand[3]


def _inside_of(ir, key, device):
    owner, dim, starts = ir.inside[key]
    return _narrowed(_own(ir, owner), dim, starts[device], _layout(ir, key)[1][dim])._replace(
        writer=ir.results.get(key, key))


# MPI 4.1 section 7.8; torch 2.14.0 torch/_tensor_docs.py, view
def _placement(ir, item, reads, device, added):
    kind, _, key, operand = item
    if key in ir.inside:
        place = _inside_of(ir, key, device)
        if kind != 'copy' or reads[0] is None:
            return place
        source = reads[0] if operand is None else _narrowed(reads[0], *_cut(ir, 'slice', operand, reads[0].sizes,
                                                                            device))
        return source if source[:4] == place[:4] else place
    if key in ir.taken:
        return added[ir.taken[key]]
    if key in ir.loads:
        return _own(ir, key)
    if _sliced(item) and reads[0] is not None:
        return _narrowed(reads[0], *_cut(ir, kind, operand, reads[0].sizes, device))
    form = ir.forms.get(key)
    if kind == 'call' and form is None and isinstance(ir.locals[operand].target, torch._ops.OpOverload) \
            and _aliases(ir.locals[operand]):
        view = _view(ir, key, ir.locals[operand], item[1], reads)
        if view is not None:
            return view
    if kind == 'collective' or isinstance(form, tuple) and any(result == key for _, result in form[1]) \
            or kind == 'call' and key in ir.sent:
        return _own(ir, key)
    return None


# torch 2.14.0 torch/_tensor_docs.py, view
def _based(local):
    schema = local.target._schema
    sets = {s for r in schema.returns if r.alias_info is not None for s in r.alias_info.before_set}
    named = _named(local, _names(local))
    found = [k for k, name in enumerate(named) for a in schema.arguments
             if a.name == name and a.alias_info is not None and sets & set(a.alias_info.before_set)]
    return found[0] if found else None


def _view(ir, key, local, keys, reads):
    k = _based(local)
    if k is None or reads[k] is None or key not in ir.metas:
        return None
    metas = [_meta(ir, r) if p is None else
             torch.empty(math.prod(_layout(ir, p.owner)[1]), dtype=_layout(ir, p.owner)[0], device='meta')
             .as_strided(p.sizes, p.strides, p.offset) for r, p in zip(keys, reads)]
    if any(m is UNKNOWN for m in metas):
        return None
    given = iter(metas)
    args, kwargs = map_arg((local.args, local.kwargs), lambda n: next(given))
    try:
        result = _call(local, args, kwargs)
    except NotImplementedError:
        return None
    except RuntimeError:
        ir.contiguous.add(key)
        return _own(ir, key)
    if not isinstance(result, torch.Tensor) or result.dtype != metas[k].dtype:
        return None
    if not torch._C._is_alias_of(result, metas[k]):
        ir.contiguous.add(key)
        return _own(ir, key)
    return Place(reads[k].owner, result.storage_offset(), tuple(result.shape), tuple(result.stride()), reads[k].writer)


def _places(ir, device):
    items = {item[2]: item for item in ir.program if item[0] != 'output'}
    added = {key: _inside_of(ir, key, device) if key in ir.inside else _own(ir, key) for key in ir.added}
    return {**mesh_graph.forward(list(items), lambda key: items[key][1],
                                 lambda key, reads: _placement(ir, items[key], reads, device, added)), **added}


# MPI 4.1 section 6.1.3, MPI_TYPE_CREATE_SUBARRAY
def _strided(ir, place):
    size, dims = _layout(ir, place.owner)[0].itemsize, []
    for n, s in zip(place.sizes, place.strides):
        if n == 1:
            continue
        if dims and dims[-1][1] == n * s:
            dims[-1] = (dims[-1][0] * n, s)
        else:
            dims.append((n, s))
    pairs = (mesh.Q * (2 * len(dims)))(*[x for n, s in dims for x in (n, s * size)])
    return mesh.Strided(offset=place.offset * size, element=size, dims=len(dims), pairs=pairs)


def _produced(ir, d, i, key, received):
    place = ir.places[d][key]
    at = 3 * ir.producer[place.writer] + 2
    return [(d, i, 'P', at, at), (d, i, 'S') + _readers(ir, {place.owner, key}, received)]


def _send(ir, i, key):
    ir.publishes.setdefault(ir.producer[ir.places[ir.device][key].writer], []).append(i)


# NCCL User Guide, In-place Operations; MPI 4.1 section 7.7
def _inside(ir):
    for kind, reads, key, operand in ir.program:
        if kind == 'copy' and operand is None:
            target = ir.taken.get(reads[0], reads[0])
            producer, _, _, collective = ir.program[ir.producer[ir.results.get(target, target)]]
            if target not in ir.inside and (target in ir.results or producer == 'call'
                                            and isinstance(ir.forms.get(target), tuple)
                                            or producer == 'collective' and collective.kind == 'reduce-scatter'):
                ir.inside[target] = ir.inside[key]


# design/aot-compilation-change-request.md A4 and appendix item 7; design/plan-to-bridge-change-request.md R4; G. Wang
# et al., Blink, MLSys 2020, section 3.3; GSPMD section 3.1; M. Cowan et al., MSCCLang, ASPLOS 2023, section 4.2; J.
# Gibbons, J. Functional Programming 1996; L. Meertens, Algorithmics, CWI Monographs 1, 1986
def _all_reduce(ir, c, groups, key, made):
    arrive, k = 3 * c.position + 1, len(groups[0])
    readers, received = _readers(ir, {made}, arrive + 1), (c.node, 'received', c.position)
    ir.shapes[received], ir.added[received] = (k - 1,) + _layout(ir, key)[1], None
    for group in groups:
        root, members = group[0], group[1:]
        up = [ir.transfer(d, [root], mesh.PAGES, _strided(ir, ir.places[d][key]),
                          {root: _strided(ir, _part(ir, received, j))})[root] for j, d in enumerate(members)]
        down = ir.transfer(root, members, _consumer(ir, {made}), _strided(ir, ir.places[root][made]),
                           {d: _strided(ir, ir.places[d][made]) for d in members})
        for d, partial in zip(members, up):
            ir.events += _produced(ir, d, partial, key, arrive) \
                + [(d, down[d], 'C') + readers, (root, partial, 'C', arrive, arrive)]
        first = min(down.values())
        ir.events += [(root, first, 'P', arrive + 1, arrive + 1), (root, first, 'S') + readers]
        if ir.device == root:
            ir.roles[c] = (0, [(i, (received, j)) for j, i in enumerate(up)])
            ir.at.update({i: made for i in down.values()})
            ir.at.update({i: received for i in up})
            ir.publishes.setdefault(c.position, []).append(first)
        elif ir.device in members:
            me = group.index(ir.device)
            ir.roles[c] = (me, [(down[ir.device], None)])
            ir.at.update({up[me - 1]: ir.places[ir.device][key].owner, down[ir.device]: made})
            _send(ir, up[me - 1], key)


# GSPMD section 4.2; design/aot-compilation-change-request.md, appendix item 7; design/disjoint-pages-change-request.md
# C3.2; design/plan-to-bridge-change-request.md R2 and R4; mesh-collective.h struct mesh_dependence,
# MESH_FROM_SOURCE_READ; NCCL, ncclAllGather
def _all_gather(ir, c, groups, key, made):
    d, received = c.dim, 3 * c.position + 1
    n = _shape(ir, key)[d]
    last = max(received, _readers(ir, {made}, received)[1])
    for group in groups:
        ranges = [ir.transfer(e, [f for f in group if f != e], _consumer(ir, {made}), _strided(ir, ir.places[e][key]),
                              {f: _strided(ir, _narrowed(ir.places[f][made], d, p * n, n)) for f in group if f != e})
                  for p, e in enumerate(group)]
        for e, transfers in zip(group, ranges):
            ir.events += _produced(ir, e, min(transfers.values()), key, received)
            ir.events += [(f, j, 'C', received, last) for f, j in transfers.items()]
        if ir.device in group:
            me = group.index(ir.device)
            ir.at.update({j: made for j in ranges[me].values()})
            ir.at.update({t[ir.device]: made for p, t in enumerate(ranges) if p != me})
            ir.roles[c] = (me, [(t[ir.device], None) for p, t in enumerate(ranges) if p != me])
            _send(ir, min(ranges[me].values()), key)


def _exchange(ir, c, groups, pairs, key, sent, landed, kind, last):
    received = 3 * c.position + 1
    for group in groups:
        moved = {(p, q): ir.transfer(group[p], [group[q]], kind, _strided(ir, sent(group[p], q)),
                                     {group[q]: _strided(ir, landed(group[q], p, q)[0])})[group[q]]
                 for p, q in pairs if p != q}
        for (p, q), x in moved.items():
            ir.events += _produced(ir, group[p], x, key, received) + [(group[q], x, 'C', received, last)]
        if ir.device in group:
            me = group.index(ir.device)
            ir.roles[c] = (me, [(x, landed(ir.device, p, me)[2]) for (p, q), x in moved.items() if q == me])
            for (p, q), x in moved.items():
                if p == me:
                    ir.at[x] = ir.places[ir.device][key].owner
                    _send(ir, x, key)
                elif q == me:
                    ir.at[x] = landed(ir.device, p, me)[1]


# MPI 4.1 section 7.8, MPI_ALLTOALL
def _all_to_all(ir, c, groups, key, made):
    (i, j), k = c.dim, len(groups[0])
    ni, nj, received = _shape(ir, key)[i], _shape(ir, key)[j] // k, 3 * c.position + 1
    _exchange(ir, c, groups, [(p, q) for p in range(k) for q in range(k)], key,
              lambda e, q: _narrowed(ir.places[e][key], j, q * nj, nj),
              lambda f, p, q: (_narrowed(ir.places[f][made], i, p * ni, ni), made, None), _consumer(ir, {made}),
              max(received, _readers(ir, {made}, received)[1]))


# design/aot-compilation-change-request.md A4; design/plan-to-bridge-change-request.md R4; GSPMD sections 3.1, 3.2, 4.2
# and 5.1; G. Wang et al., Blink, MLSys 2020, section 3.3; M. Cowan et al., MSCCLang, ASPLOS 2023, section 4.2
def _reduce_scatter(ir, c, groups, key, made):
    dim, k = c.dim, len(groups[0])
    n, received = _shape(ir, key)[dim] // k, (c.node, 'received', c.position)
    shape = _layout(ir, key)[1]
    ir.shapes[received], ir.added[received] = (k - 1,) + shape[:dim] + (n,) + shape[dim + 1:], None
    _exchange(ir, c, groups, [(p, q) for p in range(k) for q in range(k)], key,
              lambda e, q: _narrowed(ir.places[e][key], dim, q * n, n),
              lambda f, p, q: (_part(ir, received, p - (p > q)), received, (received, p - (p > q))), mesh.PAGES,
              3 * c.position + 1)


# design/aot-compilation-change-request.md A4 and appendix item 7; design/plan-to-bridge-change-request.md R2 and R4;
# GSPMD sections 4.2, 4.3 and 4.6, Appendix A.2; OpenXLA, Operation semantics, CollectivePermute
def _collective_permute(ir, c, groups, key, made):
    pairs = tuple(tuple(pair) for pair in c.dim)
    places, k = [p for pair in pairs for p in pair], len(groups[0])
    if not all(isinstance(p, int) and 0 <= p < k for p in places) \
            or len({p for p, _ in pairs}) != len(pairs) or len({q for _, q in pairs}) != len(pairs):
        raise ValueError('%s: a CollectivePermute\'s pairs %s over a group of %d: each a (source, target) place, no two '
                         'with one source or one target (XLA\'s source_target_pairs)' % (c.node.name, pairs, k))
    landed = ir.inside[made][0]
    _exchange(ir, c, groups, pairs, key, lambda e, q: ir.places[e][key],
              lambda f, p, q: (ir.places[f][made], landed, None), _consumer(ir, {landed}),
              max(3 * c.position + 1, _readers(ir, {landed}, 3 * c.position + 1)[1]))


# mesh-collective.h struct mesh_dependence; design/disjoint-pages-change-request.md C3.5
def _dependences(ir):
    bits = {'P': (mesh.FROM_PRODUCED, mesh.TO_PRODUCED), 'S': (mesh.FROM_SOURCE_READ, mesh.TO_SOURCE_READ),
            'C': (0, 0)}
    edges = []

    def link(a, b, lag):
        edges.append(mesh.Dependence(a[1], b[1], lag, bits[a[2]][0] | bits[b[2]][1]))
    for device in sorted({e[0] for e in ir.events}):
        events, classes = [e for e in ir.events if e[0] == device], {}
        for e in events:
            if e[3] == e[4]:
                classes.setdefault(e[3], []).append(e)
        points, spans = sorted(classes), [e for e in events if e[3] != e[4]]
        for p, q in zip(points, points[1:]):
            for a in classes[p]:
                for b in classes[q]:
                    link(a, b, 0)
        tails, heads = list(classes[points[-1]] if points else ()), list(classes[points[0]] if points else ())
        for x in spans:
            before, after = [p for p in points if p < x[3]], [p for p in points if p > x[4]]
            for a in classes[before[-1]] if before else ():
                link(a, x, 0)
            for b in classes[after[0]] if after else ():
                link(x, b, 0)
            for y in spans:
                if x[4] < y[3]:
                    link(x, y, 0)
            if not after:
                tails.append(x)
            if not before:
                heads.append(x)
        for a in tails:
            for b in heads:
                link(a, b, 1)
    return edges


# design/aot-compilation-change-request.md A4 and appendix item 7; GSPMD section 4.2
LOWERINGS = {'all-reduce': _all_reduce, 'all-gather': _all_gather, 'all-to-all': _all_to_all,
             'reduce-scatter': _reduce_scatter, 'collective-permute': _collective_permute}


# torch 2.14.0 torch/distributed/_functional_collectives.py, AsyncCollectiveTensor
def lower(ir):
    for item in ir.program:
        if item[0] == 'collective':
            collective, first = item[3], len(ir.declared)
            if collective.kind not in LOWERINGS:
                raise NotImplementedError('%s of %s: no lowering (A-T3\')' % (collective.kind, collective.node.name))
            LOWERINGS[collective.kind](ir, collective, _groups(ir, collective.dims), item[1][0], item[2])
            ir.declares[collective] = range(first, len(ir.declared))
    firsts = mesh_graph.ends(range(len(ir.program))[::-1], lambda g: ir.program[g][1])
    for kind, _, key, collective in ir.program:
        if kind == 'collective' and key in firsts:
            ir.waits.setdefault(firsts[key], []).append(collective)
    ir.dependences = _dependences(ir)
    _place(ir)


# design/compilation-composition-change-request.md requirement 1 and item 6, unit 6D; design/aot-compilation-change-
# request.md A4 and appendix item 7; GSPMD section 4.5 (each device holds the whole output after its reshard): function
# f's output leaf o, read by calls of the functions `wanted[f, o]` at map nodes outside f's map, is delivered there by
# f's own plan, one declared transfer from f's least device, landing at each such node in one EXECUTION value of f
# there laid out as the output's owner at the source (the same view at both ends, so the landed result has the
# producer's offset, sizes and strides), `ir.landed[o]` = (that value, its transfer) and `ir.homes[key]` its view, as
# at a node with a section; there f does no other work, so no event is added there.  An output over one of f's loads
# (ir.loads: a constant of f's module, every node's own host copy, placed and written by `make`'s loads as at the source)
# moves nothing: at each such node it is that load, one INVARIANT value with the source's view, `ir.landed[o]` = (that
# value, None).  `o` indexes pytree.tree_leaves of f's output node's args.  Called after partition and before lower
def _deliver(irs, wanted, returned, program):
    for f, o in returned:
        wanted.setdefault((f, o), set())
    for (f, o), readers in sorted(wanted.items()):
        ir = irs[f]
        everywhere = {n for n in range(64) if program >> n & 1} if (f, o) in returned else set()
        nodes = sorted(({y for k in readers for y in irs[k].map} | everywhere) - set(ir.map))
        if not nodes:
            continue
        output = next(item for item in ir.program if item[0] == 'output')[3]
        key, source = output[1][pytree.tree_leaves(output[0].args)[o]], min(ir.coords)
        place = ir.places[source][key]
        dtype, shape = _layout(ir, place.owner)
        if place.owner in ir.loads:
            if ir.index in nodes:
                v = ir.value(dtype, math.prod(shape), _device(ir, key), 0, 0, 0, kind=mesh.INVARIANT)
                ir.landed[o], ir.homes[key] = (v, None), (v, place.offset, place.sizes, place.strides)
            continue
        view = _strided(ir, place)
        transfers = ir.transfer(source, nodes, mesh.JOIN, view, dict.fromkeys(nodes, view), at={y: y for y in nodes})
        ir.events += _produced(ir, source, min(transfers.values()), key, 3 * len(ir.program))
        if ir.device == source:
            ir.at.update(dict.fromkeys(transfers.values(), place.owner))
            _send(ir, min(transfers.values()), key)
        elif ir.index in transfers:
            i = transfers[ir.index]
            ir.declared[i].value = ir.value(dtype, math.prod(shape), _device(ir, key), 1, 0, len(ir.program) - 1)
            ir.landed[o] = (ir.declared[i].value, i)
            ir.homes[key] = (ir.declared[i].value, place.offset, place.sizes, place.strides)


# design/aot-compilation-change-request.md A4 and appendix item 7; design/disjoint-pages-change-request.md C3.2 and
# C3.3; design/plan-to-bridge-change-request.md R2; mesh-collective.h struct mesh_value and struct mesh_declared;
# ExecuTorch exir/memory_planning.py, update_tensor_lifetime; MPI 4.1 sections 3.7 and 3.9, MPI_START
def _place(ir):
    if ir.device is None:
        return
    places = ir.places[ir.device]
    places.update({key: _own(ir, key) for key in ir.added if key not in places})
    waits = {g: [ir.at[i] for c in collectives for i in ir.declares[c] if i in ir.at]
             for g, collectives in ir.waits.items()}
    last = mesh_graph.ends(range(len(ir.program)),
                           lambda g: [owner for k in ir.program[g][1] for owner in ir.over.get(k, ())]
                           + waits.get(g, []))
    wired, begins = set(ir.at.values()), {}
    landed = {ir.at[i] for _, receives in ir.roles.values() for i, _ in receives}
    for key, place in places.items():
        if place is not None and place.writer == ir.results.get(key, key) and place.writer in ir.producer:
            begins[place.owner] = min(begins.get(place.owner, len(ir.program)), ir.producer[place.writer])
    for key, place in places.items():
        if place is None or place.owner != key or key in ir.added and ir.added[key] is None and key not in wired:
            continue
        dtype, shape = _layout(ir, key)
        wire, first = int(key in wired), begins.get(key, 0)
        ir.owned[key] = ir.value(dtype, math.prod(shape), _device(ir, key), wire, 0 if key in landed else first,
                                 max(first, last.get(key, first)),
                                 kind=mesh.INVARIANT if key in ir.loads else mesh.EXECUTION)
    ir.homes = {key: (ir.owned[place.owner], place.offset, place.sizes, place.strides)
                for key, place in places.items() if place is not None and place.owner in ir.owned}
    for i, owner in ir.at.items():
        ir.declared[i].value = ir.owned[owner]
    _runs(ir, places)


# design/torch-source-lowering-change-request.md item 1; design/plan-to-bridge-change-request.md R2 and R4;
# design/aot-compilation-change-request.md, appendix item 7; mesh-collective.h struct mesh_value
def _runs(ir, places):
    def accessed(g):
        if not isinstance(g, int):
            return []
        kind, reads, key, operand = ir.program[g]
        form = ir.forms.get(key) if kind == 'call' else None
        written = [result for _, result in form[1]] if isinstance(form, tuple) else [key]
        if ir.kinds.get(key) == mesh.PAGES:
            paged, pages = _paging(ir.locals[operand].target)
            reads = [] if paged is None else [k for j, k in enumerate(reads) if j not in paged]
            written = [] if pages else written
        return [owner for k in reads for owner in ir.over.get(k, ())] \
            + [places[k].owner for k in written if k is not None and places.get(k) is not None]
    synchronous = [g for g, (kind, _, key, operand) in enumerate(ir.program)
                   if not (kind in ('input', 'constant', 'slice', 'narrow') or kind == 'window' and key not in ir.inside
                           or kind == 'collective' and _local(ir, operand) is None)]
    run = mesh_graph.reach(synchronous, accessed) | set(ir.loads)
    for key, v in ir.owned.items():
        ir.values[v].run = int(key in run)


# design/aot-compilation-change-request.md A3; design/integration-finishing-change-request.md item 3; O. Sinnen, Task
# Scheduling for Parallel Systems, Wiley, 2007; mesh-collective.c plan_schedule
Weights = collections.namedtuple('Weights', 'tasks transfers')


def unit(ir):
    return Weights([1.0] * (3 * len(ir.declared)), [1.0] * len(ir.declared))


def _weighed(ir, function):
    made, count = function.weights(ir), len(ir.declared)
    if not isinstance(made, Weights) or len(made.tasks) != 3 * count or len(made.transfers) != count \
            or not all(isinstance(w, (int, float)) for w in [*made.tasks, *made.transfers]):
        raise ValueError('the weighting function gave %r for %d declared transfers: A3\'s operand is "a weight on each '
                         'task; a weight on each edge", a Weights of one number a task of mesh_plan (a declared '
                         'transfer\'s producer, source read and consumer, in that order) and one number an edge (a '
                         'declared transfer\'s, a page)' % (type(made).__name__, count))
    return [mesh.Weight(produced=made.tasks[3 * i], consumed=made.tasks[3 * i + 2], read=made.tasks[3 * i + 1],
                        page=made.transfers[i]) for i in range(count)]


# mesh.h MESH_NAME; mesh-dataflow.h struct mesh_ctx
CONTEXT = []


def _context():
    if not CONTEXT:
        context = mesh.Context()
        mesh.check(mesh.LIB.mesh_attach(C.byref(context), mesh.NAME))
        CONTEXT.append(context)
    return CONTEXT[0]


def _group():
    text = os.environ.get('MESH_GROUP', '')
    try:
        group = int(text, 0)
    except ValueError:
        group = 0
    if not 0 < group < 2 ** 64:
        raise ValueError('MESH_GROUP %r: the run\'s group is a whole nonzero number, the same at every rank' % text)
    return group


# mesh-collective.h mesh_plan; design/aot-compilation-change-request.md A7
def _program():
    text = os.environ.get('MESH_PROGRAM', '')
    try:
        nodes = {int(item, 10) for item in text.split(',')}
    except ValueError:
        nodes = set()
    if not nodes or not all(0 <= n < 64 for n in nodes):
        raise ValueError('MESH_PROGRAM %r: the program\'s nodes, the map indices its launcher started it at, whole '
                         'numbers below 64 apart by commas, the same at every rank' % text)
    return sum(1 << n for n in nodes)


# design/disjoint-pages-change-request.md D.3.3; mesh-collective.h struct mesh_value
HOST_LAG = 2


# mesh-collective.h mesh_host_arrived; mesh.h MESH_PLATFORM_NS; rdma/mesh.py Publication
def _wait(ready, progress, what):
    seen, since = progress(), time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    while not ready():
        now, value = time.clock_gettime_ns(time.CLOCK_MONOTONIC), progress()
        if value != seen:
            seen, since = value, now
        elif now - since > mesh.PLATFORM_NS:
            raise OSError(errno.ETIMEDOUT, what)


# mesh-dataflow.h mesh_host_started, the program's `started`
STARTED = C.cast(mesh.LIB.mesh_host_started, C.c_void_p)


# torch v2.14.0 include/ATen/dlpack.h, DLPACK_MAJOR_VERSION 1, DLPACK_MINOR_VERSION 3
class DLDevice(C.Structure):
    _fields_ = [('device_type', C.c_int32), ('device_id', C.c_int32)]


class DLDataType(C.Structure):
    _fields_ = [('code', C.c_uint8), ('bits', C.c_uint8), ('lanes', C.c_uint16)]


class DLTensor(C.Structure):
    _fields_ = [('data', C.c_void_p), ('device', DLDevice), ('ndim', C.c_int32), ('dtype', DLDataType),
                ('shape', C.POINTER(C.c_int64)), ('strides', C.POINTER(C.c_int64)), ('byte_offset', C.c_uint64)]


class DLPackVersion(C.Structure):
    _fields_ = [('major', C.c_uint32), ('minor', C.c_uint32)]


class DLManagedTensorVersioned(C.Structure):
    pass


DELETER = C.CFUNCTYPE(None, C.POINTER(DLManagedTensorVersioned))
DLManagedTensorVersioned._fields_ = [('version', DLPackVersion), ('manager_ctx', C.c_void_p), ('deleter', DELETER),
                                     ('flags', C.c_uint64), ('dl_tensor', DLTensor)]
# dlpack.h DLDeviceType and DLDataTypeCode; torch v2.14.0 aten/src/ATen/DLConvertor.cpp getDLDataType and toScalarType,
# include/ATen/DLConvertor.h DLPackTraits<DLManagedTensorVersioned> `capsule`; Python C API, PyCapsule_New
DEVICES, VERSION, CAPSULE = {'cpu': 1, 'mps': 8}, DLPackVersion(1, 3), b'dltensor_versioned'
CODES = {torch.int8: 0, torch.int16: 0, torch.int32: 0, torch.int64: 0, torch.uint8: 1, torch.uint16: 1,
         torch.uint32: 1, torch.uint64: 1, torch.float16: 2, torch.float32: 2, torch.float64: 2, torch.bfloat16: 4,
         torch.complex32: 5, torch.complex64: 5, torch.complex128: 5, torch.bool: 6, torch.float8_e4m3fn: 10,
         torch.float8_e4m3fnuz: 11, torch.float8_e5m2: 12, torch.float8_e5m2fnuz: 13, torch.float8_e8m0fnu: 14}
TYPES = {dtype: (code, dtype.itemsize * 8, 1) for dtype, code in CODES.items()}
TYPES[torch.float4_e2m1fn_x2] = (17, 4, 2)
CAPSULE_NEW = C.PYFUNCTYPE(C.py_object, C.c_void_p, C.c_char_p, C.c_void_p)(('PyCapsule_New', C.pythonapi))
# objc/message.h objc_msgSend; MTLResource.h -device; NSObject -retain and -release; MTLDevice.h
# -newBufferWithBytesNoCopy:length:options:deallocator:
OBJC = C.CDLL('/usr/lib/libobjc.A.dylib')
OBJC.sel_registerName.restype, OBJC.sel_registerName.argtypes = C.c_void_p, [C.c_char_p]
SEND = C.cast(OBJC.objc_msgSend, C.c_void_p).value
GET = C.CFUNCTYPE(C.c_void_p, C.c_void_p, C.c_void_p)(SEND)
PUT = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p)(SEND)
WRAP = C.CFUNCTYPE(C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_ulong, C.c_ulong, C.c_void_p)(SEND)
SELECTOR = {name: OBJC.sel_registerName(name) for name in
            (b'device', b'retain', b'release', b'newBufferWithBytesNoCopy:length:options:deallocator:')}
# MTLResource.h MTLResourceStorageModeShared
SHARED = 0
# MTLResource.h `device`; torch v2.14.0 include/ATen/mps/MPSAllocator.h, MPSHeapAllocatorImpl `m_device`;
# include/ATen/native/mps/OperationUtils.h getMTLBufferStorage and iter_tensor_offset; torch/csrc/StorageMethods.cpp
# THPStorage_dataPtr
DEVICE = []


def _metal_device():
    if not DEVICE:
        probe = torch.empty(1, device='mps')
        device = GET(probe.untyped_storage().data_ptr(), SELECTOR[b'device'])
        if not device:
            raise OSError(errno.ENODEV, 'torch\'s MPS buffer names no MTLDevice (MTLResource.h `device`)')
        DEVICE.append(device)
    return DEVICE[0]


# objc/message.h; metal-microbench metal_recording.h
def _message(restype, *argtypes):
    return C.CFUNCTYPE(restype, C.c_void_p, C.c_void_p, *argtypes)(SEND)


_SEL = OBJC.sel_registerName
OBJC.objc_getClass.restype, OBJC.objc_getClass.argtypes = C.c_void_p, [C.c_char_p]
SET = _message(None, C.c_void_p)
OBJECT = _message(C.c_void_p)
VALUE = _message(C.c_uint64)
LEVELS = _message(C.c_uint32, C.c_void_p)
VALUES = _message(C.c_ulong, C.c_void_p, C.c_uint32, C.c_void_p)
# metal_recording.h place:value:in:offsets:copies:ahead:shares:, place:step:offsets:copies:shares: and
# state:buffer:status:
PLACE_VALUE = _message(None, C.c_void_p, C.c_ulong, C.c_void_p, C.c_void_p, C.c_ulong, C.c_ulong, C.c_bool)
PLACE_STEP = _message(None, C.c_void_p, C.c_void_p, C.c_void_p, C.c_ulong, C.c_bool)
STATE_BUFFER = _message(None, C.c_void_p, C.c_void_p, C.c_ulong)


# metal_recording.h struct MetalReplay, struct MetalCall and struct MetalValue
class Replay(C.Structure):
    _fields_ = [('commands', C.c_void_p), ('residency', C.c_void_p), ('owner', C.c_void_p), ('count', mesh.U),
                ('buffers', C.c_uint16), ('memory', C.c_uint16)]


# design/aot-compilation-change-request.md, appendix item 6: "have each call launch one execution of its function";
# metal_recording.h struct MetalCall: the row's plan call `call` - 1 (mesh-collective.h planned->call, 0 a planless
# row), whose `ordinal` of its function's `every` calls in a block runs plan execution b every + ordinal at block
# execution b; `native` and `residues` its compiled CPU entries (MetalReplayNative), native[e % residues] at its call's
# execution e
class Call(C.Structure):
    _fields_ = [('replay', C.POINTER(Replay)), ('call', mesh.Q), ('native', C.c_void_p), ('residues', mesh.Q)]


class MetalValue(C.Structure):
    _fields_ = [('bytes', C.c_ulong), ('kind', mesh.U), ('state', mesh.U), ('begin', mesh.U), ('end', mesh.U)]


# rdma/mesh-metal.h struct mesh_metal_transport, struct mesh_metal_input (metal_recording.h struct MetalRemote), struct
# mesh_metal_publication and struct mesh_metal_sent (metal_recording.h struct MetalSent); mesh.h struct mesh_hop
class Transport(C.Structure):
    _fields_ = [('state', C.c_void_p), ('offset', mesh.Q), ('bytes', mesh.Q)]


class Remote(C.Structure):
    _fields_ = [('phases', mesh.Q), ('group', mesh.Q), ('count', mesh.Q), ('pages', mesh.U), ('copies', mesh.U),
                ('time', C.c_double)]


class MetalSent(C.Structure):
    _fields_ = [('counter', C.c_void_p), ('sends', mesh.Q), ('copies', mesh.Q)]


START = C.cast(mesh.LIB.mesh_transfers_start, C.c_void_p)
TRANSPORT = []


# rdma/mesh-metal.h: the transport's records in two stages, declared from the program before mesh_plan
# (mesh_metal_range_declare) and prepared from its results (mesh_metal_range_prepare, mesh_metal_release_prepare,
# mesh_metal_sending_prepare, mesh_metal_status_prepare)
def _transport():
    if not TRANSPORT:
        lib = C.CDLL(os.path.join(_repository(), '.build', 'libmesh_metal.dylib'))
        planned, transport = C.POINTER(mesh.Planned), C.POINTER(Transport)
        for name, result, arguments in (
                ('mesh_metal_transport_create', C.c_int, [mesh.CONTEXT, planned, C.c_void_p, transport]),
                ('mesh_metal_transport_destroy', None, [transport]),
                ('mesh_metal_range_declare', C.c_int, [mesh.CONTEXT, C.POINTER(mesh.U), mesh.U, mesh.U, mesh.U, mesh.U,
                                                       mesh.Q, mesh.Q, C.POINTER(Range)]),
                ('mesh_metal_range_prepare', C.c_int, [mesh.CONTEXT, transport, planned, mesh.Section,
                                                       C.POINTER(mesh.U), mesh.U, C.POINTER(Range)]),
                ('mesh_metal_release_prepare', C.c_int, [mesh.CONTEXT, transport, planned, mesh.U,
                                                         C.POINTER(Release)]),
                ('mesh_metal_sending_prepare', C.c_int, [mesh.CONTEXT, transport, planned, mesh.U, mesh.U, mesh.Section,
                                                         C.POINTER(mesh_graph.Access), C.c_size_t, C.POINTER(Sending)]),
                ('mesh_metal_status_prepare', C.c_int, [mesh.CONTEXT, transport, mesh.Q, C.POINTER(mesh.Q)])):
            function = getattr(lib, name)
            function.restype, function.argtypes = result, arguments
        TRANSPORT.append(lib)
    return TRANSPORT[0]


# dyld include/mach-o/dyld-interposing.h, DYLD_INTERPOSE; torch 2.14.0 aten/src/ATen/mps/MPSDevice.mm
RECORDER = []


def _inserted():
    path = next((p for p in os.environ.get('DYLD_INSERT_LIBRARIES', '').split(':')
                 if os.path.basename(p) == 'libmetal_recording.dylib'), None)
    if path is None:
        raise OSError(errno.ENOENT, 'no recorder in the process: the program starts with DYLD_INSERT_LIBRARIES='
                                    '<metal-microbench>/.build/libmetal_recording.dylib, whose repository holds the '
                                    'operators\' sources')
    return path


def _repository():
    return os.path.dirname(os.path.dirname(os.path.abspath(_inserted())))


def _recorder():
    if not RECORDER:
        path = _inserted()
        lib, metal = C.CDLL(path), C.CDLL('/System/Library/Frameworks/Metal.framework/Metal')
        lib.MetalRecorderOpen.restype, lib.MetalRecorderOpen.argtypes = C.c_void_p, [C.c_void_p, C.c_char_p, C.c_size_t]
        lib.MetalRecorderPrepare.restype = C.c_void_p
        lib.MetalRecorderPrepare.argtypes = [C.c_void_p, C.c_void_p, C.c_char_p, C.c_size_t]
        lib.MetalRecorderReplay.restype = C.c_int
        lib.MetalRecorderReplay.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(Replay), C.c_char_p, C.c_size_t]
        lib.MetalReplayNative.restype = C.c_void_p
        lib.MetalReplayNative.argtypes = [C.POINTER(Replay), C.POINTER(mesh.Q)]
        lib.MetalRecorderProgram.restype = C.c_void_p
        lib.MetalRecorderProgram.argtypes = [C.c_void_p, C.POINTER(Call), mesh.U, C.POINTER(C.c_double),
                                             C.POINTER(mesh.Planned), C.c_char_p, C.c_size_t]
        lib.MetalProgramRun.restype = C.c_int
        lib.MetalProgramRun.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), mesh.Q, mesh.Q, C.c_void_p, C.c_void_p,
                                        C.c_void_p, C.c_void_p, C.c_char_p, C.c_size_t]
        lib.MetalProgramEnd.restype = C.c_int
        lib.MetalProgramEnd.argtypes = [C.c_void_p, C.c_void_p, C.c_char_p, C.c_size_t]
        lib.MetalReplayDestroy.restype, lib.MetalReplayDestroy.argtypes = None, [C.POINTER(Replay)]
        lib.MetalProgramSignature.restype = C.c_int
        lib.MetalProgramSignature.argtypes = [C.c_void_p, C.c_void_p, mesh.Q, C.POINTER(mesh_graph.Signature),
                                              C.c_char_p, C.c_size_t]
        metal.MTLCreateSystemDefaultDevice.restype = C.c_void_p
        RECORDER.append((lib, metal.MTLCreateSystemDefaultDevice))
    return RECORDER[0]


# design/compilation-composition-change-request.md, "Complete replacement workgroup", numerical operations and kernel
# instructions; metal-microbench metal_ir.h (metal_ir_value, metal_ir_generic, metal_ir_states and the region builder);
# MLIR Linalg, linalg.generic; N. D. Jones, C. K. Gomard and P. Sestoft, 1993, Definition 4.1: every static operator,
# dtype, shape, stride and scalar operand of a Torch call is an instruction, literal or map coefficient of its
# structured operation, never a metadata vector a generic kernel interprets
BOOL, I8, U8, I16, U16, I32, U32, I64, U64, F16, BF16, F32, F64 = 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16
SCALAR = {torch.bool: BOOL, torch.int8: I8, torch.uint8: U8, torch.int16: I16, torch.uint16: U16, torch.int32: I32,
          torch.uint32: U32, torch.int64: I64, torch.uint64: U64, torch.float16: F16, torch.bfloat16: BF16,
          torch.float32: F32, torch.float64: F64}
# metal_ir.h ComputeType: the kind a region computes a stored kind in
CLASS = {BOOL: BOOL, I8: I32, I16: I32, I32: I32, U8: U32, U16: U32, U32: U32, I64: I64, U64: U64, F16: F16, BF16: F32,
         F32: F32, F64: F64}
STORED = {BOOL: torch.bool, I32: torch.int32, U32: torch.uint32, I64: torch.int64, U64: torch.uint64, F16: torch.float16,
        F32: torch.float32, F64: torch.float64}
REAL = tuple(SCALAR)
FLOATING = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
INTEGRAL = tuple(dtype for dtype in REAL if dtype not in FLOATING)
# metal_ir.h: linalg iterator kinds, combiners, starts, precision, placement, Tint core::BinaryOp and core::UnaryOp
PARALLEL, REDUCTION = 0, 1
ASSIGN, SUM, REGION = 0, 1, 9
IDENTITY, ACCUMULATE = 0, 1
REASSOCIATE, CONTRACT = 1, 2
GPU, CPU = 0, 1
BINARY = {'and': 0, 'or': 1, 'xor': 2, 'eq': 5, 'ne': 6, 'lt': 7, 'gt': 8, 'le': 9, 'ge': 10, 'shl': 11, 'shr': 12,
          'add': 13, 'sub': 14, 'mul': 15, 'div': 16, 'rem': 17}
COMPARE = ('eq', 'ne', 'lt', 'gt', 'le', 'ge')
NEGATION, NOT = 3, 4
# WGSL builtin functions by name (metal_ir_builtin_named, Tint core::ParseBuiltinFn)
BUILTINS = ('exp', 'log', 'sqrt', 'tanh', 'abs', 'trunc', 'floor', 'pow', 'cos')


def _class(value):
    return CLASS[SCALAR[value.dtype]]


def _floating(kind):
    return kind in (F16, F32, F64)


# metal_ir.h metal_ir_literal: two's complement, or the IEEE binary16 / binary32 bits in the low bits
def _bits(value, kind):
    if kind == BOOL:
        return int(bool(value))
    if _floating(kind):
        dtype, word, bits = {F16: (torch.float16, torch.int16, 16), F32: (torch.float32, torch.int32, 32),
                             F64: (torch.float64, torch.int64, 64)}[kind]
        return torch.tensor(float(value), dtype=dtype).view(word).item() & ((1 << bits) - 1)
    return int(value) & ((1 << 64) - 1)


# a region's builder calls in order, each (call, operands...) of metal_ir.h's builder for the region it is replayed
# into, with the kind of each result; compositions (erf, floor division, isnan, integer powers) expand here into the
# builder's binary/unary/convert/bitcast/builtin calls
class _Region:
    def __init__(self):
        self.code, self.kinds = [], []

    def add(self, record, kind):
        self.code.append(record)
        self.kinds.append(kind)
        return len(self.code) - 1

    def __call__(self, name, kind, *x, integer=0):
        if name in BINARY:
            return self.add(('binary', BINARY[name], *x), BOOL if name in COMPARE else kind)
        if name == 'neg':
            return self.add(('unary', NEGATION, x[0]), kind)
        if name == 'not':
            return self.add(('unary', NOT, x[0]), BOOL)
        if name in BUILTINS:
            return self.add(('builtin', name, x), kind)
        if name == 'select':
            return self.add(('builtin', 'select', (x[2], x[1], x[0])), kind)
        if name == 'cast':
            return self.add(('convert', kind, x[0]), kind)
        if name == 'bitcast':
            return self.add(('bitcast', kind, x[0]), kind)
        if name == 'arg':
            return self.add(('parameter', integer), kind)
        if name == 'extract':
            return self.add(('extract', integer, x), kind)
        if name == 'isnan':
            return self('ne', BOOL, x[0], x[0])
        if name == 'erf':
            return _erf(self, x[0], kind)
        if name == 'floordiv':
            return _floordiv(self, x[0], x[1], kind)
        if name == 'truncdiv':
            return _truncdiv(self, x[0], x[1], kind)
        if name == 'powi':
            return _powi(self, x[0], x[1], kind)
        if name == 'yield':
            return self.add(('yield', x), BOOL)
        raise KeyError(name)

    # WGSL §6.2.1 (and Tint core constants) hold finite values only: an infinity or NaN is its binary32 pattern's bitcast
    def const(self, value, kind):
        if _floating(kind) and not math.isfinite(value):
            made = self.add(('bitcast', F32, self.add(('literal', U32, _bits(value, F32)), U32)), F32)
            return made if kind == F32 else self('cast', kind, made)
        return self.add(('literal', kind, _bits(value, kind)), kind)

    def arg(self, k, value):
        return self('arg', _class(value), integer=k)

    def index(self, d):
        return self.add(('convert', I32, self.add(('loop', d), U32)), I32)

    def done(self, *results):
        self('yield', BOOL, *results)
        return self


# N. Juffa, Stack Overflow answer 35148199, 2016; torch 2.14.0 c10/metal/special_math.h, erf
def _erf(r, a, t):
    def c(v):
        return r.const(v, t)

    def fma(x, y, z):
        return r('add', t, r('mul', t, x, y), z)
    s, u = r('abs', t, a), r('mul', t, a, a)
    w = fma(c(-1.72853470e-5), s, c(3.83197126e-4))
    w = fma(w, u, fma(c(-3.88396438e-3), s, c(2.42546219e-2)))
    for k in (-1.06777877e-1, -6.34846687e-1, -1.28717512e-1):
        w = fma(w, s, c(k))
    w = fma(w, s, r('neg', t, s))
    far = r('sub', t, c(1), r('exp', t, w))
    far = r('select', t, r('lt', BOOL, a, c(0)), r('neg', t, far), far)
    q = c(-5.96761703e-4)
    for k in (4.99119423e-3, -2.67681349e-2, 1.12819925e-1, -3.76125336e-1, 1.28379166e-1):
        q = fma(q, u, c(k))
    return r('select', t, r('gt', BOOL, s, c(0.927734375)), far, fma(q, a, a))


# IEEE 754-2019 §5.5.1 copySign: a zero with the sign bit of x
def _signed_zero(r, x, t):
    if t == F16:
        x = r('cast', F32, x)
    sign = r('and', U32, r('bitcast', U32, x), r.const(0x80000000, U32))
    zero = r('bitcast', F32, sign)
    return r('cast', F16, zero) if t == F16 else zero


# torch 2.14.0 c10/util/generic_math.h, div_floor_floating and div_floor_integer
def _floordiv(r, a, b, t):
    def c(v):
        return r.const(v, t)
    if not _floating(t):
        safe = r('select', t, r('eq', BOOL, b, c(0)), c(1), b)
        q = r('div', t, a, safe)
        if t != U32:
            fix = r('and', BOOL, r('ne', BOOL, r('rem', t, a, safe), c(0)),
                    r('ne', BOOL, r('lt', BOOL, a, c(0)), r('lt', BOOL, b, c(0))))
            q = r('select', t, fix, r('sub', t, q, c(1)), q)
        return r('select', t, r('eq', BOOL, b, c(0)), c(0), q)
    mod = r('rem', t, a, b)
    div = r('div', t, r('sub', t, a, mod), b)
    fix = r('and', BOOL, r('ne', BOOL, mod, c(0)), r('ne', BOOL, r('lt', BOOL, b, c(0)), r('lt', BOOL, mod, c(0))))
    div = r('select', t, fix, r('sub', t, div, c(1)), div)
    floored = r('floor', t, div)
    rounded = r('select', t, r('gt', BOOL, r('sub', t, div, floored), c(0.5)), r('add', t, floored, c(1)), floored)
    quotient = r('div', t, a, b)
    zero = r('select', t, r('eq', BOOL, div, c(0)), _signed_zero(r, quotient, t), rounded)
    return r('select', t, r('eq', BOOL, b, c(0)), quotient, zero)


def _truncdiv(r, a, b, t):
    if _floating(t):
        return r('trunc', t, r('div', t, a, b))
    zero = r('eq', BOOL, b, r.const(0, t))
    return r('select', t, zero, r.const(0, t), r('div', t, a, r('select', t, zero, r.const(1, t), b)))


# torch 2.14.0 ATen/native/Pow.h, powi: exponentiation by squaring over the exponent's bits
def _powi(r, a, b, t):
    one, zero = r.const(1, t), r.const(0, t)
    result, base, rest = one, a, b
    for _ in range(64 if t in (I64, U64) else 32):
        odd = r('ne', BOOL, r('and', t, rest, one), zero)
        result = r('select', t, odd, r('mul', t, result, base), result)
        base = r('mul', t, base, base)
        rest = r('div', t, rest, r.const(2, t))
    if t in (U32, U64):
        return result
    odd = r('ne', BOOL, r('and', t, b, one), zero)
    unit = r('select', t, r('eq', BOOL, a, one), one,
             r('select', t, r('eq', BOOL, a, r.const(-1, t)), r('select', t, odd, r.const(-1, t), one), zero))
    return r('select', t, r('lt', BOOL, b, zero), unit, result)


# one structured operation, metal_ir_generic's operands: its loops `sizes` with `iterators`, inputs and outputs each
# (tensor view, affine rows over the loops, (loops + 1) coefficients a row; None: read only by metal_ir_extract), per
# output its combiner, starting value and accumulator kind, coupled REGION states (kind, identity) with their combine
# and finalize regions, and the permitted numerical transformations
class _Generic:
    def __init__(self, sizes, inputs, outputs, body, iterators=None, combiner=None, accumulator=None, start=None,
                 states=None, combine=None, finalize=None, precision=0):
        self.sizes, self.inputs, self.outputs, self.body = tuple(sizes), list(inputs), list(outputs), body
        self.iterators = tuple(iterators) if iterators is not None else (PARALLEL,) * len(self.sizes)
        self.combiner = list(combiner) if combiner is not None else [ASSIGN] * len(self.outputs)
        self.accumulator = list(accumulator) if accumulator is not None else [_class(t) for t, _ in self.outputs]
        self.start = list(start) if start is not None else [IDENTITY] * len(self.outputs)
        self.states, self.combine, self.finalize, self.precision = states, combine, finalize, precision
        self.place = GPU


def _rows(dims, picks):
    made = []
    for pick in picks:
        row = [0] * (dims + 1)
        for d, c in ({} if pick is None else {pick: 1} if isinstance(pick, int) else pick).items():
            row[dims if d is None else d] += c
        made.append(tuple(row))
    return tuple(made)


def _same(dims, rank=None):
    return _rows(dims, range(dims if rank is None else rank))


def _aligned(value, shape):
    lead = len(shape) - value.dim()
    return _rows(len(shape), [None if n == 1 and shape[lead + d] != 1 else lead + d for d, n in enumerate(value.shape)])


def _onto(value, picks, sizes):
    return _rows(len(sizes), [None if pick is not None and n == 1 and sizes[pick] != 1 else pick
                              for n, pick in zip(value.shape, picks)])


def _temporary(shape, dtype=torch.float32):
    return torch.empty(tuple(shape), dtype=dtype, device='meta')


def _typed(target, operands, outs, results=FLOATING):
    found = [value.dtype for value in operands if isinstance(value, torch.Tensor) and value.dtype not in REAL] \
        + [value.dtype for value in outs if value.dtype not in results]
    if found:
        raise NotImplementedError('%s of %s: its lowering into the structured basis is unfinished '
                                  '(design/compilation-composition-change-request.md item 1)' % (target, found[0]))


# torch 2.14.0 aten/src/ATen/OpMathType.h opmath_type: an operation's arithmetic kind
def _compute(*dtypes):
    if any(d is torch.float64 for d in dtypes):
        return F64
    if any(d.is_floating_point for d in dtypes):
        return F32
    kinds = {SCALAR[d] for d in dtypes}
    return I64 if kinds & {I64, U64} else U32 if U32 in kinds else I32


# torch 2.14.0 c10/util/TypeCast.h static_cast_with_inter_type: to bool by comparison with zero, an integer to a reduced
# float through float
def _convert(r, x, source, target):
    if source == target:
        return x
    if target == BOOL:
        return r('ne', BOOL, x, r.const(0, source))
    if not _floating(source) and target == F16:
        x = r('cast', F32, x)
    return r('cast', target, x)


def _load(r, k, value, kind):
    return _convert(r, r.arg(k, value), _class(value), kind)


def _store(r, x, kind, value):
    return _convert(r, x, kind, _class(value))


def _operands(target, args, kwargs):
    schema = target._schema.arguments
    return {**{a.name: a.default_value for a in schema if a.has_default_value()},
            **dict(zip((a.name for a in schema), args)), **kwargs}


# PyTorch v2.11.0 aten/src/ATen/native/cpu/PowKernel.cpp, pow_tensor_scalar_optimized_kernel: a static exponent
# specialized into its instructions
def _power(r, x, y, t, exponent):
    one = r.const(1, t)
    if exponent == 2:
        return r('mul', t, x, x)
    if exponent == 3:
        return r('mul', t, r('mul', t, x, x), x)
    if exponent == -2:
        return r('div', t, one, r('mul', t, x, x))
    if exponent == 0.5:
        return r('sqrt', t, x)
    if exponent == -0.5:
        return r('div', t, one, r('sqrt', t, x))
    if exponent == -1:
        return r('div', t, one, x)
    return r('pow', t, x, y)


# PyTorch 2.14 documentation of each operation named; torch 2.14.0 c10/util/MathConstants.h M_SQRT1_2, M_2_SQRTPI
SQRT1_2, TWO_SQRTPI = 0.707106781186547524401, 1.12837916709551257390


def _apply(r, op, x, y, z, alpha, t, exponent=None):
    def c(v):
        return r.const(v, t)
    if op in ('add', 'sub'):
        return r(op, t, x, y if alpha == 1 else r('mul', t, c(alpha), y))
    if op in ('mul', 'div', 'floordiv', 'truncdiv'):
        return r(op, t, x, y)
    if op in ('neg', 'tanh', 'exp', 'sqrt', 'log', 'abs'):
        return r(op, t, x)
    if op == 'relu':
        return r('select', t, r('lt', BOOL, x, c(0)), c(0), x)
    if op == 'sigmoid':
        return r('div', t, c(1), r('add', t, c(1), r('exp', t, r('neg', t, x))))
    if op == 'silu':
        return r('div', t, x, r('add', t, c(1), r('exp', t, r('neg', t, x))))
    if op == 'gelu':
        return r('mul', t, r('mul', t, c(0.5), x), r('add', t, c(1), r('erf', t, r('mul', t, x, c(SQRT1_2)))))
    if op == 'gelu_tanh':
        inner = r('add', t, x, r('mul', t, c(0.044715), r('mul', t, r('mul', t, x, x), x)))
        return r('mul', t, r('mul', t, c(0.5), x),
                 r('add', t, c(1), r('tanh', t, r('mul', t, c(TWO_SQRTPI * SQRT1_2), inner))))
    if op == 'rsqrt':
        return r('div', t, c(1), r('sqrt', t, x))
    if op in COMPARE:
        return r(op, BOOL, x, y)
    if op == 'pow':
        return r('powi', t, x, y) if not _floating(t) else _power(r, x, y, t, exponent)
    if op == 'clamp':
        lowered = x if y is None else r('select', t, r('lt', BOOL, x, y), y, x)
        return lowered if z is None else r('select', t, r('lt', BOOL, z, lowered), z, lowered)
    return r('select', t, r('ne', BOOL, x, c(0)), y, z)


# PyTorch 2.14 documentation, Broadcasting semantics and Type promotion: tensor operands broadcast by their maps' zero
# rows, scalars the region's literals
def _elementwise(op, out, a, b=None, c=None, alpha=1):
    given = [a, b, c]
    tensors = [x for x in given if isinstance(x, torch.Tensor)]
    integral = out.dtype in INTEGRAL and all(x.dtype in INTEGRAL if isinstance(x, torch.Tensor) else
                                             isinstance(x, (bool, int)) for x in given if x is not None)
    if integral and not isinstance(alpha, (bool, int)):
        raise NotImplementedError('elementwise %s into %s of a scalar no integer holds' % (op, out.dtype))
    t = _compute(out.dtype, *[x.dtype for x in tensors]) if integral else \
        _compute(out.dtype, *[x.dtype for x in tensors], torch.float32)
    r, inputs, values = _Region(), [], []
    for x in given:
        if isinstance(x, torch.Tensor):
            values.append(_load(r, len(inputs), x, t))
            inputs.append((x, _aligned(x, out.shape)))
        else:
            values.append(None if x is None else r.const(x, t))
    made = _apply(r, op, *values, alpha, t, b if op == 'pow' and not isinstance(b, torch.Tensor) else None)
    r.done(_store(r, made, BOOL if op in COMPARE else t, out))
    return [_Generic(out.shape, inputs, [(out, _same(out.dim()))], r)]


POINTWISE = {A.add: 'add', A.sub: 'sub', A.mul: 'mul', A.div: 'div', A.floor_divide: 'floordiv', A.neg: 'neg',
             A.relu: 'relu', A.tanh: 'tanh', A.sigmoid: 'sigmoid', A.exp: 'exp', A.gelu: 'gelu', A.silu: 'silu',
             A.eq: 'eq', A.ne: 'ne', A.lt: 'lt', A.le: 'le', A.gt: 'gt', A.ge: 'ge', A.pow: 'pow', A.rsqrt: 'rsqrt',
             A.sqrt: 'sqrt', A.log: 'log', A.abs: 'abs', A.clamp: 'clamp', A.clamp_min: 'clamp',
             A.clamp_max: 'clamp', A.where: 'where'}


def _arithmetic(target, given, outs):
    packet, op = target.overloadpacket, POINTWISE[target.overloadpacket]
    if packet is A.div and given.get('rounding_mode') is not None:
        op = {'floor': 'floordiv', 'trunc': 'truncdiv'}[given['rounding_mode']]
    if packet is A.gelu and given['approximate'] != 'none':
        op = 'gelu_tanh'
    operands = [given['condition'], given['self'], given['other']] if packet is A.where else \
        [given['self'], given.get('min'), given.get('max')] if op == 'clamp' else \
        [given['self'], given['exponent'], None] if packet is A.pow else [given['self'], given.get('other'), None]
    _typed(target, operands, outs, REAL)
    return _elementwise(op, outs[0], *operands, given.get('alpha', 1))


# torch 2.14.0 torch/_tensor_docs.py, copy_ and view: views of one element count correspond by their row-major
# element index, the source's coordinates computed in the region
def _coordinates(r, linear, shape):
    made, rest = [], linear
    for d in reversed(range(len(shape))):
        made.append(r('rem', I32, rest, r.const(shape[d], I32)))
        rest = r('div', I32, rest, r.const(shape[d], I32))
    return made[::-1]


def _linear(r, shape):
    at = r.const(0, I32)
    for d, n in enumerate(shape):
        at = r('add', I32, r('mul', I32, at, r.const(n, I32)), r.index(d))
    return at


def _copy(out, value):
    _typed('a copy', [value], [out], REAL)
    r = _Region()
    if tuple(value.shape) == tuple(out.shape):
        r.done(_convert(r, r.arg(0, value), _class(value), _class(out)))
        return [_Generic(out.shape, [(value, _same(out.dim()))], [(out, _same(out.dim()))], r)]
    at = _coordinates(r, _linear(r, out.shape), value.shape)
    r.done(_convert(r, r('extract', _class(value), *at, integer=0), _class(value), _class(out)))
    return [_Generic(out.shape, [(value, None)], [(out, _same(out.dim()))], r)]


def _assign(out, value):
    if not isinstance(value, torch.Tensor):
        return _fill(out, value)
    return _copy(out, value.expand(out.shape))


# torch 2.14.0 torch/_tensor_docs.py, fill_; PyTorch 2.14 documentation, torch.arange
def _fill(out, value, step=0):
    _typed('a fill', [], [out], REAL)
    t = _compute(out.dtype)
    r = _Region()
    made = r.const(value, t)
    if step:
        made = r('add', t, made, r('mul', t, _convert(r, _linear(r, out.shape), I32, t), r.const(step, t)))
    r.done(_store(r, made, t, out))
    return [_Generic(out.shape, [], [(out, _same(out.dim()))], r)]


def _copying(target, given, outs):
    return _assign(outs[0], given['src']) if target.overloadpacket in (A.copy, A.copy_) else _copy(outs[0], given['self'])


# an output kept in its arithmetic kind: the structured map out = convert(f(acc, operands...)), fused by the
# transformation with its producer
def _epilogue(out, acc, t, extra=(), apply=None):
    r = _Region()
    x = _load(r, 0, acc, t)
    loaded = [_load(r, 1 + j, v, t) for j, (v, _) in enumerate(extra)]
    r.done(_store(r, apply(r, x, *loaded) if apply else x, t, out))
    return [_Generic(out.shape, [(acc, _same(out.dim()))] + list(extra), [(out, _same(out.dim()))], r)]


# J. Gibbons, The third homomorphism theorem, J. Functional Programming 6(4), 1996, (2) and Definition 3.1: a
# reduction's states, their identity and associative join; `_identities` the finalization that is the states
def _identities(states):
    r = _Region()
    return r.done(*[r('arg', kind, integer=j) for j, (kind, _) in enumerate(states)])


def _joined(states, join):
    c = _Region()
    n = len(states)
    a = [c('arg', kind, integer=j) for j, (kind, _) in enumerate(states)]
    b = [c('arg', kind, integer=n + j) for j, (kind, _) in enumerate(states)]
    return c.done(*join(c, a, b))


def _extremum(r, a, b, t, larger):
    better = r('gt' if larger else 'lt', BOOL, b, a)
    if _floating(t):
        better = r('or', BOOL, better, r('isnan', BOOL, b))
    return r('select', t, better, b, a)


# PyTorch 2.14 documentation, torch.einsum: the output's labels PARALLEL, the summed REDUCTION (SUM in the operands'
# arithmetic kind), each operand's map its labels' loops (a broadcast extent 1 a zero row)
def _contract(out, operands, labels, output, addend=None, alpha=1, beta=0):
    sizes = dict(zip(output, out.shape))
    for value, own in zip(operands, labels):
        for size, label in zip(value.shape, own):
            sizes[label] = max(sizes.get(label, 1), size)
    summed = [label for label in dict.fromkeys(label for own in labels for label in own) if label not in output]
    domain = list(output) + summed
    at = {label: d for d, label in enumerate(domain)}
    t = _compute(out.dtype, *[value.dtype for value in operands], torch.float32)
    direct = alpha == 1 and (addend is None or beta == 0)
    acc = out if direct else _temporary(out.shape, STORED[t])
    r = _Region()
    product = None
    for j, value in enumerate(operands):
        x = _load(r, j, value, t)
        product = x if product is None else r('mul', t, product, x)
    r.done(product)
    maps = [_rows(len(domain), [None if n == 1 and sizes[label] != 1 else at[label] for n, label in zip(v.shape, own)])
            for v, own in zip(operands, labels)]
    made = [_Generic([sizes[label] for label in domain], list(zip(operands, maps)),
                     [(acc, _rows(len(domain), [at[label] for label in output]))], r,
                     [PARALLEL] * len(output) + [REDUCTION] * len(summed), [SUM], [t], precision=REASSOCIATE | CONTRACT)]
    if direct:
        return made

    def scaled(r, x, *c):
        x = x if alpha == 1 else r('mul', t, r.const(alpha, t), x)
        return x if not c else r('add', t, x, c[0] if beta == 1 else r('mul', t, r.const(beta, t), c[0]))
    extra = [(addend, _aligned(addend, out.shape))] if addend is not None and beta != 0 else []
    return made + _epilogue(out, acc, t, extra, scaled)


def _matrix(out, a, b, bias=None):
    return _contract(out, [a, b], [('n', 'k'), ('k', 'm')], ('n', 'm'), bias, 1, int(bias is not None))


def _folded(value):
    try:
        return value.view(-1, value.shape[-1])
    except RuntimeError:
        return None


# PyTorch 2.14 documentation, torch.mm, torch.bmm, torch.matmul, torch.addmm and torch.nn.functional.linear; PyTorch
# v2.11.0 aten/src/ATen/native/LinearAlgebra.cpp, _matmul_impl
def _product(target, given, outs):
    packet, (out,) = target.overloadpacket, outs
    if packet is A.addmm:
        c, a, b, alpha, beta = given['self'], given['mat1'], given['mat2'], given['alpha'], given['beta']
        _typed(target, [c, a, b], outs)
        return _contract(out, [a, b], [('n', 'k'), ('k', 'm')], ('n', 'm'), c if beta != 0 else None, alpha, beta)
    if packet is A.linear:
        x, w, bias = given['input'], given['weight'], given['bias']
        _typed(target, [x, w, bias], outs)
        rows, folded = _folded(x), _folded(out)
        if w.dim() == 2 and rows is not None and folded is not None:
            return _matrix(folded, rows, w.t(), bias)
        lead = tuple(('batch', i) for i in range(x.dim() - 1))
        return _contract(out, [x, w], [lead + ('k',), ('m', 'k')[2 - w.dim():]], lead + ('m',) * (w.dim() == 2), bias,
                         1, int(bias is not None))
    a, b = given['self'], given['mat2'] if packet in (A.mm, A.bmm) else given['other']
    _typed(target, [a, b], outs)
    if packet is A.mm or packet is A.matmul and a.dim() == b.dim() == 2:
        return _matrix(out, a, b)
    if packet is A.matmul and a.dim() > 2 and b.dim() == 2 and _folded(a) is not None and _folded(out) is not None:
        return _matrix(_folded(out), _folded(a), b)
    batch = out.dim() - (a.dim() > 1) - (b.dim() > 1)

    def side(value, matrix):
        rank = value.dim() - len(matrix)
        return tuple(('batch', batch - rank + i) for i in range(rank)) + matrix
    return _contract(out, [a, b], [side(a, ('n', 'k') if a.dim() > 1 else ('k',)),
                                   side(b, ('k', 'm') if b.dim() > 1 else ('k',))],
                     tuple(('batch', i) for i in range(batch)) + ('n',) * (a.dim() > 1) + ('m',) * (b.dim() > 1))


# PyTorch 2.14 documentation, torch.einsum: more than two operands contracted pairwise left to right, each step's
# result keeping the labels a later operand or the output uses (no product loop over every summed label at once)
def _summation(target, given, outs):
    (out,), operands = outs, list(given['tensors'])
    _typed(target, operands, outs)
    left, arrow, right = given['equation'].replace(' ', '').partition('->')
    terms = left.split(',')
    covered = max([value.dim() - len(term.replace('...', '')) for term, value in zip(terms, operands) if '...' in term],
                  default=0)
    ellipsis = tuple(('...', i) for i in range(covered))

    def labelled(term, rank):
        head, dots, tail = term.partition('...')
        return tuple(head) + (ellipsis[covered - (rank - len(head) - len(tail)):] if dots else ()) + tuple(tail)
    letters = left.replace('...', '').replace(',', '')
    output = labelled(right, out.dim()) if arrow else \
        ellipsis + tuple(sorted(c for c in set(letters) if letters.count(c) == 1))
    labels = [labelled(term, value.dim()) for term, value in zip(terms, operands)]
    if len(operands) <= 2:
        return _contract(out, operands, labels, output)
    sizes = {}
    for value, own in zip(operands, labels):
        for size, label in zip(value.shape, own):
            sizes[label] = max(sizes.get(label, 1), size)
    ops, made, own = [], operands[0], labels[0]
    for j in range(1, len(operands)):
        later = {label for others in labels[j + 1:] for label in others} | set(output)
        kept = output if j == len(operands) - 1 else tuple(dict.fromkeys(l for l in own + labels[j] if l in later))
        step = out if j == len(operands) - 1 else _temporary([sizes[label] for label in kept], out.dtype)
        ops += _contract(step, [made, operands[j]], [own, labels[j]], kept)
        made, own = step, kept
    return ops


# an index that may leave [0, size): its in-range test and the index clamped into range, so the extraction is defined
def _bounded(r, at, size):
    zero, last = r.const(0, I32), r.const(size - 1, I32)
    inside = r('and', BOOL, r('ge', BOOL, at, zero), r('le', BOOL, at, last))
    return inside, r('select', I32, r('lt', BOOL, at, zero), zero, r('select', I32, r('gt', BOOL, at, last), last, at))


def _all(r, tests):
    made = tests[0]
    for test in tests[1:]:
        made = r('and', BOOL, made, test)
    return made


def _spatial(value, k):
    for _ in range(3 - k):
        value = value.unsqueeze(2)
    return value


def _affine(r, terms, constant=0):
    made = r.const(constant, I32)
    for x, c in terms:
        made = r('add', I32, made, x if c == 1 else r('mul', I32, x, r.const(c, I32)))
    return made


# PyTorch 2.14 documentation, torch.nn.Conv1d to torch.nn.Conv3d and torch.nn.ConvTranspose1d to
# torch.nn.ConvTranspose3d; PyTorch v2.11.0 aten/src/ATen/native/Convolution.cpp, convolution_same: the loops
# (n, group, output channel of the group, o0, o1, o2 | input channel of the group, k0, k1, k2), the groups' channel
# offsets affine; an input position outside the padded image contributes nothing
def _convolving(target, given, outs):
    packet, (out,) = target.overloadpacket, outs
    x, w, bias = given['input'], given['weight'], given['bias']
    _typed(target, [x, w, bias], outs)
    transposed = given['transposed'] if packet is A.convolution else packet in TRANSPOSED
    k = w.dim() - 2
    stride, dilation, padding = _ints(given['stride'], k), _ints(given['dilation'], k), given['padding']
    lows = (0,) * k if padding == 'valid' else tuple(d * (n - 1) // 2 for d, n in zip(dilation, w.shape[2:])) \
        if padding == 'same' else _ints(padding, k)
    if x.dim() == k + 1:
        x, out = x.unsqueeze(0), out.unsqueeze(0)
    x, w, out = _spatial(x, k), _spatial(w, k), _spatial(out, k)
    stride, dilation, lows = (1,) * (3 - k) + stride, (1,) * (3 - k) + dilation, (0,) * (3 - k) + lows
    groups = given['groups']
    outputs, inputs = out.shape[1] // groups, x.shape[1] // groups
    sizes = [x.shape[0], groups, outputs, *out.shape[2:], inputs, *w.shape[2:]]
    t = _compute(out.dtype, x.dtype, w.dtype, torch.float32)
    acc = out if bias is None else _temporary(out.shape, STORED[t])
    r = _Region()
    o, kk = [r.index(3 + d) for d in range(3)], [r.index(7 + d) for d in range(3)]
    if not transposed and not any(lows):
        xmap = _rows(10, [0, {1: inputs, 6: 1}, *[{3 + d: stride[d], 7 + d: dilation[d]} for d in range(3)]])
        r.done(r('mul', t, _load(r, 0, x, t), _load(r, 1, w, t)))
    else:
        tests, at = [], []
        for d in range(3):
            if transposed:
                shifted = _affine(r, [(o[d], 1), (kk[d], -dilation[d])], lows[d])
                tests += [r('ge', BOOL, shifted, r.const(0, I32)),
                          r('eq', BOOL, r('rem', I32, shifted, r.const(stride[d], I32)), r.const(0, I32))]
                inside, clamped = _bounded(r, r('div', I32, shifted, r.const(stride[d], I32)), x.shape[2 + d])
            else:
                inside, clamped = _bounded(r, _affine(r, [(o[d], stride[d]), (kk[d], dilation[d])], -lows[d]),
                                           x.shape[2 + d])
            tests.append(inside)
            at.append(clamped)
        ci = _affine(r, [(r.index(1), inputs), (r.index(6), 1)])
        xv = _convert(r, r('extract', _class(x), r.index(0), ci, *at, integer=0), _class(x), t)
        r.done(r('select', t, _all(r, tests), r('mul', t, xv, _load(r, 1, w, t)), r.const(0, t)))
        xmap = None
    wmap = _rows(10, [{1: inputs, 6: 1}, 2, 7, 8, 9] if transposed else [{1: outputs, 2: 1}, 6, 7, 8, 9])
    made = [_Generic(sizes, [(x, xmap), (w, wmap)], [(acc, _rows(10, [0, {1: outputs, 2: 1}, 3, 4, 5]))], r,
                     [PARALLEL] * 6 + [REDUCTION] * 4, [SUM], [t], precision=REASSOCIATE | CONTRACT)]
    if bias is None:
        return made
    return made + _epilogue(out, acc, t, [(bias, _rows(5, [1]))], lambda r, a, b: r('add', t, a, b))


# PyTorch v2.11.0 aten/src/ATen/native/cpu/MaxPoolKernel.cpp, cpu_max_pool; aten/src/ATen/native/cpu/AvgPoolKernel.cpp,
# cpu_avg_pool2d and cpu_avg_pool3d: the loops (lead0, lead1, o0, o1, o2 | k0, k1, k2); the maximum's states its value
# and flat input position, NaN the greatest and the later of NaNs, the window's first element taken
INDEXED = {A.max_pool2d_with_indices: 2, A.max_pool3d_with_indices: 3}


def _pooled(target, given, outs):
    packet, x = target.overloadpacket, given['self']
    k, out, index = {**MAXIMUM, **AVERAGE, **INDEXED}[packet], outs[0], outs[1] if len(outs) > 1 else None
    _typed(target, [x], outs[:1])
    lows, dilation, lead = _ints(given['padding'], k), _ints(given.get('dilation', 1), k), x.dim() - k
    kernel, stride = _ints(given['kernel_size'], k), _ints(given['stride'] or given['kernel_size'], k)

    def shaped(value):
        for _ in range(2 - lead):
            value = value.unsqueeze(0)
        for _ in range(3 - k):
            value = value.unsqueeze(2)
        return value
    xs, os = shaped(x), shaped(out)
    kernel, stride = (1,) * (3 - k) + kernel, (1,) * (3 - k) + stride
    lows, dilation = (0,) * (3 - k) + lows, (1,) * (3 - k) + dilation
    average = packet in AVERAGE
    sizes, t, omap = [*os.shape, *kernel], _compute(out.dtype, x.dtype, torch.float32), _rows(8, range(5))
    r = _Region()
    tests, at = [], []
    for d in range(3):
        inside, clamped = _bounded(r, _affine(r, [(r.index(2 + d), stride[d]),
                                                  (r.index(5 + d), 1 if average else dilation[d])], -lows[d]),
                                   xs.shape[2 + d])
        tests.append(inside)
        at.append(clamped)
    inside = _all(r, tests)
    v = _convert(r, r('extract', _class(x), r.index(0), r.index(1), *at, integer=0), _class(x), t)
    iterators = [PARALLEL] * 5 + [REDUCTION] * 3
    if not average:
        place = _affine(r, [(_affine(r, [(at[0], xs.shape[3]), (at[1], 1)]), xs.shape[4]), (at[2], 1)])
        states = [(t, -math.inf)] + ([(I32, -1)] if index is not None else [])
        found = [r('select', t, inside, v, r.const(-math.inf, t))]
        if index is not None:
            found.append(r('select', I32, inside, place, r.const(-1, I32)))
        r.done(*found)

        def join(c, a, b):
            right = c('or', BOOL, c('gt', BOOL, b[0], a[0]), c('isnan', BOOL, b[0]))
            if index is None:
                return [c('select', t, right, b[0], a[0])]
            right = c('and', BOOL, c('ge', BOOL, b[1], c.const(0, I32)),
                      c('or', BOOL, c('lt', BOOL, a[1], c.const(0, I32)), right))
            return [c('select', t, right, b[0], a[0]), c('select', I32, right, b[1], a[1])]
        outputs = [(os, omap)] + ([(shaped(index), omap)] if index is not None else [])
        return [_Generic(sizes, [(xs, None)], outputs, r, iterators, [REGION] * len(outputs), [kind for kind, _ in states],
                         None, states, _joined(states, join), _identities(states), REASSOCIATE)]
    acc = _temporary(os.shape, STORED[t])
    r.done(r('select', t, inside, v, r.const(0, t)))
    made = [_Generic(sizes, [(xs, None)], [(acc, omap)], r, iterators, [SUM], [t], precision=REASSOCIATE)]
    override, include = given.get('divisor_override'), given.get('count_include_pad', True)

    def divided(r, total):
        pool, count = r.const(1, I32), r.const(1, I32)
        for d in range(3):
            lo = _affine(r, [(r.index(2 + d), stride[d])], -lows[d])
            hi = r('add', I32, lo, r.const(kernel[d], I32))
            edge = r.const(xs.shape[2 + d] + lows[d], I32)
            hi = r('select', I32, r('lt', BOOL, edge, hi), edge, hi)
            pool = r('mul', I32, pool, r('sub', I32, hi, lo))
            zero, size = r.const(0, I32), r.const(xs.shape[2 + d], I32)
            lo = r('select', I32, r('lt', BOOL, lo, zero), zero, lo)
            hi = r('select', I32, r('lt', BOOL, size, hi), size, hi)
            span = r('sub', I32, hi, lo)
            count = r('mul', I32, count, r('select', I32, r('lt', BOOL, span, zero), zero, span))
        divisor = r.const(override, t) if override else _convert(r, pool if include else count, I32, t)
        return r('select', t, r('eq', BOOL, count, r.const(0, I32)), r.const(0, t), r('div', t, total, divisor))
    return made + _epilogue(os, acc, t, (), divided)


# PyTorch v2.11.0 aten/src/ATen/native/cpu/PaddingKernel.cpp, ReflectionPad and ReplicationPad;
# aten/src/ATen/native/PadNd.cpp, constant_pad_nd and _pad_circular_symint
MODES = {'constant': 0, 'reflect': 1, 'replicate': 2, 'circular': 3}


def _padded(target, given, outs):
    (out,), x = outs, given['self']
    _typed(target, [x], outs, REAL)
    pad, n = list(given['pad']), x.dim()
    low, high, mode = [0] * n, [0] * n, MODES[given.get('mode', 'constant')]
    for j in range(len(pad) // 2):
        low[n - 1 - j], high[n - 1 - j] = pad[2 * j], pad[2 * j + 1]
    r, tests, at = _Region(), [], []
    for d in range(n):
        size = x.shape[d]
        if low[d] == 0 and high[d] == 0:
            at.append(r.index(d))
            continue
        j = _affine(r, [(r.index(d), 1)], -low[d])
        if mode == 0:
            inside, j = _bounded(r, j, size)
            tests.append(inside)
        elif mode == 1:
            j = r('abs', I32, j)
            j = r('select', I32, r('ge', BOOL, j, r.const(size, I32)), r('sub', I32, r.const(2 * (size - 1), I32), j), j)
        elif mode == 2:
            j = _bounded(r, j, size)[1]
        else:
            extent = r.const(max(size - max(-low[d], 0) - max(-high[d], 0), 1), I32)
            shifted = _affine(r, [(r.index(d), 1)], -max(low[d], 0))
            wrapped = r('rem', I32, r('add', I32, r('rem', I32, shifted, extent), extent), extent)
            j = r('add', I32, r.const(max(-low[d], 0), I32), wrapped)
        at.append(j)
    t = _class(x)
    v = r('extract', t, *at, integer=0)
    if tests:
        v = r('select', t, _all(r, tests), v, r.const(given.get('value') or 0, t))
    r.done(_convert(r, v, t, _class(out)))
    return [_Generic(out.shape, [(x, None)], [(out, _same(out.dim()))], r)]


# PyTorch 2.14 documentation, torch.full, torch.zeros, torch.ones, torch.empty, torch.arange, torch.scalar_tensor,
# Tensor.new_full, Tensor.new_zeros, Tensor.new_ones, Tensor.new_empty and Tensor.fill_
FILLS = {A.full: 'fill_value', A.full_like: 'fill_value', A.new_full: 'fill_value', A.scalar_tensor: 's',
         A.fill: 'value', A.zeros: 0, A.zeros_like: 0, A.new_zeros: 0, A.zero: 0, A.ones: 1, A.ones_like: 1,
         A.new_ones: 1}
EMPTY = (A.empty, A.empty_like, A.new_empty, A.empty_strided)


def _factory(target, given, outs):
    packet, (out,) = target.overloadpacket, outs
    if packet in EMPTY:
        return []
    if packet is A.arange:
        return _fill(out, given.get('start', 0), given.get('step', 1))
    value = given[FILLS[packet]] if isinstance(FILLS[packet], str) else FILLS[packet]
    return _assign(out, value)


# design/homomorphism-sources-2026-11-08.md; L. Meertens, Algorithmics, CWI Monographs 1, 1986, "Inserted-in", (i);
# Gibbons 1996, (2) and Definition 3.1: each reduction's identity and its associative operator
def _identity(packet, dtype):
    if packet in (A.sum, A.mean):
        return 0
    if dtype.is_floating_point:
        return -math.inf if packet is A.amax else math.inf
    if dtype is torch.bool:
        return int(packet is A.amin)
    return torch.iinfo(dtype).min if packet is A.amax else torch.iinfo(dtype).max


# PyTorch 2.14 documentation, torch.sum, torch.mean, torch.amax and torch.amin: the free dimensions PARALLEL and the
# reduced REDUCTION; a sum SUM in its arithmetic kind, an extremum the NaN-propagating join of its state, the mean
# its sum's epilogue; `start` ACCUMULATE for a paged part continuing its line's earlier parts
REDUCTIONS = {A.sum: 0, A.mean: 1, A.amax: 2, A.amin: 3}


def _reduction(packet, x, dims, kept, t, start=IDENTITY):
    free = [d for d in range(x.dim()) if d not in dims]
    domain = free + list(dims)
    r = _Region()
    r.done(_load(r, 0, x, t))
    loops = [x.shape[d] for d in domain]
    xmap = _rows(len(domain), [domain.index(d) for d in range(x.dim())])
    omap = _rows(len(domain), range(len(free)))
    iterators = [PARALLEL] * len(free) + [REDUCTION] * len(dims)
    if packet in (A.sum, A.mean):
        return _Generic(loops, [(x, xmap)], [(kept, omap)], r, iterators, [SUM], [t], [start], precision=REASSOCIATE)
    states = [(t, _identity(packet, x.dtype))]
    return _Generic(loops, [(x, xmap)], [(kept, omap)], r, iterators, [REGION], [t], [start], states,
                    _joined(states, lambda c, a, b: [_extremum(c, a[0], b[0], t, packet is A.amax)]),
                    _identities(states), REASSOCIATE)


def _dims(given, x):
    dims = given.get('dim')
    return sorted({d % max(x.dim(), 1) for d in _ints(dims, 1)}) if dims not in (None, []) else list(range(x.dim()))


def _reduced(target, given, outs):
    (out,), x, packet = outs, given['self'], target.overloadpacket
    _typed(target, [x], outs, REAL)
    dims, t = _dims(given, x), _compute(out.dtype, x.dtype)
    acc = _temporary(out.shape, STORED[t]) if packet is A.mean else out
    kept = acc.squeeze(tuple(dims)) if given.get('keepdim') and dims else acc
    made = [_reduction(packet, x, dims, kept, t)]
    if packet is not A.mean:
        return made
    total = math.prod(x.shape[d] for d in dims)
    return made + _epilogue(out, acc, t, (), lambda r, s: r('div', t, s, r.const(total, t)))


# M. Milakov and N. Gimelshein, Online normalizer calculation for softmax, arXiv:1805.02867, 2018, Algorithm 3 and
# section 3.1, (4): the line's state (m, s), each element x contributing (x, 1), joined by (4)
SOFTMAX = {A.softmax: 0, A._softmax: 0, A.log_softmax: 1, A._log_softmax: 1}


def _larger(r, a, b, t):
    return r('select', t, r('or', BOOL, r('isnan', BOOL, a), r('gt', BOOL, a, b)), a, b)


def _online(c, a, b, t):
    (m, s), (other, more) = a, b
    top = _larger(c, m, other, t)
    joined = c('add', t, c('mul', t, s, c('exp', t, c('sub', t, m, top))),
               c('mul', t, more, c('exp', t, c('sub', t, other, top))))
    absent, empty = c('eq', BOOL, other, c.const(-math.inf, t)), c('eq', BOOL, m, c.const(-math.inf, t))
    return [c('select', t, absent, m, c('select', t, empty, other, top)),
            c('select', t, absent, s, c('select', t, empty, more, joined))]


def _lines(x, d):
    return [j for j in range(x.dim()) if j != d] + [d]


def _statistics(x, d, m, s, t, start=IDENTITY):
    order = _lines(x, d)
    states = [(t, -math.inf), (t, 0)]
    r = _Region()
    r.done(_load(r, 0, x, t), r.const(1, t))
    xmap, lmap = _rows(x.dim(), [order.index(j) for j in range(x.dim())]), _rows(x.dim(), range(x.dim() - 1))
    return _Generic([x.shape[j] for j in order], [(x, xmap)], [(m, lmap), (s, lmap)], r,
                    [PARALLEL] * (x.dim() - 1) + [REDUCTION], [REGION] * 2, [t, t], [start] * 2, states,
                    _joined(states, lambda c, a, b: _online(c, a, b, t)), _identities(states), REASSOCIATE)


def _normalizing(x, d, m, s, out, t, log):
    order = _lines(x, d)
    xmap, lmap = _rows(x.dim(), [order.index(j) for j in range(x.dim())]), _rows(x.dim(), range(x.dim() - 1))
    e = _Region()
    shifted = e('sub', t, _load(e, 0, x, t), e.arg(1, m))
    made = e('sub', t, shifted, e('log', t, e.arg(2, s))) if log else \
        e('mul', t, e('exp', t, shifted), e('div', t, e.const(1, t), e.arg(2, s)))
    e.done(_store(e, made, t, out))
    return _Generic([x.shape[j] for j in order], [(x, xmap), (m, lmap), (s, lmap)], [(out, xmap)], e)


def _softened(target, given, outs):
    (out,), x = outs, given['self']
    _typed(target, [x], outs)
    d = given['dim'] % max(x.dim(), 1)
    if not x.dim():
        x, out, d = x.view(1), out.view(1), 0
    lines = [x.shape[j] for j in _lines(x, d)[:-1]]
    t = _compute(out.dtype, x.dtype, torch.float32)
    m, s = _temporary(lines, STORED[t]), _temporary(lines, STORED[t])
    return [_statistics(x, d, m, s, t), _normalizing(x, d, m, s, out, t, SOFTMAX[target.overloadpacket])]


# PyTorch 2.14 documentation, torch.cumsum: element i of a line the sum of its elements j <= i, the loops
# (line..., i | j)
def _scanned(target, given, outs):
    (out,), x = outs, given['self']
    _typed(target, [x], outs, REAL)
    d = given['dim'] % max(x.dim(), 1)
    if not x.dim():
        x, out, d = x.view(1), out.view(1), 0
    order, n = _lines(x, d), x.dim()
    t = _compute(out.dtype, x.dtype)
    r = _Region()
    r.done(r('select', t, r('le', BOOL, r.index(n), r.index(n - 1)), _load(r, 0, x, t), r.const(0, t)))
    xmap = _rows(n + 1, [order.index(j) if j != d else n for j in range(n)])
    return [_Generic([x.shape[j] for j in order] + [x.shape[d]], [(x, xmap)],
                     [(out, _rows(n + 1, [order.index(j) for j in range(n)]))], r, [PARALLEL] * n + [REDUCTION], [SUM],
                     [t])]


# PyTorch 2.14 documentation, torch.sort and torch.topk: element j's rank the count of the line's elements before it
# in the requested order (NaN last ascending, first descending, ties by index), then each result position p takes the
# element whose rank it is
def _before(r, a, ia, b, ib, descending, t):
    earlier = r('lt', BOOL, ia, ib)
    ordered = r('or', BOOL, r('gt' if descending else 'lt', BOOL, a, b), r('and', BOOL, r('eq', BOOL, a, b), earlier))
    if not _floating(t):
        return ordered
    na, nb = r('isnan', BOOL, a), r('isnan', BOOL, b)
    nan = r('select', BOOL, r('eq', BOOL, na, nb), earlier, na if descending else nb)
    return r('select', BOOL, r('or', BOOL, na, nb), nan, ordered)


def _ranked(target, given, outs):
    (values, indices), x = outs, given['self']
    _typed(target, [x], [values], REAL)
    d = given['dim'] % max(x.dim(), 1)
    if not x.dim():
        x, values, indices, d = x.view(1), values.view(1), indices.view(1), 0
    length = x.shape[d]
    k, descending = (given['k'], given['largest']) if target.overloadpacket is A.topk else (length, given['descending'])
    order, n, t = _lines(x, d), x.dim(), _class(x)
    rank = _temporary([x.shape[j] for j in order], torch.int32)
    r = _Region()
    counted = _before(r, r.arg(1, x), r.index(n), r.arg(0, x), r.index(n - 1), descending, t)
    r.done(_convert(r, counted, BOOL, I32))
    xmap = _rows(n + 1, [order.index(j) for j in range(n)])
    tmap = _rows(n + 1, [order.index(j) if j != d else n for j in range(n)])
    counting = _Generic([x.shape[j] for j in order] + [length], [(x, xmap), (x, tmap)], [(rank, _rows(n + 1, range(n)))],
                        r, [PARALLEL] * n + [REDUCTION], [SUM], [I32], precision=REASSOCIATE)
    s = _Region()
    hit = s('eq', BOOL, s.arg(1, rank), s.index(n - 1))
    s.done(s('select', t, hit, s.arg(0, x), s.const(0, t)), s('select', I32, hit, s.index(n), s.const(-1, I32)))
    states = [(t, 0), (I32, -1)]

    def join(c, a, b):
        taken = c('ge', BOOL, b[1], c.const(0, I32))
        return [c('select', t, taken, b[0], a[0]), c('select', I32, taken, b[1], a[1])]
    pmap = _rows(n + 1, [order.index(j) for j in range(n)])
    placing = _Generic([x.shape[j] if j != d else k for j in order] + [length],
                       [(x, tmap), (rank, tmap)], [(values, pmap), (indices, pmap)], s, [PARALLEL] * n + [REDUCTION],
                       [REGION] * 2, [t, I32], None, states, _joined(states, join), _identities(states), REASSOCIATE)
    return [counting, placing]


# PyTorch 2.14 documentation, torch.nn.LayerNorm, torch.nn.GroupNorm and torch.nn.BatchNorm2d; torch 2.14.0
# torch/_refs/__init__.py, native_layer_norm, native_group_norm and _normalize; torch/_decomp/decompositions.py,
# native_batch_norm_helper: the statistics' groups and the normalized elements both affine loops over x (group norm's
# channel split into (group, channel of the group)); the mean, then the variance about it, two SUM reductions
NORMS = (A.layer_norm, A.native_layer_norm, A.group_norm, A.native_group_norm, A.batch_norm, A.native_batch_norm,
         A._native_batch_norm_legit, A._native_batch_norm_legit_functional, A._native_batch_norm_legit_no_training)


def _flat(value):
    try:
        return value.view(-1)
    except RuntimeError:
        return value.as_strided((value.numel(),), (value.stride(-1) if value.dim() else 1,), value.storage_offset())


def _grouped(value, count):
    n, c = value.shape[0], value.shape[1]
    return value.as_strided((n, count, c // count, *value.shape[2:]),
                            (value.stride(0), value.stride(1) * (c // count), value.stride(1), *value.stride()[2:]),
                            value.storage_offset())


def _normalized(target, given, outs):
    packet, x, out = target.overloadpacket, given['input'], outs[0]
    weight, bias = given.get('weight'), given.get('bias')
    _typed(target, [x, weight, bias, given.get('running_mean'), given.get('running_var')], outs[:1])
    eps, t = float(given['eps']), _compute(out.dtype, x.dtype, torch.float32)
    if packet in (A.layer_norm, A.native_layer_norm):
        lead = x.dim() - len(given['normalized_shape'])
        view, place, group_of = x, out, list(range(lead))
        reduced = channel_of = list(range(lead, x.dim()))
    elif packet in (A.group_norm, A.native_group_norm):
        count = given['num_groups'] if packet is A.group_norm else given['group']
        view, place, group_of, channel_of = _grouped(x, count), _grouped(out, count), [0, 1], [1, 2]
        reduced = list(range(2, view.dim()))
    else:
        view, place, group_of, channel_of = x, out, [1], [1]
        reduced = [0] + list(range(2, x.dim()))
    dims = view.dim()
    groups, affine = [view.shape[j] for j in group_of], [view.shape[j] for j in channel_of]
    stats = [o.view(groups) for o in outs[1:]] if len(outs) == 3 and outs[1].numel() == math.prod(groups) else \
        [_temporary(groups, STORED[t]), _temporary(groups, STORED[t])]
    mean, rstd = stats
    training = bool(given.get('training'))
    known = not training and given.get('running_mean') is not None and packet in (
        A.batch_norm, A.native_batch_norm, A._native_batch_norm_legit, A._native_batch_norm_legit_no_training)
    g, ops = _same(len(groups)), []
    if known:
        rm, rv = _flat(given['running_mean']).view(groups), _flat(given['running_var']).view(groups)
        f = _Region()
        f.done(_store(f, _load(f, 0, rm, t), t, mean),
               _store(f, f('div', t, f.const(1, t), f('sqrt', t, f('add', t, _load(f, 1, rv, t), f.const(eps, t)))),
                      t, rstd))
        ops.append(_Generic(groups, [(rm, g), (rv, g)], [(mean, g), (rstd, g)], f))
    else:
        total = math.prod(view.shape[j] for j in reduced)
        domain = group_of + reduced
        loops, iterators = [view.shape[j] for j in domain], [PARALLEL] * len(group_of) + [REDUCTION] * len(reduced)
        vmap, gmap = _rows(dims, [domain.index(j) for j in range(dims)]), _rows(dims, range(len(group_of)))
        sums, squares = _temporary(groups, STORED[t]), _temporary(groups, STORED[t])
        r = _Region()
        r.done(_load(r, 0, view, t))
        ops.append(_Generic(loops, [(view, vmap)], [(sums, gmap)], r, iterators, [SUM], [t], precision=REASSOCIATE))
        r = _Region()
        deviation = r('sub', t, _load(r, 0, view, t), r('div', t, r.arg(1, sums), r.const(total, t)))
        r.done(r('mul', t, deviation, deviation))
        ops.append(_Generic(loops, [(view, vmap), (sums, gmap)], [(squares, gmap)], r, iterators, [SUM], [t],
                            precision=REASSOCIATE))
        f = _Region()
        variance = f('div', t, f.arg(1, squares), f.const(total, t))
        f.done(_store(f, f('div', t, f.arg(0, sums), f.const(total, t)), t, mean),
               _store(f, f('div', t, f.const(1, t), f('sqrt', t, f('add', t, variance, f.const(eps, t)))), t, rstd))
        ops.append(_Generic(groups, [(sums, g), (squares, g)], [(mean, g), (rstd, g)], f))
        if training and given.get('running_mean') is not None:
            # torch 2.14.0 torch/_decomp/decompositions.py, native_batch_norm_helper: the running statistics' update,
            # into the functional variant's results or the running statistics' own storage
            rm, rv = _flat(given['running_mean']).view(groups), _flat(given['running_var']).view(groups)
            updated = [o.view(groups) for o in outs[3:5]] if len(outs) == 5 else [rm, rv]
            momentum = float(given['momentum'])
            u = _Region()

            def moved(old, new):
                return u('add', t, u('mul', t, u.const(1 - momentum, t), old), u('mul', t, u.const(momentum, t), new))
            unbiased = u('div', t, u.arg(1, squares), u.const(max(total - 1, 1), t))
            u.done(_store(u, moved(_load(u, 2, rm, t), u('div', t, u.arg(0, sums), u.const(total, t))), t, updated[0]),
                   _store(u, moved(_load(u, 3, rv, t), unbiased), t, updated[1]))
            ops.append(_Generic(groups, [(sums, g), (squares, g), (rm, g), (rv, g)], [(updated[0], g), (updated[1], g)],
                                u))
    e = _Region()
    y = e('mul', t, e('sub', t, _load(e, 0, view, t), _load(e, 1, mean, t)), _load(e, 2, rstd, t))
    inputs = [(view, _same(dims)), (mean, _rows(dims, group_of)), (rstd, _rows(dims, group_of))]
    for affine_value in (weight, bias):
        if affine_value is not None:
            shaped = _flat(affine_value).view(affine)
            y = e('mul' if affine_value is weight else 'add', t, y, _load(e, len(inputs), shaped, t))
            inputs.append((shaped, _rows(dims, channel_of)))
    e.done(_store(e, y, t, out))
    return ops + [_Generic(view.shape, inputs, [(place, _same(dims))], e)]


# an index tensor's elements as region indices: int32 as stored, int64 as its low 32-bit word (little-endian; equal on
# every index of a 32-bit extent); the value the builder declares is that typed view of the same storage
class _Words:
    def __init__(self, tensor):
        self.tensor, self.dtype, self.shape = tensor, torch.int32, tuple(tensor.shape)

    def dim(self):
        return self.tensor.dim()


def _indexing(value):
    return _Words(value) if value.dtype is torch.int64 else value


def _wrapped(r, k, index, extent):
    c = _convert(r, r.arg(k, index), _class(index), I32)
    return r('select', I32, r('lt', BOOL, c, r.const(0, I32)), r('add', I32, c, r.const(extent, I32)), c)


# PyTorch 2.14 documentation, torch.gather, torch.index_select, torch.nn.functional.embedding and Tensor Indexing API;
# NumPy User Guide, Indexing on ndarrays, Advanced indexing: each source dimension's index a result coordinate or an
# index tensor's element there (its view over the result's dimensions), a negative index counted from the end
def _take(out, x, dims, indices):
    r = _Region()
    inputs = [(x, None)]
    for t, strides in indices:
        inputs.append((_indexing(t.as_strided(out.shape, strides, t.storage_offset())), _same(out.dim())))
    at = [r.index(source) if mode < 0 else _wrapped(r, 1 + mode, inputs[1 + mode][0], x.shape[d])
          for d, (mode, source) in enumerate(dims)]
    r.done(_convert(r, r('extract', _class(x), *at, integer=0), _class(x), _class(out)))
    return [_Generic(out.shape, inputs, [(out, _same(out.dim()))], r)]


def _advanced(x, indices):
    tensors = [(j, t) for j, t in enumerate(indices) if t is not None]
    for _, t in tensors:
        if t.dtype not in (torch.int64, torch.int32):
            raise NotImplementedError('an index of %s: its lowering is unfinished' % (t.dtype,))
    shape = tuple(torch.broadcast_shapes(*[t.shape for _, t in tensors]))
    positions = [j for j, _ in tensors]
    adjacent = positions == list(range(positions[0], positions[-1] + 1))
    start = positions[0] if adjacent else 0
    rest = [j for j in range(x.dim()) if j not in positions]
    places = {j: (j if j < start else j - len(positions) + len(shape)) if adjacent else len(shape) + rest.index(j)
              for j in rest}
    return tensors, shape, start, places


def _taken(target, given, outs):
    packet, (out,) = target.overloadpacket, outs
    _typed(target, [], outs, REAL)
    if packet is A.embedding:
        x, index = given['weight'], given['indices']
        return _take(out, x, [(0, 0), (-1, index.dim())], [(index, [*index.stride(), 0])])
    x = given['self']
    if packet is A.index:
        tensors, shape, start, places = _advanced(x, list(given['indices']))
        modes = {j: k for k, (j, _) in enumerate(tensors)}
        strides = [[0] * start + list(t.expand(shape).stride()) + [0] * (out.dim() - start - len(shape))
                   for _, t in tensors]
        return _take(out, x, [(modes[j], 0) if j in modes else (-1, places[j]) for j in range(x.dim())],
                     [(t, s) for (_, t), s in zip(tensors, strides)])
    d, index = given['dim'] % max(x.dim(), 1), given['index']
    if packet is A.gather:
        return _take(out, x, [(0, 0) if j == d else (-1, j) for j in range(x.dim())], [(index, index.stride())])
    return _take(out, x, [(0, 0) if j == d else (-1, j) for j in range(x.dim())],
                 [(index, [index.stride(0) if index.dim() and j == d else 0 for j in range(out.dim())])])


# PyTorch 2.14 documentation, Tensor.scatter_, Tensor.scatter_add_, Tensor.scatter_reduce_, Tensor.index_put_,
# Tensor.index_add_, Tensor.index_copy_ and Tensor.index_fill_; PyTorch v2.11.0
# aten/src/ATen/native/TensorAdvancedIndexing.cpp, scatter_reduce_two and scatter_reduce_exclude_self_helper: each
# result element reduces the candidates whose indices name it (loops (result..., candidate...), the candidates
# REDUCTION, states its accumulator and match count), then joins them with its base element
SCATTERS = {'last': 0, 'sum': 1, 'add': 1, 'prod': 2, 'multiply': 2, 'mean': 3, 'amax': 4, 'amin': 5}


def _scatter(out, base, src, targets, candidates, limits, near, far, op, include, alpha=1):
    integral = out.dtype in INTEGRAL
    t = _compute(out.dtype, *([src.dtype] if isinstance(src, torch.Tensor) else [])) if integral else F32
    n, m = out.dim(), len(candidates)
    acc, count = _temporary(out.shape, STORED[t]), _temporary(out.shape, torch.int32)
    shape = list(out.shape) + list(candidates)
    r, inputs, match = _Region(), [], None
    for index, dim, wrap, across, along in targets:
        view = _indexing(index.as_strided(shape, list(across) + list(along), index.storage_offset()))
        hit = r('eq', BOOL, _wrapped(r, len(inputs), view, wrap), r.index(dim))
        match = hit if match is None else r('and', BOOL, match, hit)
        inputs.append((view, _same(n + m)))
    within = [r('lt', BOOL, r.index(d), r.const(limits[d], I32)) for d in range(n) if limits[d] < out.shape[d]]
    if within:
        match = r('and', BOOL, match, _all(r, within))
    if isinstance(src, torch.Tensor):
        view = src.as_strided(shape, list(near) + list(far), src.storage_offset())
        v = _load(r, len(inputs), view, t)
        inputs.append((view, _same(n + m)))
    else:
        v = r.const(src, t)
    r.done(r('select', t, match, v, r.const(0, t)), _convert(r, match, BOOL, I32))

    def combined(c, a, b):
        return c('add', t, a, b) if op in (1, 3) else c('mul', t, a, b) if op == 2 else \
            _extremum(c, a, b, t, op == 4) if op in (4, 5) else b

    def join(c, a, b):
        none = c.const(0, I32)
        value = c('select', t, c('eq', BOOL, b[1], none), a[0],
                  c('select', t, c('eq', BOOL, a[1], none), b[0], combined(c, a[0], b[0])))
        return [value, c('add', I32, a[1], b[1])]
    states = [(t, 0), (I32, 0)]
    imap = _rows(n + m, range(n))
    made = [_Generic(shape, inputs, [(acc, imap), (count, imap)], r, [PARALLEL] * n + [REDUCTION] * m, [REGION] * 2,
                     [t, I32], None, states, _joined(states, join), _identities(states))]
    e = _Region()
    a, k, b = e.arg(0, acc), e.arg(1, count), _load(e, 2, base, t)
    own = include and op != 0
    if op == 0:
        result = a
    elif op == 1:
        scaled = a if alpha == 1 else e('mul', t, e.const(alpha, t), a)
        result = e('add', t, b, scaled) if own else scaled
    elif op == 2:
        result = e('mul', t, b, a) if own else a
    elif op == 3:
        total = e('add', t, b, a) if own else a
        counted = _convert(e, e('add', I32, k, e.const(int(own), I32)), I32, t)
        result = e('floordiv' if integral else 'div', t, total, counted)
    else:
        result = _extremum(e, a, b, t, op == 4) if own else a
    e.done(_store(e, e('select', t, e('eq', BOOL, k, e.const(0, I32)), b, result), t, out))
    return made + [_Generic(out.shape, [(acc, _same(n)), (count, _same(n)), (base, _same(n))], [(out, _same(n))], e)]


def _scattered(target, given, outs):
    packet, (out,), x = target.overloadpacket, outs, given['self']
    _typed(target, [x], outs, REAL)
    if packet is A.index_put:
        tensors, shape, start, places = _advanced(x, list(given['indices']))
        made = [x.shape[j] for j in range(x.dim()) if j in places and places[j] < start] + list(shape) + \
            [x.shape[j] for j in range(x.dim()) if j in places and places[j] >= start + len(shape)]
        values = given['values'].expand(made)
        near = [values.stride(places[j]) if j in places else 0 for j in range(x.dim())]
        far = [values.stride(start + i) for i in range(len(shape))]
        targets = [(t, j, x.shape[j], [0] * x.dim(), list(t.expand(shape).stride())) for j, t in tensors]
        return _scatter(out, x, values, targets, shape, list(x.shape), near, far,
                        SCATTERS['sum' if given['accumulate'] else 'last'], True)
    d, index = given['dim'] % max(x.dim(), 1), given['index']
    src = given.get('src', given.get('source', given.get('value')))
    if packet in (A.index_add, A.index_copy, A.index_fill):
        far = [src.stride(d) if src.dim() else 0] if isinstance(src, torch.Tensor) else [0]
        near = [src.stride(j) if isinstance(src, torch.Tensor) and src.dim() and j != d else 0 for j in range(x.dim())]
        targets = [(index, d, x.shape[d], [0] * x.dim(), [index.stride(0) if index.dim() else 0])]
        return _scatter(out, x, src, targets, [index.numel()], list(x.shape), near, far,
                        SCATTERS['sum' if packet is A.index_add else 'last'], True, given.get('alpha', 1))
    reduce = given.get('reduce')
    op = SCATTERS['sum' if packet is A.scatter_add else reduce if reduce is not None else 'last']
    near = [src.stride(j) if isinstance(src, torch.Tensor) and j != d else 0 for j in range(x.dim())]
    far = [src.stride(d) if isinstance(src, torch.Tensor) else 0]
    targets = [(index, d, x.shape[d], [index.stride(j) if j != d else 0 for j in range(x.dim())], [index.stride(d)])]
    return _scatter(out, x, src, targets, [index.shape[d]],
                    [index.shape[j] if j != d else x.shape[j] for j in range(x.dim())], near, far, op,
                    given.get('include_self', True))


# PyTorch 2.14 documentation, torch.nn.functional.scaled_dot_product_attention; Milakov and Gimelshein 2018; T. Dao et
# al., FlashAttention, NeurIPS 2022, section 3.1 and Algorithm 1, lines 10 to 13: the scores one SUM over the head
# dimension, loops (lead..., kv head, group, l, t | f), then the output's states (m, s, acc) one REDUCTION over the
# keys t, each key contributing (score, 1, v), joined by Algorithm 1's rescaling, finalized to acc / s
def _heads(x, group, kv):
    if x.dim() < 3:
        return x.unsqueeze(0).unsqueeze(0)
    heads = x.shape[-3] if kv else x.shape[-3] // group
    shape = (*x.shape[:-3], heads, 1 if kv else group, *x.shape[-2:])
    stride = (*x.stride()[:-3], x.stride(-3) * (1 if kv else group), 0 if kv else x.stride(-3), *x.stride()[-2:])
    return x.as_strided(shape, stride, x.storage_offset())


def _attention(q, k, v, mask, out, causal, scale, group):
    length, size, other, value = q.shape[-2], q.shape[-1], k.shape[-2], v.shape[-1]
    t = _compute(q.dtype, k.dtype, v.dtype, torch.float32)
    qs, ks, vs = _heads(q, group, False), _heads(k, group, True), _heads(v, group, True)
    b = len(qs.shape) - 2
    scores = _temporary([*qs.shape[:-2], length, other], STORED[t])
    domain = [*qs.shape[:-2], length, other, size]
    r = _Region()
    r.done(r('mul', t, _load(r, 0, qs, t), _load(r, 1, ks, t)))
    ops = [_Generic(domain, [(qs, _rows(b + 3, [*range(b), b, b + 2])), (ks, _onto(ks, [*range(b), b + 1, b + 2], domain))],
                    [(scores, _rows(b + 3, [*range(b), b, b + 1]))], r, [PARALLEL] * (b + 2) + [REDUCTION], [SUM],
                    [t], precision=REASSOCIATE | CONTRACT)]
    shape = [*qs.shape[:-2], length, value, other]
    r = _Region()
    score = r('mul', t, r.arg(0, scores), r.const(scale, t))
    inputs = [(scores, _rows(b + 3, [*range(b), b, b + 2])), (vs, _onto(vs, [*range(b), b + 2, b + 1], shape))]
    if mask is not None:
        ms = _heads(mask, group, False)
        inputs.append((ms, _onto(ms, [*range(b), b, b + 2], shape)))
        given = r.arg(2, ms)
        score = r('select', t, given, score, r.const(-math.inf, t)) if mask.dtype is torch.bool else \
            r('add', t, score, _convert(r, given, _class(mask), t))
    if causal:
        score = r('select', t, r('gt', BOOL, r.index(b + 2), r.index(b)), r.const(-math.inf, t), score)
    r.done(score, r.const(1, t), _load(r, 1, vs, t))
    states = [(t, -math.inf), (t, 0), (t, 0)]

    def join(c, a, b):
        top = c('select', t, c('gt', BOOL, b[0], a[0]), b[0], a[0])
        e1, e2 = c('exp', t, c('sub', t, a[0], top)), c('exp', t, c('sub', t, b[0], top))
        joined = [top, c('add', t, c('mul', t, a[1], e1), c('mul', t, b[1], e2)),
                  c('add', t, c('mul', t, a[2], e1), c('mul', t, b[2], e2))]
        absent_a, absent_b = c('eq', BOOL, a[0], c.const(-math.inf, t)), c('eq', BOOL, b[0], c.const(-math.inf, t))
        return [c('select', t, absent_b, x, c('select', t, absent_a, y, z)) for x, y, z in zip(a, b, joined)]
    smap = _rows(b + 3, range(b + 2))
    f = _Region()
    total = f('arg', t, integer=1)
    f.done(f('select', t, f('eq', BOOL, total, f.const(0, t)), f.const(math.nan, t),
             f('div', t, f('arg', t, integer=2), total)))
    ops.append(_Generic(shape, inputs, [(_heads(out, group, False), smap)], r, [PARALLEL] * (b + 2) + [REDUCTION],
                        [REGION], [t], None, states, _joined(states, join), f, REASSOCIATE))
    return ops


def _attending(given):
    q, k, v, mask = given['query'], given['key'], given['value'], given['attn_mask']
    lead = tuple(q.shape[:-2])
    group = q.shape[-3] // k.shape[-3] if given['enable_gqa'] and lead else 1
    kv = (*lead[:-1], lead[-1] // group) if lead else ()
    scale = given['scale'] if given['scale'] is not None else 1 / math.sqrt(q.shape[-1])
    return (q, k.expand(*kv, *k.shape[-2:]), v.expand(*kv, *v.shape[-2:]),
            mask.expand(*lead, q.shape[-2], k.shape[-2]) if mask is not None else None), given['is_causal'], scale, group


def _attended(target, given, outs):
    _typed(target, [given['query'], given['key'], given['value'], given['attn_mask']], outs)
    if given['dropout_p']:
        raise NotImplementedError('%s with dropout_p %r: the lowering of its random draws is unfinished'
                                  % (target, given['dropout_p']))
    operands, causal, scale, group = _attending(given)
    return _attention(*operands, outs[0], causal, scale, group)


# J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, Parallel random numbers: as easy as 1, 2, 3, SC 2011, §3.1 and
# §4.2 (Philox4x32-10: its multipliers, Weyl key increments, ten rounds): each draw a counter-based function of its key
# and counter, the key the program's seed, the counter (element, execution, draw site, stream); the generator a STATE
# value {seed low, seed high, execution, 0} the program advances once an execution
PHILOX = (0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85)
RANDOM = {'state': None, 'sites': 0, 'stream': 0}
GENERATORS = set()


def _generator():
    if RANDOM['state'] is None:
        seed = torch.initial_seed() ^ (RANDOM['stream'] << 32)
        words = [seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF, 0, 0]
        RANDOM['state'] = torch.tensor([w - (1 << 32) if w >= 1 << 31 else w for w in words], dtype=torch.int32)
        GENERATORS.add(_key(RANDOM['state']))
    RANDOM['sites'] += 1
    return RANDOM['state'], RANDOM['sites']


def _mulhilo(r, a, m):
    def c(v):
        return r.const(v, U32)
    low, high = c(m & 0xFFFF), c(m >> 16)
    al, ah = r('and', U32, a, c(0xFFFF)), r('shr', U32, a, c(16))
    t0, t1, t2, t3 = r('mul', U32, al, low), r('mul', U32, ah, low), r('mul', U32, al, high), r('mul', U32, ah, high)
    middle = r('add', U32, r('add', U32, r('shr', U32, t0, c(16)), r('and', U32, t1, c(0xFFFF))),
               r('and', U32, t2, c(0xFFFF)))
    hi = r('add', U32, r('add', U32, r('add', U32, t3, r('shr', U32, t1, c(16))), r('shr', U32, t2, c(16))),
           r('shr', U32, middle, c(16)))
    return hi, r('mul', U32, a, c(m))


def _philox(r, counter, key):
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for round in range(10):
        hi0, lo0 = _mulhilo(r, c0, PHILOX[0])
        hi1, lo1 = _mulhilo(r, c2, PHILOX[1])
        c0, c1, c2, c3 = r('xor', U32, r('xor', U32, hi1, c1), k0), lo1, r('xor', U32, r('xor', U32, hi0, c3), k1), lo0
        if round < 9:
            k0, k1 = r('add', U32, k0, r.const(PHILOX[2], U32)), r('add', U32, k1, r.const(PHILOX[3], U32))
    return c0, c1, c2, c3


# the draws of operation `out`'s elements: the generator's words as inputs (their constant maps), the element's row-major
# index the counter's first word
def _draws(r, out, inputs):
    state, site = _generator()
    for word in range(3):
        inputs.append((state, _rows(out.dim(), [{None: word}])))
    k0, k1, execution = [r('bitcast', U32, r.arg(len(inputs) - 3 + word, state)) for word in range(3)]
    element = r('bitcast', U32, _linear(r, out.shape))
    return _philox(r, (element, execution, r.const(site, U32), r.const(0, U32)), (k0, k1))


def _uniform(r, word):
    return r('mul', F32, r('cast', F32, r('shr', U32, word, r.const(8, U32))), r.const(2.0 ** -24, F32))


# PyTorch 2.14 documentation, torch.rand, torch.randn, torch.randint, torch.bernoulli, Tensor.uniform_ and
# torch.nn.functional.dropout (torch 2.14.0 aten native_dropout: mask = U < 1 - p, out = x mask / (1 - p)); the normal
# draws by G. E. P. Box and M. E. Muller, Ann. Math. Statist. 29(2), 1958
DRAWS = (A.rand, A.rand_like, A.randn, A.randn_like, A.randint, A.randint_like, A.bernoulli, A.uniform, A.native_dropout)


def _random(target, given, outs):
    packet, out = target.overloadpacket, outs[0]
    _typed(target, [], outs[:1], REAL)
    r, inputs = _Region(), []
    if packet is A.native_dropout:
        x, p, train = given['input'], float(given['p']), given['train']
        inputs.append((x, _same(x.dim())))
        v = _load(r, 0, x, F32)
        if train is False:
            r.done(_store(r, v, F32, out), r.const(True, BOOL))
        else:
            keep = r('lt', BOOL, _uniform(r, _draws(r, out, inputs)[0]), r.const(1 - p, F32))
            scaled = r('mul', F32, v, r.const(1 / (1 - p) if p < 1 else 0, F32))
            r.done(_store(r, r('select', F32, keep, scaled, r.const(0, F32)), F32, out), keep)
        return [_Generic(out.shape, inputs, [(out, _same(out.dim())), (outs[1], _same(out.dim()))], r)]
    if packet is A.bernoulli and given.get('p') is None:
        inputs.append((given['self'], _same(out.dim())))
    words = _draws(r, out, inputs)
    u = _uniform(r, words[0])
    if packet in (A.randn, A.randn_like):
        first = r('add', F32, u, r.const(2.0 ** -24, F32))
        radius = r('sqrt', F32, r('mul', F32, r.const(-2, F32), r('log', F32, first)))
        made = r('mul', F32, radius, r('cos', F32, r('mul', F32, r.const(2 * math.pi, F32), _uniform(r, words[1]))))
    elif packet in (A.randint, A.randint_like):
        low, high = given.get('low') or 0, given['high']
        made = r('add', I32, r.const(low, I32), r('bitcast', I32, r('rem', U32, words[0], r.const(high - low, U32))))
        r.done(_store(r, made, I32, out))
        return [_Generic(out.shape, inputs, [(out, _same(out.dim()))], r)]
    elif packet is A.bernoulli:
        p = given.get('p')
        made = r('cast', F32, r('lt', BOOL, u, _load(r, 0, given['self'], F32) if p is None else r.const(p, F32)))
    elif packet is A.uniform:
        low, high = float(given.get('from', 0)), float(given.get('to', 1))
        made = r('add', F32, r.const(low, F32), r('mul', F32, u, r.const(high - low, F32)))
    else:
        made = u
    r.done(_store(r, made, F32, out))
    return [_Generic(out.shape, inputs, [(out, _same(out.dim()))], r)]


# the generator's advance: its execution word incremented once an execution, after the program's draws
def _stepped(state):
    r = _Region()
    r.done(r('add', I32, r.arg(0, state), r.const(1, I32)))
    return _Generic((1,), [(state, _rows(1, [{None: 2}]))], [(state, _rows(1, [{None: 2}]))], r)


# PyTorch 2.14 documentation, torch.cat
def _concatenated(target, given, outs):
    (out,), d = outs, given['dim'] % max(outs[0].dim(), 1)
    ops, at = [], 0
    for t in given['tensors']:
        if t.dim() == 1 and t.shape[0] == 0 and out.dim() != 1:
            continue
        ops += _copy(out.narrow(d, at, t.shape[d]), t)
        at += t.shape[d]
    return ops


# design/torch-source-lowering-change-request.md item 1; design/compilation-composition-change-request.md, "Complete
# replacement workgroup": each call's structured operations, a view's materialization its copy
SOURCES = {**dict.fromkeys(POINTWISE, _arithmetic), **dict.fromkeys((A.clone, A._to_copy, A._unsafe_view, A.copy,
                                                                     A.copy_), _copying),
           **dict.fromkeys((A.mm, A.bmm, A.matmul, A.linear, A.addmm), _product), A.einsum: _summation,
           **dict.fromkeys((*CONVOLUTION, *TRANSPOSED, A.convolution), _convolving),
           **dict.fromkeys((*MAXIMUM, *AVERAGE, *INDEXED), _pooled), **dict.fromkeys(PAD, _padded),
           **dict.fromkeys((*FILLS, *EMPTY, A.arange), _factory), **dict.fromkeys(REDUCTIONS, _reduced),
           **dict.fromkeys(SOFTMAX, _softened), A.cumsum: _scanned, A.sort: _ranked, A.topk: _ranked,
           **dict.fromkeys(NORMS, _normalized),
           **dict.fromkeys((A.gather, A.index_select, A.embedding, A.index), _taken),
           **dict.fromkeys((A.scatter, A.scatter_add, A.scatter_reduce, A.index_put, A.index_add, A.index_copy,
                            A.index_fill), _scattered),
           A.scaled_dot_product_attention: _attended, A.cat: _concatenated, **dict.fromkeys(DRAWS, _random)}
# PyTorch 2.14 documentation, Tensor Views
VIEWS = {A.as_strided, A.detach, A.diagonal, A.expand, A.expand_as, A.movedim, A.narrow, A.permute, A.select,
         A.squeeze, A.transpose, A.t, A.adjoint, A.real, A.imag, A.view_as_real, A.unflatten, A.unfold, A.unsqueeze,
         A.view, A.view_as, A.unbind, A.split, A.split_with_sizes, A.tensor_split, A.hsplit, A.vsplit, A.swapaxes,
         A.swapdims, A.chunk, A.slice, A.alias, operator.getitem}


def _lowering(local):
    packet = getattr(local.target, 'overloadpacket', local.target)
    if packet in SOURCES:
        return lambda args, kwargs, outs: SOURCES[packet](local.target, _operands(local.target, args, kwargs), outs)
    if packet in VIEWS:
        return lambda args, kwargs, outs: _copy(outs[0], _call(local, args, kwargs))
    raise NotImplementedError('%s: its lowering into the structured basis is unfinished '
                              '(design/torch-source-lowering-change-request.md item 1)' % (local.target,))


# dlpack.h DLManagedTensorVersioned `deleter`; torch v2.14.0 torch/csrc/utils/tensor_new.cpp tensor_fromDLPackImpl,
# aten/src/ATen/DLConvertor.cpp fromDLPackImpl; G. E. Collins, A method for overlapping and erasure of lists, CACM
# 3(12), 1960; design/aot-compilation-change-request.md, appendix item 7; design/torch-capture-change-request.md item 6
class _Mapping:
    def __init__(self, view):
        self.view, self.tensors, self.retired = view, 0, False

    def release(self):
        if self.retired and not self.tensors and self.view.address:
            self.view = mesh.View()


# mesh-collective.h mesh_view_values; mesh-call.h mesh_section_address; mesh-section.h struct mesh_section;
# design/torch-capture-change-request.md item 6; design/plan-to-bridge-change-request.md R2
mesh.LIB.mesh_section_address.restype = C.c_void_p
mesh.LIB.mesh_section_address.argtypes = [mesh.CONTEXT, mesh.Section, mesh.U]


def _mapped(context, functions, count, values, row):
    view = mesh.View()
    first = (mesh.Q * (sum(len(v) for v in values) + 1))()
    offsets = (mesh.Q * max(sum(v.section.count for function in values for v in function), 1))()
    mesh.LIB.mesh_view_values(C.byref(context), functions, count, first, offsets, C.byref(view))
    at, addresses = 0, []
    for function in values:
        addresses.append([[view.address + offsets[first[at + i] + c] for c in range(v.section.count)] if v.run else
                          [[mesh.LIB.mesh_section_address(C.byref(context), v.section, c * v.section.stride + k)
                            for k in range(-(-v.section.bytes // row))] for c in range(v.section.count)]
                          for i, v in enumerate(function)])
        at += len(function)
    if not all(address for function in addresses for value in function for copy in value if isinstance(copy, list)
               for address in copy):
        raise OSError(errno.EFAULT, 'a page-table row of the program\'s values with no page (mesh-call.h '
                                    'mesh_section_address)')
    return view, addresses


# MTLDevice.h newBufferWithBytesNoCopy:length:options:deallocator:; design/plan-to-bridge-change-request.md R1 and R2
def _hull(values, addresses, row):
    starts, ends = [], []
    for function, made in zip(values, addresses):
        for v, at in zip(function, made):
            for copy in at:
                pages = copy if isinstance(copy, list) else [copy]
                starts += pages
                ends += [a + (row if isinstance(copy, list) else v.section.bytes) for a in pages]
    return mesh.View(min(starts), -(-(max(ends) - min(starts)) // row) * row) if starts else mesh.View()


HELD = {}


@DELETER
def _delete(managed):
    held = HELD.pop(C.addressof(managed.contents), None)
    if held:
        if held[0].manager_ctx:
            PUT(held[0].manager_ctx, SELECTOR[b'release'])
        held[3].tensors -= 1
        held[3].release()


# design/aot-compilation-change-request.md, appendix item 7; design/plan-to-bridge-change-request.md R2; MTLResource.h
# hazardTrackingMode; torch 2.14.0 include/ATen/native/mps/OperationUtils.h needsGather
def _placed_buffer(view):
    buffer = WRAP(_metal_device(), SELECTOR[b'newBufferWithBytesNoCopy:length:options:deallocator:'], view.address,
                  view.bytes, SHARED, None)
    if not buffer:
        raise OSError(errno.ENOMEM, 'Metal made no buffer over the program\'s mapping, %d bytes (MTLDevice.h '
                                    'newBufferWithBytesNoCopy)' % view.bytes)
    return buffer


# torch v2.14.0 torch/utils/dlpack.py; aten/src/ATen/DLConvertor.cpp fromDLPackImpl, dlDeviceToTorchDevice,
# torchDeviceToDLDevice and toDLPackImpl; torch/csrc/utils/tensor_new.cpp tensor_fromDLPack; dlpack.h
def _dlpack_tensor(device, data, offset, owner, dtype, shape, mapping):
    if dtype not in TYPES:
        raise NotImplementedError('%s: no DLPack type torch v2.14.0 gives it and takes back as itself (DLConvertor.cpp '
                                  'getDLDataType, toScalarType)' % dtype)
    n = len(shape)
    sizes = (C.c_int64 * max(n, 1))(*shape)
    strides = (C.c_int64 * max(n, 1))(*(math.prod(shape[i + 1:]) for i in range(n)))
    managed = DLManagedTensorVersioned(VERSION, owner, _delete, 0,
                                       DLTensor(data, DLDevice(DEVICES[device], 0), n, DLDataType(*TYPES[dtype]),
                                                sizes, strides, offset))
    if owner:
        GET(owner, SELECTOR[b'retain'])
    HELD[C.addressof(managed)] = (managed, sizes, strides, mapping)
    mapping.tensors += 1
    try:
        return torch.from_dlpack(CAPSULE_NEW(C.addressof(managed), CAPSULE, None))
    except BaseException:
        _delete(C.pointer(managed))
        raise


def _tensor(device, address, buffer, base, dtype, shape, mapping):
    if device == 'cpu':
        return _dlpack_tensor('cpu', address, 0, None, dtype, shape, mapping)
    return _dlpack_tensor('mps', buffer, address - base, buffer, dtype, shape, mapping)


# design/disjoint-pages-change-request.md C3.3 and C3.8; design/plan-to-bridge-change-request.md R2 and R4;
# design/torch-source-lowering-change-request.md item 1
def _bind(ir, values, addresses, buffer, base, row, mapping):
    copies = []
    for v, (dtype, numel, device), at in zip(values, ir.layouts, addresses):
        n = row // dtype.itemsize
        copies.append([_tensor(device, copy, buffer, base, dtype, (numel,), mapping) if v.run else
                       _Paged([_tensor(device, a, buffer, base, dtype, (min(n, numel - k * n),), mapping)
                               for k, a in enumerate(copy)]) for copy in at])
    places, given = _homes(ir, copies)
    hosts = {i: [copy.view(shape) if ir.layouts[v][2] == 'cpu' else
                 _dlpack_tensor('cpu', a, 0, None, ir.layouts[v][0], shape, mapping) for copy, a in
                 zip(copies[v], addresses[v])] for i, (v, shape) in ir.inputs.items()}
    return places, {i: ir.layouts[d.value][2] for i, d in enumerate(ir.declared) if d.value != mesh.ABSENT}, \
        {i: (given[i], hosts[i]) for i in given}, copies


# torch 2.14.0 torch/_torch_docs.py as_strided
def _homes(ir, copies):
    places = {key: [copy.as_strided(sizes, strides, copy.storage_offset() + offset)
                    if isinstance(copy, torch.Tensor) else copy for copy in copies[v]]
              for key, (v, offset, sizes, strides) in ir.homes.items()}
    return places, {i: [copy.view(shape) for copy in copies[v]] for i, (v, shape) in ir.inputs.items()}


def _aliases(local):
    return local.target is operator.getitem or isinstance(local.target, torch._ops.OpOverload) \
        and any(r.alias_info is not None for r in local.target._schema.returns)


def _writes(local):
    return isinstance(local.target, torch._ops.OpOverload) and any(
        a.alias_info is not None and a.alias_info.is_write for a in local.target._schema.arguments)


# design/plan-to-bridge-change-request.md R2 and R4; mesh-section.h struct mesh_section: a placed value of pages of
# `row` bytes in the region's page-table rows, each page's placed tensor
class _Paged:
    def __init__(self, pages):
        self.pages = pages


# design/torch-source-lowering-change-request.md item 1: the storage identity of a tensor of the compilation (its
# placed stand-ins, temporaries and views are meta tensors; a constant is a tensor holding its data)
def _key(value):
    value = value.tensor if isinstance(value, _Words) else value
    return value.untyped_storage()._cdata if value.device.type == 'meta' else value.untyped_storage().data_ptr()


# torch 2.14.0 torch/_tensor_docs.py, view and copy_; design/torch-source-lowering-change-request.md item 1: a call's
# structured operations over its operands' placed views, its result its own place, a placed view of an operand it
# writes in place, an alias, or a temporary the compilation realizes
def _called(ir, key, operands, local, form, placed):
    lower = _lowering(local)
    given = iter(operands)
    args, kwargs = map_arg((local.args, local.kwargs), lambda n: next(given))
    mine = placed(key)
    if isinstance(form, tuple):
        outs = [placed(result) for _, result in form[1]]
        return tuple(outs) if len(outs) > 1 else mine, lower(args, kwargs, outs)
    if key in ir.contiguous:
        base = operands[_based(local)]
        return mine, _assign(mine.view(base.shape), base)
    # torch 2.14.0 torchgen/model.py Annotation.is_write (`Tensor(a!)`): a call writing an operand in place writes its
    # result's place, the operand's bytes (_view); a pure alias (`Tensor(a)`) has no work of its own
    if mine is not None and (ir.places[ir.device][key].owner == key or _writes(local)):
        return mine, lower(args, kwargs, [mine])
    if mine is not None:
        return mine, []
    if _aliases(local):
        return _call(local, args, kwargs), []
    outs, spec = _made(ir, key, local, args, kwargs)
    return pytree.tree_unflatten(outs, spec), lower(args, kwargs, outs)


def _made(ir, key, local, args, kwargs):
    made, spec = pytree.tree_flatten(_call(local, args, kwargs))
    return [torch.empty_strided(m.shape, m.stride(), dtype=m.dtype, device='meta') for m in made], spec


# OpenXLA operation semantics, CollectivePermute; MPICH alltoall_intra_pairwise.c, MPIR_Localcopy
def _local(ir, collective):
    if collective not in ir.roles:
        return None
    me = ir.roles[collective][0]
    pairs = [tuple(pair) for pair in collective.dim] if collective.kind == 'collective-permute' else []
    if collective.kind == 'reduce-scatter' or collective.kind == 'all-reduce' and me == 0:
        return 'reduce'
    if (me, me) in pairs:
        return 'copy'
    return 'fill' if pairs and me not in [q for _, q in pairs] else None


# design/plan-to-bridge-change-request.md R4; design/homomorphism-sources-2026-11-08.md; J. Gibbons, The third
# homomorphism theorem, J. Functional Programming 6(4), 1996, (2) and Definition 3.1: a reduction's local part, its
# total the sum of its own piece and every received piece j (flat elements [j size, (j + 1) size) of the landing
# value), one structured operation whose lowering acquires each piece's readiness (each landing value a received range
# of the program, _programs)
def _collected(ir, placed, item, operands):
    _, reads, key, collective = item
    me, receives = ir.roles[collective]
    local = _local(ir, collective)
    if local == 'reduce':
        total, first = placed(key), operands[0]
        if collective.kind == 'reduce-scatter':
            n = first.shape[collective.dim] // (len(receives) + 1)
            first = first.narrow(collective.dim, me * n, n)
        _typed(collective.kind, [first], [total])
        flat = total.view(-1)
        whole = [first.reshape(-1)] + [placed(received).view(-1).narrow(0, j * flat.numel(), flat.numel())
                                       for _, (received, j) in receives]
        t = _compute(total.dtype, *[x.dtype for x in whole])
        r = _Region()
        made = None
        for k, piece in enumerate(whole):
            x = _load(r, k, piece, t)
            made = x if made is None else r('add', t, made, x)
        r.done(_store(r, made, t, total))
        return [_Generic(flat.shape, [(piece, _same(1)) for piece in whole], [(flat, _same(1))], r)]
    places = ir.places[ir.device]
    if local == 'copy' and places[reads[0]][:4] != places[key][:4]:
        return _assign(placed(key), operands[0])
    return _fill(placed(key), 0) if local == 'fill' else []


# GSPMD section 4.5; torch 2.14.0 torch/export/exported_program.py, ExportedProgram; Megatron-LM core_r0.12.0
# megatron/core/tensor_parallel/layers.py:125–169, _initialize_affine_weight_cpu;
# design/problems-of-14b9a757-change-request.md, "Sources for the second stops", 8, and "Sources for the third stops", 8
def _hosted(ir, item, values):
    kind, reads, _, operand = item
    if kind == 'constant':
        return operator.attrgetter(operand)(ir.gm)
    if kind == 'call':
        given, local = iter(values), ir.locals[operand]
        return _call(local, *map_arg((local.args, local.kwargs), lambda n: next(given)))
    if kind == 'slice':
        (partition_dim, m), world_size = operand, ir.shape[operand[1]]
        weight_list = torch.split(values[0], _layout(ir, reads[0])[1][partition_dim] // world_size, dim=partition_dim)
        (piece,) = weight_list[ir.coords[ir.device][m]::world_size]
        return piece
    return values[0].narrow(*_cut(ir, kind, operand, _layout(ir, reads[0])[1], ir.device))


# MPICH allgather_intra_ring.c, MPIR_Localcopy; torch 2.14.0 torch/_tensor_docs.py, copy_
def _copied(ir, read, key, cut, source, target):
    if ir.places[ir.device][key].writer != key:
        return []
    return _assign(target, source if cut is None else source.narrow(*_cut(ir, 'slice', cut, _layout(ir, read)[1],
                                                                            ir.device)))


# N. D. Jones, C. K. Gomard and P. Sestoft, 1993, Definition 4.1: one item of the function's program over its placed
# stand-ins (`placed`) and the call's input leaves (`given`): its value and its structured operations
def _specialized(ir, placed, item, operands, ops, given):
    kind, reads, key, operand = item
    value = placed(key)
    if kind == 'input':
        value = given[operand]
    elif kind == 'constant' and value is None:
        value = operator.attrgetter(operand)(ir.gm)
    elif kind == 'copy':
        ops += _copied(ir, reads[0], key, operand, operands[0], value)
    elif kind == 'window' and key in ir.inside:
        dim, m, _, length, starts, size, fill = operand
        q = ir.coords[ir.device][m]
        below, above = min(length, max(0, -starts[q])), min(length, max(0, size - starts[q]))
        for start, extent in ((0, below), (above, length - above)):
            ops += _fill(value.narrow(dim, start, extent), fill) if fill is not None and extent else []
    elif kind in ('slice', 'narrow', 'window') and value is None:
        value = operands[0].narrow(*_cut(ir, kind, operand, _layout(ir, reads[0])[1], ir.device))
    elif kind == 'call':
        value, made = _called(ir, key, operands, ir.locals[operand], ir.forms.get(key), placed)
        ops += made
    elif kind == 'collective':
        ops += _collected(ir, placed, item, operands)
    return value


# a call's structured operations in the function's program order and its result over the placed stand-ins `tensors`
def _instantiate(ir, tensors, given, spec, stream=0):
    ops = []
    RANDOM.update(state=None, sites=0, stream=stream)
    items = {item[2]: item for item in ir.program if item[0] != 'output'}

    def placed(key):
        return tensors[key][0] if key in tensors else None

    def specialized(key, operands):
        first = len(ops)
        value = _specialized(ir, placed, items[key], operands, ops, given)
        for g in ops[first:]:
            g.place = CPU if _device(ir, key) == 'cpu' else GPU
        return value
    values = mesh_graph.forward(list(items), lambda key: items[key][1], specialized)
    if RANDOM['state'] is not None:
        ops.append(_stepped(RANDOM['state']))
    node, keys = next(item[3] for item in ir.program if item[0] == 'output')
    return ops, pytree.tree_unflatten(list(map_arg(node.args[0], lambda n: values[keys[n]])), spec)


# design/plan-to-bridge-change-request.md R4
def _paging(target):
    packet = getattr(target, 'overloadpacket', None)
    if packet in REDUCTIONS:
        return {0}, False
    if packet in SOFTMAX:
        return {0}, True
    if packet is A.scaled_dot_product_attention:
        return {1, 2}, False
    return None, True


# design/disjoint-pages-change-request.md C3.4; design/aot-compilation-change-request.md, appendix item 7; P. E.
# McKenney and J. D. Slingwine, Read-copy update, PDCS 1998; mesh-dataflow.h mesh_retire; mesh-collective.h mesh_plan
# metal_recording.h MetalProgramEnd: 0 ended, EIO a commit's error (ended), ECANCELED stalled with its cancelled work
# and callbacks drained — the operands are released, then the status raised; ETIMEDOUT the drain did not finish, so
# the operands stay live
def _retire(context, state):
    replay, status = state['replay'], 0
    if replay is not None:
        status = replay['lib'].MetalProgramEnd(replay['program'], replay['ended'], replay['error'], len(replay['error']))
        if status == errno.ETIMEDOUT:
            raise OSError(status, replay['error'].value.decode())
    mesh.LIB.mesh_retire(context.M, context.client)
    _released(state['owned'])
    mapping = state['mapping']
    state.update(replay=None, mapping=None, devices=(), executions=0, stale=True)
    if mapping is not None:
        mapping.retired = True
        mapping.release()
    if status:
        raise OSError(status, replay['error'].value.decode())


# the preparation's resources, each with its release, made in order and released in the reverse order, on retirement
# or on a preparation that raised
def _owned(owned, made):
    if made:
        owned[made] = lambda: PUT(made, SELECTOR[b'release'])
    return made


def _released(owned):
    while owned:
        owned.popitem()[1]()


# Python documentation, weakref.finalize; Lib/weakref.py, finalize
def _dropped(context, state):
    if state['mapping'] is not None or state['owned']:
        _retire(context, state)


# metal-microbench metal_ir.h: one program per call row, built from the call's structured operations over its placed
# stand-ins (each storage one value class, planned by its plan value), lowered before planning, finalized after the
# plan's transport words, bound to the placed storage (design/compilation-composition-change-request.md, "Complete
# replacement workgroup"; 6's stages A to E)
INPUT, CONSTANT, TEMPORARY, OUTPUT, STATE = range(5)
BODY, COMBINE, FINALIZE = range(3)
RELEASE = C.CFUNCTYPE(None, C.c_void_p)
HOLDERS = {}


@RELEASE
def _released_constant(owner):
    HOLDERS.pop(owner, None)


def _array(kind, values):
    values = list(values)
    return (kind * max(len(values), 1))(*values)


class _Builder:
    def __init__(self, lib, owners, role, row, identity):
        self.lib, self.p, self.owners, self.role, self.row = lib, lib.metal_ir_program_new(), owners, role, row
        self.values, self.roots, self.roles, self.planned, self.builtins = {}, {}, {}, {}, {}
        self.identity = identity

    def root(self, t):
        key = _key(t)
        if key in self.roots:
            return self.roots[key]
        base = t.tensor if isinstance(t, _Words) else t
        numel = base.untyped_storage().nbytes() // base.element_size()
        owner = self.owners.get(key)
        role = self.role(owner) if owner is not None else STATE if key in GENERATORS else \
            CONSTANT if base.device.type != 'meta' else TEMPORARY
        v = self.lib.metal_ir_value(self.p, role, SCALAR[base.dtype], 1, _array(mesh.Q, [numel]), 1,
                                    _array(Term, [Term(0, 0, 1, 1, 0)]), 0, self.identity)
        if owner is not None:
            k, index, _, pages, row = owner
            self.lib.metal_ir_value_planned(self.p, v, k, index, 0)
            if pages:
                self.lib.metal_ir_value_paged(self.p, v, row, pages)
            self.planned[v] = owner
        elif base.device.type != 'meta':
            storage = base.untyped_storage()
            HOLDERS[id(base)] = base
            buffer = storage.data_ptr() if base.device.type == 'mps' else None
            data = OBJECT(buffer, _SEL(b'contents')) if buffer else storage.data_ptr()
            self.lib.metal_ir_value_data(self.p, v, data, storage.nbytes(), buffer, 0, id(base), _released_constant)
        self.roots[key], self.roles[key] = v, role
        return v

    def value(self, t):
        base, words = (t.tensor, 2) if isinstance(t, _Words) else (t, 1)
        strides = tuple(s * words for s in base.stride())
        signature = (_key(t), SCALAR[t.dtype], tuple(t.shape), strides, base.storage_offset() * words)
        if signature not in self.values:
            root = self.root(t)
            v = self.lib.metal_ir_value(self.p, self.roles[_key(t)], SCALAR[t.dtype], len(t.shape), _array(mesh.Q, t.shape), len(strides),
                                        _array(Term, [Term(d, 0, s, 1, 0) for d, s in enumerate(strides)]),
                                        signature[4], self.identity)
            self.lib.metal_ir_value_view(self.p, v, root)
            self.values[signature] = v
        return self.values[signature]

    def builtin(self, name):
        if name not in self.builtins:
            self.builtins[name] = self.lib.metal_ir_builtin_named(name.encode())
        return self.builtins[name]

    def replay(self, r, region, inputs):
        lib, p, h = self.lib, self.p, []
        for record in region.code:
            call = record[0]
            if call == 'parameter':
                h.append(lib.metal_ir_parameter(p, r, record[1]))
            elif call == 'loop':
                h.append(lib.metal_ir_parameter(p, r, inputs + record[1]))
            elif call == 'literal':
                h.append(lib.metal_ir_literal(p, r, record[1], record[2]))
            elif call == 'binary':
                h.append(lib.metal_ir_binary(p, r, record[1], h[record[2]], h[record[3]]))
            elif call == 'unary':
                h.append(lib.metal_ir_unary(p, r, record[1], h[record[2]]))
            elif call == 'convert':
                h.append(lib.metal_ir_convert(p, r, record[1], h[record[2]]))
            elif call == 'bitcast':
                h.append(lib.metal_ir_bitcast(p, r, record[1], h[record[2]]))
            elif call == 'builtin':
                given = _array(mesh.U, [h[x] for x in record[2]])
                h.append(lib.metal_ir_builtin(p, r, self.builtin(record[1]), len(record[2]), given))
            elif call == 'extract':
                given = _array(mesh.U, [h[x] for x in record[2]])
                h.append(lib.metal_ir_extract(p, r, record[1], len(record[2]), given))
            else:
                given = _array(mesh.U, [h[x] for x in record[1]])
                lib.metal_ir_yield(p, r, len(record[1]), given)
                h.append(None)

    def generic(self, g):
        lib, p = self.lib, self.p
        inputs, outputs = [self.value(t) for t, _ in g.inputs], [self.value(t) for t, _ in g.outputs]
        exprs, terms, loops = [], [], len(g.sizes)
        for t, rows in g.inputs + g.outputs:
            for d in range(len(t.shape)):
                row = rows[d] if rows is not None else (0,) * (loops + 1)
                first = len(terms)
                terms += [Term(l, 0, c, 1, 0) for l, c in enumerate(row[:loops]) if c]
                exprs.append(Expr(first, len(terms) - first, row[loops]))
        op = lib.metal_ir_generic(p, loops, _array(mesh.Q, g.sizes), _array(mesh.U, g.iterators), len(inputs),
                                  _array(mesh.U, inputs), len(outputs), _array(mesh.U, outputs),
                                  _array(mesh.U, g.combiner), _array(mesh.U, g.start), _array(mesh.U, g.accumulator),
                                  _array(Expr, exprs), len(terms), _array(Term, terms), g.precision, g.place,
                                  self.identity)
        if g.states:
            lib.metal_ir_states(p, op, len(g.states), _array(mesh.U, [kind for kind, _ in g.states]),
                                _array(mesh.Q, [_bits(identity, kind) for kind, identity in g.states]))
        self.replay(lib.metal_ir_region(p, op, BODY), g.body, len(inputs))
        if g.states:
            self.replay(lib.metal_ir_region(p, op, COMBINE), g.combine, 0)
            self.replay(lib.metal_ir_region(p, op, FINALIZE), g.finalize, 0)
        return op


# metal-microbench metal_ir.h (its C ABI over .build/libmetal_ir.dylib); mesh-graph.h struct mesh_graph_term (metal_ir.h
# struct metal_ir_term) and struct mesh_graph_generation; rdma/mesh-metal.h struct mesh_metal_range, struct
# mesh_metal_release and struct mesh_metal_sending
class Term(C.Structure):
    _fields_ = [('dim', mesh.U), ('padding', mesh.U), ('coefficient', C.c_int64), ('divisor', mesh.Q), ('modulus', mesh.Q)]


class Expr(C.Structure):
    _fields_ = [('first', mesh.U), ('terms', mesh.U), ('constant', C.c_int64)]


class Generated(C.Structure):
    _fields_ = [('every', mesh.U), ('producer', mesh.U), ('ordinal', mesh.U)]


class Range(C.Structure):
    _fields_ = [('input', Remote), ('generation', Generated), ('partitions', mesh.U), ('page', mesh.Q),
                ('field', C.POINTER(mesh.U)), ('at', mesh.Q), ('transfer', mesh.U), ('owner', mesh.U)]


class Release(C.Structure):
    _fields_ = [('supplied', mesh.Q), ('status', mesh.Q), ('total', mesh.Q), ('transfer', mesh.U), ('padding', mesh.U)]


class Sending(C.Structure):
    _fields_ = [('sent', MetalSent), ('extent', C.POINTER(mesh_graph.Access)), ('extents', C.c_size_t),
                ('transfer', mesh.U), ('padding', mesh.U)]


# metal_ir.h struct metal_ir_dispatch
class Dispatch(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('kind', 'kernel', 'binding', 'bindings', 'effect', 'effects', 'source',
                                            'sources', 'allocation', 'allocations', 'arguments', 'later', 'flags',
                                            'fill')] + [('groups', mesh.U * 3), ('threads', mesh.U * 3)]


TASK_DISPATCH = 1
MACHINES = []


# Roofline (S. Williams, A. Waterman and D. Patterson, CACM 52(4), 2009, §3) operands of this node's resources as
# configuration (etc/mesh-capacity.json, its model's "machines" {"gpu": {...}, "cpu": {...}} with struct
# mesh_graph_machine's fields), read once; none recorded: no estimate
def _machines():
    if not MACHINES:
        model = subprocess.run(['/usr/sbin/sysctl', '-n', 'hw.model'], capture_output=True, text=True).stdout.strip()
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'etc', 'mesh-capacity.json')
        with open(path) as records:
            given = json.load(records).get('models', {}).get(model, {}).get('machines', {})
        MACHINES.append({resource: mesh_graph.Machine(**given[name]) for resource, name in ((GPU, 'gpu'), (CPU, 'cpu'))
                         if name in given})
    return MACHINES[0]


COMPILER = []


def _compiler():
    if not COMPILER:
        lib = C.CDLL(os.path.join(_repository(), '.build', 'libmetal_ir.dylib'))
        P, U, Q = C.c_void_p, mesh.U, mesh.Q
        for name, result, arguments in (
                ('metal_ir_program_new', P, []), ('metal_ir_program_free', None, [P]),
                ('metal_ir_value', U, [P, U, U, U, C.POINTER(Q), U, C.POINTER(Term), C.c_int64, Q]),
                ('metal_ir_value_view', None, [P, U, U]),
                ('metal_ir_value_data', None, [P, U, P, C.c_size_t, P, Q, P, RELEASE]),
                ('metal_ir_value_planned', None, [P, U, U, U, Q]), ('metal_ir_value_paged', None, [P, U, Q, U]),
                ('metal_ir_value_receives', None, [P, U, C.POINTER(Range)]),
                ('metal_ir_value_sends', None, [P, U, C.POINTER(Release)]),
                ('metal_ir_value_received', None, [P, U, C.POINTER(Range)]),
                ('metal_ir_value_sent', None, [P, U, C.POINTER(Release)]),
                ('metal_ir_value_sending', None, [P, U, C.POINTER(Sending)]),
                ('metal_ir_transport', None, [P, Q]),
                ('metal_ir_generic', U, [P, U, C.POINTER(Q), C.POINTER(U), U, C.POINTER(U), U, C.POINTER(U),
                                         C.POINTER(U), C.POINTER(U), C.POINTER(U), C.POINTER(Expr), U, C.POINTER(Term),
                                         U, U, Q]),
                ('metal_ir_states', None, [P, U, U, C.POINTER(U), C.POINTER(Q)]),
                ('metal_ir_region', U, [P, U, U]), ('metal_ir_parameter', U, [P, U, U]),
                ('metal_ir_literal', U, [P, U, U, Q]), ('metal_ir_binary', U, [P, U, U, U, U]),
                ('metal_ir_unary', U, [P, U, U, U]), ('metal_ir_convert', U, [P, U, U, U]),
                ('metal_ir_bitcast', U, [P, U, U, U]), ('metal_ir_builtin', U, [P, U, U, U, C.POINTER(U)]),
                ('metal_ir_extract', U, [P, U, U, U, C.POINTER(U)]), ('metal_ir_builtin_named', U, [C.c_char_p]),
                ('metal_ir_yield', None, [P, U, U, C.POINTER(U)]),
                ('metal_ir_lower', C.c_int, [P, P, C.c_char_p, C.c_size_t]),
                ('metal_ir_finalize', C.c_int, [P, C.c_char_p, C.c_size_t]), ('metal_ir_values', U, [P]),
                ('metal_ir_dispatches', U, [P, C.POINTER(C.POINTER(Dispatch))]),
                ('metal_ir_dispatch_actor', None, [P, U, C.POINTER(mesh_graph.Actor)])):
            function = getattr(lib, name)
            function.restype, function.argtypes = result, arguments
        COMPILER.append(lib)
    return COMPILER[0]


# design/torch-capture-change-request.md items 3 and 9; design/plan-to-bridge-change-request.md R4; NCCL User Guide
# 2.32.3, "NCCL operations can be captured by CUDA Graphs"
def _endpoints(ir):
    receives = [i for _, landed in ir.roles.values() for i, _ in landed if ir.declared[i].value != mesh.ABSENT]
    sends = [i for g in sorted(ir.publishes) for i in ir.publishes[g] if ir.declared[i].value != mesh.ABSENT]
    return receives, sends


# design/torch-capture-change-request.md items 3, 6 and 9; design/aot-compilation-change-request.md, appendix item 6;
# design/compilation-composition-change-request.md items 1 and 6, "Complete replacement workgroup" (A and B): each
# call of the block, in the block's order (row j of the program is the plan's call j), its structured operations over
# meta stand-ins of its function's plan values (copy j of a call's values, a block input's the holder's), declared in
# its own program with the transport's declared ranges and releases (mesh-metal.h mesh_metal_range_declare), lowered,
# and prepared by the executor for its schedule and lifetimes
def _programs(context, irs, compiled, gm, of, constants, holders, owned, calls):
    lib, made = _recorder()
    compiler, transport = _compiler(), _transport()
    error = C.create_string_buffer(4096)
    device = made()
    recorder = _owned(owned, lib.MetalRecorderOpen(device, error, len(error)))
    PUT(device, SELECTOR[b'release'])
    if not recorder:
        raise OSError(errno.ENODEV, 'the recorder: %s' % error.value.decode())
    owners, raw, standins, rows, ordinals = {}, {}, {}, [], {}
    rated = {k: {v for v, _ in ir.inputs.values()} for k, ir in enumerate(irs)}
    landing = {k: {ir.declared[i].value for i in _endpoints(ir)[0]} | {v for v, _ in ir.landed.values()}
               for k, ir in enumerate(irs)}
    results = {k: {ir.homes[key][0] for key in next(item[3] for item in ir.program if item[0] == 'output')[1].values()
                   if key in ir.homes} for k, ir in enumerate(irs)}
    listed = (mesh.U * max(len(calls), 1))(*calls)

    def role(owner):
        k, v = owner[:2]
        if v in rated[k] or v in landing[k]:
            return INPUT
        if irs[k].values[v].kind == mesh.INVARIANT:
            return STATE
        return OUTPUT if v in results[k] else TEMPORARY

    def standin(owner, dtype, numel):
        made = torch.empty(numel, dtype=dtype, device='meta')
        owners[_key(made)] = owner
        return made

    # a plan value's one stand-in, over its whole logical extent; a value the plan places in page-table rows (run 0)
    # its pages of `ir.page` bytes (metal_ir_value_paged)
    def instance(k, j):
        if (k, j) not in standins:
            ir, copies = irs[k], []
            for v, (dtype, numel, _) in enumerate(ir.layouts):
                pages = 0 if ir.values[v].run else -(-numel * dtype.itemsize // ir.page)
                copies.append(raw[k, 0][v] if j and v in rated[k] else [standin((k, v, j, pages, ir.page), dtype, numel)])
            raw[k, j], standins[k, j] = copies, _homes(ir, copies)
        return standins[k, j]

    def storage(k, v, j):
        return raw[k, 0 if v in rated[k] else j][v][0]

    def call(node, args, kwargs):
        k = of[node]
        j = ordinals[k] = ordinals.get(k, -1) + 1
        ir, row = irs[k], len(rows)
        inputs = pytree.arg_tree_leaves(*args, **kwargs)
        if ir.device is None:
            # design/compilation-composition-change-request.md items 1 and 6 (unit 6E): at a node with no section of
            # the function, its call's result is its delivered leaves (_deliver `ir.landed`), each over its landing
            # value's stand-in, and its passed-through leaves (Compiled `passed`) the call's own inputs here
            output, keys = next(item[3] for item in ir.program if item[0] == 'output')
            places, passed = instance(k, j)[0] if ir.landed else {}, compiled[k].passed
            leaves = [places[keys[leaf]][0] if o in ir.landed else inputs[passed[o]] if o in passed else None
                      for o, leaf in enumerate(pytree.tree_leaves(output.args))]
            result = pytree.tree_unflatten(leaves, compiled[k].spec) if any(x is not None for x in leaves) else None
            rows.append({'node': node, 'k': k, 'j': j, 'result': result, 'builder': None})
            return result
        ops, result = _instantiate(ir, instance(k, j)[0], inputs, compiled[k].spec, row)
        builder = _Builder(compiler, owners, role, row, (k << 48) | (j << 32))
        owned[id(builder)] = lambda p=builder.p: compiler.metal_ir_program_free(p)
        for g in ops:
            builder.generic(g)
        receives, sends = _endpoints(ir)
        marks = {'received': [], 'sent': [], 'sending': {}}
        # 6's stage A: each received range (this call's own transfer i landing in its value v; a result of call r of
        # function f delivered here, f's transfer i landing in f's value v) and each sent range, declared from the
        # static call graph and the value's bytes; the plan's words follow at D
        landed = [(row, i, k, ir.declared[i].value, j) for i in receives]
        landed += [(r, i, f, v, rows[r]['j']) for _, r, f, _, v, i, _ in ir.delivered.get(j, ())]
        for owner, i, f, v, at in landed:
            whole, declared = storage(f, v, at), Range()
            mesh.check(transport.mesh_metal_range_declare(C.byref(context), listed, len(calls), row, owner, i,
                                                          whole.untyped_storage().nbytes(), 0, C.byref(declared)))
            value = builder.root(whole)
            compiler.metal_ir_value_receives(builder.p, value, C.byref(declared))
            marks['received'].append((value, f, v, declared))
        for i in sends:
            whole = storage(k, ir.declared[i].value, j)
            release, value = Release(transfer=i), builder.root(whole)
            compiler.metal_ir_value_sends(builder.p, value, C.byref(release))
            marks['sent'].append((value, release))
        for i in receives + sends:
            marks['sending'][i] = builder.root(storage(k, ir.declared[i].value, j))
        if compiler.metal_ir_lower(builder.p, device, error, len(error)):
            raise OSError(errno.EINVAL, 'call %s of compiled function %d: its lowering: %s'
                          % (node.name, k, error.value.decode()))
        rows_work = _work(compiler, builder.p)
        prepared = _owned(owned, lib.MetalRecorderPrepare(recorder, builder.p, error, len(error)))
        if not prepared:
            raise OSError(errno.EINVAL, 'call %s of compiled function %d: its preparation: %s'
                          % (node.name, k, error.value.decode()))
        rows.append({'node': node, 'k': k, 'j': j, 'result': result, 'builder': builder, 'prepared': prepared,
                     'marks': marks, 'receives': receives, 'sends': sends, 'work': rows_work[0],
                     'launches': rows_work[1]})
        return result
    _Program(gm, call).run(*[instance(k, 0)[1][i][0] if k is not None else constant
                             for i, (constant, k) in enumerate(zip(constants, holders))])
    return {'lib': lib, 'recorder': recorder, 'owners': owners, 'rated': rated, 'rows': rows, 'compiler': compiler,
            'device': device}


# mesh-graph.h mesh_graph_actor_time over each dispatch's actor facts (metal_ir_dispatch_actor), on the machine of its
# resource: a row's device part in ns, and its GPU launches
def _work(compiler, p):
    machines, made = _machines(), C.POINTER(Dispatch)()
    count = compiler.metal_ir_dispatches(p, C.byref(made))
    total = launches = 0
    for d in range(count):
        resource = CPU if made[d].kind == TASK_DISPATCH else GPU
        launches += resource == GPU
        if resource in machines:
            actor = mesh_graph.Actor()
            compiler.metal_ir_dispatch_actor(p, d, C.byref(actor))
            total += mesh_graph.LIB.mesh_graph_actor_time(C.byref(machines[resource]), C.byref(actor))
    return total, launches


def _times(programs, irs):
    machines, made = _machines(), [[] for _ in irs]
    for row in programs['rows']:
        if row['builder'] is not None:
            made[row['k']].append((row['work'], row['launches']))
    times = []
    for k, ir in enumerate(irs):
        work = sum(w for w, _ in made[k]) / max(len(made[k]), 1)
        host = 0.0
        if CPU in machines:
            written = sum(ir.values[v].section.bytes for v, _ in ir.inputs.values())
            launches = sum(n for _, n in made[k]) / max(len(made[k]), 1)
            host = mesh_graph.LIB.mesh_graph_actor_time(C.byref(machines[CPU]), C.byref(mesh_graph.Actor(
                0.0, 1, 1, written, int(math.ceil(launches)), 0, None)))
        times.append((host, work))
    return times


# M. Lam, PLDI 1988, p. 322; ExecuTorch exir/memory_planning.py:348–354; metal_recording.h levels: and
# values:level:into:: each row's schedule's lifetimes, a plan value's interval the union of its calls' (every call an
# execution of the one plan), and the program values the lowering realized beyond the plan's (bytes > 0, no plan
# identity) new values of the row's function, `firsts[row]` the first of them
def _lifetimes(programs, irs):
    own = [[] for _ in irs]
    spanned = {}
    for at, row in enumerate(programs['rows']):
        if row['builder'] is None:
            continue
        k, recorder, prepared = row['k'], programs['recorder'], row['prepared']
        levels = LEVELS(recorder, _SEL(b'levels:'), prepared)
        made = (MetalValue * max(VALUES(recorder, _SEL(b'values:level:into:'), prepared, 0, None), 1))()
        count = VALUES(recorder, _SEL(b'values:level:into:'), prepared, 0, C.cast(made, C.c_void_p))
        row['own'] = {}
        for vi in range(count):
            value = made[vi]
            owner = row['builder'].planned.get(vi)
            if owner is not None:
                f, v = owner[:2]
                if f == k and irs[f].values[v].kind == mesh.EXECUTION and not irs[f].values[v].lag:
                    spans = [(value.begin, value.end)] + ([spanned[f, v]] if (f, v) in spanned else [])
                    spanned[f, v] = irs[f].values[v].begin, irs[f].values[v].end = \
                        min(s for s, _ in spans), max(e for _, e in spans)
            elif value.bytes:
                row['own'][vi] = len(irs[k].values) + len(own[k])
                own[k].append(mesh.Value(section=mesh.Section(bytes=value.bytes), kind=value.kind, state=value.state,
                                         begin=value.begin, end=value.end, run=1))
        row['levels'] = levels
    return own


# design/compilation-composition-change-request.md, "Complete replacement workgroup" (6's stage D): each row's received
# ranges, releases and sends' lifetimes prepared from the plan (mesh-metal.h mesh_metal_range_prepare,
# mesh_metal_release_prepare, mesh_metal_sending_prepare) and its plan's status word, then its final emission
def _finalized(programs, context, irs, planned, values, owned):
    transport, compiler, error = _transport(), programs['compiler'], C.create_string_buffer(4096)
    made = Transport()
    mesh.check(transport.mesh_metal_transport_create(C.byref(context), C.byref(planned), programs['device'],
                                                     C.byref(made)))
    owned[id(made)] = lambda: transport.mesh_metal_transport_destroy(C.byref(made))
    for at, row in enumerate(programs['rows']):
        builder = row['builder']
        if builder is None:
            continue
        p, k = builder.p, row['k']
        for value, f, v, declared in row['marks']['received']:
            field = (mesh.U * max(declared.partitions, 1))()
            status = transport.mesh_metal_range_prepare(C.byref(context), C.byref(made), C.byref(planned),
                                                        values[f][v].section, field, declared.partitions,
                                                        C.byref(declared))
            mesh.check(status)
            compiler.metal_ir_value_received(p, value, C.byref(declared))
        for value, release in row['marks']['sent']:
            mesh.check(transport.mesh_metal_release_prepare(C.byref(context), C.byref(made), C.byref(planned), at,
                                                            C.byref(release)))
            compiler.metal_ir_value_sent(p, value, C.byref(release))
        for i in row['receives'] + row['sends']:
            v, sending = irs[k].declared[i].value, Sending()
            section = values[k][v].section
            status = transport.mesh_metal_sending_prepare(C.byref(context), C.byref(made), C.byref(planned), at, i,
                                                          section, None, 0, C.byref(sending))
            if status == errno.ERANGE:
                extent = (mesh_graph.Access * sending.extents)()
                status = transport.mesh_metal_sending_prepare(C.byref(context), C.byref(made), C.byref(planned), at,
                                                              i, section, extent, sending.extents, C.byref(sending))
            mesh.check(status)
            if sending.sent.sends:
                compiler.metal_ir_value_sending(p, row['marks']['sending'][i], C.byref(sending))
        word = mesh.Q()
        mesh.check(transport.mesh_metal_status_prepare(C.byref(context), C.byref(made), planned.call[at].plan,
                                                       C.byref(word)))
        compiler.metal_ir_transport(p, word.value)
        row['status'] = word.value
        if compiler.metal_ir_finalize(p, error, len(error)):
            raise OSError(errno.EINVAL, 'call %s of compiled function %d: its emission: %s'
                          % (row['node'].name, k, error.value.decode()))
    return made


# design/aot-compilation-change-request.md, appendix item 6; design/disjoint-pages-change-request.md C3.3; ExecuTorch,
# "Concepts", Memory planning: copy c of a value written by call j of the n calls of its function in the block is the
# copy of its plan's execution t n + j, and a reader whose function the block calls m times runs at executions t m + i,
# so its offsets repeat with period m x copies; a block input's copy is the block's execution t
def _expanded(selection, owned, reader, addresses, base):
    k, v, j = owned[:3]
    at, m, n = addresses[k][v], selection['every'][reader], selection['every'][k]
    block = v in selection['rated'][k]
    picked = [at[(x // m) % len(at) if block else ((x // m) * n + j) % len(at)] for x in range(m * len(at))]
    flat = [a for copy in picked for a in (copy if isinstance(copy, list) else [copy])]
    return (C.c_uint64 * len(flat))(*[a - base for a in flat]), len(picked)


# ExecuTorch, "Concepts", Memory planning; torch 2.14.0 torch/_inductor/cudagraph_trees.py; M. Lam, PLDI 1988, §2.3;
# metal_recording.h place:value:in:offsets:copies:ahead:shares:, place:step:, state:buffer:status:, replay: and
# MetalReplayNative; MTL4CommandQueue.h addResidencySet:; design/compilation-composition-change-request.md, "Complete
# replacement workgroup" (6's stage E): each row's values bound to their placed copies (a plan value's copies as its
# reader's executions select them, _expanded; a realized value's own copies), its step word and transport view, its
# replay and compiled CPU entries the row (MetalCall `call` = j + 1) of the one native program, prepared with the
# plan's results (mesh-collective.h struct mesh_planned) and retained by it until retirement
def _replays(programs, context, irs, planned, horizon, values, addresses, base, buffer, period, owned, made, steps,
             every, times):
    lib, recorder, error = programs['lib'], programs['recorder'], C.create_string_buffer(4096)
    queue, rows, selection = OBJECT(recorder, _SEL(b'queue')), [], {'every': every, 'rated': programs['rated']}

    def offsets(at):
        return (C.c_uint64 * len(at))(*[a - base for a in at])
    for entry in programs['rows']:
        if entry['builder'] is None:
            rows.append(None)
            continue
        k, prepared = entry['k'], entry['prepared']
        for vi, owner in entry['builder'].planned.items():
            at, width = _expanded(selection, owner, k, addresses, base)
            PLACE_VALUE(recorder, _SEL(b'place:value:in:offsets:copies:ahead:shares:'), prepared, vi, buffer, at, width,
                        0, bool(values[owner[0]][owner[1]].shares))
        for vi, index in entry['own'].items():
            value = values[k][index]
            PLACE_VALUE(recorder, _SEL(b'place:value:in:offsets:copies:ahead:shares:'), prepared, vi, buffer,
                        offsets(addresses[k][index]), value.section.count, 0, bool(value.shares))
        step = values[k][steps[k]]
        PLACE_STEP(recorder, _SEL(b'place:step:offsets:copies:shares:'), prepared, buffer,
                   offsets(addresses[k][steps[k]]), step.section.count, bool(step.shares))
        STATE_BUFFER(recorder, _SEL(b'state:buffer:status:'), prepared, made.state, entry['status'])
        replay = Replay()
        status = lib.MetalRecorderReplay(recorder, prepared, C.byref(replay), error, len(error))
        owned.pop(prepared)()
        if status:
            raise OSError(status, 'call %s of compiled function %d: the replay: %s'
                          % (entry['node'].name, k, error.value.decode()))
        residues = mesh.Q()
        native = lib.MetalReplayNative(C.byref(replay), C.byref(residues))
        if replay.commands:
            SET(queue, _SEL(b'addResidencySet:'), replay.residency)
        owned[id(replay)] = lambda replay=replay: (SET(queue, _SEL(b'removeResidencySet:'), replay.residency)
                                                   if replay.commands else None, lib.MetalReplayDestroy(C.byref(replay)))
        rows.append((replay, native, residues.value))
    calls = (Call * len(rows))(*[Call(C.pointer(r[0]) if r is not None and r[0].commands else None, row + 1,
                                      r[1] if r is not None else None, r[2] if r is not None else 0)
                                 for row, r in enumerate(rows)])
    spans = (C.c_double * max(2 * len(rows), 1))(*[t for row in programs['rows'] for t in times[row['k']]])
    program = _owned(owned, lib.MetalRecorderProgram(recorder, calls, len(calls), spans, C.byref(planned), error,
                                                     len(error)))
    if not program:
        raise OSError(errno.EINVAL, 'the program\'s calls: %s' % error.value.decode())
    # design/synchronization-signature-change-request.md items 1 and 2: the compiled program's one signature over its
    # warm-up and one period of its residues, the plan's composed in by its producer (MetalProgramSignature), kept with
    # the compiled program as the graph module's forms and its record's bytes
    replayed = mesh_graph.Signature()
    status = lib.MetalProgramSignature(program, C.addressof(context), horizon + period, C.byref(replayed), error,
                                       len(error))
    if status:
        raise OSError(status, 'the program\'s signature: %s' % error.value.decode())
    try:
        signed = dict(mesh_graph.forms(replayed), record=mesh_graph.record(replayed))
    finally:
        mesh_graph.free(replayed)
    # tools/e2b_replay.m signature_write: the record and its three printed forms beside the prefix MESH_SIGNATURE names,
    # the mesh-sync-monitor.py operand
    prefix = os.environ.get('MESH_SIGNATURE')
    if prefix:
        for suffix, data in (('.record', signed['record']), ('.mtx', signed['matrix_market'].encode()),
                             ('.dot', signed['dot'].encode()), ('.txt', signed['space_time'].encode())):
            with open(prefix + suffix, 'wb') as written:
                written.write(data)
        print('prepared signature %s digest %s' % (prefix, signed['digest']), flush=True)
    gpu = OBJECT(queue, _SEL(b'device'))
    allocators = (C.c_void_p * horizon)(*[_owned(owned, OBJECT(gpu, _SEL(b'newCommandAllocator')))
                                          for _ in range(horizon)])
    return {'lib': lib, 'queue': queue, 'calls': calls, 'program': program, 'allocators': allocators,
            'ended': _owned(owned, OBJECT(gpu, _SEL(b'newSharedEvent'))), 'error': error, 'signature': signed}


# design/aot-compilation-change-request.md, appendix item 6: block execution p's copy of a call's result, the copy of
# its plan's execution p every + ordinal (a block input's: p)
def _period(copies, horizon):
    return math.lcm(horizon, *(len(c) for function in copies.values() for c in function))


# N. D. Jones, C. K. Gomard and P. Sestoft, 1993, Definition 4.1; design/aot-compilation-change-request.md, appendix
# item 6; torch 2.14.0 torch/_inductor/cudagraph_trees.py: the block's outputs specialized once to each block execution
# residue p, each call's results the placed tensors of their stand-ins at the copies of its execution p every + ordinal
# (a block input's at p)
def _specialize(programs, horizon, bound, gm, constants, inputs, every):
    copies = {k: b[3] for k, b in enumerate(bound)}
    results = {row['node']: row['result'] for row in programs['rows']}
    owners, rated = programs['owners'], programs['rated']

    def leaf(x, p):
        if not isinstance(x, torch.Tensor):
            return x
        k, v, j = owners[_key(x)][:3]
        at = copies[k][v]
        copy = at[(p if v in rated[k] else p * every[k] + j) % len(at)]
        return copy.as_strided(x.shape, x.stride(), copy.storage_offset() + x.storage_offset())
    return [_Program(gm, lambda node, args, kwargs, p=p: pytree.tree_map(lambda x: leaf(x, p), results[node])).run(
        *[placed[0][p % len(placed[0])] if placed is not None else constant for constant, placed in zip(constants, inputs)])
        for p in range(_period(copies, horizon))]


# mesh-call.h mesh_transfers_start; MPI 4.1 §3.9, MPI_START; NCCL User Guide, CUDA Stream Semantics
def _replayed(context, replay, b):
    status = replay['lib'].MetalProgramRun(replay['program'], replay['allocators'], b, 1, replay['ended'],
                                           C.addressof(context), START, STARTED, replay['error'], len(replay['error']))
    if status:
        raise OSError(status, 'execution %d of the program: %s' % (b, replay['error'].value.decode()))
    return replay['outputs'][b % len(replay['outputs'])]


# MTLEvent.h `signaledValue`
def _ended(replay):
    return VALUE(replay['ended'], _SEL(b'signaledValue'))


# design/aot-compilation-change-request.md A1 to A3 and A10, "The calls' graph"; P. Barham et al., MLSys 2022, section
# 3; J. Reed, Z. DeVito, H. He, A. Ussery and J. Ansel, MLSys 2022; torch v2.14.0 torch/fx/_symbolic_trace.py,
# Tracer.is_leaf_module; torch/nn/modules/module.py, _forward_unimplemented
class Function(torch.nn.Module):
    def __init__(self, module, placement, weights, device=None):
        super().__init__()
        self.module, self.placement, self.weights, self.device = module, placement, weights, device


class Tracer(torch.fx.Tracer):
    def is_leaf_module(self, m, module_qualified_name):
        return isinstance(m, Function) or super().is_leaf_module(m, module_qualified_name)


# M. C. Rinard and P. C. Diniz, Commutativity analysis: a new analysis technique for parallelizing compilers, ACM TOPLAS
# 19(6), 1997, section 4.9.2 and footnote 4; design/aot-compilation-change-request.md A10
ALGEBRA = {operator.add: 'add', torch.add: 'add', 'add': 'add', operator.mul: 'mul', torch.mul: 'mul', 'mul': 'mul',
           operator.sub: 'sub', torch.sub: 'sub', 'sub': 'sub', operator.neg: 'neg', torch.neg: 'neg', 'neg': 'neg',
           torch.maximum: 'maximum', 'maximum': 'maximum', torch.minimum: 'minimum', 'minimum': 'minimum'}


def _rewrite(name, operands):
    def of(e, kind):
        return isinstance(e, tuple) and e[:1] == (kind,)
    if name == 'sub':
        return _rewrite('add', (operands[0], _rewrite('neg', operands[1:])))
    if name == 'neg':
        return operands[0][1] if of(operands[0], 'neg') else ('neg', operands[0])
    for i, e in enumerate(operands):
        if name == 'mul' and of(e, 'add'):
            return _rewrite('add', tuple(_rewrite('mul', operands[:i] + operands[i + 1:] + (term,)) for term in e[1:]))
    flat = [f for e in operands for f in (e[1:] if of(e, name) else (e,))]
    return (name,) + tuple(sorted(flat, key=repr))


# Rinard and Diniz, sections 3.2.1, 3.4.1, 4.9.1 and 4.9.2; torch v2.14.0 torch/fx/interpreter.py;
# design/aot-compilation-change-request.md A10
def _expression(op, target, args, kwargs):
    name = ALGEBRA.get(target) if op in ('call_function', 'call_method') and not kwargs else None
    if name and len(args) == 1 + (name != 'neg'):
        return _rewrite(name, tuple(args))
    return op, target, tuple(args), tuple(sorted(kwargs.items()))


class _Symbolic(torch.fx.Interpreter):
    def call_function(self, target, args, kwargs):
        return _expression('call_function', target, args, kwargs)

    def call_method(self, target, args, kwargs):
        return _expression('call_method', target, args, kwargs)

    def call_module(self, target, args, kwargs):
        return _expression('call_module', target, args, kwargs)

    def get_attr(self, target, args, kwargs):
        return _expression('get_attr', target, args, kwargs)


# Rinard and Diniz, sections 3.1 and 4.8, Figure 15; torch/fx/proxy.py TraceError
def _commutes(module, k):
    tracer = Tracer()
    try:
        graph = tracer.trace(module)
    except torch.fx.proxy.TraceError:
        return False
    body, count = torch.fx.GraphModule(tracer.root, graph), sum(node.op == 'placeholder' for node in graph.nodes)

    def run(first, second):
        inner = _Symbolic(body).run(*[('symbol', 'receiver') if i == k else ('symbol', first, i) for i in range(count)])
        return _Symbolic(body).run(*[inner if i == k else ('symbol', second, i) for i in range(count)])
    return k < count and run(1, 2) == run(2, 1)


# design/aot-compilation-change-request.md A10, "Its bound", and appendix item 6; A. J. Bernstein, Analysis of programs
# for parallel processing, IEEE Trans. Electronic Computers EC-15(5), 1966; Rinard and Diniz, section 2; R. P. Dilworth,
# Annals of Math., 1950; D. R. Fulkerson, Proc. AMS, 1956; `sources[j]` call j's flat input leaves' producers,
# `every[f]` the block's calls of function f
# (_Export), `horizon` the block executions the program issues ahead
def _analyse(gm, of, compiled, sources, every, horizon):
    modules = [record.function.module for record in compiled]
    nodes = list(gm.graph.nodes)
    callee = {node: of.get(node) for node in nodes}
    calls = [node for node in nodes if callee[node] is not None]
    place = {node: i for i, node in enumerate(calls)}
    link, follower = {}, {}
    for b in calls:
        reads = []
        map_arg((b.args, b.kwargs), reads.append)
        for k, a in enumerate(b.args):
            if isinstance(a, Node) and a in place and callee[a] == callee[b] and len(a.users) == 1 \
                    and reads.count(a) == 1 and link.get(a, (a, k))[1] == k and _commutes(modules[callee[b]], k):
                link[b], follower[a] = (a, k), b
                break
    chains = mesh_graph.paths(calls, lambda a: [follower[a]] if a in follower else [])
    step = {node: chain for chain in chains for node in chain}

    def inputs(node):
        found = set(node.all_input_nodes)
        if node in link:
            receiver = step[node][0].args[link[node][1]]
            found = (found - {link[node][0]}) | ({receiver} if isinstance(receiver, Node) else set())
        for i in node.all_input_nodes:
            if i in step and step[i][-1] is i:
                found |= set(step[i])
        return found
    ordered = mesh_graph.closure(nodes, inputs)
    flows = {(place[a], place[b]) for b in calls for a in ordered[b] if a in place}
    # Bernstein's conditions on a call's writes of its inputs (Compiled `mutated`): ordered against every other call
    # reading the same leaf, in program order; mesh-collective.h MESH_ORDER_FLOW for a pair of the inputs' closure,
    # MESH_ORDER_CONFLICT for a pair of the conditions (a pair of both)
    conflicts = set()
    for j, b in enumerate(calls):
        for q in compiled[callee[b]].mutated:
            written = sources[j][q]
            conflicts |= {(min(c, j), max(c, j)) for c, read in enumerate(sources)
                          if c != j and written is not None and written in read}
    pairs = sorted(flows | conflicts)
    # the effects that persist between executions (Compiled `written`, `read`): every call of the functions sharing a
    # storage one of them writes is one chain in program order, which mesh_plan keeps and closes from one block's last
    # call to the next block's first (delay 1, mesh-collective.h `chains`), with the commuting chains merged into it:
    # one labelling (mesh_graph.components) over the links between successive calls of each commuting chain and of
    # each written storage's calls, both directions; each call's head the least call of its label
    groups = [[place[node] for node in chain] for chain in chains] + \
        [[j for j, node in enumerate(calls) if storage in compiled[callee[node]].written | compiled[callee[node]].read]
         for storage in {s for record in compiled for s in record.written}]
    label = mesh_graph.components(len(calls), [link for group in groups for a, b in zip(group, group[1:])
                                               for link in ((a, b), (b, a))])
    least = {}
    for j, c in enumerate(label):
        least.setdefault(c, j)
    head = [least[c] for c in label]
    # A10, "Its bound": the most calls of a function outstanding at once is the width (Dilworth and Fulkerson,
    # mesh_graph.width) of its calls in the order the program issues them over its `horizon` block executions:
    # vertices (b, j), b < horizon; inside a block A10's pairs and each chain's successive calls; across blocks each
    # closed chain's (two calls or more, or calls writing persistent state) last call of block b before its first of
    # block b + 1, as mesh_plan closes chains; that order's transitive closure by mesh_graph.closure. Storage for the
    # executions in flight is mesh_plan's from `horizon` (plan_unroll), not this bound
    members = {}
    for j, h in enumerate(head):
        members.setdefault(h, []).append(j)
    before = {j: {a for a, b in pairs if b == j} for j in range(len(calls))}
    for chain in members.values():
        for a, b in zip(chain, chain[1:]):
            before[b].add(a)
    closed = {h: chain[-1] for h, chain in members.items()
              if len(chain) > 1 or any(compiled[callee[calls[j]]].written for j in chain)}
    unrolled = [(b, j) for b in range(horizon) for j in range(len(calls))]
    after = mesh_graph.closure(unrolled, lambda v: [(v[0], a) for a in before[v[1]]] +
                               ([(v[0] - 1, closed[v[1]])] if v[0] and v[1] in closed else []))
    outstanding = {}
    for f in every:
        mine = [v for v in unrolled if callee[calls[v[1]]] == f]
        outstanding[f] = mesh_graph.width(mine, lambda a: [x for x in mine if a in after[x]])
    return Calls(outstanding, (mesh.U * max(2 * len(pairs), 1))(*[j for pair in pairs for j in pair]),
                 (mesh.U * max(len(pairs), 1))(*[mesh.ORDER_CONFLICT if pair in conflicts else mesh.ORDER_FLOW
                                                 for pair in pairs]), len(pairs),
                 (mesh.U * max(len(calls), 1))(*head))


# design/aot-compilation-change-request.md A10; P. Barham et al., MLSys 2022, section 4.2; torch v2.14.0
# torch/fx/interpreter.py
class _Program(torch.fx.Interpreter):
    def __init__(self, gm, execute):
        super().__init__(gm)
        self.execute = execute

    def run_node(self, n):
        if n.op == 'call_module':
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            return self.execute(n, args, kwargs)
        if n.op == 'call_function' and self.env[n.args[0]] is None:
            return None
        return super().run_node(n)


# torch 2.14.0 torch/export/graph_signature.py ExportGraphSignature buffers_to_mutate, parameters_to_mutate,
# user_inputs_to_mutate: `written` and `read` the storages of the module state the function writes and reads (its
# effects that persist from one execution to the next), `mutated` the flat input leaves it writes; `passed` its result
# leaves that are its own inputs, {flat output leaf: flat input leaf} (an output that is placeholder q, the
# placeholders being the flat input leaves in order)
Compiled = collections.namedtuple('Compiled', 'function signature gm examples result spec written read mutated '
                                              'passed')


# a flat leaf of a value of the traced program: leaf `leaf` of call `call`'s result, or of block input leaf `leaf`
# (call None)
class _Produced:
    __slots__ = ('call', 'leaf')

    def __init__(self, call, leaf):
        self.call, self.leaf = call, leaf

    def __eq__(self, other):
        return isinstance(other, _Produced) and (self.call, self.leaf) == (other.call, other.leaf)

    def __hash__(self):
        return hash((self.call, self.leaf))


# torch 2.14.0 torch/export/graph_signature.py: the storages of a function's state named by `names` (fully qualified
# names in the exported module, as torch/nn/modules/module.py get_submodule resolves them)
def _storages(module, names):
    return frozenset(operator.attrgetter(name)(module).untyped_storage().data_ptr() for name in names)
# design/torch-source-lowering-change-request.md item 2; PyTorch docs/source/notes/cuda.md L1683; torch 2.14.0
# torch/_higher_order_ops/cond.py and torch/_higher_order_ops/while_loop.py
CONTROL = (torch.ops.higher_order.cond, torch.ops.higher_order.while_loop)


def _static(gm):
    def held(module, node):
        return operator.attrgetter(node.target)(module) if node.op == 'get_attr' else None
    for module in mesh_graph.walk([gm], lambda m: [held(m, n) for n in m.graph.nodes
                                                   if isinstance(held(m, n), torch.fx.GraphModule)]):
        for node in module.graph.nodes:
            if node.op != 'call_function' or node.target not in CONTROL:
                continue
            data = [node.args[0]]
            if node.target is torch.ops.higher_order.while_loop:
                test = held(module, node.args[0]).graph
                read = mesh_graph.reach([test.output_node()], lambda n: n.all_input_nodes)
                data = [o for p, o in zip([n for n in test.nodes if n.op == 'placeholder'],
                                          [*node.args[2], *node.args[3]]) if p in read]
            on = {v.meta['val'].device.type if isinstance(v, Node) and isinstance(v.meta.get('val'), torch.Tensor)
                  else 'cpu' for v in data}
            if on - {'mps'}:
                raise ValueError('%s (%s): dynamic control flow based on CPU data is not compiled '
                                 '(design/torch-source-lowering-change-request.md item 2; PyTorch '
                                 'docs/source/notes/cuda.md L1683)' % (node.name, node.target))


def _metadata(value):
    return (tuple(value.shape), tuple(value.stride()), value.dtype, value.device) \
        if isinstance(value, torch.Tensor) else value


# design/aot-compilation-change-request.md A1, A10 and appendix item 6; torch 2.14.0 torch/export/__init__.py;
# torch/export/exported_program.py, run_decompositions and ExportedProgram.module; torch/export/_trace.py,
# _export_for_training; torch/_export/non_strict_utils.py, fakify; torch/_subclasses/fake_tensor.py;
# torch/_subclasses/fake_impls.py, constructors; torch/_subclasses/meta_utils.py, describe_tensor; torch/_guards.py,
# detect_fake_mode; torch/_export/passes/remove_runtime_assertions.py, _RemoveRuntimeAssertionsPass; R. Frostig, M. J.
# Johnson and C. Leary, SysML 2018; mesh-collective.h struct mesh_function
class _Export(torch.fx.Interpreter):
    def __init__(self, gm, mode):
        super().__init__(gm)
        self.mode, self.compiled, self.of, self.origin, self.calls = mode, [], {}, {}, []
        self.marks, self.sources, self.inputs = {}, [], 0

    # each node's value with every flat leaf replaced by the _Produced naming it
    def marked(self, n, produced):
        leaves, spec = pytree.tree_flatten(self.env[n])
        self.marks[n] = pytree.tree_unflatten([produced(o) for o in range(len(leaves))], spec)

    def fake(self, value, device=None):
        if not isinstance(value, torch.Tensor):
            return value
        with self.mode:
            return torch.empty_strided(tuple(value.shape), tuple(value.stride()), dtype=value.dtype,
                                       device=device or value.device)

    def run_node(self, n):
        function = self.module.get_submodule(n.target) if n.op == 'call_module' else None
        if isinstance(function, Function):
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            return self.export(n, function, args, kwargs)
        source = n.args[0] if n.op == 'call_function' and n.target is operator.getitem and n.args else None
        if isinstance(source, Node) and source in self.origin and not isinstance(self.env[source], torch.Tensor):
            self.origin[n] = self.origin[source]
            self.marks[n] = self.marks[source][n.args[1]]
            return super().run_node(n)
        if n.op == 'placeholder':
            value = super().run_node(n)
            self.env[n], first = value, self.inputs
            self.marked(n, lambda o: _Produced(None, first + o))
            self.inputs += len(pytree.tree_leaves(value))
            return value
        if n.op == 'output':
            return super().run_node(n)
        raise NotImplementedError(
            '%s (%s %s): a node of the traced program that is no placeholder, no call of a compiled function, no '
            'operator.getitem of such a call\'s result and no output; nothing runs eagerly between compiled calls '
            '(design/aot-compilation-change-request.md A1, corrected 2026-11-05)' % (n.name, n.op, n.target))

    def export(self, n, function, args, kwargs):
        flat, spec = pytree.tree_flatten((args, kwargs))
        signature = (spec, tuple(_metadata(v) for v in flat))
        found = [k for k, c in enumerate(self.compiled) if c.function is function and c.signature == signature]
        if not found:
            if function.device is not None:
                args, kwargs = pytree.tree_map(lambda v: self.fake(v, function.device), (args, kwargs))
            exported = torch.export.export(function.module, tuple(args), dict(kwargs), strict=False) \
                .run_decompositions({})
            gm = exported.module(check_guards=False)
            _RemoveRuntimeAssertionsPass()(gm)
            _static(gm)
            out = [a.meta['val'] if isinstance(a, Node) else a for a in gm.graph.output_node().args[0]]
            result = pytree.tree_unflatten([self.fake(v) for v in out], exported.call_spec.out_spec)
            given, inputs = exported.graph_signature, [node for node in gm.graph.nodes if node.op == 'placeholder']
            mutated = set(given.user_inputs_to_mutate.values())
            self.compiled.append(Compiled(function, signature, gm, pytree.tree_flatten((args, kwargs))[0], result,
                                          exported.call_spec.out_spec,
                                          _storages(function.module, [*given.buffers_to_mutate.values(),
                                                                      *given.parameters_to_mutate.values()]),
                                          _storages(function.module, [*given.inputs_to_buffers.values(),
                                                                      *given.inputs_to_parameters.values()]),
                                          frozenset(q for q, name in enumerate(given.user_inputs) if name in mutated),
                                          {o: inputs.index(a) for o, a in enumerate(gm.graph.output_node().args[0])
                                           if isinstance(a, Node) and a in inputs}))
            found = [len(self.compiled) - 1]
        self.of[n], self.origin[n] = found[0], len(self.calls)
        # design/compilation-composition-change-request.md item 6, unit 6E: each flat input leaf's producer
        self.sources.append([leaf if isinstance(leaf, _Produced) else None
                             for leaf in pytree.tree_leaves(map_arg((n.args, n.kwargs), lambda a: self.marks[a]))])
        origin, record, read = len(self.calls), self.compiled[found[0]], self.sources[-1]
        leaves, spec = pytree.tree_flatten(record.result)
        # a passed-through result leaf (Compiled `passed`) is its input's value: its producer is the input's
        self.marks[n] = pytree.tree_unflatten(
            [read[record.passed[o]] if o in record.passed and read[record.passed[o]] is not None
             else _Produced(origin, o) for o in range(len(leaves))], spec)
        self.calls.append(found[0])
        return record.result


# design/aot-compilation-change-request.md A1 to A3, A8 to A10 and appendix item 7;
# design/disjoint-pages-change-request.md C3.1, D.3.2 and D.3.3; mesh-collective.h mesh_plan, struct mesh_function and
# struct mesh_value; mesh-dataflow.h mesh_nodes_read, mesh_retire and struct mesh_ctx
# `horizon`: the block executions the program issues ahead of their ends, the one operand of the analysis's bound
# (outstanding), the storage's copies of block inputs and step words, the native program's lag and its allocators
def program(block, *example_inputs, horizon=HOST_LAG):
    tracer = Tracer()
    graph = tracer.trace(block)
    gm = torch.fx.GraphModule(tracer.root, graph)
    exporter = _Export(gm, FakeTensorMode())
    exporter.run(*[exporter.fake(x) for x in example_inputs])
    compiled, of, calls, sources = exporter.compiled, exporter.of, exporter.calls, exporter.sources
    # the block's returned leaves (_Export `marks` of its output's arguments): a call's result leaf the caller receives
    # at every node of the program, delivered where its function has no section (_deliver)
    returned = {(calls[leaf.call], leaf.leaf) for leaf in pytree.tree_leaves(map_arg(
        next(n for n in gm.graph.nodes if n.op == 'output').args, lambda a: exporter.marks.get(a)))
        if isinstance(leaf, _Produced) and leaf.call is not None}
    # the traced block's calls of each function (`every`) and each call's ordinal among them (`place`): the static
    # call identities the analysis's bound, the capture's generations, step words and block-copy selection take
    # (mesh_plan receives `calls` and publishes the same, planned.call)
    first, grouped = mesh_graph.rows(len(compiled), calls)
    every, place = {f: first[f + 1] - first[f] for f in range(len(compiled))}, [0] * len(calls)
    for q, r in enumerate(grouped):
        place[r] = q - first[calls[r]]
    analysed = _analyse(gm, of, compiled, sources, every, horizon)
    operands = [record.function.placement(record.gm, record.examples) for record in compiled]
    placeholders = [n for n in gm.graph.nodes if n.op == 'placeholder']
    # the block's reads, (consumer call j, its flat input leaf q, the producing leaf), one relation from _Export
    # `sources` (passed-through results resolved): block inputs' readers, deliveries and delivered operands take it
    reads = [(j, q, leaf) for j, read in enumerate(sources) for q, leaf in enumerate(read) if leaf is not None]
    readers = [[calls[j] for j, _, leaf in reads if leaf.call is None and leaf.leaf == i]
               for i in range(len(placeholders))]
    given = [(x.dtype, x.numel(), x.device.type, tuple(x.shape)) if isinstance(x, torch.Tensor) else None
             for x in example_inputs]
    constants = [None if isinstance(x, torch.Tensor) else x for x in example_inputs]
    context, state = _context(), {'stale': True, 'mapping': None, 'devices': (), 'executions': 0, 'horizon': 1,
                                  'inputs': [None] * len(given), 'timeout': None, 'replay': None, 'owned': {}}

    def make():
        deadline = time.clock_gettime_ns(time.CLOCK_MONOTONIC) + mesh.PLATFORM_NS
        if state['mapping'] is not None or state['owned']:
            _retire(context, state)
        nodes, header = mesh.Nodes(), mesh.Header.from_address(context.M)
        mesh.check(mesh.LIB.mesh_nodes_read(C.byref(context), C.byref(nodes)))
        page = header.pgsz * header.block
        irs = [partition(record.gm, operand, nodes, page) for record, operand in zip(compiled, operands)]
        # design/compilation-composition-change-request.md items 1 and 6 (unit 6E); design/aot-compilation-change-
        # request.md A4: a function's results are whole at every node of its map (_partition), so a consumer at a node
        # outside its producer's map reads the producer's delivery there (_deliver). wanted[f, o]: the functions of the
        # calls reading function f's flat result leaf o (_Export `sources`), the same at every node
        wanted = {}
        for j, _, leaf in reads:
            if leaf.call is not None:
                wanted.setdefault((calls[leaf.call], leaf.leaf), set()).add(calls[j])
        _deliver(irs, wanted, returned, _program())
        for ir, record in zip(irs, compiled):
            lower(ir)
            ir.weights = _weighed(ir, record.function)
            ir.delivered = {}
        # each consumer call's delivered operands here, before capture (the recorder's masked readers are emitted
        # there): ir.delivered[ordinal] = [(q, r, f, o, v, i, generation)], its flat input leaf q the leaf o of call r
        # (function f) landed in f's value v by f's transfer i (`irs[f].landed`), and the range's generation
        # (metal_recording.h struct MetalGeneration) {every, producer, ordinal} = {the consumer's calls of its
        # function a block, the producer's, call r's ordinal}: the consumer's execution t reads r's execution
        # (t div every) producer + ordinal (`every`, `place`)
        for j, q, leaf in reads:
            landed = irs[calls[leaf.call]].landed.get(leaf.leaf, (None, None)) if leaf.call is not None \
                else (None, None)
            irs[calls[j]].delivered.setdefault(place[j], [])
            if landed[1] is not None:
                irs[calls[j]].delivered[place[j]].append(
                    (q, leaf.call, calls[leaf.call], leaf.leaf, *landed,
                     (every[calls[j]], every[calls[leaf.call]], place[leaf.call])))
        holders = []
        for i, (x, functions) in enumerate(zip(given, readers)):
            k = next((k for k in functions if irs[k].device is not None), None)
            if x is not None and k is not None:
                irs[k].inputs[i] = (irs[k].value(*x[:3], 0, 0, 0, lag=1), x[3])
            holders.append(k if x is not None else None)
        # design/torch-capture-change-request.md item 9: "every node runs its captured program by replay, its
        # collectives included"; design/aot-compilation-change-request.md, appendix item 6
        # design/compilation-composition-change-request.md item 5
        try:
            # design/compilation-composition-change-request.md, "Complete replacement workgroup", 6's stages A to C:
            # each call's program declared, lowered and prepared, its schedule's lifetimes the plan's operands
            programs = _programs(context, irs, compiled, gm, of, constants, holders, state['owned'], calls)
            own = _lifetimes(programs, irs)
            # metal_recording.h place:step:; the later passes: each function's step word, its rows' plan execution
            # e = b every + ordinal, which the native program writes before row execution e; lag copies
            # (mesh-collective.c plan copies of a lagged value), horizon every of them, so each of the horizon block
            # executions in flight owns copy e mod copies; 8 bytes, e's 64 bits as the host's executions,
            # generations and count targets are
            steps = {}
            for k in dict.fromkeys(row['k'] for row in programs['rows'] if row['builder'] is not None):
                steps[k] = len(irs[k].values) + len(own[k])
                own[k].append(mesh.Value(section=mesh.Section(bytes=8), kind=mesh.EXECUTION, lag=1, run=1))
            arrays = [((mesh.Declared * len(ir.declared))(*ir.declared),
                       (mesh.Weight * len(ir.weights))(*ir.weights),
                       (mesh.Value * (len(ir.values) + len(mine)))(*ir.values, *mine),
                       (mesh.Dependence * len(ir.dependences))(*ir.dependences)) for ir, mine in zip(irs, own)]
            # design/torch-capture-change-request.md item 2 and its 2026-11-09 appendix: each function's t(v) at this
            # node a call, its device part (_work) and its host part, the host's writes of its block inputs and its
            # launches' encoding as a CPU actor; mesh_plan's Bff of their edges the horizon and host copies
            times = _times(programs, irs)
            functions = (mesh.Function * len(irs))(*[
                mesh.Function(C.cast(d, C.POINTER(mesh.Declared)), C.cast(w, C.POINTER(mesh.Weight)),
                              C.cast(v, C.POINTER(mesh.Value)), C.cast(e, C.POINTER(mesh.Dependence)), len(d), len(v),
                              len(e), 0, analysed.outstanding[k], 0, *times[k])
                for k, (d, w, v, e) in enumerate(arrays)])
            # design/compilation-composition-change-request.md item 6, unit 6D: the plan's results for native
            # preparation (mesh-collective.h struct mesh_planned), each call's plan, lead, ordinal and every the
            # identities the specialization and the program's rows take, kept until the program is retired
            planned = mesh.Planned()
            mesh.check(mesh.LIB.mesh_plan(C.byref(context), _group(), _program(), functions, len(irs),
                                          (mesh.U * len(calls))(*calls), len(calls), analysed.orders,
                                          analysed.causes, analysed.norders, analysed.chains, deadline,
                                          C.byref(planned)))
            state['owned'][id(planned)] = lambda: mesh.LIB.mesh_planned_free(C.byref(planned))
            state.update(mapping=_Mapping(mesh.View()), devices={layout[2] for ir in irs for layout in ir.layouts})
            # JGroups ViewId.java
            views = [mesh.State.from_address(context.M + f.plan) for f in functions]
            if any((s.serial, s.sequencer, s.boot) != (nodes.serial, nodes.sequencer, nodes.boot) for s in views):
                raise OSError(errno.ESTALE, 'a plan\'s view is not the one its IR was built on: plan again')
            listed = [values for _, _, values, _ in arrays]
            # 6's stage D: the plan's transport words into each row's program, then its final emission
            made = _finalized(programs, context, irs, planned, listed, state['owned'])
            view, addresses = _mapped(context, functions, len(irs), listed, page)
            state['mapping'].view, hull = view, _hull(listed, addresses, page)
            buffer = _placed_buffer(hull)
            try:
                bound = [_bind(ir, values, at, buffer, hull.address, page, state['mapping'])
                         for ir, values, at in zip(irs, listed, addresses)]
                state['inputs'] = [next((b[2][i] for b in bound if i in b[2]), None) for i in range(len(given))]
                state['horizon'] = planned.horizon
                outputs = _specialize(programs, planned.horizon, bound, gm, constants, state['inputs'], every)
                # 6's stage E: each row bound to its placed storage and replayed into the one native program
                state['replay'] = _replays(programs, context, irs, planned, planned.horizon, listed, addresses,
                                           hull.address, buffer, len(outputs), state['owned'], made, steps, every,
                                           times)
                state['replay'].update(outputs=outputs)
            finally:
                if buffer:
                    PUT(buffer, SELECTOR[b'release'])
            # torch 2.14.0 torch/export/exported_program.py, ExportedProgram; GSPMD section 4.5; Megatron-LM
            # core_r0.12.0 megatron/core/tensor_parallel/layers.py:156–165; design/problems-of-14b9a757-change-
            # request.md, "Sources for the second stops", 8, and "Sources for the third stops", 8;
            # design/torch-source-lowering-change-request.md item 1
            launched = False
            for ir, (places, _, _, _) in zip(irs, bound):
                loaded = ir.loads & places.keys()
                needed = mesh_graph.reach(loaded, lambda key: ir.attributes[key][1])
                host = mesh_graph.forward([key for key in ir.attributes if key in needed],
                                          lambda key: ir.attributes[key][1],
                                          lambda key, values: _hosted(ir, ir.attributes[key], values))
                for key in loaded:
                    places[key][0].copy_(host[key])
                    launched = launched or places[key][0].device.type == 'mps'
            if launched:
                torch.mps.synchronize()
        except BaseException:
            _retire(context, state)
            raise
        state.update(stale=False)

    def timed(work):
        if state['timeout'] is not None:
            raise OSError(errno.ETIMEDOUT, 'the program\'s run ended at its first ETIMEDOUT (R-7): %s'
                          % state['timeout'])
        try:
            return work()
        except OSError as failure:
            if failure.errno == errno.ETIMEDOUT:
                state['timeout'] = failure
            raise
    timed(make)

    def ended():
        return _ended(state['replay']) if state['replay'] is not None else 0

    def writable():
        if state['stale']:
            make()
        b = state['executions']
        _wait(lambda: ended() + state['horizon'] > b, ended, 'execution %d of the program: execution %d not ended, no '
              'end for MESH_PLATFORM_NS, so its inputs\' copy is not free (D.3.3)' % (b, b - state['horizon']))
        return tuple(None if placed is None else placed[1][b % len(placed[1])] for placed in state['inputs'])

    # design/compilation-composition-change-request.md item 5
    def execution():
        if state['stale']:
            make()
        b = state['executions']
        state['executions'] = b + 1
        try:
            return _replayed(context, state['replay'], b)
        except OSError as failure:
            if failure.errno in (errno.ESTALE, errno.ECANCELED):
                state['stale'] = True
            raise

    def inputs():
        return timed(writable)

    def run():
        return timed(execution)
    # MTL4CommandQueue.h signalEvent:value:
    run.inputs, run.ended = inputs, ended
    run.signature = lambda: state['replay']['signature'] if state['replay'] is not None else None
    weakref.finalize(run, _dropped, context, state)
    return run
