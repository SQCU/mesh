# design/aot-compilation-change-request.md A1 to A4, A8 to A10 and appendix item 7;
# design/plan-to-bridge-change-request.md R1; Y. Xu et al., GSPMD, arXiv 2105.04663, 2021
import collections
import ctypes as C
import errno
import functools
import hashlib
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
ELEMENTWISE = {A.add, A.sub, A.mul, A.div, A.neg, A.relu, A.tanh, A.sigmoid, A.exp, A.gelu, A.silu}
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
# design/aot-compilation-change-request.md A10, "Its output"; M. C. Rinard and P. C. Diniz, ACM TOPLAS 19(6), 1997
Calls = collections.namedtuple('Calls', 'mine outstanding orders commuting')
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
        self.made, self.calls, self.locals, self.shapes = set(), None, {}, {}
        self.metas, self.forms, self.over, self.producer, self.places = {}, {}, {}, {}, {}
        self.homes, self.at, self.inside, self.added, self.owned = {}, {}, {}, {}, {}
        self.publishes, self.inputs, self.declares, self.waits = {}, {}, {}, {}
        self.attributes, self.loads, self.kinds, self.page, self.tasks = {}, set(), {}, None, 0
        self.taken, self.results, self.sent, self.contiguous = {}, {}, set(), set()

    # GSPMD section 3.1
    def local(self, node, sharding):
        return tuple(s // self.shape[m] if m >= 0 else s for s, m in zip(_value(node).shape, sharding))

    # mesh-collective.h struct mesh_declared; design/plan-to-bridge-change-request.md R2 and R4;
    # design/aot-compilation-change-request.md, appendix item 5
    def transfer(self, source, destinations, kind, part, landings):
        self.identity += 1
        transfers = {}
        for d in destinations:
            transfers[d] = len(self.declared)
            self.declared.append(mesh.Declared(self.map[source], self.map[d], self.identity, kind, mesh.ABSENT, 0,
                                               part, landings[d]))
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
# design/aot-compilation-change-request.md, appendix item 7
def _reshard(ir, node, have, want):
    def step(item):
        if item[2] not in ir.made:
            ir.made.add(item[2])
            ir.program.append(item)
    if node.op in ('placeholder', 'get_attr'):
        have = (-1,) * len(have)
    for m in range(len(ir.shape)):
        if m in have and m not in want:
            after, d, shape = tuple(-1 if s == m else s for s in have), have.index(m), ir.local(node, have)
            own = (node, after, 'own')
            ir.shapes[own] = shape
            ir.inside[own] = ((node, after), d, {e: ir.coords[e][m] * shape[d] for e in ir.coords})
            step(('copy', ((node, have),), own, None))
            step(('collective', (own,), (node, after), Collective('all-gather', node, (m,), d, len(ir.program))))
            have = after
    for m in range(len(ir.shape)):
        if m in have and have.index(m) != want.index(m):
            i, j = have.index(m), want.index(m)
            if have[j] >= 0:
                raise NotImplementedError('%s: an AllToAll along mesh dimension %d into dimension %d, which mesh '
                                          'dimension %d holds: refused (GSPMD section 4.5)'
                                          % (node.name, m, j, have[j]))
            after, shape = tuple(-1 if k == i else m if k == j else s for k, s in enumerate(have)), ir.local(node, have)
            own = (node, after, 'own')
            ir.shapes[own] = tuple(n // ir.shape[m] if k == j else n for k, n in enumerate(shape))
            ir.inside[own] = ((node, after), i, {e: ir.coords[e][m] * shape[i] for e in ir.coords})
            step(('copy', ((node, have),), own, (j, m)))
            step(('collective', ((node, have),), (node, after),
                  Collective('all-to-all', node, (m,), (i, j), len(ir.program))))
            have = after
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
def partition(gm, placement, nodes):
    if not isinstance(placement, Placement):
        raise TypeError('the placement function returned %r, not a Placement (A2)' % (placement,))
    ir, operators, annotated = IR(gm, placement, nodes), {}, set()
    for node in ir.order:
        if node.op == 'call_function':
            ir.locals[node] = Local(node.op, node.target, node.args, node.kwargs)
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
        return items[key][0] == 'input' and _value(key[0]).is_contiguous()
    shape = _layout(ir, place.owner)[1]
    return place.offset == 0 and place.sizes == shape and place.strides == _contiguous(shape)


def _paged(ir, places, items, key, reads, local):
    packet = getattr(local.target, 'overloadpacket', None)
    if packet not in POINTWISE and packet is not A.copy and packet not in COPIES or not reads \
            or not all(isinstance(k[0].meta.get('val'), torch.Tensor) for k in (key, *reads)):
        return False
    dtype, shape = _layout(ir, key)
    for operand, size in (_layout(ir, k) for k in reads):
        if operand.itemsize != dtype.itemsize or (math.prod(size) != math.prod(shape) if packet in COPIES
                                                  else size != shape):
            return False
    return places.get(key) is not None and places[key].owner == key \
        and all(_whole(ir, places, items, k) for k in (key, *reads))


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
    def read(g):
        return [owner for k in ir.program[g][1] for owner in ir.over.get(k, ())] if isinstance(g, int) else []
    joined = [g for g, item in enumerate(ir.program) if ir.kinds.get(item[2]) != mesh.PAGES]
    return mesh.PAGES if roots & mesh_graph.reach(range(len(ir.program)), read) \
        and not roots & mesh_graph.reach(joined, read) else mesh.JOIN


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
        parts = [part[0] for _, part in ir.roles[operand][1] if part is not None] if kind == 'collective' else []
        return [owner for k in reads for owner in ir.over.get(k, ())] \
            + [places[k].owner for k in written + parts if k is not None and places.get(k) is not None]
    synchronous = [g for g, (kind, _, key, operand) in enumerate(ir.program)
                   if not (ir.kinds.get(key) == mesh.PAGES or kind in ('input', 'constant', 'slice', 'narrow')
                           or kind == 'window' and key not in ir.inside
                           or kind == 'collective' and _local(ir, operand) is None)]
    run = mesh_graph.reach(synchronous, accessed) | set(ir.loads)
    for key, v in ir.owned.items():
        ir.values[v].run = int(key in run)


# design/aot-compilation-change-request.md A3; O. Sinnen, Task Scheduling for Parallel Systems, Wiley, 2007;
# metal_recording.h tasks: and weigh:lengths:
Weights = collections.namedtuple('Weights', 'tasks transfers')


def unit(ir):
    return Weights([1.0] * ir.tasks, [mesh.Weight(1.0, 1.0, 1.0, 1.0) for _ in ir.declared])


def _weighed(ir, function):
    made = function.weights(ir)
    if not isinstance(made, Weights) or len(made.tasks) != ir.tasks or len(made.transfers) != len(ir.declared) \
            or not all(isinstance(w, (int, float)) for w in made.tasks) \
            or not all(isinstance(w, mesh.Weight) for w in made.transfers):
        raise ValueError('the weighting function gave %r for %d tasks and %d declared transfers: A3\'s operand is "a '
                         'weight on each task; a weight on each edge", a Weights of one number a task and one mesh.Weight '
                         'a declared transfer' % (type(made).__name__, ir.tasks, len(ir.declared)))
    return made


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


def _arrived(context, plan, transfer, t, count):
    def ready():
        status = mesh.LIB.mesh_host_arrived(C.byref(context), plan, transfer, t)
        if status == mesh.FAILED:
            raise OSError(errno.ECANCELED, 'execution %d of declared transfer %d: a field ahead, or the range cancelled '
                                           '(mesh-collective.h mesh_host_arrived)' % (t, transfer))
        return status == 1
    _wait(ready, lambda: count.value, 'execution %d of declared transfer %d: no landing for MESH_PLATFORM_NS'
          % (t, transfer))


# design/aot-compilation-change-request.md, appendix item 7, "A publish waits for nothing"; rdma/mesh-torch.mm
# mesh_torch_publish
TORCH = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(mesh.__file__)), 'libmesh-torch.dylib'))
TORCH.mesh_torch_publish.restype, TORCH.mesh_torch_publish.argtypes = C.c_int, [C.c_void_p, C.c_void_p, mesh.Q,
                                                                                C.POINTER(mesh.U), mesh.U, mesh.Q]
TORCH.mesh_torch_notifying.restype, TORCH.mesh_torch_notifying.argtypes = mesh.Q, []
PUBLISH = C.cast(mesh.LIB.mesh_host_publish, C.c_void_p)
STARTED = C.cast(mesh.LIB.mesh_host_started, C.c_void_p)
# rdma/mesh-torch.mm mesh_torch_retire, mesh_torch_end and mesh_torch_ended
TORCH.mesh_torch_retire.restype, TORCH.mesh_torch_retire.argtypes = C.c_int, [C.c_void_p, C.c_void_p, mesh.Q]
TORCH.mesh_torch_end.restype, TORCH.mesh_torch_end.argtypes = C.c_int, [mesh.Q]
TORCH.mesh_torch_ended.restype, TORCH.mesh_torch_ended.argtypes = mesh.Q, []
RETIRE = C.cast(mesh.LIB.mesh_retire, C.c_void_p)


def _publishes(context, plan, devices, transfers):
    metal = [i for i in transfers if devices[i] == 'mps']
    steps = [lambda env, inputs, t, i=i: mesh.LIB.mesh_host_publish(C.byref(context), plan, i, t)
             for i in transfers if devices[i] != 'mps']
    if metal:
        array = (mesh.U * len(metal))(*metal)
        steps.append(lambda env, inputs, t: mesh.check(TORCH.mesh_torch_publish(PUBLISH, C.addressof(context), plan,
                                                                                array, len(array), t)))
    return steps


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
NEW_BUFFER = _message(C.c_void_p, C.c_ulong, C.c_ulong)
STRING = _message(C.c_void_p, C.c_char_p)
SET = _message(None, C.c_void_p)
OBJECT = _message(C.c_void_p)
VALUE = _message(C.c_uint64)
LEVELS = _message(C.c_uint32, C.c_void_p)
VALUES = _message(C.c_ulong, C.c_void_p, C.c_uint32, C.c_void_p)
PLACE_OPERAND = _message(None, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_ulong, C.c_ulong, C.c_bool)
PLACE_VALUE = _message(None, C.c_void_p, C.c_ulong, C.c_void_p, C.c_void_p, C.c_ulong, C.c_bool)
SIGNAL = _message(None, C.c_void_p, C.c_uint64)
WAIT = _message(C.c_bool, C.c_uint64, C.c_uint64)


# metal_recording.h struct MetalReplay, struct MetalCall and struct MetalValue
class Replay(C.Structure):
    _fields_ = [('commands', C.c_void_p), ('residency', C.c_void_p), ('owner', C.c_void_p), ('count', mesh.U),
                ('buffers', C.c_uint16), ('memory', C.c_uint16)]


class Call(C.Structure):
    _fields_ = [('replay', C.POINTER(Replay)), ('plan', mesh.Q), ('passes', mesh.Q), ('landed', C.c_void_p),
                ('lead', mesh.Q)]


class MetalValue(C.Structure):
    _fields_ = [('bytes', C.c_ulong), ('kind', mesh.U), ('state', mesh.U), ('begin', mesh.U), ('end', mesh.U)]


ENCODE = C.CFUNCTYPE(None, C.c_void_p)
START = C.cast(mesh.LIB.mesh_transfers_start, C.c_void_p)


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
        lib.MetalRecorderCapture.restype = C.c_void_p
        lib.MetalRecorderCapture.argtypes = [C.c_void_p, ENCODE, C.c_void_p, C.c_char_p, C.c_size_t]
        lib.MetalRecorderReplay.restype = C.c_int
        lib.MetalRecorderReplay.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(Replay), C.c_char_p, C.c_size_t]
        lib.MetalRecorderOperand.restype = C.c_int
        lib.MetalRecorderOperand.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(mesh.U), C.POINTER(mesh.U)]
        lib.MetalRecorderTasks.restype, lib.MetalRecorderTasks.argtypes = mesh.U, [C.c_void_p, C.c_void_p]
        lib.MetalRecorderWeigh.restype = C.c_int
        lib.MetalRecorderWeigh.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(C.c_double), mesh.U, C.c_char_p,
                                           C.c_size_t]
        lib.MetalRecorderRun.restype = C.c_int
        lib.MetalRecorderRun.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), mesh.U, C.POINTER(Call), mesh.U, mesh.Q,
                                         C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_char_p, C.c_size_t]
        lib.MetalReplayDestroy.restype, lib.MetalReplayDestroy.argtypes = None, [C.POINTER(Replay)]
        metal.MTLCreateSystemDefaultDevice.restype = C.c_void_p
        RECORDER.append((lib, metal.MTLCreateSystemDefaultDevice))
    return RECORDER[0]


# design/joint-compilation.md §3, row 11
ENGINE = {}


def _engine(self, mat2, out):
    m, k, n = self.shape[0], self.shape[1], mat2.shape[1]
    key = (m, k, n, self.stride(0), mat2.stride(1), out.stride(0), out.stride(1))
    if key not in ENGINE:
        ENGINE[key] = torch.mps.compile_shader(
            '#define MMB_PREFILL_MPP 1\n#define MMB_TENSOR_REDUCTION_TILE 0\n#define MMB_TENSOR_BARRIER 0\n'
            '#define MMB_MATRIX_OUTPUT_TRANSPOSED %d\n#define MMB_MATRIX_SHAPE uint4(%du, %du, %du, %du)\n'
            '#define MMB_MATRIX_INPUTS uint4(%du, 0u, 1u, %du)\n#define MMB_MATRIX_OUTPUT uint2(%du, %du)\n'
            % (out.stride(1) != 1, m, k, n, m, self.stride(0), mat2.stride(1), out.stride(0), out.stride(1))
            + ENGINE['source'])
    ENGINE[key].dense_tensor_f16(self, mat2, out, out, threads=[(m + 63) // 64 * 128, (n + 63) // 64, 1],
                                 group_size=[128, 1, 1])
    return out


# design/torch-source-lowering-change-request.md item 1; metal-microbench operator_shaders.metal; torch 2.14.0
# torch/mps/__init__.py, compile_shader
OPERATORS = {}
ELEMENTS = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2, torch.int64: 3, torch.int32: 4, torch.int16: 5,
            torch.int8: 6, torch.uint8: 7, torch.bool: 8}
FLOATING = (torch.float32, torch.float16, torch.bfloat16)
INTEGRAL = (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool)


def _library():
    if 'library' not in OPERATORS:
        root = _repository()
        with open(os.path.join(root, 'matrix_arithmetic.metal')) as arithmetic, \
                open(os.path.join(root, 'matrix_shaders.metal')) as shaders, \
                open(os.path.join(root, 'operator_shaders.metal')) as operators:
            ENGINE['source'] = arithmetic.read() + '\n' + shaders.read()
            OPERATORS['library'] = torch.mps.compile_shader(operators.read())
    return OPERATORS['library']


# design/torch-source-lowering-change-request.md item 1; metal-microbench operator_shaders.metal; Metal Shading Language
# Specification 4.1, §1.5 and §1.6.2; torch 2.14.0 torch/mps/__init__.py, compile_shader
COMPILE = ('c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-Wno-unknown-attributes', '-x', 'c++',
           '-dynamiclib')


def _cpu():
    if 'cpu' not in OPERATORS:
        root = _repository()
        with open(os.path.join(root, 'operator_shaders.metal'), 'rb') as source:
            text = source.read()
        digest = hashlib.sha256(b'\0'.join((text,) + tuple(word.encode() for word in COMPILE))).hexdigest()
        built = os.path.join(root, '.build', 'operator_shaders-%s.dylib' % digest[:16])
        if not os.path.exists(built):
            os.makedirs(os.path.dirname(built), exist_ok=True)
            made = '%s.%d' % (built, os.getpid())
            done = subprocess.run(COMPILE + ('-', '-o', made), input=text, capture_output=True)
            if done.returncode:
                raise OSError(errno.ENOEXEC, 'the CPU build of %s: %s' % (source.name, done.stderr.decode()[-4000:]))
            os.replace(made, built)
        OPERATORS['cpu'] = C.CDLL(built)
    return OPERATORS['cpu']


# metal-microbench operator_shaders.metal Grid; Python documentation, ctypes, Arrays
def _grid(name, count, arguments):
    if ('cpu', name) not in OPERATORS:
        entry = getattr(_cpu(), name + '_grid')
        entry.restype, entry.argtypes = None, [C.c_void_p, C.c_uint]
        OPERATORS['cpu', name] = entry
    entry = OPERATORS['cpu', name]
    held = [value if isinstance(value, torch.Tensor) else
            ((C.c_float if any(isinstance(x, float) for x in value) else C.c_int64) * len(value))(*value)
            for value in arguments]
    pointers = (C.c_void_p * len(held))(*[value.data_ptr() if isinstance(value, torch.Tensor) else C.addressof(value)
                                          for value in held])
    return lambda held=held: entry(pointers, count)


# MTLComputeCommandEncoder.h dispatchThreads:threadsPerThreadgroup:; design/torch-source-lowering-change-request.md
# item 1
def _dispatch(name, count, *arguments):
    devices = {value.device.type for value in arguments if isinstance(value, torch.Tensor)}
    if count >= 1 << 32:
        raise NotImplementedError('%s over %d elements: past a grid\'s 32-bit thread position' % (name, count))
    if not count or devices == {'meta'}:
        return []
    if devices == {'cpu'}:
        return [_grid(name, count, arguments)]
    if devices != {'mps'}:
        raise NotImplementedError('%s over tensors of the %s: one task\'s operands are on the one backend it runs on'
                                  % (name, ' and the '.join(sorted(devices))))
    if name not in OPERATORS:
        OPERATORS[name] = getattr(_library(), name)
    kernel = OPERATORS[name]
    group = min(256, kernel.max_threads_per_threadgroup, count)
    return [functools.partial(kernel, *arguments, threads=count, group_size=group)]


def _each(thunks):
    for thunk in thunks:
        thunk()


def _typed(target, operands, outs, results=FLOATING):
    found = [value.dtype for value in operands if isinstance(value, torch.Tensor) and value.dtype not in ELEMENTS] \
        + [value.dtype for value in outs if value.dtype not in results]
    if found:
        raise NotImplementedError('%s of %s: no source computes it (design/torch-source-lowering-change-request.md item '
                                  '1)' % (target, found[0]))


# torch 2.14.0 torch/_C/__init__.pyi, FunctionSchema `arguments`, Argument `default_value`
def _operands(target, args, kwargs):
    schema = target._schema.arguments
    return {**{a.name: a.default_value for a in schema if a.has_default_value()},
            **dict(zip((a.name for a in schema), args)), **kwargs}


# PyTorch 2.14 documentation, Broadcasting semantics
def _broadcast(value, out):
    return list(value.expand(out.shape).stride()) if isinstance(value, torch.Tensor) else [0] * out.dim()


def _scalar(value, integral):
    return (int(value) if isinstance(value, (bool, int)) else 0) if integral else float(value)


def _pointwise(op, out, a, b=None, c=None, alpha=1):
    operands = [a, b, c]
    tensor = next(x for x in operands if isinstance(x, torch.Tensor))
    integral = out.dtype in INTEGRAL and all(x.dtype in INTEGRAL if isinstance(x, torch.Tensor) else isinstance(x, int)
                                            for x in operands if x is not None)
    g = [ELEMENTS[out.dtype], *[ELEMENTS[x.dtype] if isinstance(x, torch.Tensor) else -2 if x is None else -1
                                for x in operands], op, int(integral), out.dim(), *out.shape, *out.stride(),
         *[s for x in operands for s in _broadcast(x, out)]]
    plain = [x for x in operands if x is not None and not isinstance(x, torch.Tensor)]
    floats = [0.0 if x is None or isinstance(x, torch.Tensor) else float(x) for x in operands] + [float(alpha)]
    longs = [0 if x is None or isinstance(x, torch.Tensor) else _scalar(x, True) for x in operands] + [_scalar(alpha, True)]
    if integral and any(not isinstance(x, int) for x in plain + [alpha]):
        raise NotImplementedError('elementwise %d into %s of a scalar no integer holds' % (op, out.dtype))
    return _dispatch('elementwise', out.numel(), out, *[x if isinstance(x, torch.Tensor) else tensor for x in operands],
                     g, floats, longs)


# metal-microbench operator_shaders.metal pointwise
POINTWISE = {A.add: 0, A.sub: 1, A.mul: 2, A.div: 3, A.floor_divide: 4, A.neg: 6, A.relu: 7, A.tanh: 8, A.sigmoid: 9,
             A.exp: 10, A.gelu: 11, A.silu: 13, A.eq: 14, A.ne: 15, A.lt: 16, A.le: 17, A.gt: 18, A.ge: 19, A.pow: 20,
             A.rsqrt: 21, A.sqrt: 22, A.log: 23, A.abs: 24, A.clamp: 25, A.clamp_min: 25, A.clamp_max: 25, A.where: 26}


def _arithmetic(target, given, outs):
    packet, op = target.overloadpacket, POINTWISE[target.overloadpacket]
    if packet is A.div and given.get('rounding_mode') is not None:
        op = {'floor': 4, 'trunc': 5}[given['rounding_mode']]
    if packet is A.gelu and given['approximate'] != 'none':
        op = 12
    operands = [given['condition'], given['self'], given['other']] if packet is A.where else \
        [given['self'], given.get('min'), given.get('max')] if op == 25 else \
        [given['self'], given['exponent'], None] if packet is A.pow else [given['self'], given.get('other'), None]
    _typed(target, operands, outs, ELEMENTS)
    return _pointwise(op, outs[0], *operands, given.get('alpha', 1))


# torch 2.14.0 torch/_tensor_docs.py, copy_ and view
def _copy(out, value):
    _typed('a copy', [value], [out], ELEMENTS)
    g = [ELEMENTS[out.dtype], ELEMENTS[value.dtype], out.dim(), *out.shape, *out.stride(), value.dim(), *value.shape,
         *value.stride()]
    return _dispatch('copy', out.numel(), out, value, g)


def _assign(out, value):
    return _copy(out, value.expand(out.shape))


# torch 2.14.0 torch/_tensor_docs.py, fill_; PyTorch 2.14 documentation, torch.arange
def _fill(out, value, step=0):
    _typed('a fill', [], [out], ELEMENTS)
    integral = out.dtype in INTEGRAL
    return _dispatch('fill', out.numel(), out, [ELEMENTS[out.dtype], out.dim(), *out.shape, *out.stride()],
                     [float(value), float(step)], [int(value) if integral else 0, int(step) if integral else 0])


def _copying(target, given, outs):
    return _assign(outs[0], given['src']) if target.overloadpacket is A.copy else _copy(outs[0], given['self'])


# PyTorch 2.14 documentation, torch.einsum
def _contract(out, operands, labels, output, addend=None, alpha=1, beta=0):
    if len(operands) > 4:
        raise NotImplementedError('a contraction of %d operands: the source takes at most four (metal-microbench '
                                  'operator_shaders.metal contraction)' % len(operands))
    sizes = dict(zip(output, out.shape))
    for value, own in zip(operands, labels):
        for size, label in zip(value.shape, own):
            sizes[label] = max(sizes.get(label, 1), size)
    summed = [label for label in dict.fromkeys(label for own in labels for label in own) if label not in output]

    def strides(value, own, letters):
        return [sum(s for n, s, label in zip(value.shape, value.stride(), own) if label == letter and n != 1)
                for letter in letters]
    g = [ELEMENTS[out.dtype], ELEMENTS[addend.dtype] if addend is not None else -1, len(operands),
         *[ELEMENTS[value.dtype] for value in operands], *[-1] * (4 - len(operands)), len(output), len(summed),
         *[sizes[label] for label in output], *[sizes[label] for label in summed], *out.stride(),
         *_broadcast(addend, out)]
    for value, own in zip(operands, labels):
        g += strides(value, own, output) + strides(value, own, summed)
    buffers = [*operands, *[operands[0]] * (4 - len(operands)), addend if addend is not None else operands[0]]
    return _dispatch('contraction', out.numel(), out, *buffers, g, [float(alpha), float(beta)])


# design/torch-source-lowering-change-request.md item 1
def _engined(self, mat2, out):
    return self.dtype == mat2.dtype == out.dtype == torch.float16 and self.numel() and mat2.numel() \
        and self.stride(1) == 1 and mat2.stride(0) == 1 and out.device.type == 'mps'


def _matrix(out, a, b, bias=None):
    if _engined(a, b, out):
        _library()
        return [functools.partial(_engine, a, b, out)] + (_pointwise(0, out, out, bias) if bias is not None else [])
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
        if alpha == 1 and beta == 1:
            return _matrix(out, a, b, c)
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


# PyTorch 2.14 documentation, torch.einsum
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
    if len(operands) <= 4:
        return _contract(out, operands, labels, output)
    sizes = {}
    for value, own in zip(operands, labels):
        for size, label in zip(value.shape, own):
            sizes[label] = max(sizes.get(label, 1), size)
    thunks, made, own = [], operands[0], labels[0]
    for j in range(1, len(operands)):
        later = {label for others in labels[j + 1:] for label in others} | set(output)
        kept = output if j == len(operands) - 1 else tuple(dict.fromkeys(l for l in own + labels[j] if l in later))
        step = out if j == len(operands) - 1 else \
            torch.empty([sizes[label] for label in kept], dtype=out.dtype, device=out.device)
        thunks += _contract(step, [made, operands[j]], [own, labels[j]], kept)
        made, own = step, kept
    return thunks


# PyTorch 2.14 documentation, torch.nn.Conv1d to torch.nn.Conv3d and torch.nn.ConvTranspose1d to
# torch.nn.ConvTranspose3d; PyTorch v2.11.0 aten/src/ATen/native/Convolution.cpp, convolution_same
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

    def spatial(values, fill):
        return [fill] * (3 - k) + list(values)

    def strides(value):
        return list(value.stride()[:2]) + spatial(value.stride()[2:], 0)
    g = [ELEMENTS[out.dtype], ELEMENTS[x.dtype], ELEMENTS[w.dtype], ELEMENTS[bias.dtype] if bias is not None else -1,
         int(transposed), given['groups'], x.shape[0], out.shape[1], x.shape[1], *spatial(out.shape[2:], 1),
         *spatial(x.shape[2:], 1), *spatial(w.shape[2:], 1), *spatial(stride, 1), *spatial(lows, 0),
         *spatial(dilation, 1), *strides(out), *strides(x), *strides(w), bias.stride(0) if bias is not None else 0]
    return _dispatch('convolution', out.numel(), out, x, w, bias if bias is not None else w, g)


# PyTorch v2.11.0 aten/src/ATen/native/cpu/MaxPoolKernel.cpp, cpu_max_pool; aten/src/ATen/native/cpu/AvgPoolKernel.cpp,
# cpu_avg_pool2d and cpu_avg_pool3d
INDEXED = {A.max_pool2d_with_indices: 2, A.max_pool3d_with_indices: 3}


def _pooled(target, given, outs):
    packet, x = target.overloadpacket, given['self']
    k, out, index = {**MAXIMUM, **AVERAGE, **INDEXED}[packet], outs[0], outs[1] if len(outs) > 1 else None
    _typed(target, [x], outs[:1])
    lows, dilation, lead = _ints(given['padding'], k), _ints(given.get('dilation', 1), k), x.dim() - k

    def spatial(values, fill):
        return [fill] * (3 - k) + list(values)

    def strides(value):
        return [0] * (2 - lead) + list(value.stride()[:lead]) + spatial(value.stride()[lead:], 0)
    g = [ELEMENTS[out.dtype], ELEMENTS[x.dtype], int(packet not in AVERAGE), int(index is not None),
         int(given.get('count_include_pad', True)), given.get('divisor_override') or 0, lead,
         *[1] * (2 - lead), *x.shape[:lead], *spatial(out.shape[lead:], 1), *spatial(x.shape[lead:], 1),
         *spatial(_ints(given['kernel_size'], k), 1), *spatial(_ints(given['stride'] or given['kernel_size'], k), 1),
         *spatial(lows, 0), *spatial(dilation, 1), *strides(out), *strides(x),
         *(strides(index) if index is not None else [0] * 5)]
    return _dispatch('pool', out.numel(), out, index if index is not None else out, x, g)


# PyTorch v2.11.0 aten/src/ATen/native/cpu/PaddingKernel.cpp, ReflectionPad and ReplicationPad;
# aten/src/ATen/native/PadNd.cpp, constant_pad_nd and _pad_circular_symint
MODES = {'constant': 0, 'reflect': 1, 'replicate': 2, 'circular': 3}


def _padded(target, given, outs):
    (out,), x = outs, given['self']
    _typed(target, [x], outs)
    pad, n = list(given['pad']), x.dim()
    low, high, mode = [0] * n, [0] * n, [0] * n
    for j in range(len(pad) // 2):
        low[n - 1 - j], high[n - 1 - j], mode[n - 1 - j] = pad[2 * j], pad[2 * j + 1], MODES[given.get('mode', 'constant')]
    extent = [max(size - max(-a, 0) - max(-b, 0), 1) for size, a, b in zip(x.shape, low, high)]
    g = [ELEMENTS[out.dtype], ELEMENTS[x.dtype], n, *out.shape, *out.stride(), *x.shape, *x.stride(), *low, *mode,
         *[max(-a, 0) for a in low], *extent]
    return _dispatch('pad', out.numel(), out, x, g, [float(given.get('value') or 0)])


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
    return _assign(out, value) if isinstance(value, torch.Tensor) else _fill(out, value)


# PyTorch 2.14 documentation, torch.sum, torch.mean, torch.amax and torch.amin
REDUCTIONS = {A.sum: 0, A.mean: 1, A.amax: 2, A.amin: 3}


def _reduced(target, given, outs):
    (out,), x = outs, given['self']
    _typed(target, [x], outs, ELEMENTS)
    dims = given.get('dim')
    dims = sorted({d % max(x.dim(), 1) for d in _ints(dims, 1)}) if dims not in (None, []) else list(range(x.dim()))
    free = [d for d in range(x.dim()) if d not in dims]
    strides = [out.stride(d) for d in free] if given.get('keepdim') else list(out.stride())
    g = [ELEMENTS[out.dtype], ELEMENTS[x.dtype], REDUCTIONS[target.overloadpacket], int(out.dtype in INTEGRAL), len(free),
         len(dims), *[x.shape[d] for d in free], *strides, *[x.stride(d) for d in free], *[x.shape[d] for d in dims],
         *[x.stride(d) for d in dims]]
    return _dispatch('reduce', out.numel(), out, x, g)


def _lines(x, out, d):
    other = [j for j in range(x.dim()) if j != d]
    return [x.shape[d] if x.dim() else 1, out.stride(d) if x.dim() else 0, x.stride(d) if x.dim() else 0, len(other),
            *[x.shape[j] for j in other], *[out.stride(j) for j in other], *[x.stride(j) for j in other]]


# PyTorch 2.14 documentation, torch.nn.Softmax and torch.nn.LogSoftmax
SOFTMAX = {A.softmax: 0, A._softmax: 0, A.log_softmax: 1, A._log_softmax: 1}


def _softened(target, given, outs):
    (out,), x = outs, given['self']
    _typed(target, [x], outs)
    lines = _lines(x, out, given['dim'] % max(x.dim(), 1))
    return _dispatch('softmax', x.numel() // max(lines[0], 1), out, x,
                     [ELEMENTS[out.dtype], ELEMENTS[x.dtype], SOFTMAX[target.overloadpacket], *lines])


# PyTorch 2.14 documentation, torch.cumsum
def _scanned(target, given, outs):
    (out,), x = outs, given['self']
    _typed(target, [x], outs, ELEMENTS)
    lines = _lines(x, out, given['dim'] % max(x.dim(), 1))
    return _dispatch('scan', x.numel() // max(lines[0], 1), out, x,
                     [ELEMENTS[out.dtype], ELEMENTS[x.dtype], int(out.dtype in INTEGRAL), *lines])


# PyTorch 2.14 documentation, torch.sort and torch.topk
def _ranked(target, given, outs):
    (values, indices), x = outs, given['self']
    _typed(target, [x], [values], ELEMENTS)
    d = given['dim'] % max(x.dim(), 1)
    other = [j for j in range(x.dim()) if j != d]
    length = x.shape[d] if x.dim() else 1
    k, descending = (given['k'], given['largest']) if target.overloadpacket is A.topk else (length, given['descending'])
    g = [ELEMENTS[values.dtype], ELEMENTS[x.dtype], int(x.dtype in INTEGRAL), int(descending), k, length,
         *[t.stride(d) if x.dim() else 0 for t in (x, values, indices)], len(other), *[x.shape[j] for j in other],
         *[t.stride(j) for t in (x, values, indices) for j in other]]
    return _dispatch('rank', x.numel(), values, indices, x, g)


def _flat(value):
    try:
        return value.view(-1)
    except RuntimeError:
        raise NotImplementedError('a normalization\'s statistics or weight of strides %s: the source reads them as one '
                                  'run' % (value.stride(),))


# PyTorch 2.14 documentation, torch.nn.LayerNorm, torch.nn.GroupNorm and torch.nn.BatchNorm2d; torch 2.14.0
# torch/_refs/__init__.py, native_layer_norm, native_group_norm and _normalize; torch/_decomp/decompositions.py,
# native_batch_norm_helper
NORMS = (A.layer_norm, A.native_layer_norm, A.group_norm, A.native_group_norm, A.batch_norm,
         A._native_batch_norm_legit_no_training)


def _normalized(target, given, outs):
    packet, x, out = target.overloadpacket, given['input'], outs[0]
    weight, bias = given.get('weight'), given.get('bias')
    _typed(target, [x, weight, bias, given.get('running_mean'), given.get('running_var')], outs[:1])
    if packet in (A.layer_norm, A.native_layer_norm):
        inner = math.prod(given['normalized_shape'])
        groups, layout = x.numel() // max(inner, 1), (inner, x.numel() // max(inner, 1), 1, inner)
        step = inner
    elif packet in (A.group_norm, A.native_group_norm):
        n, c, count = x.shape[0], x.shape[1], given['num_groups'] if packet is A.group_norm else given['group']
        spatial = x.numel() // max(n * c, 1)
        groups, step = n * count, c // count * spatial
        layout = (step, groups, spatial, c)
    else:
        if given.get('training'):
            raise NotImplementedError('%s in training: its running statistics are updated in place, which no '
                                      'source here writes' % (target,))
        n, c = x.shape[0], x.shape[1]
        spatial = x.numel() // max(n * c, 1)
        groups, step, layout = c, spatial, (spatial, c, spatial, c)
    made = [_flat(o) for o in outs[1:]] if len(outs) == 3 and outs[1].numel() == groups else \
        [torch.empty(groups, dtype=torch.float32, device=x.device) for _ in range(2)]
    mean, rstd = made
    known = packet in (A.batch_norm, A._native_batch_norm_legit_no_training)
    rm, rv = (_flat(given['running_mean']), _flat(given['running_var'])) if known else (mean, rstd)
    first = [ELEMENTS[mean.dtype], ELEMENTS[rstd.dtype], ELEMENTS[x.dtype], int(known), ELEMENTS[rm.dtype],
             ELEMENTS[rv.dtype], mean.stride(0), rstd.stride(0), rm.stride(0), rv.stride(0), 1, 0, step, step, x.dim(),
             *x.shape, *x.stride()]
    w, b = _flat(weight) if weight is not None else None, _flat(bias) if bias is not None else None
    second = [ELEMENTS[out.dtype], ELEMENTS[x.dtype], ELEMENTS[mean.dtype], ELEMENTS[rstd.dtype],
              ELEMENTS[w.dtype] if w is not None else -1, ELEMENTS[b.dtype] if b is not None else -1, *layout,
              mean.stride(0), rstd.stride(0), w.stride(0) if w is not None else 0, b.stride(0) if b is not None else 0,
              x.dim(), *x.shape, *out.stride(), *x.stride()]
    return _dispatch('moments', groups, mean, rstd, x, rm, rv, first, [float(given['eps'])]) + \
        _dispatch('normalize', x.numel(), out, x, mean, rstd, w if w is not None else x, b if b is not None else x,
                  second)


# PyTorch 2.14 documentation, torch.gather, torch.index_select, torch.nn.functional.embedding and Tensor Indexing API;
# NumPy User Guide, Indexing on ndarrays, Advanced indexing
def _take(out, x, dims, indices):
    if len(indices) > 4:
        raise NotImplementedError('an index of %d tensors: the source takes at most four (metal-microbench '
                                  'operator_shaders.metal take)' % len(indices))
    g = [ELEMENTS[out.dtype], ELEMENTS[x.dtype], out.dim(), x.dim(), *[ELEMENTS[t.dtype] for t, _ in indices],
         *[-1] * (4 - len(indices)), *out.shape, *out.stride()]
    for d, (mode, source) in enumerate(dims):
        g += [mode, source, x.stride(d), x.shape[d]]
    for _, strides in indices:
        g += list(strides)
    return _dispatch('take', out.numel(), out, x, *[t for t, _ in indices], *[x] * (4 - len(indices)), g)


def _advanced(x, indices):
    tensors = [(j, t) for j, t in enumerate(indices) if t is not None]
    for _, t in tensors:
        if t.dtype not in (torch.int64, torch.int32):
            raise NotImplementedError('an index of %s: only integer indices have static shapes' % (t.dtype,))
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
    _typed(target, [], outs, ELEMENTS)
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
# aten/src/ATen/native/TensorAdvancedIndexing.cpp, scatter_reduce_two and scatter_reduce_exclude_self_helper
SCATTERS = {'last': 0, 'sum': 1, 'add': 1, 'prod': 2, 'multiply': 2, 'mean': 3, 'amax': 4, 'amin': 5}


def _scatter(out, base, src, targets, candidates, limits, near, far, op, include, alpha=1):
    if len(targets) > 4:
        raise NotImplementedError('a scatter of %d index tensors: the source takes at most four (metal-microbench '
                                  'operator_shaders.metal scatter)' % len(targets))
    integral = out.dtype in INTEGRAL
    tensor = isinstance(src, torch.Tensor)
    g = [ELEMENTS[out.dtype], ELEMENTS[base.dtype], ELEMENTS[src.dtype] if tensor else -1, op, int(include),
         int(integral), out.dim(), len(candidates), len(targets), *[ELEMENTS[t.dtype] for t, *_ in targets],
         *[-1] * (4 - len(targets)), *out.shape, *out.stride(), *base.stride(), *limits, *candidates, *near, *far]
    for _, dim, wrap, across, along in targets:
        g += [dim, wrap, *across, *along]
    return _dispatch('scatter', out.numel(), out, base, src if tensor else base, *[t for t, *_ in targets],
                     *[base] * (4 - len(targets)), g, [0.0 if tensor else float(src), float(alpha)],
                     [0 if tensor else _scalar(src, True), _scalar(alpha, True)])


def _scattered(target, given, outs):
    packet, (out,), x = target.overloadpacket, outs, given['self']
    _typed(target, [x], outs, ELEMENTS)
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


# PyTorch 2.14 documentation, torch.nn.functional.scaled_dot_product_attention
def _attended(target, given, outs):
    (out,), q, k, v, mask = outs, given['query'], given['key'], given['value'], given['attn_mask']
    _typed(target, [q, k, v, mask], outs)
    if given['dropout_p']:
        raise NotImplementedError('%s with dropout_p %r: its dropout draws random numbers, which no source here '
                                  'draws' % (target, given['dropout_p']))
    lead, (length, size), (other, value) = q.shape[:-2], q.shape[-2:], (k.shape[-2], v.shape[-1])
    group = q.shape[-3] // k.shape[-3] if given['enable_gqa'] else 1
    heads = lead if group == 1 else (*lead[:-1], k.shape[-3])
    keys, values = k.expand(*heads, other, size), v.expand(*heads, other, value)
    masks = mask.expand(*lead, length, other) if mask is not None else None
    n = len(lead)
    g = [ELEMENTS[out.dtype], ELEMENTS[q.dtype], ELEMENTS[k.dtype], ELEMENTS[v.dtype],
         ELEMENTS[mask.dtype] if mask is not None else -1, int(given['is_causal']), length, other, size, value, group, n,
         *lead, *[t.stride()[j] if t is not None else 0 for t in (out, q, keys, values, masks) for j in range(n)],
         *out.stride()[-2:], *q.stride()[-2:], *keys.stride()[-2:], *values.stride()[-2:],
         *(masks.stride()[-2:] if masks is not None else (0, 0))]
    scale = given['scale'] if given['scale'] is not None else 1 / math.sqrt(size)
    return _dispatch('attention', out.numel(), out, q, k, v, mask if mask is not None else q, g, [float(scale)])


# PyTorch 2.14 documentation, torch.cat
def _concatenated(target, given, outs):
    (out,), d = outs, given['dim'] % max(outs[0].dim(), 1)
    thunks, at = [], 0
    for t in given['tensors']:
        if t.dim() == 1 and t.shape[0] == 0 and out.dim() != 1:
            continue
        thunks += _copy(out.narrow(d, at, t.shape[d]), t)
        at += t.shape[d]
    return thunks


# design/torch-source-lowering-change-request.md item 1
SOURCES = {**dict.fromkeys(POINTWISE, _arithmetic), **dict.fromkeys((A.clone, A._to_copy, A._unsafe_view, A.copy),
                                                                    _copying),
           **dict.fromkeys((A.mm, A.bmm, A.matmul, A.linear, A.addmm), _product), A.einsum: _summation,
           **dict.fromkeys((*CONVOLUTION, *TRANSPOSED, A.convolution), _convolving),
           **dict.fromkeys((*MAXIMUM, *AVERAGE, *INDEXED), _pooled), **dict.fromkeys(PAD, _padded),
           **dict.fromkeys((*FILLS, *EMPTY, A.arange), _factory), **dict.fromkeys(REDUCTIONS, _reduced),
           **dict.fromkeys(SOFTMAX, _softened), A.cumsum: _scanned, A.sort: _ranked, A.topk: _ranked,
           **dict.fromkeys(NORMS, _normalized),
           **dict.fromkeys((A.gather, A.index_select, A.embedding, A.index), _taken),
           **dict.fromkeys((A.scatter, A.scatter_add, A.scatter_reduce, A.index_put, A.index_add, A.index_copy,
                            A.index_fill), _scattered),
           A.scaled_dot_product_attention: _attended, A.cat: _concatenated}
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
    raise NotImplementedError('%s: no source lowers it (design/torch-source-lowering-change-request.md item 1)'
                              % (local.target,))


def _devices(node):
    values = [v for n in [node, *node.all_input_nodes] for v in pytree.tree_leaves(n.meta.get('val'))]
    return {v.device.type for v in values if isinstance(v, torch.Tensor)}


def _sourced(gm):
    for node in gm.graph.nodes:
        if node.op == 'call_function':
            _lowering(Local(node.op, node.target, node.args, node.kwargs))
            devices = _devices(node)
            if len(devices) > 1:
                raise NotImplementedError('%s (%s) moves a value between the %s inside one function: design/torch-'
                                          'source-lowering-change-request.md item 1 types and schedules a function on '
                                          'the backend it is placed on, and names no backend for a move between two'
                                          % (node.name, node.target, ' and the '.join(sorted(devices))))


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


# mesh-collective.h struct mesh_value; design/aot-compilation-change-request.md A9 and appendix items 6 and 7;
# design/disjoint-pages-change-request.md C3.3; M. Cowan et al., MSCCLang, ASPLOS 2023, section 4.2; GSPMD section 4.1;
# torch v2.14.0 torch/_inductor/cudagraph_trees.py; Y. Futamura, Partial evaluation of computation process: an approach
# to a compiler-compiler, 1971; rdma/mesh.py Publication
class _Dynamic:
    def __init__(self, key):
        self.key = key


def _bound(structure, value_of):
    built = map_arg(structure, value_of)
    leaves, spec = pytree.tree_flatten(built)
    dynamic = [(j, value.key) for j, value in enumerate(leaves) if isinstance(value, _Dynamic)]
    if not dynamic:
        return (lambda env: built), False

    def arguments(env):
        for j, key in dynamic:
            leaves[j] = env[key]
        return pytree.tree_unflatten(leaves, spec)
    return arguments, True


# design/torch-source-lowering-change-request.md item 1
class _Stand(dict):
    def __init__(self, ir):
        super().__init__()
        self.ir = ir

    def __missing__(self, key):
        dtype, shape = _layout(self.ir, key)
        return torch.empty(shape, dtype=dtype, device='meta')


def _metas(value):
    return pytree.tree_map(lambda v: torch.empty_strided(v.shape, v.stride(), dtype=v.dtype, device='meta')
                           if isinstance(v, torch.Tensor) else v, value)


def _launched(ir, local, arguments, dynamic, outs):
    lower = _lowering(local)
    lower(*_metas(arguments(_Stand(ir) if dynamic else None)), _metas(outs))
    if not dynamic:
        thunks = lower(*arguments(None), outs)
        return [lambda env, inputs, t: _each(thunks)]
    return [lambda env, inputs, t: _each(lower(*arguments(env), outs))]


def _aliases(local):
    return local.target is operator.getitem or isinstance(local.target, torch._ops.OpOverload) \
        and any(r.alias_info is not None for r in local.target._schema.returns)


def _entry(context, state, transfer):
    return mesh.Publication.from_address(context.M + state.transfers + transfer * mesh.PUBLICATION_BYTES)


def _count(context, state, word, c):
    return C.c_uint64.from_address(context.M + word + 8 * (c % state.copies))


# design/torch-source-lowering-change-request.md item 1; design/plan-to-bridge-change-request.md R2 and R4; mesh-section.h
# struct mesh_section
class _Paged:
    def __init__(self, pages):
        self.pages = pages


def _page(value, k, n):
    if isinstance(value, _Paged):
        return value.pages[k]
    if not isinstance(value, torch.Tensor):
        return value
    flat = value.view(-1)
    return flat.narrow(0, k * n, min(n, flat.numel() - k * n))


def _pages(ir, key, local, arguments, dynamic, out):
    dtype, shape = _layout(ir, key)
    n = ir.page // dtype.itemsize
    return [step for k in range(-(-math.prod(shape) // n))
            for step in _launched(ir, local, lambda env, k=k: pytree.tree_map(lambda v: _page(v, k, n), arguments(env)),
                                  dynamic, [_page(out, k, n)])]


# torch 2.14.0 torch/_tensor_docs.py, view and copy_; design/torch-source-lowering-change-request.md item 1
def _called(ir, key, operands, local, form, placed):
    _lowering(local)
    given = iter(operands)
    arguments, dynamic = _bound((local.args, local.kwargs), lambda n: next(given))
    mine = placed(key)
    if ir.kinds.get(key) == mesh.PAGES:
        return mine, _pages(ir, key, local, arguments, dynamic, mine)
    if isinstance(form, tuple):
        outs = [placed(result) for _, result in form[1]]
        return tuple(outs) if len(outs) > 1 else mine, _launched(ir, local, arguments, dynamic, outs)
    if key in ir.contiguous:
        base = operands[_based(local)]
        thunks = _assign(mine.view(base.shape), base)
        return mine, [lambda env, inputs, t: _each(thunks)]
    if mine is not None and ir.places[ir.device][key].owner == key:
        return mine, _launched(ir, local, arguments, dynamic, [mine])
    if mine is not None:
        return mine, []
    if _aliases(local):
        if not dynamic:
            return _call(local, *arguments(None)), []
        return _Dynamic(key), [lambda env, inputs, t: env.__setitem__(key, _call(local, *arguments(env)))]
    made, spec = pytree.tree_flatten(_call(local, *_metas(arguments(_Stand(ir) if dynamic else None))))
    outs = [torch.empty_strided(m.shape, m.stride(), dtype=m.dtype, device=v.device)
            for m, v in zip(made, pytree.tree_leaves(key[0].meta.get('val')))]
    return pytree.tree_unflatten(outs, spec), _launched(ir, local, arguments, dynamic, outs)


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


def _collected(ir, context, plan, state, placed, c, item, operands):
    _, reads, key, collective = item
    me, receives = ir.roles[collective]
    local = _local(ir, collective)
    if local == 'reduce':
        waits = [[lambda env, inputs, t, i=i, count=_count(context, state, _entry(context, state, i).count, c):
                  _arrived(context, plan, i, t, count)] if state is not None else [] for i, _ in receives]
        total, first, steps = placed(key), operands[0], []
        if collective.kind == 'reduce-scatter':
            n = first.shape[collective.dim] // (len(receives) + 1)
            first = first.narrow(collective.dim, me * n, n)
        for wait, (_, (received, j)) in zip(waits, receives):
            _typed(collective.kind, [first, placed(received)[j]], [total])
            add = _pointwise(0, total, first, placed(received)[j])
            steps += wait + [lambda env, inputs, t, add=add: _each(add)]
            first = total
        return steps
    places = ir.places[ir.device]
    if local == 'copy' and places[reads[0]][:4] != places[key][:4]:
        thunks = _assign(placed(key), operands[0])
        return [lambda env, inputs, t: _each(thunks)]
    return _filled(placed(key), 0) if local == 'fill' else []


# torch 2.14.0 torch/_tensor_docs.py, fill_; design/torch-source-lowering-change-request.md item 1
def _filled(part, fill):
    thunks = _fill(part, fill)
    return [lambda env, inputs, t: _each(thunks)]


# torch 2.14.0 torch/distributed/_functional_collectives.py, wait_tensor; MPI 4.1 section 7.1; libfabric
# man/fi_cntr.3.md, fi_cntr_readerr; man/fi_endpoint.3.md, fi_cancel
def _sent(context, plan, transfer, t, counter):
    def ready():
        status = mesh.LIB.mesh_host_sent(C.byref(context), plan, transfer, t)
        if status == mesh.FAILED:
            raise OSError(errno.ECANCELED, 'execution %d of declared transfer %d: a send cancelled '
                                           '(mesh-collective.h mesh_host_sent)' % (t, transfer))
        return status == 1
    _wait(ready, lambda: counter[0] + counter[1],
          'execution %d of declared transfer %d: no send completion for MESH_PLATFORM_NS' % (t, transfer))


def _completed(ir, context, plan, state, c, collective):
    steps = []
    for i in ir.declares[collective]:
        entry = _entry(context, state, i)
        if entry.pages:
            steps.append(lambda env, inputs, t, i=i, count=_count(context, state, entry.count, c):
                         _arrived(context, plan, i, t, count))
        if entry.sends:
            steps.append(lambda env, inputs, t, i=i, counter=(C.c_uint64 * 2).from_address(
                context.M + entry.completion + 16 * (c % state.copies)): _sent(context, plan, i, t, counter))
    return steps


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
    at = None if cut is None else _cut(ir, 'slice', cut, _layout(ir, read)[1], ir.device)

    def part(value):
        return value if at is None else value.narrow(*at)
    if isinstance(source, _Dynamic):
        _assign(target, part(_Stand(ir)[read]))
        return [lambda env, inputs, t: _each(_assign(target, part(env[read])))]
    thunks = _assign(target, part(source))
    return [lambda env, inputs, t: _each(thunks)]


def _specialized(ir, context, plan, state, devices, placed, c, item, operands, steps):
    kind, reads, key, operand = item
    value = placed(key)
    for collective in ir.waits.get(ir.producer[key], ()) if state is not None else ():
        steps += _completed(ir, context, plan, state, c, collective)
    if kind == 'input':
        value = _Dynamic(key)
        steps.append(lambda env, inputs, t: env.__setitem__(key, inputs[operand]))
    elif kind == 'constant' and value is None:
        value = operator.attrgetter(operand)(ir.gm)
    elif kind == 'copy':
        steps += _copied(ir, reads[0], key, operand, operands[0], value)
    elif kind == 'window' and key in ir.inside:
        dim, m, _, length, starts, size, fill = operand
        q = ir.coords[ir.device][m]
        below, above = min(length, max(0, -starts[q])), min(length, max(0, size - starts[q]))
        for start, extent in ((0, below), (above, length - above)):
            steps += _filled(value.narrow(dim, start, extent), fill) if fill is not None and extent else []
    elif kind in ('slice', 'narrow', 'window') and value is None:
        at = _cut(ir, kind, operand, _layout(ir, reads[0])[1], ir.device)
        if isinstance(operands[0], _Dynamic):
            value = _Dynamic(key)
            steps.append(lambda env, inputs, t: env.__setitem__(key, env[reads[0]].narrow(*at)))
        else:
            value = operands[0].narrow(*at)
    elif kind == 'call':
        value, made = _called(ir, key, operands, ir.locals[operand], ir.forms.get(key), placed)
        steps += made
    elif kind == 'collective':
        steps += _collected(ir, context, plan, state, placed, c, item, operands)
    if state is not None:
        steps += _publishes(context, plan, devices, ir.publishes.get(ir.producer[key], ()))
    return value


def _instantiate(ir, context, plan, devices, tensors, c, spec):
    state, steps = mesh.State.from_address(context.M + plan) if plan else None, []
    items = {item[2]: item for item in ir.program if item[0] != 'output'}

    def placed(key):
        return tensors[key][c % len(tensors[key])] if key in tensors else None
    values = mesh_graph.forward(list(items), lambda key: items[key][1],
                                lambda key, operands: _specialized(ir, context, plan, state, devices, placed, c,
                                                                   items[key], operands, steps))
    for collective in ir.waits.get(len(ir.program) - 1, ()) if state is not None else ():
        steps += _completed(ir, context, plan, state, c, collective)
    node, keys = next(item[3] for item in ir.program if item[0] == 'output')
    arguments = _bound(node.args[0], lambda n: values[keys[n]])[0]
    return steps, lambda env: pytree.tree_unflatten(list(arguments(env)), spec)


def _steps(ir, context, plan, tensors, devices, spec):
    if ir.device is None:
        return [[]], [lambda env: None]
    made = [_instantiate(ir, context, plan, devices, tensors, c, spec)
            for c in range(math.lcm(1, *(len(x) for x in tensors.values())))]
    return [steps for steps, _ in made], [result for _, result in made]



# design/disjoint-pages-change-request.md C3.4; design/aot-compilation-change-request.md, appendix item 7; P. E.
# McKenney and J. D. Slingwine, Read-copy update, PDCS 1998; mesh-dataflow.h mesh_retire; rdma/mesh-torch.mm
# mesh_torch_retire and mesh_torch_notifying; mesh-collective.h mesh_plan
def _retire(context, state, deadline):
    if state['replay'] is not None:
        _unreplayed(context, state, deadline)
    elif 'mps' in state['devices']:
        mesh.check(TORCH.mesh_torch_retire(RETIRE, context.M, context.client))
    else:
        mesh.LIB.mesh_retire(context.M, context.client)
    while TORCH.mesh_torch_notifying():
        if time.clock_gettime_ns(time.CLOCK_MONOTONIC) > deadline:
            raise OSError(errno.ETIMEDOUT, 'a notification after torch\'s work not run by the preparation\'s deadline: '
                                           'the program\'s operand may stand (rdma/mesh-torch.mm)')
    mapping = state['mapping']
    mapping.retired = True
    state.update(mapping=None, runs=None, devices=(), stale=True)
    mapping.release()


# Python documentation, weakref.finalize; Lib/weakref.py, finalize
def _dropped(context, state):
    if state['mapping'] is not None:
        _retire(context, state, time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                + (0 if state['timeout'] is not None else mesh.PLATFORM_NS))


# design/aot-compilation-change-request.md, appendix item 6; design/disjoint-pages-change-request.md C3.7; mesh-call.h
# mesh_transfers_start; torch 2.14.0 torch/fx/graph.py _PyTreeCodeGen, torch/utils/_pytree.py; MPI 4.1 section 3.9,
# MPI_START; NCCL User Guide, CUDA Stream Semantics
def _started(context, plan, metal):
    if metal:
        array = (mesh.U * 1)(0)
        return lambda t: mesh.check(TORCH.mesh_torch_publish(STARTED, C.addressof(context), plan, array, 1, t))
    return lambda t: mesh.LIB.mesh_host_started(C.byref(context), plan, 0, t)


def _execute(context, run, args, kwargs):
    t, k = run['executions'], run['state'].submissions
    status = mesh.LIB.mesh_transfers_start(C.byref(context), run['plan'], t + 1)
    if run['state'].submissions > k:
        run['executions'] = t + 1
    mesh.check(status)
    run['started'](t)
    c, env, inputs = t % len(run['steps']), {}, pytree.arg_tree_leaves(*args, **kwargs)
    for step in run['steps'][c]:
        step(env, inputs, t)
    return run['results'][c](env)


# MTLDevice.h newBufferWithLength:options:; MTLResource.h `label`; metal-microbench metal_recording.m
# MetalProgramOperand
def _labelled(device, nbytes, name):
    buffer = NEW_BUFFER(device, _SEL(b'newBufferWithLength:options:'), max(nbytes, 1), SHARED)
    if not buffer:
        raise OSError(errno.ENOMEM, 'Metal made no buffer of %d bytes (MTLDevice.h newBufferWithLength:options:)'
                      % nbytes)
    _label(buffer, name)
    return buffer


def _label(buffer, name):
    text = STRING(OBJECT(OBJC.objc_getClass(b'NSString'), _SEL(b'alloc')), _SEL(b'initWithUTF8String:'), name.encode())
    SET(buffer, _SEL(b'setLabel:'), text)
    PUT(text, SELECTOR[b'release'])


# torch 2.14.0 torch/export/__init__.py, aten/src/ATen/mps/MPSStream.mm, aten/src/ATen/native/mps/OperationUtils.mm;
# metal-microbench metal_recording.h capture:, metal_recording.m MetalDelegateValues and MetalMaskReaders;
# design/torch-source-lowering-change-request.md item 1
def _captured(context, irs, compiled, gm, of, constants, holders):
    lib, made = _recorder()
    _library()
    error = C.create_string_buffer(4096)
    device = made()
    recorder = lib.MetalRecorderOpen(device, error, len(error))
    PUT(device, SELECTOR[b'release'])
    if not recorder:
        raise OSError(errno.ENODEV, 'the recorder: %s' % error.value.decode())
    owners, standins, counted = {}, {}, _Mapping(mesh.View())

    def standin(owned, bytes, dtype, numel):
        buffer = _labelled(_metal_device(), bytes, 'function %d value %d' % owned[:2]
                           + (' page %d' % owned[2] if len(owned) > 2 else ''))
        owners[buffer] = owned
        made = _dlpack_tensor('mps', buffer, 0, buffer, dtype, (numel,), counted)
        PUT(buffer, SELECTOR[b'release'])
        return made
    for k, ir in enumerate(irs):
        if ir.device is None:
            continue
        copies = []
        for v, (dtype, numel, _) in enumerate(ir.layouts):
            n = ir.page // dtype.itemsize
            copies.append([standin((k, v), dtype.itemsize * numel, dtype, numel)] if ir.values[v].run else
                          [_Paged([standin((k, v, p), ir.page, dtype, min(n, numel - p * n))
                                   for p in range(-(-numel // n))])])
        standins[k] = _homes(ir, copies)
    calls = []

    def capture(node, args, kwargs):
        k = of[node]
        ir = irs[k]
        if ir.device is None:
            calls.append((node, k, None, None, 0, []))
            return None
        steps, results = _steps(ir, context, 0, standins[k][0], {}, compiled[k].spec)
        inputs, env, result, failure = pytree.arg_tree_leaves(*args, **kwargs), {}, [], []

        def encode(_):
            try:
                for step in steps[0]:
                    step(env, inputs, 0)
                result.append(results[0](env))
                for leaf in pytree.tree_leaves(result[0]):
                    buffer = leaf.untyped_storage().data_ptr() if isinstance(leaf, torch.Tensor) else None
                    if buffer is not None and buffer not in owners:
                        owners[buffer] = (k, ir.value(leaf.dtype, leaf.untyped_storage().nbytes() // leaf.element_size(),
                                                      'mps', 0, 0, 0))
                        _label(buffer, 'function %d value %d' % owners[buffer])
                torch.mps.synchronize()
            except BaseException as raised:
                failure.append(raised)
        torch.mps.synchronize()
        callback = ENCODE(encode)
        handle = lib.MetalRecorderCapture(recorder, callback, None, error, len(error))
        if failure:
            raise failure[0]
        if not handle:
            raise OSError(errno.EINVAL, 'call %s of compiled function %d: the capture: %s'
                          % (node.name, k, error.value.decode()))
        ir.tasks = lib.MetalRecorderTasks(recorder, handle)
        made = _weighed(ir, compiled[k].function)
        if lib.MetalRecorderWeigh(recorder, handle, (C.c_double * max(ir.tasks, 1))(*made.tasks), ir.tasks, error,
                                  len(error)):
            raise OSError(errno.EINVAL, 'call %s of compiled function %d: the weighing: %s'
                          % (node.name, k, error.value.decode()))
        ir.weights = list(made.transfers)
        levels = LEVELS(recorder, _SEL(b'levels:'), handle)
        own = (MetalValue * max(VALUES(recorder, _SEL(b'values:level:into:'), handle, 0, None), 1))()
        count = VALUES(recorder, _SEL(b'values:level:into:'), handle, 0, C.cast(own, C.c_void_p))
        calls.append((node, k, handle, result[0], levels, [mesh.Value(
            section=mesh.Section(bytes=v.bytes), kind=v.kind, state=v.state, begin=v.begin, end=v.end, run=1)
            for v in own[:count]], env, steps))
        return result[0]
    _Program(gm, capture).run(*[standins[k][1][i][0] if k is not None else constant
                                for i, (constant, k) in enumerate(zip(constants, holders))])
    return {'lib': lib, 'recorder': recorder, 'owners': owners, 'calls': calls}


# ExecuTorch, "Concepts", Memory planning; torch 2.14.0 torch/_inductor/cudagraph_trees.py; M. Lam, PLDI 1988, §2.3;
# metal_recording.h place:value:; mesh-collective.h mesh_view_values; MTL4CommandQueue.h addResidencySet:; mesh.h struct
# mesh_state; design/torch-capture-change-request.md item 6; design/torch-source-lowering-change-request.md item 1
def _replays(captured, functions, states, values, addresses, base, buffer):
    lib, recorder, error = captured['lib'], captured['recorder'], C.create_string_buffer(4096)
    queue, replays = OBJECT(recorder, _SEL(b'queue')), []

    def offsets(at):
        return (C.c_uint64 * len(at))(*[a - base for a in at])
    for node, k, handle, _, _, own, *_ in captured['calls']:
        if handle is None:
            replays.append(None)
            continue
        for operand, owned in captured['owners'].items():
            value, at = values[owned[0]][owned[1]], addresses[owned[0]][owned[1]]
            PLACE_OPERAND(recorder, _SEL(b'place:operand:in:offsets:copies:ahead:shares:'), handle, operand, buffer,
                          offsets(at if len(owned) == 2 else [pages[owned[2]] for pages in at]), value.section.count,
                          0, bool(value.shares))
        first = len(values[k]) - len(own)
        for i in range(len(own)):
            value = values[k][first + i]
            PLACE_VALUE(recorder, _SEL(b'place:value:in:offsets:copies:shares:'), handle, i, buffer,
                        offsets(addresses[k][first + i]), value.section.count, bool(value.shares))
        replay = Replay()
        status = lib.MetalRecorderReplay(recorder, handle, C.byref(replay), error, len(error))
        PUT(handle, SELECTOR[b'release'])
        if status:
            raise OSError(status, 'call %s of compiled function %d: the replay: %s' % (node.name, k,
                                                                                         error.value.decode()))
        SET(queue, _SEL(b'addResidencySet:'), replay.residency)
        replays.append(replay)
    calls = (Call * len(replays))(*[Call(C.pointer(replay) if replay is not None else None, functions[k].plan, 0, None,
                                         states[k].copies - 1)
                                    for replay, (_, k, *_) in zip(replays, captured['calls'])])
    gpu = OBJECT(queue, _SEL(b'device'))
    allocators = (C.c_void_p * HOST_LAG)(*[OBJECT(gpu, _SEL(b'newCommandAllocator')) for _ in range(HOST_LAG)])
    return {'lib': lib, 'recorder': recorder, 'queue': queue, 'replays': replays, 'calls': calls,
            'allocators': allocators, 'ended': OBJECT(gpu, _SEL(b'newSharedEvent')), 'error': error}


# torch 2.14.0 torch/_inductor/cudagraph_trees.py
def _outputs(captured, copies, gm, constants, inputs):
    period = math.lcm(HOST_LAG, *(len(c) for function in copies.values() for c in function))

    def leaf(x, p):
        if not isinstance(x, torch.Tensor):
            return x
        k, v = captured['owners'][x.untyped_storage().data_ptr()]
        copy = copies[k][v][p % len(copies[k][v])]
        return copy.as_strided(x.shape, x.stride(), copy.storage_offset() + x.storage_offset())
    results = {node: result for node, _, _, result, *_ in captured['calls']}
    return [_Program(gm, lambda node, args, kwargs, p=p: pytree.tree_map(lambda x: leaf(x, p), results[node])).run(
        *[placed[0][p % len(placed[0])] if placed is not None else constant for constant, placed in zip(constants, inputs)])
        for p in range(period)]


# mesh-call.h mesh_transfers_start; MPI 4.1 §3.9, MPI_START; NCCL User Guide, CUDA Stream Semantics
def _replayed(context, replay, b):
    status = replay['lib'].MetalRecorderRun(replay['recorder'], replay['allocators'], HOST_LAG, replay['calls'],
                                            len(replay['calls']), b, replay['ended'], C.addressof(context), START,
                                            STARTED, replay['error'], len(replay['error']))
    if status:
        raise OSError(status, 'execution %d of the program: %s' % (b, replay['error'].value.decode()))
    return replay['outputs'][b % len(replay['outputs'])]


# MTLEvent.h `signaledValue`
def _ended(replay):
    return VALUE(replay['ended'], _SEL(b'signaledValue'))


# MTL4CommandQueue.h signalEvent:value:; MTLEvent.h waitUntilSignaledValue:timeoutMS:; mesh-dataflow.h mesh_retire
def _unreplayed(context, state, deadline):
    replay, last = state['replay'], state['executions'] + 1
    SIGNAL(replay['queue'], _SEL(b'signalEvent:value:'), replay['ended'], last)
    while _ended(replay) < last:
        left = deadline - time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        if left <= 0 or not WAIT(replay['ended'], _SEL(b'waitUntilSignaledValue:timeoutMS:'), last,
                                 max(left // 1000000, 1)) and _ended(replay) < last:
            raise OSError(errno.ETIMEDOUT, 'the replay\'s work not ended by the deadline: the program\'s operand may '
                                           'stand')
    mesh.LIB.mesh_retire(context.M, context.client)
    for made in replay['replays']:
        if made is not None:
            SET(replay['queue'], _SEL(b'removeResidencySet:'), made.residency)
            replay['lib'].MetalReplayDestroy(C.byref(made))
    for held in (*replay['allocators'], replay['ended'], replay['recorder']):
        PUT(held, SELECTOR[b'release'])
    state['replay'] = None


# design/aot-compilation-change-request.md A1 to A3 and A10, "The calls' graph"; P. Barham et al., MLSys 2022, section
# 3; J. Reed, Z. DeVito, H. He, A. Ussery and J. Ansel, MLSys 2022; torch v2.14.0 torch/fx/_symbolic_trace.py,
# Tracer.is_leaf_module; torch/nn/modules/module.py, _forward_unimplemented
class Function(torch.nn.Module):
    def __init__(self, module, placement, weights):
        super().__init__()
        self.module, self.placement, self.weights = module, placement, weights


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
# Annals of Math., 1950; D. R. Fulkerson, Proc. AMS, 1956
def _analyse(gm, of, modules):
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
    held, ordered = mesh_graph.closure(nodes, lambda node: node.all_input_nodes), mesh_graph.closure(nodes, inputs)
    orders = frozenset((place[a], place[b]) for b in calls for a in ordered[b] if a in place)
    commuting = tuple(tuple(place[node] for node in chain) for chain in chains)
    entries = {}
    for f in dict.fromkeys(callee[node] for node in calls):
        mine = [node for node in calls if callee[node] == f]
        entries[f] = Calls(tuple(place[node] for node in mine),
                           mesh_graph.width(mine, lambda a: [b for b in mine if a in held[b]]), orders, commuting)
    return entries


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


Compiled = collections.namedtuple('Compiled', 'function signature gm examples result spec')
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
        self.mode, self.compiled, self.of, self.origin, self.calls, self.reads = mode, [], {}, {}, [], []

    def fake(self, value):
        if not isinstance(value, torch.Tensor):
            return value
        with self.mode:
            return torch.empty_strided(tuple(value.shape), tuple(value.stride()), dtype=value.dtype,
                                       device=value.device)

    def run_node(self, n):
        function = self.module.get_submodule(n.target) if n.op == 'call_module' else None
        if isinstance(function, Function):
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            return self.export(n, function, args, kwargs)
        source = n.args[0] if n.op == 'call_function' and n.target is operator.getitem and n.args else None
        if isinstance(source, Node) and source in self.origin and not isinstance(self.env[source], torch.Tensor):
            self.origin[n] = self.origin[source]
            return super().run_node(n)
        if n.op in ('placeholder', 'output'):
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
            exported = torch.export.export(function.module, tuple(args), dict(kwargs), strict=False) \
                .run_decompositions({})
            gm = exported.module(check_guards=False)
            _RemoveRuntimeAssertionsPass()(gm)
            _static(gm)
            _sourced(gm)
            out = [a.meta['val'] if isinstance(a, Node) else a for a in gm.graph.output_node().args[0]]
            result = pytree.tree_unflatten([self.fake(v) for v in out], exported.call_spec.out_spec)
            self.compiled.append(Compiled(function, signature, gm, flat, result, exported.call_spec.out_spec))
            found = [len(self.compiled) - 1]
        self.of[n], self.origin[n] = found[0], len(self.calls)
        self.reads.append({self.origin[a] for a in n.all_input_nodes if a in self.origin})
        self.calls.append(found[0])
        return self.compiled[found[0]].result


# design/aot-compilation-change-request.md A1 to A3, A8 to A10 and appendix item 7;
# design/disjoint-pages-change-request.md C3.1, D.3.2 and D.3.3; mesh-collective.h mesh_plan, struct mesh_function and
# struct mesh_value; mesh-dataflow.h mesh_nodes_read, mesh_retire and struct mesh_ctx; rdma/mesh-torch.mm mesh_torch_end
def program(block, *example_inputs):
    tracer = Tracer()
    graph = tracer.trace(block)
    gm = torch.fx.GraphModule(tracer.root, graph)
    exporter = _Export(gm, FakeTensorMode())
    exporter.run(*[exporter.fake(x) for x in example_inputs])
    compiled, of, calls, reads = exporter.compiled, exporter.of, exporter.calls, exporter.reads
    entries = _analyse(gm, of, [record.function.module for record in compiled])
    operands = [record.function.placement(record.gm, record.examples) for record in compiled]
    placeholders = [n for n in gm.graph.nodes if n.op == 'placeholder']
    readers = [[of[n] for n in gm.graph.nodes if n in of and p in n.all_input_nodes] for p in placeholders]
    given = [(x.dtype, x.numel(), x.device.type, tuple(x.shape)) if isinstance(x, torch.Tensor) else None
             for x in example_inputs]
    constants = [None if isinstance(x, torch.Tensor) else x for x in example_inputs]
    context, state = _context(), {'stale': True, 'mapping': None, 'runs': None, 'devices': (), 'executions': 0,
                                  'ended': 0, 'inputs': [None] * len(given), 'timeout': None, 'replay': None}

    def make():
        nodes, header = mesh.Nodes(), mesh.Header.from_address(context.M)
        mesh.check(mesh.LIB.mesh_nodes_read(C.byref(context), C.byref(nodes)))
        irs, page = [], header.pgsz * header.block
        for k, (record, operand) in enumerate(zip(compiled, operands)):
            ir = partition(record.gm, operand, nodes)
            ir.calls, ir.page = entries[k], page
            lower(ir)
            irs.append(ir)
        for j, k in enumerate(calls):
            absent = sorted(r for r in reads[j] if irs[calls[r]].device is None)
            if irs[k].device is not None and absent:
                raise ValueError('call %d of the program, of compiled function %d, has its section at map index %d '
                                 'and reads the result of call %d, whose function has none there: such a node "has '
                                 'no value for any program code to read" (design/aot-compilation-change-request.md '
                                 'A4, withdrawn text)' % (j, k, nodes.self, absent[0]))
        holders = []
        for i, (x, functions) in enumerate(zip(given, readers)):
            k = next((k for k in functions if irs[k].device is not None), None)
            if x is not None and k is not None:
                irs[k].inputs[i] = (irs[k].value(*x[:3], 0, 0, 0, lag=HOST_LAG), x[3])
            holders.append(k if x is not None else None)
        here = {layout[2] for ir in irs for layout in ir.layouts}
        replayed = here == {'mps'} and not any(d.value != mesh.ABSENT for ir in irs for d in ir.declared)
        if replayed and len(set(calls)) != len(calls):
            raise NotImplementedError('a compiled function called more than once in the block: its replay binds its '
                                      'values by its plan\'s executions, the block\'s inputs by the block\'s')
        captured = _captured(context, irs, compiled, gm, of, constants, holders) if replayed else None
        for ir, record in zip(irs, compiled):
            if ir.weights is None:
                ir.weights = list(_weighed(ir, record.function).transfers)
        own = [[] for _ in irs]
        # M. Lam, PLDI 1988, p. 322; ExecuTorch exir/memory_planning.py:348–354; torch/fx/graph.py:1498–1502; TensorFlow
        # Lite arena_planner.cc:334–335; metal_recording.h MetalRecorderOperand
        buffers = {}
        for buffer, owned in captured['owners'].items() if captured else ():
            buffers.setdefault(owned[:2], []).append(buffer)

        def interval(handle, buffer, levels, result):
            first, last = mesh.U(), mesh.U()
            bound = captured['lib'].MetalRecorderOperand(handle, buffer, C.byref(first), C.byref(last))
            if result or bound & 4:
                return first.value if bound & 2 else 0, max(levels, 1) - 1 if result else last.value
            return 0, max(levels, 1) - 1
        for _, k, handle, result, levels, values, *_ in captured['calls'] if captured else ():
            results = {captured['owners'].get(leaf.untyped_storage().data_ptr()) for leaf in pytree.tree_leaves(result)
                       if isinstance(leaf, torch.Tensor)}
            for v, value in enumerate(irs[k].values) if handle is not None else ():
                if value.kind == mesh.EXECUTION and not value.lag:
                    spans = [interval(handle, buffer, levels, (k, v) in results) for buffer in buffers.get((k, v), ())]
                    value.begin, value.end = (min(s for s, _ in spans), max(e for _, e in spans)) if spans else \
                        (0, max(levels, 1) - 1)
            own[k] = values
        arrays = [((mesh.Declared * len(ir.declared))(*ir.declared), (mesh.Weight * len(ir.weights))(*ir.weights),
                   (mesh.Value * (len(ir.values) + len(mine)))(*ir.values, *mine),
                   (mesh.Dependence * len(ir.dependences))(*ir.dependences)) for ir, mine in zip(irs, own)]
        functions = (mesh.Function * len(irs))(*[
            mesh.Function(C.cast(d, C.POINTER(mesh.Declared)), C.cast(w, C.POINTER(mesh.Weight)),
                          C.cast(v, C.POINTER(mesh.Value)), C.cast(e, C.POINTER(mesh.Dependence)), len(d), len(v),
                          len(e), 0, ir.calls.outstanding, 0)
            for ir, (d, w, v, e) in zip(irs, arrays)])
        deadline = time.clock_gettime_ns(time.CLOCK_MONOTONIC) + mesh.PLATFORM_NS
        if state['mapping'] is not None:
            _retire(context, state, deadline)
        mesh.check(mesh.LIB.mesh_plan(C.byref(context), _group(), _program(), functions, len(irs),
                                      (mesh.U * len(calls))(*calls), len(calls), deadline))
        state.update(mapping=_Mapping(mesh.View()), devices={layout[2] for ir in irs for layout in ir.layouts})
        try:
            # JGroups ViewId.java
            views = [mesh.State.from_address(context.M + f.plan) for f in functions]
            if any((s.serial, s.sequencer, s.boot) != (nodes.serial, nodes.sequencer, nodes.boot) for s in views):
                raise OSError(errno.ESTALE, 'a plan\'s view is not the one its IR was built on: plan again')
            listed = [values for _, _, values, _ in arrays]
            view, addresses = _mapped(context, functions, len(irs), listed, page)
            state['mapping'].view, hull = view, _hull(listed, addresses, page)
            buffer = _placed_buffer(hull) if 'mps' in state['devices'] else None
            try:
                bound = [_bind(ir, values, at, buffer, hull.address, page, state['mapping'])
                         for ir, values, at in zip(irs, listed, addresses)]
                if captured:
                    state['replay'] = _replays(captured, functions, views, listed, addresses, hull.address, buffer)
            finally:
                if buffer:
                    PUT(buffer, SELECTOR[b'release'])
            # torch 2.14.0 torch/export/exported_program.py, ExportedProgram; GSPMD section 4.5; Megatron-LM core_r0.12.0
            # megatron/core/tensor_parallel/layers.py:156–165; design/problems-of-14b9a757-change-request.md, "Sources
            # for the second stops", 8, and "Sources for the third stops", 8; design/torch-source-lowering-change-
            # request.md item 1
            launched = False
            for ir, (places, _, _, _) in zip(irs, bound):
                loaded = ir.loads & places.keys()
                needed = mesh_graph.reach(loaded, lambda key: ir.attributes[key][1])
                host = mesh_graph.forward([key for key in ir.attributes if key in needed],
                                          lambda key: ir.attributes[key][1],
                                          lambda key, values: _hosted(ir, ir.attributes[key], values))
                for key in loaded:
                    _each(_assign(places[key][0], host[key]))
                    launched = launched or places[key][0].device.type == 'mps'
            if launched:
                torch.mps.synchronize()
            runs = None if captured else [
                dict(zip(('steps', 'results'), _steps(ir, context, f.plan, places, devices, record.spec)),
                     plan=f.plan, executions=0, state=mesh.State.from_address(context.M + f.plan),
                     started=_started(context, f.plan, any(layout[2] == 'mps' for layout in ir.layouts)))
                for ir, f, record, (places, devices, _, _) in zip(irs, functions, compiled, bound)]
            if captured:
                state['replay']['outputs'] = _outputs(captured, {k: b[3] for k, b in enumerate(bound)}, gm, constants,
                                                      [next((b[2][i] for b in bound if i in b[2]), None)
                                                       for i in range(len(given))])
        except BaseException:
            _retire(context, state, deadline)
            raise
        state['inputs'] = [next((b[2][i] for b in bound if i in b[2]), None) for i in range(len(given))]
        state.update(stale=False, runs=runs)

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
        if state['replay'] is not None:
            return _ended(state['replay'])
        return TORCH.mesh_torch_ended() if 'mps' in state['devices'] else state['ended']

    def end(b):
        if 'mps' in state['devices']:
            mesh.check(TORCH.mesh_torch_end(b))
        else:
            state['ended'] = b + 1

    def writable():
        if state['stale']:
            make()
        b = state['executions']
        _wait(lambda: ended() + HOST_LAG > b, ended, 'execution %d of the program: execution %d not ended, no end for '
              'MESH_PLATFORM_NS, so its inputs\' copy is not free (D.3.3)' % (b, b - HOST_LAG))
        return tuple(None if placed is None else placed[1][b % len(placed[1])] for placed in state['inputs'])

    def execution():
        if state['stale']:
            make()
        b = state['executions']
        state['executions'] = b + 1
        if state['replay'] is not None:
            try:
                return _replayed(context, state['replay'], b)
            except OSError as failure:
                if failure.errno in (errno.ESTALE, errno.ECANCELED):
                    state['stale'] = True
                raise
        arguments = [constant if placed is None else placed[0][b % len(placed[0])]
                     for constant, placed in zip(constants, state['inputs'])]
        try:
            result = _Program(gm, lambda node, args, kwargs: _execute(context, state['runs'][of[node]], args,
                                                                      kwargs)).run(*arguments)
        except BaseException as failure:
            if isinstance(failure, OSError) and failure.errno in (errno.ESTALE, errno.ECANCELED):
                state['stale'] = True
            end(b)
            raise
        end(b)
        return result

    def inputs():
        return timed(writable)

    def run():
        return timed(execution)
    # MTL4CommandQueue.h signalEvent:value:
    run.inputs, run.ended = inputs, ended
    weakref.finalize(run, _dropped, context, state)
    return run
