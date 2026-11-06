import collections
import ctypes as C
import errno
import math
import operator
import os
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

# torch 2.14.0 torch/export/__init__.py, export: "produces normalized operators in the functional ATen operator set";
# torch/_ops.py: "Each OpOverload object contains pointer to a specific operator overload, a pointer to the parent
# `OpOverloadPacket` object"
A = torch.ops.aten
# Y. Xu et al., GSPMD, arXiv 2105.04663, 2021, section 3.5
ELEMENTWISE = {A.add, A.sub, A.mul, A.div, A.neg, A.relu, A.tanh, A.sigmoid, A.exp, A.gelu, A.silu}
# GSPMD, sections 3.2 and 3.5: "string equation, e.g., ABC, ACD -> ABD"
DOT = {A.matmul, A.mm, A.bmm}
# GSPMD, section 4.3: "Windowed operators.  The most typical use case of halo exchange is for partitinoning window-based
# operators (e.g., Convolution, ReduceWindow)"; "Halo exchange for data formatting ... after a Slice or Pad operator"
CONVOLUTION = {A.conv1d: 1, A.conv2d: 2, A.conv3d: 3}
TRANSPOSED = {A.conv_transpose1d: 1, A.conv_transpose2d: 2, A.conv_transpose3d: 3}
MAXIMUM = {A.max_pool1d: 1, A.max_pool2d: 2, A.max_pool3d: 3}
AVERAGE = {A.avg_pool1d: 1, A.avg_pool2d: 2, A.avg_pool3d: 3}
PAD = {A.pad, A.constant_pad_nd}
# GSPMD, section 3.1: the device mesh, "a logical multi-dimensional tensor"; mesh_split(tensor, device_mesh,
# dims_mapping).  L. Zheng et al., Alpa, OSDI 2022, section 4.1: "For a set of physical devices, there can be multiple
# logical views"
Placement = collections.namedtuple('Placement', 'mesh nodes splits')
# GSPMD, sections 2.1, 3.2, 4.2 and 4.5
Collective = collections.namedtuple('Collective', 'kind node dims dim position')
Calls = collections.namedtuple('Calls', 'mine outstanding orders commuting')
# GSPMD, section 4.1, "Static operator configurations"
Local = collections.namedtuple('Local', 'op target args kwargs')
# GSPMD, sections 3.2, 3.5, 4 and 4.6
Rule = collections.namedtuple('Rule', 'kind level operands labels out propagate contracting windows fused merged '
                                      'configure')
# GSPMD, section 4.3 and Appendix A.2; section 4.1, masking by "the identity value of the reduction"
Window = collections.namedtuple('Window', 'label operand dim extent stride low transposed fill')


# M. Cowan et al., MSCCLang, ASPLOS 2023, sections 4 and 5.  NCCL User Guide, "Collective Operations": "Collective
# operations have to be called for each rank (hence CUDA device), using the same count and the same datatype, to form a
# complete collective operation.  Failure to do so will result in undefined behavior, including hangs, crashes, or data
# corruption"
class IR:
    def __init__(self, gm, placement, nodes):
        self.gm, self.placement, self.weights = gm, placement, None
        self.order = list(gm.graph.nodes)
        self.shape, self.coords, self.map = _mesh(placement)
        here = [d for d, index in enumerate(self.map) if index == nodes.self]
        self.device, self.identity = here[0] if here else None, 0
        self.sharding, self.program, self.roles, self.events = {}, [], {}, []
        self.declared, self.dependences, self.values, self.layouts = [], [], [], []
        self.made, self.calls, self.locals, self.shapes = set(), None, {}, {}
        self.metas, self.forms, self.over, self.producer, self.places = {}, {}, {}, {}, {}
        self.homes, self.at, self.inside, self.added, self.owned = {}, {}, {}, {}, {}
        self.publishes, self.inputs, self.declares, self.waits = {}, {}, {}, {}
        self.attributes, self.loads = {}, set()
        self.taken, self.results, self.sent, self.contiguous = {}, {}, set(), set()

    # GSPMD, section 3.1, tiled: local(s) = s / D_m along mesh dimension m, s where m = -1
    def local(self, node, sharding):
        return tuple(s // self.shape[m] if m >= 0 else s for s, m in zip(_value(node).shape, sharding))

    def transfer(self, source, destinations, kind, part, landings):
        self.identity += 1
        transfers = {}
        for d in destinations:
            transfers[d] = len(self.declared)
            self.declared.append(mesh.Declared(self.map[source], self.map[d], self.identity, kind, mesh.ABSENT, 0,
                                               part, landings[d]))
        return transfers

    def value(self, dtype, numel, device, wire, begin, end, kind=mesh.EXECUTION, lag=0):
        self.values.append(mesh.Value(section=mesh.Section(bytes=dtype.itemsize * numel), kind=kind, wire=wire,
                                      begin=begin, end=end, lag=lag, run=1))
        self.layouts.append((dtype, numel, device))
        return len(self.values) - 1


# J. Ansel et al., PyTorch 2, ASPLOS 2024, sections 3.5 and 5.1: "the metadata of a tensor without its actual data".
# torch 2.14.0 torch/fx/experimental/proxy_tensor.py, set_meta: `proxy.node.meta["val"] = extract_val(...)`;
# torch/_subclasses/fake_tensor.py, FakeTensor: "FakeTensor extends MetaTensors to also carry an additional
# `fake_device` which tracks devices that would have been used"
def _value(node):
    return node.meta['val']


# GSPMD, section 3.1
def _shape(ir, key):
    return ir.shapes[key] if key in ir.shapes else ir.local(key[0], key[1])


def _layout(ir, key):
    if key in ir.metas:
        return ir.metas[key].dtype, tuple(ir.metas[key].shape)
    return _value(key[0]).dtype, tuple(_shape(ir, key))


def _device(ir, key):
    return (key[0].meta['val'][key[3]] if key in ir.results else _value(key[0])).device.type


# GSPMD, section 3.1; Alpa, section 4.1
def _mesh(placement):
    def item(at):
        value = placement.mesh
        for i in at:
            value = value[i]
        return value
    first = mesh_graph.walk([()], lambda at: [at + (0,)] if isinstance(item(at), (list, tuple)) else [])
    shape = [len(item(at)) for at in first[:-1]]
    places = mesh_graph.walk([()], lambda at: [at + (i,) for i in range(len(item(at)))] if len(at) < len(shape) else [])
    return tuple(shape), {item(at): at for at in places if len(at) == len(shape)}, list(placement.nodes)


# GSPMD, section 3.5, "Merging compatible shardings"
def _merge(a, b):
    merged = tuple(x if y < 0 else y for x, y in zip(a, b))
    used = [m for m in merged if m >= 0]
    if any(0 <= x != y >= 0 for x, y in zip(a, b)) or len(used) != len(set(used)):
        return None
    return merged


# GSPMD, section 3.5, "Preserved dimensions in operators"
def _carry(sharding, source, target):
    return tuple(sharding[source.index(label)] if label in source else -1 for label in target)


# torch 2.14.0 torch/_C/__init__.pyi: FunctionSchema `arguments`, Argument `name`
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


# GSPMD, sections 3.2 and 3.5
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


# GSPMD, section 3.2: "An Einsum can be expressed as a string equation, e.g., ABC, ACD -> ABD, where A is an
# embarrassingly parallel batch dimension in both operands and the output, C is a contracting dimension that only exist
# in the operands and will be sum-reduced, while B and D are non-contracting dimensions that exist in one operand and
# are inherited by the output"
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


# GSPMD, sections 4.3 and 4.4: Convolution "allows the user to combine spatial partitioning and feature partitioning in
# the same operator"; Appendix A.2: "If the RHS is replicated, window dilation only affects the effective window size
# when the operator is partitioned based on its LHS"; "Base dilation is the dilation factor of the LHS"; section 4.1,
# "Static operator configurations".  native_functions.yaml: conv2d(Tensor input, Tensor weight, Tensor? bias=None,
# SymInt[2] stride=1, ...).  Window extent: d(k - 1) + 1
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


# GSPMD, sections 4.1 and 4.3, ReduceWindow.  native_functions.yaml: max_pool2d(Tensor self, int[2] kernel_size, int[2]
# stride=[], ...).  Window extent: d(k - 1) + 1
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


# GSPMD, section 4.3: "Halo exchange for data formatting ... after a Slice or Pad operator, the shape of the tensor
# changes, and so do the boundaries between partitions.  This requires us to realign the data on different partitions,
# which can be handled as a form of halo exchange (Figure 5b)".  native_functions.yaml: pad(Tensor self, SymInt[] pad,
# str mode="constant", float? value=None); constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0)
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


# GSPMD, section 4.3.  native_functions.yaml: slice(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None,
# SymInt step=1); select(Tensor(a) self, int dim, SymInt index); unsqueeze(Tensor(a) self, int dim)
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


# GSPMD, section 4.5: "GSPMD always produces a valid partitioned graph regardless of what sharding annotations are
# provided.  If the provided shardings are not the typical supported cases, the partitioner will perform resharding on
# the inputs and/or outputs"; section 3.1: "All devices have the same full data"
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


# GSPMD, section 4.6: "Similar optimizations include merging a sequence of Pad and Slice operators, which is important
# to partition the pipelined program in Section 3.3 where the shifting using Pad and Slice can be done with a single
# CollectivePermute"; section 3.3: "pad_left(state, 1)[:-1]"
def _merged(pad, padding, slicing):
    pads, fill = {w.label: w for w in padding.windows}, padding.windows[0].fill if padding.windows else 0
    windows = [w._replace(low=w.low + (pads.pop(w.label).low if w.label in pads else 0), fill=fill)
               for w in slicing.windows]
    windows += [w for label, w in pads.items() if label in slicing.out]

    def configure(sharded):
        return tuple(padding.configure(sharded)) + tuple(slicing.configure(sharded))
    return slicing._replace(operands=padding.operands, windows=tuple(windows), merged=pad, configure=configure)


# GSPMD, section 3.5: it "runs the propagation pass on the entire graph in multiple iterations, and alternates between
# forward propagation (input to output) and backward propagation (output to input)"; it "preserves initial user
# annotations" and "changes the sharding on a tensor only when it finds a more fine-grained sharding (possibily by
# combining with existing compatible sharding)"; "Elementwise operators are assigned the highest priority to propagate
# through in both directions"; "Operators like Dot ... are assigned lower priority".  A dimension of size s takes mesh
# dimension m only where s mod D_m = 0
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


# GSPMD, section 4.5: it "uses AllGather to replicate data along sharded dimensions, AllToAll to switch sharded
# dimensions, CollectivePermute to change device order, and DynamicSlice to shard replicated dimensions", and "could use
# multiple steps of resharding to reach the desired sharding".  MPICH allgather_intra_ring.c: "First, load the "local"
# version in the recvbuf"; alltoall_intra_pairwise.c: "Make local copy first" (MPIR_Localcopy)
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


# GSPMD, Appendix A.2: "we calculate the maximum sizes of left and right halos across partitions and perform the halo
# exchange of the maximum size"; the halo sizes a "linear function of the partition ID"; "we can use the maximum low
# padding on all partitions to ensure that each partition calculates all required windows, then perform a DynamicSlice
# on the output of the partitioned operator to remove unnecessary data".
# Stride s, extent e, low padding p: output j reads [j s - p, j s - p + e); partition q reads from q o s - p for
# (o - 1) s + e.  Transposed, base dilation s, kernel reach d(k - 1): output j takes operand i where i s + t d = j + p,
# so partition q reads from floor((q o + p - d(k - 1)) / s)
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


# GSPMD, section 4.3: "We use the CollectivePermute operator to exchange halo data between partitions"; "GSPMD uses
# maximum halo size across partitions, then uses DynamicSlice to remove excessive data in the halos"; Figure 9b:
# "Concatenate exchanged left and right halos", "DynamicSlice on the region actually needed", "Mask out invalid regions
# with the identity value"; Appendix A.2: "some partitions may include garbage values (e.g., halos from out-of-range
# input data), so we apply masking as described in Section 4.1"; section 4.1: "a Select according to a mask calculated
# by comparing Iota and an offset based on PartitionId".  Section 4.5: "DynamicSlice to shard replicated dimensions".
# MPICH MPIR_Localcopy
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


# GSPMD, section 2.1: "AllReduce when the layer sums over a partitioned dimension"; section 3.2: "gradients will be
# communicated with ReduceScatter instead of AllReduce", "unsharded on demand"; section 5.1: "replace the AllReduce on
# the output BSM with a ReduceScatter"; Figure 6: "AllGather non-matching Bx".  A bias b after a contraction partial
# along k shards: sum_q (x_q W_q + b) = x W + k b.
# XLA spmd_partitioner.cc, PartitionedHlo::ReshardAsWindowedInput: "for (int dim = 0; dim <
# base_shape_.dimensions().size(); ++dim)", "if (shard_count == 1 ...) continue", "visiting_hlo = *resharded"
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


# GSPMD, section 3: "sharding completion and per-operator partitioning"; section 3.1: "All devices have the same full
# data".  A dimension over a mesh dimension of one device: s / 1 = s
def partition(gm, placement, nodes):
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
        ir.sharding[names[name]] = tuple(m if m >= 0 and ir.shape[m] > 1 else -1 for m in mapping)
        annotated.add(names[name])
    # GSPMD, section 4.5: "DynamicSlice to shard replicated dimensions".  torch 2.14.0 torch/export/exported_program.py,
    # ExportedProgram: "a state_dict containing tensor values of all lifted parameters and buffers"
    replicated = mesh_graph.forward(ir.order, lambda node: node.all_input_nodes,
                                    lambda node, operands: node.op == 'get_attr' or node.op == 'call_function'
                                    and bool(operands) and all(operands) and _aliases(ir.locals[node]))
    annotated |= {node for node, whole in replicated.items() if whole and ir.sharding[node] is not None}
    _complete(ir, operators, annotated)
    _partition(ir, operators)
    ir.producer = {item[2]: g for g, item in enumerate(ir.program) if item[0] != 'output'}
    items = {item[2]: item for item in ir.program if item[0] != 'output'}
    ir.sent = mesh_graph.reach([item[1][0] for item in ir.program if item[0] == 'collective'],
                               lambda key: items[key][1][:1] if _sliced(items[key]) else ())
    _forms(ir)
    # GSPMD, section 4.5, as above
    bases = mesh_graph.forward(list(items), lambda key: items[key][1],
                               lambda key, reads: _bases(ir, items[key], reads))
    ir.attributes = {key: items[key] for key, base in bases.items() if base is not None}
    read = {k for _, reads, key, _ in ir.program if key not in ir.attributes for k in reads if k in ir.attributes}
    ir.loads = {k for k in read if _sliced(ir.attributes[k])} \
        | {b for k in read if not _sliced(ir.attributes[k]) for b in bases[k]}
    _inside(ir)
    ir.places = {device: _places(ir, device) for device in ir.coords}
    _over(ir)
    return ir


def _call(local, args, kwargs):
    return local.target(*args, **kwargs)


# ExecuTorch, "Concepts", Out variant: "Instead of allocating returned tensors in kernel implementations, an operator's
# out variant will take in a pre-allocated tensor to its out kwarg, and store the result there".  PyTorch, "Meta
# device": "you cannot perform data-dependent operations like torch.nonzero() or item()".  torch 2.14.0
# torch/_library/_out_variant.py: to_out_variant, "Given a functional operator overload, return its corresponding out
# variant"; _is_functional, "no argument is written to and the name doesn't end with '_'"; `if len(mutable_args) !=
# len(schema.returns): continue`; _has_valid_out_variant_returns, "Each return must alias exactly one mutable arg, in
# order".  torchgen/native_function_generation.py: `name="out" if len(func.returns) == 1 else f"out{i}"`
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


def _viewed(item):
    return item[0] == 'window' and len(item[1]) == 1 and item[3][6] is None


def _sliced(item):
    return item[0] in ('slice', 'narrow') or _viewed(item)


# GSPMD, section 4.5, "DynamicSlice to shard replicated dimensions"
def _bases(ir, item, reads):
    kind, _, key, operand = item
    if kind == 'constant':
        return frozenset((key,)) if isinstance(key[0].meta.get('val'), torch.Tensor) else None
    if reads and None not in reads and (_sliced(item) or kind == 'call' and key in ir.metas
                                        and _aliases(ir.locals[operand])):
        return frozenset().union(*reads)
    return None


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


# GSPMD, section 4.4; section 3.1: "GSPMD preserves the order of devices in device_mesh"
def _groups(ir, dims):
    groups = {}
    for device in sorted(ir.coords, key=ir.coords.get):
        groups.setdefault(tuple(c for m, c in enumerate(ir.coords[device]) if m not in dims), []).append(device)
    return list(groups.values())


# Item g: reads at 3g, arrivals at 3g + 1, writes at 3g + 2
def _readers(ir, roots, default):
    spans = [(3 * g, 3 * g + (item[0] == 'collective' and item[3].kind != 'all-reduce'))
             for g, item in enumerate(ir.program) if any(ir.over.get(k, frozenset()) & roots for k in item[1])]
    return (min(s for s, _ in spans), max(e for _, e in spans)) if spans else (default, default)


# J. Gibbons, J. Functional Programming, 1996; L. Meertens, Algorithmics, CWI Monographs 1, 1986: f(a ++ b) =
# f(a) (+) f(b), (+) associative and commutative
def _consumer(ir, roots):
    readers = [(item, k) for item in ir.program for k in item[1] if ir.over.get(k, frozenset()) & roots]
    if readers and all(item[0] == 'call' and isinstance(ir.locals[item[3]].target, torch._ops.OpOverload)
                       and ir.locals[item[3]].target.overloadpacket in ELEMENTWISE
                       and _shape(ir, k) == _shape(ir, item[2]) for item, k in readers):
        return mesh.PAGES
    return mesh.JOIN


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


# GSPMD, section 4.5, DynamicSlice
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


# MPI 4.1, section 7.8, in place: "Data sent and received must have the same type map"
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


# torch 2.14.0 torch/_tensor_docs.py, view: "Returns a new tensor with the same data as the self tensor but of a
# different shape.  The returned tensor shares the same data"; "When it is unclear whether a view can be performed, it
# is advisable to use reshape, which returns a view if the shapes are compatible, and copies (equivalent to calling
# contiguous) otherwise"
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


# torch 2.14.0 torch/_prims/__init__.py, collapse_view: "The dimensions can be merged if and only if they are all
# "nested" with each other.  That is, they all have the property that stride[i] = stride[i+1] * shape[i+1]".  MPI 4.1,
# section 6.1.3, MPI_TYPE_CREATE_SUBARRAY: "IN ndims: number of array dimensions (positive integer)"
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


# MPI 4.1, section 3.7: "The sender should not modify any part of the send buffer after a nonblocking send operation is
# called, until the send completes"
def _produced(ir, d, i, key, received):
    place = ir.places[d][key]
    at = 3 * ir.producer[place.writer] + 2
    return [(d, i, 'P', at, at), (d, i, 'S') + _readers(ir, {place.owner, key}, received)]


def _send(ir, i, key):
    ir.publishes.setdefault(ir.producer[ir.places[ir.device][key].writer], []).append(i)


# NCCL User Guide, In-place Operations: "For ncclReduceScatter and ncclAllGather, in place operations are done when the
# per-rank pointer is located at the rank offset of the global buffer"; MPI 4.1, section 7.7, in place: "the input data
# of each MPI process is assumed to be in the area where that MPI process would receive its own contribution to the
# receive buffer"
def _inside(ir):
    for kind, reads, key, operand in ir.program:
        if kind == 'copy' and operand is None:
            target = ir.taken.get(reads[0], reads[0])
            producer, _, _, collective = ir.program[ir.producer[ir.results.get(target, target)]]
            if target not in ir.inside and (target in ir.results or producer == 'call'
                                            and isinstance(ir.forms.get(target), tuple)
                                            or producer == 'collective' and collective.kind == 'reduce-scatter'):
                ir.inside[target] = ir.inside[key]


# G. Wang et al., Blink, MLSys 2020, section 3.3: "to do an AllReduce operation on the directed graph, we first run a
# reduce operation to a chosen root vertex using the undirected graph and then do a broadcast operation from the root
# vertex using the same tree but with links going in the reverse direction".  MSCCLang, section 4.2: it "receives a
# chunk, reduces it with a source chunk, and locally copies it to the destination".  GSPMD, section 3.1: "GSPMD
# preserves the order of devices in device_mesh"
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


# GSPMD, section 4.2: it "concatenates tensors from all participants following a specified order"; NCCL: "receiving data
# from rank `i` at offset `i*sendcount`"
def _all_gather(ir, c, groups, key, made):
    d, received = c.dim, 3 * c.position + 1
    n = _shape(ir, key)[d]
    last = max(received, _readers(ir, {made}, received)[1])
    for group in groups:
        ranges = [ir.transfer(e, [f for f in group if f != e], mesh.JOIN, _strided(ir, ir.places[e][key]),
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


# MPI 4.1, section 7.8, MPI_ALLTOALL: "The j-th block sent from MPI process i is received by MPI process j and is placed
# in the i-th block of recvbuf"
def _all_to_all(ir, c, groups, key, made):
    (i, j), k = c.dim, len(groups[0])
    ni, nj, received = _shape(ir, key)[i], _shape(ir, key)[j] // k, 3 * c.position + 1
    _exchange(ir, c, groups, [(p, q) for p in range(k) for q in range(k)], key,
              lambda e, q: _narrowed(ir.places[e][key], j, q * nj, nj),
              lambda f, p, q: (_narrowed(ir.places[f][made], i, p * ni, ni), made, None), mesh.JOIN,
              max(received, _readers(ir, {made}, received)[1]))


# GSPMD, section 4.2: it "is semantically equivalent to an AllReduce followed by a DynamicSlice where each partition
# gets one slice of the fully reduced data.  An efficient implementation has only half the cost of AllReduce".  Blink,
# section 3.3: "we first run a reduce operation to a chosen root vertex".  MSCCLang, section 4.2
def _reduce_scatter(ir, c, groups, key, made):
    dim, k = c.dim, len(groups[0])
    n, received = _shape(ir, key)[dim] // k, (c.node, 'received', c.position)
    shape = _layout(ir, key)[1]
    ir.shapes[received], ir.added[received] = (k - 1,) + shape[:dim] + (n,) + shape[dim + 1:], None
    _exchange(ir, c, groups, [(p, q) for p in range(k) for q in range(k)], key,
              lambda e, q: _narrowed(ir.places[e][key], dim, q * n, n),
              lambda f, p, q: (_part(ir, received, p - (p > q)), received, (received, p - (p > q))), mesh.PAGES,
              3 * c.position + 1)


# GSPMD, section 4.2: it "exchanges data among a list of source-destination pairs".  OpenXLA, Operation semantics,
# CollectivePermute (https://openxla.org/xla/operation_semantics): "For each pair, the operand is sent from source
# replica to target replica"; "Any two pairs should not have the same target replica id, and they should not have the
# same source replica id"
def _collective_permute(ir, c, groups, key, made):
    pairs = tuple(tuple(pair) for pair in c.dim)
    places, k = [p for pair in pairs for p in pair], len(groups[0])
    if not all(isinstance(p, int) and 0 <= p < k for p in places) \
            or len({p for p, _ in pairs}) != len(pairs) or len({q for _, q in pairs}) != len(pairs):
        raise ValueError('%s: a CollectivePermute\'s pairs %s over a group of %d: each a (source, target) place, no two '
                         'with one source or one target (XLA\'s source_target_pairs)' % (c.node.name, pairs, k))
    landed = ir.inside[made][0]
    _exchange(ir, c, groups, pairs, key, lambda e, q: ir.places[e][key],
              lambda f, p, q: (ir.places[f][made], landed, None), mesh.JOIN,
              max(3 * c.position + 1, _readers(ir, {landed}, 3 * c.position + 1)[1]))


# S. S. Bhattacharyya, S. Sriram and E. A. Lee, "Optimizing synchronization in multiprocessor DSP systems", IEEE Trans.
# Signal Processing 45(6), 1997, section 2: "retain the processor assignment and actor ordering specified by the fully
# static schedule ... Each processor executes the actors assigned to it in the order specified at compile time".
# Event a over [a.first, a.last] precedes b where a.last < b.first, lag 0; each event of execution t precedes each of
# execution t + 1, lag 1
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


# GSPMD, section 4.2
LOWERINGS = {'all-reduce': _all_reduce, 'all-gather': _all_gather, 'all-to-all': _all_to_all,
             'reduce-scatter': _reduce_scatter, 'collective-permute': _collective_permute}


# torch 2.14.0 torch/distributed/_functional_collectives.py, AsyncCollectiveTensor: "A Tensor wrapper subclass that is
# used to trigger a call to wait prior to first use of the underlying tensor"
def lower(ir):
    for item in ir.program:
        if item[0] == 'collective':
            collective, first = item[3], len(ir.declared)
            LOWERINGS[collective.kind](ir, collective, _groups(ir, collective.dims), item[1][0], item[2])
            ir.declares[collective] = range(first, len(ir.declared))
    firsts = mesh_graph.ends(range(len(ir.program))[::-1], lambda g: ir.program[g][1])
    for kind, _, key, collective in ir.program:
        if kind == 'collective':
            ir.waits.setdefault(firsts.get(key, len(ir.program) - 1), []).append(collective)
    ir.dependences = _dependences(ir)
    _place(ir)


# ExecuTorch, exir/memory_planning.py, update_tensor_lifetime: "A tensor's lifetime are represented by the index of the
# first and last node referring that tensor in its inputs/outputs."  MPI 4.1, section 3.7: "The receiver should not
# access any part of the receive buffer after a nonblocking receive operation is called, until the receive completes"
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


def unit(ir):
    return [mesh.Weight(1.0, 1.0, 1.0, 1.0) for _ in ir.declared]


CONTEXT = []


def _context():
    if not CONTEXT:
        context = mesh.Context()
        mesh.check(mesh.LIB.mesh_attach(C.byref(context), mesh.NAME))
        CONTEXT.append(context)
    return CONTEXT[0]


def _group():
    return int(os.environ['MESH_GROUP'], 0)


def _program():
    return sum(1 << n for n in {int(item, 10) for item in os.environ['MESH_PROGRAM'].split(',')})


HOST_LAG = 2


def _wait(ready, progress, what):
    seen, since = progress(), time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    while not ready():
        now, value = time.clock_gettime_ns(time.CLOCK_MONOTONIC), progress()
        if value != seen:
            seen, since = value, now
        elif now - since > mesh.PLATFORM_NS:
            raise OSError(errno.ETIMEDOUT, what)


# libfabric, man/fi_endpoint.3.md, fi_cancel: "An error queue entry will be available in the associated error queue
# with error code FI_ECANCELED"
def _arrived(context, plan, transfer, t, count):
    def ready():
        status = mesh.LIB.mesh_host_arrived(C.byref(context), plan, transfer, t)
        if status == mesh.FAILED:
            raise OSError(errno.ECANCELED, 'execution %d of declared transfer %d: a field ahead, or the range cancelled '
                                           '(mesh-collective.h mesh_host_arrived)' % (t, transfer))
        return status == 1
    _wait(ready, lambda: count.value, 'execution %d of declared transfer %d: no landing for MESH_PLATFORM_NS'
          % (t, transfer))


TORCH = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(mesh.__file__)), 'libmesh-torch.dylib'))
TORCH.mesh_torch_publish.restype, TORCH.mesh_torch_publish.argtypes = C.c_int, [C.c_void_p, C.c_void_p, mesh.Q,
                                                                                C.POINTER(mesh.U), mesh.U, mesh.Q]
TORCH.mesh_torch_notifying.restype, TORCH.mesh_torch_notifying.argtypes = mesh.Q, []
PUBLISH = C.cast(mesh.LIB.mesh_host_publish, C.c_void_p)
STARTED = C.cast(mesh.LIB.mesh_host_started, C.c_void_p)
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


# torch 2.14.0 include/ATen/dlpack.h, DLPACK_MAJOR_VERSION 1, DLPACK_MINOR_VERSION 3: DLManagedTensor "is deprecated
# after DLPack v0.8 Use DLManagedTensorVersioned instead"
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
# dlpack.h DLDeviceType: kDLCPU ("CPU device") 1, kDLMetal ("Metal for Apple GPU") 8.  DLDataTypeCode: kDLInt 0,
# kDLUInt 1, kDLFloat 2, kDLBfloat 4, kDLComplex 5, kDLBool 6, kDLFloat8_e4m3fn 10, kDLFloat8_e4m3fnuz 11,
# kDLFloat8_e5m2 12, kDLFloat8_e5m2fnuz 13, kDLFloat8_e8m0fnu 14, kDLFloat4_e2m1fn 17.  torch 2.14.0
# aten/src/ATen/DLConvertor.cpp getDLDataType: Float4_e2m1fn_x2's "dtype.lanes = 2" and "dtype.bits = 4";
# include/ATen/DLConvertor.h DLPackTraits<DLManagedTensorVersioned> `capsule`
DEVICES, VERSION, CAPSULE = {'cpu': 1, 'mps': 8}, DLPackVersion(1, 3), b'dltensor_versioned'
CODES = {torch.int8: 0, torch.int16: 0, torch.int32: 0, torch.int64: 0, torch.uint8: 1, torch.uint16: 1,
         torch.uint32: 1, torch.uint64: 1, torch.float16: 2, torch.float32: 2, torch.float64: 2, torch.bfloat16: 4,
         torch.complex32: 5, torch.complex64: 5, torch.complex128: 5, torch.bool: 6, torch.float8_e4m3fn: 10,
         torch.float8_e4m3fnuz: 11, torch.float8_e5m2: 12, torch.float8_e5m2fnuz: 13, torch.float8_e8m0fnu: 14}
TYPES = {dtype: (code, dtype.itemsize * 8, 1) for dtype, code in CODES.items()}
TYPES[torch.float4_e2m1fn_x2] = (17, 4, 2)
CAPSULE_NEW = C.PYFUNCTYPE(C.py_object, C.c_void_p, C.c_char_p, C.c_void_p)(('PyCapsule_New', C.pythonapi))
# objc/message.h: "These functions must be cast to an appropriate function pointer type before being called".
# MTLDevice.h, newBufferWithBytesNoCopy:length:options:deallocator:: "Create a buffer by wrapping an existing part of
# the address space"
OBJC = C.CDLL('/usr/lib/libobjc.A.dylib')
OBJC.sel_registerName.restype, OBJC.sel_registerName.argtypes = C.c_void_p, [C.c_char_p]
SEND = C.cast(OBJC.objc_msgSend, C.c_void_p).value
GET = C.CFUNCTYPE(C.c_void_p, C.c_void_p, C.c_void_p)(SEND)
PUT = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p)(SEND)
WRAP = C.CFUNCTYPE(C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_ulong, C.c_ulong, C.c_void_p)(SEND)
SELECTOR = {name: OBJC.sel_registerName(name) for name in
            (b'device', b'retain', b'release', b'newBufferWithBytesNoCopy:length:options:deallocator:')}
# MTLResource.h: MTLResourceStorageModeShared = MTLStorageModeShared << MTLResourceStorageModeShift,
# MTLStorageModeShared 0
SHARED = 0
# MTLResource.h, device: "The device this resource was created against.  This resource can only be used with this
# device".  torch 2.14.0 include/ATen/native/mps/OperationUtils.h, getMTLBufferStorage:
# `__builtin_bit_cast(id<MTLBuffer>, tensor.storage().data())`; iter_tensor_offset: "MPS storage data is not the real
# GPU pointer, but rather a pointer to id<MTLBuffer> object"; torch/csrc/StorageMethods.cpp THPStorage_dataPtr:
# `self_.mutable_data()`
DEVICE = []


def _metal_device():
    if not DEVICE:
        probe = torch.empty(1, device='mps')
        device = GET(probe.untyped_storage().data_ptr(), SELECTOR[b'device'])
        if not device:
            raise OSError(errno.ENODEV, 'torch\'s MPS buffer names no MTLDevice (MTLResource.h `device`)')
        DEVICE.append(device)
    return DEVICE[0]


# G. E. Collins, A method for overlapping and erasure of lists, CACM 3(12), 1960.  dlpack.h, DLManagedTensorVersioned
# `deleter`: "This should be called to destruct manager_ctx which holds the DLManagedTensorVersioned"
class _Mapping:
    def __init__(self, view):
        self.view, self.tensors, self.retired = view, 0, False

    def release(self):
        if self.retired and not self.tensors and self.view.address:
            mesh.LIB.mesh_view_release(self.view)
            self.view = mesh.View()


def _mapped(context, functions, count, values):
    header, view = mesh.Header.from_address(context.M), mesh.View()
    offsets = (mesh.Q * max(sum(len(v) for v in values), 1))()
    mesh.check(mesh.LIB.mesh_view_values(C.byref(context), functions, count, offsets, C.byref(view)))
    at, addresses = 0, []
    for function in values:
        addresses.append([[view.address + offsets[at + i] + c * v.section.pages * header.pgsz
                           for c in range(v.section.count)] for i, v in enumerate(function)])
        at += len(function)
    return view, addresses


HELD = {}


# dlpack.h, `deleter`: "The destructor deletes the argument self as well".  torch 2.14.0
# torch/csrc/utils/tensor_new.cpp, tensor_fromDLPackImpl: `tensor->deleter(tensor)`
@DELETER
def _delete(managed):
    held = HELD.pop(C.addressof(managed.contents), None)
    if held:
        if held[0].manager_ctx:
            PUT(held[0].manager_ctx, SELECTOR[b'release'])
        held[3].tensors -= 1
        held[3].release()


# torch 2.14.0 include/ATen/native/mps/OperationUtils.h, needsGather: `!is_macOS_15_0_or_newer && (!t.is_contiguous() ||
# t.storage_offset())`
def _placed_buffer(view):
    buffer = WRAP(_metal_device(), SELECTOR[b'newBufferWithBytesNoCopy:length:options:deallocator:'], view.address,
                  view.bytes, SHARED, None)
    if not buffer:
        raise OSError(errno.ENOMEM, 'Metal made no buffer over the program\'s mapping, %d bytes (MTLDevice.h '
                                    'newBufferWithBytesNoCopy)' % view.bytes)
    return buffer


# torch 2.14.0 torch/utils/dlpack.py: "The returned PyTorch tensor will share the memory with the input tensor".
# aten/src/ATen/DLConvertor.cpp: dlDeviceToTorchDevice, `at::Device(DeviceType::MPS, index)`; toDLPackImpl,
# `byte_offset` "src.storage_offset() * c10::elementSize(src.scalar_type())".  dlpack.h: "For vanilla CPU memory ...
# this is set to 0"; strides "can not be NULL if ndim != 0"; "By default the flags should be set to 0"
def _dlpack_tensor(device, data, offset, owner, dtype, shape, mapping):
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


# torch 2.14.0 torch/_torch_docs.py, as_strided: "the offset in the underlying storage of the output tensor"
def _bind(context, ir, values, addresses, buffer, mapping):
    copies = []
    for v, (dtype, numel, device), at in zip(values, ir.layouts, addresses):
        copies.append([_dlpack_tensor('cpu', a, 0, None, dtype, (numel,), mapping) if device == 'cpu' else
                       _dlpack_tensor('mps', buffer, a - mapping.view.address, buffer, dtype, (numel,), mapping)
                       for a in at])
    places = {key: [copy.as_strided(sizes, strides, copy.storage_offset() + offset) for copy in copies[v]]
              for key, (v, offset, sizes, strides) in ir.homes.items()}
    given = {i: ([copy.view(shape) for copy in copies[v]],
                 [copy.view(shape) if ir.layouts[v][2] == 'cpu' else
                  _dlpack_tensor('cpu', a, 0, None, ir.layouts[v][0], shape, mapping) for copy, a in
                  zip(copies[v], addresses[v])]) for i, (v, shape) in ir.inputs.items()}
    return places, {i: ir.layouts[d.value][2] for i, d in enumerate(ir.declared) if d.value != mesh.ABSENT}, given


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


def _apply(target, arguments, env, outs):
    args, kwargs = arguments(env)
    return target(*args, **kwargs, **outs)


def _aliases(local):
    return local.target is operator.getitem or isinstance(local.target, torch._ops.OpOverload) \
        and any(r.alias_info is not None for r in local.target._schema.returns)


def _entry(context, state, transfer):
    return mesh.Publication.from_address(context.M + state.transfers + transfer * mesh.PUBLICATION_BYTES)


def _count(context, state, word, c):
    return C.c_uint64.from_address(context.M + word + 8 * (c % state.copies))


def _called(ir, key, operands, local, form, placed):
    given = iter(operands)
    arguments, dynamic = _bound((local.args, local.kwargs), lambda n: next(given))
    mine = placed(key)
    if isinstance(form, tuple):
        outs = {name: placed(result) for name, result in form[1]}
        return tuple(outs.values()) if len(outs) > 1 else mine, \
            [lambda env, inputs, t: _apply(form[0], arguments, env, outs)]
    if key in ir.contiguous:
        base = operands[_based(local)]
        laid = mine.view(base.shape)
        return mine, [lambda env, inputs, t: laid.copy_(base)]
    if mine is not None and ir.places[ir.device][key].owner == key:
        return mine, [lambda env, inputs, t: mine.copy_(_apply(local.target, arguments, env, {}))]
    if mine is not None:
        return mine, []
    if not dynamic and _aliases(local):
        return _apply(local.target, arguments, None, {}), []
    return _Dynamic(key), [lambda env, inputs, t: env.__setitem__(key, _apply(local.target, arguments, env, {}))]


def _collected(ir, context, plan, state, placed, c, item, operands):
    _, reads, key, collective = item
    me, receives = ir.roles[collective]
    if collective.kind == 'reduce-scatter' or collective.kind == 'all-reduce' and me == 0:
        waits = [lambda env, inputs, t, i=i, count=_count(context, state, _entry(context, state, i).count, c):
                 _arrived(context, plan, i, t, count) for i, _ in receives]
        total, first, steps = placed(key), operands[0], []
        if collective.kind == 'reduce-scatter':
            n = first.shape[collective.dim] // (len(receives) + 1)
            first = first.narrow(collective.dim, me * n, n)
        for wait, (_, (received, j)) in zip(waits, receives):
            steps += [wait, lambda env, inputs, t, a=first, b=placed(received)[j]: torch.add(a, b, out=total)]
            first = total
        return steps
    if collective.kind == 'collective-permute':
        place, pairs, places = placed(key), [tuple(pair) for pair in collective.dim], ir.places[ir.device]
        if (me, me) in pairs:
            return [] if places[reads[0]][:4] == places[key][:4] else \
                [lambda env, inputs, t, a=operands[0]: place.copy_(a)]
        if me not in [q for _, q in pairs]:
            return [lambda env, inputs, t: place.zero_()]
    return []


# torch 2.14.0 torch/distributed/_functional_collectives.py, wait_tensor: "Wait on a tensor returned by the collectives
# ops".  libfabric, man/fi_cntr.3.md: "Counters record the number of requested operations that have completed";
# "fi_cntr_readerr: Reads the number of operations which have completed in error."
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


# GSPMD, section 4.5, "DynamicSlice to shard replicated dimensions"; torch 2.14.0 torch/export/exported_program.py,
# ExportedProgram: "a state_dict containing tensor values of all lifted parameters and buffers"
def _hosted(ir, item, values):
    kind, reads, _, operand = item
    if kind == 'constant':
        return operator.attrgetter(operand)(ir.gm)
    if kind == 'call':
        given, local = iter(values), ir.locals[operand]
        return _call(local, *map_arg((local.args, local.kwargs), lambda n: next(given)))
    return values[0].narrow(*_cut(ir, kind, operand, _layout(ir, reads[0])[1], ir.device))


# MPICH MPIR_Localcopy; torch 2.14.0 torch/_tensor_docs.py, copy_: "Copies the elements from src into self tensor and
# returns self"; MPI_IN_PLACE
def _copied(ir, read, key, cut, source, target):
    if ir.places[ir.device][key].writer != key:
        return []
    at = None if cut is None else _cut(ir, 'slice', cut, _layout(ir, read)[1], ir.device)
    if isinstance(source, _Dynamic):
        if at is None:
            return [lambda env, inputs, t: target.copy_(env[read])]
        return [lambda env, inputs, t: target.copy_(env[read].narrow(*at))]
    source = source if at is None else source.narrow(*at)
    return [lambda env, inputs, t: target.copy_(source)]


def _specialized(ir, context, plan, state, devices, placed, c, item, operands, steps):
    kind, reads, key, operand = item
    value = placed(key)
    for collective in ir.waits.get(ir.producer[key], ()):
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
        steps += [lambda env, inputs, t, part=value.narrow(dim, start, extent): part.fill_(fill)
                  for start, extent in ((0, below), (above, length - above)) if fill is not None and extent]
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
    steps += _publishes(context, plan, devices, ir.publishes.get(ir.producer[key], ()))
    return value


def _instantiate(ir, context, plan, devices, tensors, c, spec):
    state, steps = mesh.State.from_address(context.M + plan), []
    items = {item[2]: item for item in ir.program if item[0] != 'output'}

    def placed(key):
        return tensors[key][c % len(tensors[key])] if key in tensors else None
    values = mesh_graph.forward(list(items), lambda key: items[key][1],
                                lambda key, operands: _specialized(ir, context, plan, state, devices, placed, c,
                                                                   items[key], operands, steps))
    for collective in ir.waits.get(len(ir.program) - 1, ()):
        steps += _completed(ir, context, plan, state, c, collective)
    node, keys = next(item[3] for item in ir.program if item[0] == 'output')
    arguments = _bound(node.args[0], lambda n: values[keys[n]])[0]
    return steps, lambda env: pytree.tree_unflatten(list(arguments(env)), spec)


# Copy c of value v at execution t: c = t mod count_v; period lcm_v(count_v)
def _steps(ir, context, plan, tensors, devices, spec):
    if ir.device is None:
        return [[]], [lambda env: None]
    made = [_instantiate(ir, context, plan, devices, tensors, c, spec)
            for c in range(math.lcm(1, *(len(x) for x in tensors.values())))]
    return [steps for steps, _ in made], [result for _, result in made]


# P. E. McKenney and J. D. Slingwine, Read-copy update, PDCS 1998
def _retire(context, state, deadline):
    if 'mps' in state['devices']:
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


# The Python documentation, weakref.finalize: "Return a callable finalizer object which will be called when obj is
# garbage collected"; func, args and kwargs must "not own any references to obj"; "When the program exits, each
# remaining live finalizer is called unless its atexit attribute has been set to false"
def _dropped(context, state):
    if state['mapping'] is not None:
        _retire(context, state, time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                + (0 if state['timeout'] is not None else mesh.PLATFORM_NS))


# MPI 4.1, section 3.9, MPI_START: "The request becomes active once the call is made. [...] The communication buffer
# should not be modified after the call, and until the operation completes."  NCCL User Guide, CUDA Stream Semantics:
# "NCCL calls are associated to a stream which is passed as the last argument of the collective communication function.
# The NCCL call returns when the operation has been effectively enqueued to the given stream, or returns an error.  The
# collective operation is then executed asynchronously on the CUDA device."
def _started(context, plan, metal):
    if metal:
        array = (mesh.U * 1)(0)
        return lambda t: mesh.check(TORCH.mesh_torch_publish(STARTED, C.addressof(context), plan, array, 1, t))
    return lambda t: mesh.LIB.mesh_host_started(C.byref(context), plan, 0, t)


# torch 2.14.0 torch/fx/graph.py, _PyTreeCodeGen: process_inputs, `pytree.arg_tree_leaves(*inputs)`; process_outputs,
# `pytree.tree_unflatten(out, self.pytree_info.out_spec)`
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


# P. Barham et al., Pathways, MLSys 2022, section 3: "a new program tracer (Figure 2) that a user can wrap around a
# block of Python code that calls many compiled functions.  The tracer generates a single PATHWAYS program where each
# compiled function is represented by a computation node in a dataflow graph".  J. Reed, Z. DeVito, H. He, A. Ussery and
# J. Ansel, torch.fx, MLSys 2022
class Function(torch.nn.Module):
    def __init__(self, module, placement, weights):
        super().__init__()
        self.module, self.placement, self.weights = module, placement, weights


# torch 2.14.0 torch/fx/_symbolic_trace.py, Tracer.is_leaf_module: "Leaf modules are the atomic units that appear in the
# IR, referenced by call_module calls"
class Tracer(torch.fx.Tracer):
    def is_leaf_module(self, m, module_qualified_name):
        return isinstance(m, Function) or super().is_leaf_module(m, module_qualified_name)


# M. C. Rinard and P. C. Diniz, Commutativity analysis: a new analysis technique for parallelizing compilers, ACM TOPLAS
# 19(6), 1997, section 4.9.2: "simple arithmetic rewrite rules such as ex1-ex2 => ex1+(-ex2), -(-ex) => ex and
# ex1*(ex2+ex3) => (ex1*ex2)+(ex1*ex3)", and "rules such as (ex1+ex2)+ex3 => (ex1+ex2+ex3) that convert binary
# applications of commutative and associative operators to n-ary applications.  It then sorts the operands according to
# an arbitrary order on expressions"; footnote 4: "ignore[s] here potential anomalies caused by the finite
# representation of numbers"
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


# Rinard and Diniz, sections 3.2.1 and 4.9.1: symbolic execution "simply executes the operations, computing with
# expressions instead of values".  torch 2.14.0 torch/fx/interpreter.py: "executes an FX graph Node-by-Node"
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


# Rinard and Diniz, section 4.8: executed symbolically "in both execution orders", the simplified expressions "denote
# the same value"; "If it cannot symbolically execute the methods, the commutativity-testing algorithm conservatively
# assumes that the methods may not commute".  torch.fx TraceError: "symbolically traced variables cannot be used as
# inputs to control flow"
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


# A. J. Bernstein, Analysis of programs for parallel processing, IEEE Trans. Electronic Computers EC-15(5), 1966: i and
# j are independent where W(i) & R(j) = R(i) & W(j) = W(i) & W(j) = {}.  Rinard and Diniz, section 2: "can execute in
# any order without affecting the overall behavior".  Width by Dilworth's theorem, Fulkerson's matching, Hopcroft and
# Karp
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


# Pathways, section 4.2: "When the user wants to run a traced program, it calls the PATHWAYS client library which first
# assigns virtual devices to any computations that have not been run before, and registers the computations with the
# resource manager, triggering the servers to compile the computations"
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


def _metadata(value):
    return (tuple(value.shape), tuple(value.stride()), value.dtype, value.device) \
        if isinstance(value, torch.Tensor) else value


# torch 2.14.0 torch/export/__init__.py, export: it "produces a traced graph representing only the Tensor computation of
# the function in an Ahead-of-Time (AOT) fashion, which can subsequently be executed with different inputs or
# serialized"; torch/export/exported_program.py, run_decompositions: "If empty, we don't decompose any operator";
# ExportedProgram.module: "a self contained GraphModule with all the parameters/buffers inlined";
# torch/_subclasses/fake_impls.py, constructors: `new_kwargs["device"] = torch.device("meta")`; torch/_guards.py,
# detect_fake_mode: "fake mode ... doesn't match mode".  R. Frostig, M. J. Johnson and C. Leary, SysML 2018: "on a trace
# cache miss"
class _Export(torch.fx.Interpreter):
    def __init__(self, gm, mode):
        super().__init__(gm)
        self.mode, self.compiled, self.of, self.origin, self.calls = mode, [], {}, {}, []

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
            'operator.getitem of such a call\'s result and no output; nothing runs eagerly between compiled calls'
            % (n.name, n.op, n.target))

    def export(self, n, function, args, kwargs):
        flat, spec = pytree.tree_flatten((args, kwargs))
        signature = (spec, tuple(_metadata(v) for v in flat))
        found = [k for k, c in enumerate(self.compiled) if c.function is function and c.signature == signature]
        if not found:
            exported = torch.export.export(function.module, tuple(args), dict(kwargs), strict=False) \
                .run_decompositions({})
            gm = exported.module(check_guards=False)
            _RemoveRuntimeAssertionsPass()(gm)
            out = [a.meta['val'] if isinstance(a, Node) else a for a in gm.graph.output_node().args[0]]
            result = pytree.tree_unflatten([self.fake(v) for v in out], exported.call_spec.out_spec)
            self.compiled.append(Compiled(function, signature, gm, flat, result, exported.call_spec.out_spec))
            found = [len(self.compiled) - 1]
        self.of[n], self.origin[n] = found[0], len(self.calls)
        self.calls.append(found[0])
        return self.compiled[found[0]].result


def program(block, *example_inputs):
    tracer = Tracer()
    gm = torch.fx.GraphModule(tracer.root, tracer.trace(block))
    exporter = _Export(gm, FakeTensorMode())
    exporter.run(*[exporter.fake(x) for x in example_inputs])
    compiled, of, calls = exporter.compiled, exporter.of, exporter.calls
    entries = _analyse(gm, of, [record.function.module for record in compiled])
    operands = [record.function.placement(record.gm, record.examples) for record in compiled]
    placeholders = [n for n in gm.graph.nodes if n.op == 'placeholder']
    readers = [[of[n] for n in gm.graph.nodes if n in of and p in n.all_input_nodes] for p in placeholders]
    given = [(x.dtype, x.numel(), x.device.type, tuple(x.shape)) if isinstance(x, torch.Tensor) else None
             for x in example_inputs]
    constants = [None if isinstance(x, torch.Tensor) else x for x in example_inputs]
    context, state = _context(), {'stale': True, 'mapping': None, 'runs': None, 'devices': (), 'executions': 0,
                                  'ended': 0, 'inputs': [None] * len(given), 'timeout': None}

    def make():
        nodes = mesh.Nodes()
        mesh.check(mesh.LIB.mesh_nodes_read(C.byref(context), C.byref(nodes)))
        irs = []
        for k, (record, operand) in enumerate(zip(compiled, operands)):
            ir = partition(record.gm, operand, nodes)
            ir.calls = entries[k]
            lower(ir)
            ir.weights = list(record.function.weights(ir))
            irs.append(ir)
        for i, (x, functions) in enumerate(zip(given, readers)):
            k = next((k for k in functions if irs[k].device is not None), None)
            if x is not None and k is not None:
                irs[k].inputs[i] = (irs[k].value(*x[:3], 0, 0, 0, lag=HOST_LAG), x[3])
        arrays = [((mesh.Declared * len(ir.declared))(*ir.declared), (mesh.Weight * len(ir.weights))(*ir.weights),
                   (mesh.Value * len(ir.values))(*ir.values),
                   (mesh.Dependence * len(ir.dependences))(*ir.dependences)) for ir in irs]
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
            views = [mesh.State.from_address(context.M + f.plan) for f in functions]
            # JGroups ViewId.java: "The comparison is done on the IDs, if they are equal, we use the creator"
            if any((s.serial, s.sequencer, s.boot) != (nodes.serial, nodes.sequencer, nodes.boot) for s in views):
                raise OSError(errno.ESTALE, 'a plan\'s view is not the one its IR was built on: plan again')
            view, addresses = _mapped(context, functions, len(irs), [values for _, _, values, _ in arrays])
            state['mapping'].view = view
            buffer = _placed_buffer(view) if 'mps' in state['devices'] else None
            try:
                bound = [_bind(context, ir, values, at, buffer, state['mapping']) for ir, (_, _, values, _), at in
                         zip(irs, arrays, addresses)]
            finally:
                if buffer:
                    PUT(buffer, SELECTOR[b'release'])
            for ir, (places, _, _) in zip(irs, bound):
                loaded = ir.loads & places.keys()
                needed = mesh_graph.reach(loaded, lambda key: ir.attributes[key][1])
                host = mesh_graph.forward([key for key in ir.attributes if key in needed],
                                          lambda key: ir.attributes[key][1],
                                          lambda key, values: _hosted(ir, ir.attributes[key], values))
                for key in loaded:
                    places[key][0].copy_(host[key])
            runs = [dict(zip(('steps', 'results'), _steps(ir, context, f.plan, places, devices, record.spec)),
                         plan=f.plan, executions=0, state=mesh.State.from_address(context.M + f.plan),
                         started=_started(context, f.plan, any(layout[2] == 'mps' for layout in ir.layouts)))
                    for ir, f, record, (places, devices, _) in zip(irs, functions, compiled, bound)]
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
    run.inputs = inputs
    weakref.finalize(run, _dropped, context, state)
    return run
