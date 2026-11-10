# design/aot-compilation-change-request.md A1 to A4, A8 to A10 and appendix item 7;
# design/plan-to-bridge-change-request.md R1; Y. Xu et al., GSPMD, arXiv 2105.04663, 2021
import collections
import ctypes as C
import ctypes.util
import errno
import math
import operator
import os
import pickle
import sys
import time
import types
import weakref

import torch
import torch.utils._pytree as pytree
from torch._library._out_variant import _is_functional, get_out_arg_names, to_out_variant
from torch._prims_common import is_non_overlapping_and_dense_or_false
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch._subclasses.meta_utils import MetaConverter
from torch.fx.node import Node, map_arg
from torch.utils._mode_utils import no_dispatch

import mesh
import mesh_code
import mesh_disk
import mesh_graph

# torch 2.14.0 torch/export/__init__.py, export; torch/_ops.py, OpOverload
A = torch.ops.aten
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
Local = collections.namedtuple('Local', 'op target args kwargs annotation', defaults=(None,))
AttentionOrigin = collections.namedtuple('AttentionOrigin', 'query key page cuts shape offsets',
                                         defaults=(0, 0, (), (), ()))
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
        self.prototypes = {}
        self.homes, self.at, self.inside, self.added, self.owned = {}, {}, {}, {}, {}
        self.publishes, self.inputs, self.declares, self.waits = {}, {}, {}, {}
        self.attributes, self.loads, self.kinds, self.page, self.hull = {}, set(), {}, None, None
        self.taken, self.results, self.sent, self.contiguous = {}, {}, set(), set()
        self.given, self.landed = None, {}
        self.trims = {}
        self.initial, self.constant_values = None, set()

    def emit(self, item):
        self.program.append(item)
        if item[2] is not None:
            self.made.add(item[2])

    def describe(self, key, result):
        self.prototypes[key] = result
        if isinstance(result, torch.Tensor):
            self.metas[key] = result
        return result

    # GSPMD section 3.1
    def local(self, node, sharding):
        return tuple(-(-s // self.shape[m]) if m >= 0 else s for s, m in zip(_value(node).shape, sharding))

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
def _prototype(node):
    values = pytree.tree_leaves(node.meta.get('val'))
    return values[0] if values and all(isinstance(v, torch.Tensor) and v.shape == values[0].shape
                                     for v in values) else None


def _value(node):
    value = _prototype(node)
    if value is None:
        raise NotImplementedError('%s (%s): no tensor value (torch.export: its meta[\'val\'])'
                                  % (node.name, node.target))
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
    value = key[0].meta['val'][key[3]] if key in ir.results else key[0].meta.get('val')
    return next((v.device.type for v in pytree.tree_leaves(value) if isinstance(v, torch.Tensor)), 'cpu')


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
    if packet is A.squeeze and _named(local, names) == ['self']:
        dimensions = _dims(given, _value(operands[0]))
        kept = tuple(d for d in range(r) if d not in dimensions or x[d] != 1)
        return Rule('slice', 1, operands, [tuple(range(r))], kept, (0,), (), (), False, None, lambda sharded: ())
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


# GSPMD sections 3.5 and 4.1: a reshape's unchanged dimension boundaries preserve their partition;
# PyTorch Tensor Views, aten::view, aten::_unsafe_view and aten::expand
def _view_rule(node, local, operands, shapes, out):
    shape, packet, names = shapes[0], local.target.overloadpacket, _names(local)
    if packet is A.expand:
        lead = len(out) - len(shape)
        labels = tuple(d - lead if d >= lead and shape[d - lead] == size else ('expanded', d)
                       for d, size in enumerate(out))
    else:
        boundaries = {(math.prod(shape[:d]), size): d for d, size in enumerate(shape) if size > 1}
        labels = tuple(boundaries.get((math.prod(out[:d]), size), ('reshaped', d)) for d, size in enumerate(out))
    def configure(sharded):
        return ((node, _with(local, names, {names[1]: [sharded.get(label, size)
                                                      for label, size in zip(labels, out)]})),)
    return Rule('format', 0, operands, [tuple(range(len(shape)))], labels, (0,), (), (), False, None, configure)


# GSPMD section 3.5; T. Dao et al., FlashAttention, NeurIPS 2022, Algorithm 1: independent query rows;
# PyTorch 2.14 torch.nn.functional.scaled_dot_product_attention
def _attention_rule(node, local, operands, shapes, out):
    rank, labels, given = len(out), [], _given(local, _names(local))
    named = _named(local, _names(local))
    for k, (name, shape) in enumerate(zip(named, shapes)):
        if name not in ('query', 'key', 'value', 'attn_mask') or not shape or len(shape) > rank:
            return None
        dimensions = list(range(rank - len(shape), rank))
        dimensions[-1] = ('feature',) if name in ('query', 'key') else ('keys',) if name == 'attn_mask' else rank - 1
        if name in ('key', 'value'):
            dimensions[-2] = ('keys',)
        for d, size in enumerate(shape[:-2]):
            if size != out[rank - len(shape) + d] or given.get('enable_gqa') and d == len(shape) - 3:
                dimensions[d] = ('broadcast', k, d)
        labels.append(tuple(dimensions))
    return Rule('attention', 1, operands, labels, tuple(range(rank)), tuple(range(len(operands))), (), (), False,
                None, lambda sharded: ())


def _factory_rule(node, local, out):
    names, given = _names(local), _given(local, _names(local))
    template = given.get('self')
    operands = []
    map_arg({name: value for name, value in given.items() if name != 'self'}, operands.append)
    def configure(sharded):
        shape = tuple(sharded.get(d, size) for d, size in enumerate(out))
        changes = {'size': shape} if 'size' in names else {}
        if 'device' in names:
            changes['device'] = torch.device('meta')
        if isinstance(template, Node):
            source = _value(template)
            order = sorted(range(len(shape)), key=lambda d: (-source.stride(d), d)) \
                if len(shape) == source.dim() and is_non_overlapping_and_dense_or_false(source) \
                else tuple(range(len(shape)))
            changes['self'] = torch.empty_permuted(shape, order, dtype=source.dtype, device='meta')
        return ((node, _with(local, names, changes)),)
    return Rule('factory', 0, operands, [()] * len(operands), tuple(range(len(out))), (), (), (), False, None, configure)


# design/aot-compilation-change-request.md A2 and A4; GSPMD sections 3.1, 3.2, 3.5, 4.3 to 4.6 and Appendix A.2; torch
# 2.14.0 torch/_export/verifier.py
def _operator(node, local):
    operands = []
    map_arg((local.args, local.kwargs), operands.append)
    general = Rule('general', None, operands, None, None, (), (), (), False, None, lambda sharded: ())
    if local.target is operator.getitem and len(operands) == 1 and _prototype(operands[0]) is not None \
            and not isinstance(operands[0].meta.get('val'), torch.Tensor):
        rank = _value(node).dim()
        return Rule('format', 0, operands, [tuple(range(rank))], tuple(range(rank)), (0,), (), (), False, None,
                    general.configure)
    if not isinstance(local.target, torch._ops.OpOverload) or _prototype(node) is None \
            or not all(_prototype(o) is not None for o in operands):
        return general
    out, shapes = tuple(_value(node).shape), [tuple(_value(o).shape) for o in operands]
    packet, rule = local.target.overloadpacket, None
    if packet in FILLS or packet in EMPTY:
        rule = _factory_rule(node, local, out)
    elif (packet in POINTWISE or packet is A.copy_) and all(len(s) <= len(out) for s in shapes):
        rule = Rule('elementwise', 0, operands,
                    [tuple(len(out) - len(s) + i if size == out[len(out) - len(s) + i] else ('broadcast', k, i)
                           for i, size in enumerate(s)) for k, s in enumerate(shapes)],
                    tuple(range(len(out))), tuple(range(len(operands))), (), (), False, None,
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
    elif packet is A.sum and len(operands) == 1:
        given, rank = _given(local, _names(local)), len(shapes[0])
        dimensions = tuple(_dims(given, _value(operands[0])))
        labels = tuple(('reduced', d) if d in dimensions else d for d in range(rank)) \
            if given.get('keepdim', False) else tuple(d for d in range(rank) if d not in dimensions)
        rule = Rule('reduction', 1, operands, [tuple(range(rank))], labels, (0,), dimensions, (), False, None,
                    general.configure)
    elif packet in (A.sort, A.topk, A.cumsum, A.cumprod, *SOFTMAX) and len(operands) == 1:
        given, rank = _given(local, _names(local)), len(shapes[0])
        dim = given.get('dim', -1) % max(rank, 1)
        labels = tuple(('whole', d) if d == dim else d for d in range(rank))
        rule = Rule('format', 0, operands, [labels], tuple(range(rank)), (0,), (), (), False, None,
                    general.configure)
    elif packet in (A.gather, A.scatter, A.scatter_add, A.scatter_reduce):
        given, rank = _given(local, _names(local)), len(out)
        dim = given['dim'] % max(rank, 1)
        labels = [tuple(d if d != dim and shape[d] == out[d] else ('indexed', k, d) for d in range(rank))
                  for k, shape in enumerate(shapes)]
        rule = Rule('format', 0, operands, labels, tuple(range(rank)), tuple(range(len(operands))), (), (), False,
                    None, general.configure)
    elif packet is A.cat:
        dim, rank = _given(local, _names(local)).get('dim', 0) % max(len(out), 1), len(out)
        labels = [tuple(('part', k, d) if d == dim else d for d in range(rank)) for k in range(len(operands))]
        rule = Rule('format', 0, operands, labels, tuple(range(rank)), tuple(range(len(operands))), (), (), False,
                    None, general.configure)
    elif packet in (A.permute, A.transpose, A.t) and len(operands) == 1:
        given, rank = _given(local, _names(local)), len(shapes[0])
        dimensions = list(range(rank))
        if packet is A.permute:
            dimensions = [d % rank for d in given['dims']]
        elif rank > 1:
            a, b = (0, 1) if packet is A.t else (given['dim0'] % rank, given['dim1'] % rank)
            dimensions[a], dimensions[b] = dimensions[b], dimensions[a]
        rule = Rule('format', 0, operands, [tuple(range(rank))], tuple(dimensions), (0,), (), (), False, None,
                    general.configure)
    elif packet in (A.view, A._unsafe_view, A.expand) and len(operands) == 1:
        rule = _view_rule(node, local, operands, shapes, out)
    elif packet in (A.alias, A.detach, A.clone, A.lift_fresh_copy, A._to_copy) and len(operands) == 1 and shapes[0] == out:
        rule = Rule('format', 0, operands, [tuple(range(len(out)))], tuple(range(len(out))), (0,), (), (), False,
                    None, general.configure)
    elif packet is A.scaled_dot_product_attention:
        rule = _attention_rule(node, local, operands, shapes, out)
    elif packet in CONVOLUTION or packet in TRANSPOSED:
        rule = _convolution(node, local, operands, shapes)
    elif packet in MAXIMUM or packet in AVERAGE:
        rule = _pooling(node, local, operands, shapes)
    elif packet in PAD:
        rule = _pad(node, local, operands, shapes)
    elif packet in (A.slice, A.select, A.unsqueeze, A.squeeze):
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
def _partition_pad(ir, key, dimension, extent):
    shape = _shape(ir, key)
    if shape[dimension] == extent:
        return key
    made = key + ('padded', dimension, extent)
    if made not in ir.made:
        ir.shapes[made] = shape[:dimension] + (extent,) + shape[dimension + 1:]
        ir.emit(('pad', (key,), made, (dimension, extent)))
    return made


def _reshard(ir, node, have, want):
    def step(item):
        if item[2] not in ir.made:
            ir.emit(item)

    def gather(have, m):
        after, d, shape = tuple(-1 if s == m else s for s in have), have.index(m), ir.local(node, have)
        target = (node, after)
        if target in ir.made:
            return after
        size = ir.local(node, after)[d]
        padded = shape[d] * ir.shape[m]
        made = target if padded == size else target + ('gather', m)
        if made != target:
            ir.shapes[made] = shape[:d] + (padded,) + shape[d + 1:]
        own = (node, after, 'own')
        ir.shapes[own] = shape
        ir.inside[own] = (made, d, {e: ir.coords[e][m] * shape[d] for e in ir.coords})
        step(('copy', ((node, have),), own, None))
        step(('collective', (own,), made, Collective('all-gather', node, (m,), d, len(ir.program))))
        if made != target:
            step(('narrow', (made,), target, (d, 0, size)))
        return after

    def exchange(have, m):
        i, j = have.index(m), want.index(m)
        after, shape = tuple(-1 if k == i else m if k == j else s for k, s in enumerate(have)), ir.local(node, have)
        padded = shape[i] * ir.shape[m]
        target, size = (node, after), ir.local(node, after)[i]
        if target in ir.made:
            return after
        source = _partition_pad(ir, (node, have), j, -(-shape[j] // ir.shape[m]) * ir.shape[m])
        made = target if padded == size else target + ('exchange', m)
        ir.shapes[made] = tuple(padded if k == i else -(-n // ir.shape[m]) if k == j else n
                               for k, n in enumerate(shape))
        own = (node, after, 'own')
        ir.shapes[own] = tuple(-(-n // ir.shape[m]) if k == j else n for k, n in enumerate(shape))
        ir.inside[own] = (made, i, {e: ir.coords[e][m] * shape[i] for e in ir.coords})
        step(('copy', (source,), own, (j, m)))
        step(('collective', (source,), made,
              Collective('all-to-all', node, (m,), (i, j), len(ir.program))))
        if made != target:
            step(('narrow', (made,), target, (i, 0, size)))
        return after
    if (node, want) in ir.made:
        return node, want
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
            d = want.index(m)
            source = _partition_pad(ir, (node, have), d, -(-ir.local(node, have)[d] // ir.shape[m]) * ir.shape[m])
            step(('slice', (source,), (node, after), (d, m)))
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
        ir.emit(('copy', (key,), middle, None))
    for side, width in () if whole else (('left', left), ('right', right)):
        pieces = []
        for j in range(1, -(-width // n) + 1):
            part = min(n, width - (j - 1) * n)
            piece, arrival = (node, key[1], 'halo', consumer, dim, side, j), (node, key[1], 'arrival', consumer, dim,
                                                                               side, j)
            ir.shapes[piece] = ir.shapes[arrival] = tuple(part if i == dim else s for i, s in enumerate(shape))
            ir.emit(('narrow', (middle,), piece, (dim, n - part if side == 'left' else 0, part)))
            pairs = tuple((p, p + j) if side == 'left' else (p + j, p) for p in range(parts - j))
            ir.emit(('collective', (piece,), arrival,
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
    ir.emit(('window', tuple(reads), region, (dim, m, offsets, length, tuple(starts), size, fill)))
    return region, after


# GSPMD sections 2.1, 3, 3.2, 4, 4.2, 4.5 and 5.1, Figures 6 and 7; design/aot-compilation-change-request.md A4 and
# appendix item 7; XLA spmd_partitioner.cc, PartitionedHlo::ReshardAsWindowedInput
def _partition(ir, operators):
    inputs = 0
    for node in ir.order:
        if node.op in ('placeholder', 'get_attr'):
            sharding = ir.sharding[node]
            whole = None if sharding is None else (-1,) * len(sharding)
            ir.emit(('input', (), (node, whole), inputs) if node.op == 'placeholder'
                              else ('constant', (), (node, whole), node.target))
            if sharding is not None:
                _reshard(ir, node, whole, sharding)
            inputs += node.op == 'placeholder'
        elif node.op == 'output':
            keys = {value: (value, None) if ir.sharding[value] is None
                    else _reshard(ir, value, ir.sharding[value], (-1,) * len(ir.sharding[value]))
                    for value in node.all_input_nodes}
            ir.emit(('output', tuple(keys.values()), None, (node, keys)))
        elif operators.get(node) is not None:
            rule, sharding = operators[node], ir.sharding[node]
            if rule.labels is None:
                keys = [(o, None) if ir.sharding[o] is None
                        else _reshard(ir, o, ir.sharding[o], (-1,) * len(ir.sharding[o])) for o in rule.operands]
                whole = None if sharding is None else (-1,) * len(sharding)
                ir.emit(('call', tuple(keys), (node, whole), node))
                if sharding is not None:
                    _reshard(ir, node, whole, sharding)
                continue
            out, labels = rule.out, rule.labels
            free = set() if rule.kind == 'factory' else {label for label in out if not any(label in own for own in labels)}
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
            sharded = {label: ir.local(node, local)[d] for d, label in enumerate(out) if local[d] >= 0}
            cuts, dims = [], {w.dim: w for w in rule.windows}
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
            if rule.kind == 'attention':
                q, m = len(local) - 2, local[-2]
                start = ir.coords[ir.device][m] * ir.local(node, local)[q] if m >= 0 and ir.device is not None else 0
                key = rule.operands[1]
                part = ir.sharding[key][-2]
                owned = ir.coords[ir.device][part] * ir.local(key, ir.sharding[key])[-2] \
                    if part >= 0 and ir.device is not None else 0
                keycuts = tuple(range(0, _value(key).shape[-2], max(1, ir.local(key, ir.sharding[key])[-2]))) \
                    if part >= 0 else ()
                query = rule.operands[0]
                offsets = tuple(ir.coords[ir.device][m] * extent if m >= 0 and ir.device is not None else 0
                                for m, extent in zip(wants[0], ir.local(query, wants[0])))
                ir.locals[node] = ir.locals[node]._replace(annotation=AttentionOrigin(
                    start, owned, ir.page, keycuts, tuple(_value(query).shape), offsets))
            reads = tuple(keys)
            if rule.merged is not None:
                ir.emit(('call', reads, (rule.merged, 'merged', node), rule.merged))
                reads = ((rule.merged, 'merged', node),)
            key = (node, local, 'partial') if partial else (node, local)
            made = [key + ('cut', j) for j in range(len(cuts))] + [key]
            ir.trims[made[0]] = [(k, own.index(label), m, _value(rule.operands[k]).shape[own.index(label)])
                                 for label, m in contracted.items() if m >= 0
                                 for k, own in enumerate(labels) if label in own
                                 and _value(rule.operands[k]).shape[own.index(label)] % ir.shape[m]]
            ir.emit(('call', reads, made[0], node))
            for j, (at, m, after, o) in enumerate(cuts):
                ir.emit(('window', (made[j],), made[j + 1], (at, m, after, o, None, None, None)))
            if any(m in used for m in partial):
                (m,) = partial
                d = target.index(m)
                source = _partition_pad(ir, key, d, -(-_shape(ir, key)[d] // ir.shape[m]) * ir.shape[m])
                ir.emit(('collective', (source,), (node, target),
                                   Collective('reduce-scatter', node, (m,), target.index(m), len(ir.program))))
            elif partial:
                ir.emit(('collective', (key,), (node, target),
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
            ir.sharding[node] = (-1,) * _value(node).dim() if _prototype(node) is not None else None
    for node, rule in list(operators.items()):
        pad = rule.operands[0] if rule is not None and rule.kind == 'slice' else None
        if pad is not None and operators.get(pad) is not None and operators[pad].kind == 'pad' and len(pad.users) == 1:
            operators[node], operators[pad] = _merged(pad, operators[pad], rule), None
    names = {node.name: node for node in ir.order if node.op != 'output' and ir.sharding[node] is not None}
    names.update({node.target: node for node in ir.order if node.op == 'get_attr'})
    for name, mapping in placement.splits.items():
        node, mapping = names.get(name), tuple(mapping)
        if node is None or ir.sharding[node] is None or len(mapping) != len(ir.sharding[node]) \
                or not all(isinstance(m, int) and -1 <= m < len(ir.shape) for m in mapping) \
                or len([m for m in mapping if m >= 0]) != len({m for m in mapping if m >= 0}):
            raise ValueError('splits[%r] = %s: no tensor result\'s dims_mapping over a mesh of %d dimensions, each at '
                             'most once (GSPMD section 3.1)' % (name, mapping, len(ir.shape)))
        ir.sharding[node] = tuple(m if m >= 0 and ir.shape[m] > 1 else -1 for m in mapping)
        annotated.add(node)
    # GSPMD sections 3.1, 3.5 and 4.5; torch 2.14.0 torch/export/exported_program.py, ExportedProgram;
    # design/problems-of-14b9a757-change-request.md, "Sources for the third stops", 8
    replicated = mesh_graph.forward(ir.order, lambda node: node.all_input_nodes,
                                    lambda node, operands: node.op == 'get_attr'
                                    and not any(m >= 0 for m in ir.sharding[node] or ()) or node.op == 'call_function'
                                    and bool(operands) and all(operands) and _aliases(ir.locals[node]))
    annotated |= {node for node, whole in replicated.items() if whole and ir.sharding[node] is not None}
    _complete(ir, operators, annotated)
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
    if key in ir.shapes or isinstance(key[1], tuple):
        return pytree.tree_map(lambda value: torch.empty(_shape(ir, key), dtype=value.dtype, device='meta'),
                               key[0].meta['val'])
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
            and isinstance(ir.prototypes.get(reads[0]), (tuple, list)):
        ir.taken[key] = ir.forms[reads[0]][1][local.args[1]][1]
    if any(o is UNKNOWN for o in operands):
        return ir.describe(key, _meta(ir, key))
    given = iter(operands)
    args, kwargs = map_arg((local.args, local.kwargs), lambda n: next(given))
    try:
        result = _call(local, args, kwargs)
    except (NotImplementedError, RuntimeError):
        return ir.describe(key, _meta(ir, key))
    if getattr(local.target, 'overloadpacket', None) is A.scaled_dot_product_attention:
        source = operand.meta['val']
        order = sorted(range(source.dim()), key=lambda d: (-source.stride(d), d))
        result = torch.empty_permuted(result.shape, order, dtype=result.dtype, device='meta')
    ir.describe(key, result)
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
    return item[0] in ('slice', 'narrow', 'pad') or _viewed(item)


# GSPMD section 4.5; torch 2.14.0 torch/export/exported_program.py, ExportedProgram
def _bases(ir, item, reads):
    kind, _, key, operand = item
    if kind == 'constant':
        return frozenset((key,)) if isinstance(key[0].meta.get('val'), torch.Tensor) else None
    if reads and None not in reads and (_sliced(item) or kind == 'call' and key in ir.metas
                                        and _aliases(ir.locals[operand])):
        return frozenset((key,)) if kind == 'pad' else frozenset().union(*reads)
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
COPIES = {A.clone, A.lift_fresh_copy, A._to_copy, A._unsafe_view}


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
    metadata = ir.metas.get(key)
    strides = tuple(metadata.stride()) if key not in ir.contiguous and metadata is not None \
        and is_non_overlapping_and_dense_or_false(metadata) \
        else _contiguous(shape)
    return Place(key, 0, shape, strides, ir.results.get(key, key))


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
        n = -(-sizes[d] // ir.shape[m])
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
    if kind == 'pad':
        return None if key in ir.attributes and key not in ir.loads else _own(ir, key)
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


class CheckFailure(C.Structure):
    _fields_ = [('identity', mesh.Q), ('batch', mesh.Q), ('info', C.c_int64), ('code', mesh.U), ('padding', mesh.U)]


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
                ('mesh_metal_status_prepare', C.c_int, [mesh.CONTEXT, transport, mesh.Q, C.POINTER(mesh.Q)]),
                ('mesh_metal_machines', C.c_int, [C.c_char_p, C.POINTER(mesh_graph.Machine),
                                                  C.POINTER(mesh_graph.Machine)])):
            function = getattr(lib, name)
            function.restype, function.argtypes = result, arguments
        TRANSPORT.append(lib)
    return TRANSPORT[0]


# dlfcn.h dladdr; metal-microbench metal_recording.h
RECORDER = []


class _LibraryInfo(C.Structure):
    _fields_ = [('path', C.c_char_p), ('base', C.c_void_p), ('symbol', C.c_char_p), ('address', C.c_void_p)]


def _repository():
    info = _LibraryInfo()
    locate = C.CDLL(None).dladdr
    locate.argtypes, locate.restype = [C.c_void_p, C.POINTER(_LibraryInfo)], C.c_int
    if not locate(_recorder()[0].MetalRecorderOpen, C.byref(info)):
        raise OSError(errno.ENOENT, 'the loaded recorder has no library path')
    return os.path.dirname(os.path.dirname(os.path.realpath(os.fsdecode(info.path))))


def _recorder():
    if not RECORDER:
        path = next((p for p in os.environ.get('DYLD_INSERT_LIBRARIES', '').split(':')
                     if os.path.basename(p) == 'libmetal_recording.dylib'), None) \
            or ctypes.util.find_library('metal_recording') or 'libmetal_recording.dylib'
        lib, metal = C.CDLL(path), C.CDLL('/System/Library/Frameworks/Metal.framework/Metal')
        lib.MetalRecorderOpen.restype, lib.MetalRecorderOpen.argtypes = C.c_void_p, [C.c_void_p, C.c_char_p, C.c_size_t]
        lib.MetalRecorderPrepare.restype = C.c_void_p
        lib.MetalRecorderPrepare.argtypes = [C.c_void_p, C.c_void_p, C.c_char_p, C.c_size_t]
        lib.MetalRecorderReplay.restype = C.c_int
        lib.MetalRecorderReplay.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(Replay), C.c_char_p, C.c_size_t]
        lib.MetalReplayNative.restype = C.c_void_p
        lib.MetalReplayNative.argtypes = [C.POINTER(Replay), C.POINTER(mesh.Q)]
        lib.MetalRecorderProgram.restype = C.c_void_p
        lib.MetalRecorderProgram.argtypes = [C.c_void_p, C.POINTER(Call), mesh.U, C.POINTER(mesh_graph.Machine),
                                             C.POINTER(mesh.Planned), C.c_char_p, C.c_size_t]
        for name in ('MetalRecorderSave', 'MetalProgramSave'):
            function = getattr(lib, name)
            function.restype = C.c_int
            function.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t), C.c_char_p, C.c_size_t]
        lib.MetalRecorderLoad.restype = C.c_void_p
        lib.MetalRecorderLoad.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t, C.c_char_p, C.c_size_t]
        lib.MetalProgramLoad.restype = C.c_void_p
        lib.MetalProgramLoad.argtypes = lib.MetalRecorderProgram.argtypes[:-2] + [C.c_void_p, C.c_size_t,
                                                                               C.c_char_p, C.c_size_t]
        lib.MetalProgramRun.restype = C.c_int
        lib.MetalProgramRun.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), mesh.Q, mesh.Q, C.c_void_p, C.c_void_p,
                                        C.c_void_p, C.c_void_p, C.c_char_p, C.c_size_t]
        lib.MetalProgramHorizon.restype, lib.MetalProgramHorizon.argtypes = mesh.U, [C.c_void_p]
        lib.MetalProgramEnd.restype = C.c_int
        lib.MetalProgramEnd.argtypes = [C.c_void_p, C.c_void_p, C.c_char_p, C.c_size_t]
        lib.MetalProgramCheckFailure.restype = C.c_int
        lib.MetalProgramCheckFailure.argtypes = [C.c_void_p, C.POINTER(CheckFailure)]
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
KINDS = dict(zip((BOOL, I8, U8, I16, U16, I32, U32, I64, U64, F16, BF16, F32, F64),
                 ('bool', 'i8', 'u8', 'i16', 'u16', 'i32', 'u32', 'i64', 'u64', 'f16', 'bf16', 'f32', 'f64')))
NONE = 0xFFFFFFFF
SCALAR = {torch.bool: BOOL, torch.int8: I8, torch.uint8: U8, torch.int16: I16, torch.uint16: U16, torch.int32: I32,
          torch.uint32: U32, torch.int64: I64, torch.uint64: U64, torch.float16: F16, torch.bfloat16: BF16,
          torch.float32: F32, torch.float64: F64}
# metal_ir.h ComputeType: the kind a region computes a stored kind in
CLASS = {BOOL: BOOL, I8: I32, I16: I32, I32: I32, U8: U32, U16: U32, U32: U32, I64: I64, U64: U64, F16: F16, BF16: F32,
         F32: F32, F64: F64}
STORED = {BOOL: torch.bool, I32: torch.int32, U32: torch.uint32, I64: torch.int64, U64: torch.uint64, F16: torch.float16,
        F32: torch.float32, F64: torch.float64}
REAL = tuple(SCALAR)
COMPLEXES = (torch.complex64, torch.complex128)
FLOATING = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
INTEGRAL = tuple(dtype for dtype in REAL if dtype not in FLOATING)
# torch 2.14.0 aten/src/ATen/native/LinearAlgebra.cpp addmm_impl_cpu_ (AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16,
# kHalf)): the contractions' result kinds, every number kind but bool
CONTRACTED = FLOATING + COMPLEXES + tuple(dtype for dtype in INTEGRAL if dtype is not torch.bool)
# metal_ir.h: linalg iterator kinds, combiners, starts, precision, placement, Tint core::BinaryOp and core::UnaryOp
PARALLEL, REDUCTION = 0, 1
ASSIGN, SUM, REGION = 0, 1, 9
IDENTITY, ACCUMULATE = 0, 1
REASSOCIATE, CONTRACT = 1, 2
GPU, CPU, ANE = 0, 1, 2
BINARY = {'and': 0, 'or': 1, 'xor': 2, 'eq': 5, 'ne': 6, 'lt': 7, 'gt': 8, 'le': 9, 'ge': 10, 'shl': 11, 'shr': 12,
          'add': 13, 'sub': 14, 'mul': 15, 'div': 16, 'rem': 17}
COMPARE = ('eq', 'ne', 'lt', 'gt', 'le', 'ge')
BOOLEAN = COMPARE + ('isnan', 'isinf', 'logical_and', 'logical_or', 'logical_xor', 'logical_not')
COMPLEMENT, NEGATION, NOT = 1, 3, 4
# WGSL builtin functions by name (metal_ir_builtin_named, Tint core::ParseBuiltinFn)
BUILTINS = ('exp', 'log', 'sqrt', 'tanh', 'abs', 'trunc', 'floor', 'ceil', 'round', 'pow', 'sin', 'cos', 'tan', 'asin',
            'acos', 'atan', 'atan2', 'sinh', 'cosh', 'asinh', 'acosh', 'atanh', 'exp2', 'log2')


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
        if name == 'complement':
            return self.add(('unary', COMPLEMENT, x[0]), kind)
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

    def scatter(self, output, indices, value, predicate):
        self.add(('scatter', output, tuple(indices), value, predicate), BOOL)

    def extract_if(self, input, kind, indices, predicate, fallback):
        return self.add(('extract_if', input, tuple(indices), predicate, fallback), kind)


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
        self.place, self.node = GPU, 0


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
    if found and (found[0].is_complex or results is CONTRACTED and found[0] is torch.bool):
        raise NotImplementedError('%s of %s: PyTorch 2.14 defines no form of it for that type (its CPU kernel admits '
                                  'none)' % (target, found[0]))
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
    if op in ('floor', 'ceil', 'round', 'trunc'):
        return x if not _floating(t) else r(op, t, x)
    if op in ('sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'asinh', 'acosh', 'atanh', 'erf'):
        return r(op, t, x)
    if op == 'atan2':
        return r('atan2', t, x, y)
    if op == 'reciprocal':
        return r('div', t, c(1), x)
    if op in ('exp2', 'log2'):
        return r(op, t, x)
    if op == 'log10':
        return r('div', t, r('log', t, x), c(math.log(10)))
    # W. Kahan's log1p (D. Goldberg, ACM Comput. Surv. 23(1), 1991, Theorem 4) and expm1 by the same correction
    if op == 'log1p':
        u = r('add', t, c(1), x)
        corrected = r('mul', t, r('log', t, u), r('div', t, x, r('sub', t, u, c(1))))
        return r('select', t, r('eq', BOOL, u, c(1)), x,
                 r('select', t, r('eq', BOOL, u, c(math.inf)), u, corrected))
    if op == 'expm1':
        u = r('exp', t, x)
        difference = r('sub', t, u, c(1))
        corrected = r('mul', t, difference, r('div', t, x, r('log', t, u)))
        return r('select', t, r('eq', BOOL, u, c(1)), x,
                 r('select', t, r('eq', BOOL, u, c(math.inf)), u,
                   r('select', t, r('eq', BOOL, difference, c(-1)), c(-1), corrected)))
    if op == 'isnan':
        return r('isnan', BOOL, x)
    if op == 'isinf':
        return r('eq', BOOL, r('abs', t, x), c(math.inf))
    if op == 'sign':
        return r('sub', t, _convert(r, r('lt', BOOL, c(0), x), BOOL, t), _convert(r, r('lt', BOOL, x, c(0)), BOOL, t))
    if op in ('maximum', 'minimum'):
        better = r('select', t, r('gt' if op == 'maximum' else 'lt', BOOL, x, y), x, y)
        if not _floating(t):
            return better
        return r('select', t, r('isnan', BOOL, x), x, r('select', t, r('isnan', BOOL, y), y, better))
    if op == 'fmod':
        return r('rem', t, x, y if _floating(t) else r('select', t, r('eq', BOOL, y, c(0)), c(1), y))
    if op == 'remainder':
        m = r('rem', t, x, y if _floating(t) else r('select', t, r('eq', BOOL, y, c(0)), c(1), y))
        fix = r('and', BOOL, r('ne', BOOL, m, c(0)), r('ne', BOOL, r('lt', BOOL, m, c(0)), r('lt', BOOL, y, c(0))))
        return r('select', t, fix, r('add', t, m, y), m)
    if op in ('bitwise_and', 'bitwise_or', 'bitwise_xor'):
        return r(op[8:], t, x, y)
    if op == 'bitwise_not':
        return r('not', BOOL, x) if t == BOOL else r('complement', t, x)
    if op in ('logical_and', 'logical_or', 'logical_xor'):
        a, b = _convert(r, x, t, BOOL), _convert(r, y, t, BOOL)
        return r('ne', BOOL, a, b) if op == 'logical_xor' else r(op[8:], BOOL, a, b)
    if op == 'logical_not':
        return r('not', BOOL, _convert(r, x, t, BOOL))
    return r('select', t, r('ne', BOOL, x, c(0)), y, z)


# PyTorch 2.14 documentation, Broadcasting semantics and Type promotion: tensor operands broadcast by their maps' zero
# rows, scalars the region's literals
def _elementwise(op, out, a, b=None, c=None, alpha=1):
    if any(isinstance(x, torch.Tensor) and x.is_complex() or isinstance(x, complex) for x in (out, a, b, c)):
        if op == 'where':
            return [g for k in (0, 1) for g in _elementwise(op, _component(out, k), a,
                                                            _component(b, k), _component(c, k), alpha)]
        return _complex(op, out, a, b, alpha)
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
    r.done(_store(r, made, BOOL if op in BOOLEAN or op == 'bitwise_not' and t == BOOL else t, out))
    return [_Generic(out.shape, inputs, [(out, _same(out.dim()))], r)]


POINTWISE = {A.add: 'add', A.sub: 'sub', A.mul: 'mul', A.div: 'div', A.floor_divide: 'floordiv', A.neg: 'neg',
             A.relu: 'relu', A.tanh: 'tanh', A.sigmoid: 'sigmoid', A.exp: 'exp', A.gelu: 'gelu', A.silu: 'silu',
             A.eq: 'eq', A.ne: 'ne', A.lt: 'lt', A.le: 'le', A.gt: 'gt', A.ge: 'ge', A.pow: 'pow', A.rsqrt: 'rsqrt',
             A.sqrt: 'sqrt', A.log: 'log', A.abs: 'abs', A.clamp: 'clamp', A.clamp_min: 'clamp',
             A.clamp_max: 'clamp', A.where: 'where',
             **{getattr(A, name): name for name in (
                 'floor', 'ceil', 'round', 'trunc', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh',
                 'asinh', 'acosh', 'atanh', 'erf', 'atan2', 'reciprocal', 'exp2', 'log2', 'log10', 'log1p', 'expm1',
                 'isnan',
                 'isinf', 'sign', 'maximum', 'minimum', 'fmod', 'remainder', 'bitwise_and', 'bitwise_or', 'bitwise_xor',
                 'bitwise_not', 'logical_and', 'logical_or', 'logical_xor', 'logical_not')},
             A.__and__: 'bitwise_and', A.__or__: 'bitwise_or', A.__xor__: 'bitwise_xor'}
# PyTorch 2.14 documentation, torch.nn.functional.leaky_relu, hardtanh and elu, and torch.nan_to_num (its infinities'
# defaults the dtype's finite extremes): their scalar parameters the region's literals, a composition of the pointwise
# basis
PARAMETERIZED = {A.leaky_relu: ('negative_slope',), A.hardtanh: ('min_val', 'max_val'),
                 A.elu: ('alpha', 'scale', 'input_scale'), A.nan_to_num: ('nan', 'posinf', 'neginf')}


def _parameterized(target, given, outs):
    packet, x, out = target.overloadpacket, given['self'], outs[0]
    if packet is A.nan_to_num and not out.dtype.is_floating_point:
        return _copy(out, x)
    _typed(target, [x], outs)
    a = [given.get(name) for name in PARAMETERIZED[packet]]
    r, t = _Region(), _compute(out.dtype, x.dtype, torch.float32)
    v = _load(r, 0, x, t)

    def c(value):
        return r.const(value, t)
    zero = c(0)
    if packet is A.leaky_relu:
        made = r('select', t, r('gt', BOOL, v, zero), v, r('mul', t, v, c(a[0])))
    elif packet is A.hardtanh:
        made = r('select', t, r('lt', BOOL, v, c(a[0])), c(a[0]), r('select', t, r('gt', BOOL, v, c(a[1])), c(a[1]), v))
    elif packet is A.nan_to_num:
        largest = torch.finfo(out.dtype).max
        high, low = c(largest if a[1] is None else a[1]), c(-largest if a[2] is None else a[2])
        made = r('select', t, r('isnan', BOOL, v), c(a[0] or 0), r('select', t, r('eq', BOOL, v, c(math.inf)), high,
                                                                     r('select', t, r('eq', BOOL, v, c(-math.inf)), low, v)))
    else:
        negative = r('mul', t, c(a[1] * a[0]), r('sub', t, r('exp', t, r('mul', t, v, c(a[2]))), c(1)))
        made = r('select', t, r('gt', BOOL, v, zero), r('mul', t, v, c(a[1])), negative)
    r.done(_store(r, made, t, out))
    return [_Generic(out.shape, [(x, _same(x.dim()))], [(out, _same(out.dim()))], r)]


def _arithmetic(target, given, outs):
    packet, op = target.overloadpacket, POINTWISE[target.overloadpacket]
    if any(isinstance(v, torch.Tensor) and v.is_complex() or isinstance(v, complex)
           for v in [*given.values(), *outs]):
        if packet is A.where:
            return [g for k in (0, 1) for g in _elementwise('where', _component(outs[0], k), given['condition'],
                                                           _component(given['self'], k), _component(given['other'], k))]
        return _complex(op, outs[0], given['self'], given['exponent'] if packet is A.pow else given.get('other'),
                        given.get('alpha', 1))
    if packet is A.div and given.get('rounding_mode') is not None:
        op = {'floor': 'floordiv', 'trunc': 'truncdiv'}[given['rounding_mode']]
    if packet is A.gelu and given['approximate'] != 'none':
        op = 'gelu_tanh'
    if packet is A.bitwise_not and given['self'].dtype is torch.bool:
        op = 'logical_not'
    operands = [given['condition'], given['self'], given['other']] if packet is A.where else \
        [given['self'], given.get('min'), given.get('max')] if op == 'clamp' else \
        [given['self'], given['exponent'], None] if packet is A.pow else [given['self'], given.get('other'), None]
    _typed(target, operands, outs, REAL)
    return _elementwise(op, outs[0], *operands, given.get('alpha', 1))


# torch 2.14.0 torch/_tensor_docs.py, copy_ and view: views of one element count correspond by their row-major
# element index, the source's coordinates computed in the region
def _coordinates(r, linear, shape):
    made, rest, kind = [], linear, r.kinds[linear]
    for d in reversed(range(len(shape))):
        made.append(r('rem', kind, rest, r.const(shape[d], kind)))
        rest = r('div', kind, rest, r.const(shape[d], kind))
    return made[::-1]


def _linear(r, shape):
    at = r.const(0, I32)
    for d, n in enumerate(shape):
        at = r('add', I32, r('mul', I32, at, r.const(n, I32)), r.index(d))
    return at


def _copy(out, value):
    if out.is_complex() or value.is_complex():
        if out.is_complex() and value.is_complex():
            return _copy(torch.view_as_real(out), torch.view_as_real(value))
        return _complex('copy', out, value) if out.is_complex() else _copy(out, torch.view_as_real(value)[..., 0])
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


# PyTorch 2.14 documentation, Complex Numbers, torch.view_as_real
def _widened(rows, extra):
    return tuple(row[:-1] + (0,) * extra + row[-1:] for row in rows)


def _pair(r, inputs, x, rows, loops, t):
    pairs, made = torch.view_as_real(x), []
    for c in (0, 1):
        made.append(_load(r, len(inputs), pairs, t))
        inputs.append((pairs, rows + (_rows(loops, [{None: c}])[0],)))
    return tuple(made)


def _component(x, k):
    if isinstance(x, torch.Tensor):
        return torch.view_as_real(x)[..., k] if x.is_complex() else x if k == 0 else 0
    return (complex(x).real if k == 0 else complex(x).imag) if isinstance(x, complex) else x if k == 0 else 0


def _complex(op, out, a, b=None, alpha=1):
    shape = list(out.shape)
    loops = shape
    t = F64 if any(isinstance(x, torch.Tensor) and x.dtype in (torch.complex128, torch.float64) for x in (a, b, out)) \
        else F32
    r, inputs, values = _Region(), [], []
    for x in (a, b):
        if x is None:
            values.append(None)
        elif isinstance(x, torch.Tensor) and x.is_complex():
            values.append(_pair(r, inputs, x, _aligned(x, shape), len(loops), t))
        elif isinstance(x, torch.Tensor):
            values.append((_load(r, len(inputs), x, t), r.const(0, t)))
            inputs.append((x, _aligned(x, shape)))
        else:
            z = complex(x)
            values.append((r.const(z.real, t), r.const(z.imag, t)))
    made = _complex_value(r, t, op, values[0], values[1], alpha, b if op == 'pow' and not isinstance(b, torch.Tensor)
                          else None)
    if op in BOOLEAN:
        r.done(_store(r, made, BOOL, out))
        return [_Generic(loops, inputs, [(out, _same(len(loops)))], r)]
    re, im = made if isinstance(made, tuple) else (made, made)
    if out.is_complex():
        r.done(_store(r, re, t, out.real), _store(r, im, t, out.imag))
        return [_Generic(loops, inputs, [(out.real, _same(len(loops))), (out.imag, _same(len(loops)))], r)]
    r.done(_store(r, re, t, out))
    return [_Generic(loops, inputs, [(out, _same(len(loops)))], r)]


# the complex arithmetic of (re, im) pairs: R. L. Smith, Algorithm 116, Complex division, CACM 5(8), 1962; the modulus
# scaled by the larger part (no overflow of the squares); W. Kahan, Branch cuts for complex elementary functions, in
# The State of the Art in Numerical Analysis, 1987 (sqrt by the half-sum of |z| and |re|; tanh by tan and sinh);
# the elementary functions by their identities (Abramowitz and Stegun 4.3.55-4.3.58, 4.5.49-4.5.54, 4.4.14-4.4.16,
# 4.6.14-4.6.16: sin, cos, sinh, cosh and the inverses through log and sqrt)
def _complex_value(r, t, op, z, w, alpha, exponent):
    def c(v):
        return r.const(v, t)

    def add(x, y):
        return r('add', t, x[0], y[0]), r('add', t, x[1], y[1])

    def sub(x, y):
        return r('sub', t, x[0], y[0]), r('sub', t, x[1], y[1])

    def mul(x, y):
        return (r('sub', t, r('mul', t, x[0], y[0]), r('mul', t, x[1], y[1])),
                r('add', t, r('mul', t, x[0], y[1]), r('mul', t, x[1], y[0])))

    def scale(x, k):
        return r('mul', t, x[0], c(k)), r('mul', t, x[1], c(k))

    def div(x, y):
        wide = r('ge', BOOL, r('abs', t, y[0]), r('abs', t, y[1]))
        first, second = r('select', t, wide, y[0], y[1]), r('select', t, wide, y[1], y[0])
        ratio = r('div', t, second, first)
        den = r('add', t, first, r('mul', t, second, ratio))
        re_w = r('div', t, r('add', t, x[0], r('mul', t, x[1], ratio)), den)
        im_w = r('div', t, r('sub', t, x[1], r('mul', t, x[0], ratio)), den)
        re_n = r('div', t, r('add', t, r('mul', t, x[0], ratio), x[1]), den)
        im_n = r('div', t, r('sub', t, r('mul', t, x[1], ratio), x[0]), den)
        return r('select', t, wide, re_w, re_n), r('select', t, wide, im_w, im_n)

    def modulus(x):
        a, b = r('abs', t, x[0]), r('abs', t, x[1])
        big, small = r('select', t, r('ge', BOOL, a, b), a, b), r('select', t, r('ge', BOOL, a, b), b, a)
        ratio = r('div', t, small, r('select', t, r('eq', BOOL, big, c(0)), c(1), big))
        return r('mul', t, big, r('sqrt', t, r('add', t, c(1), r('mul', t, ratio, ratio))))

    def exp(x):
        e = r('exp', t, x[0])
        return r('mul', t, e, r('cos', t, x[1])), r('mul', t, e, r('sin', t, x[1]))

    def log(x):
        return r('log', t, modulus(x)), r('atan2', t, x[1], x[0])

    def sqrt(x):
        m = r('sqrt', t, r('mul', t, c(0.5), r('add', t, r('abs', t, x[0]), modulus(x))))
        zero = r('eq', BOOL, m, c(0))
        safe = r('select', t, zero, c(1), m)
        other = r('div', t, r('abs', t, x[1]), r('mul', t, c(2), safe))
        negative = r('lt', BOOL, x[0], c(0))
        re = r('select', t, negative, other, m)
        im = r('select', t, negative, r('select', t, r('lt', BOOL, x[1], c(0)), r('neg', t, m), m),
               r('div', t, x[1], r('mul', t, c(2), safe)))
        return r('select', t, zero, c(0), re), r('select', t, zero, x[1], im)

    def sinh_cosh(a):
        e, f = r('exp', t, a), r('exp', t, r('neg', t, a))
        return r('mul', t, c(0.5), r('sub', t, e, f)), r('mul', t, c(0.5), r('add', t, e, f))

    def sin(x):
        sh, ch = sinh_cosh(x[1])
        return r('mul', t, r('sin', t, x[0]), ch), r('mul', t, r('cos', t, x[0]), sh)

    def cos(x):
        sh, ch = sinh_cosh(x[1])
        return r('mul', t, r('cos', t, x[0]), ch), r('neg', t, r('mul', t, r('sin', t, x[0]), sh))

    def sinh(x):
        sh, ch = sinh_cosh(x[0])
        return r('mul', t, sh, r('cos', t, x[1])), r('mul', t, ch, r('sin', t, x[1]))

    def cosh(x):
        sh, ch = sinh_cosh(x[0])
        return r('mul', t, ch, r('cos', t, x[1])), r('mul', t, sh, r('sin', t, x[1]))

    def tanh(x):
        tn = r('tan', t, x[1])
        beta = r('add', t, c(1), r('mul', t, tn, tn))
        sh = sinh_cosh(x[0])[0]
        rho = r('sqrt', t, r('add', t, c(1), r('mul', t, sh, sh)))
        den = r('add', t, c(1), r('mul', t, beta, r('mul', t, sh, sh)))
        return r('div', t, r('mul', t, beta, r('mul', t, rho, sh)), den), r('div', t, tn, den)

    def times_i(x):
        return r('neg', t, x[1]), x[0]

    def over_i(x):
        return x[1], r('neg', t, x[0])
    one = (c(1), c(0))

    def asinh(x):
        return log(add(x, sqrt(add(mul(x, x), one))))

    def acosh(x):
        return log(add(x, mul(sqrt(add(x, one)), sqrt(sub(x, one)))))

    def atanh(x):
        return scale(sub(log(add(one, x)), log(sub(one, x))), 0.5)

    def real(name, x):
        return _apply(r, name, x, None, None, 1, t)
    if op in ('add', 'sub'):
        u = w if alpha == 1 else mul(w, (c(complex(alpha).real), c(complex(alpha).imag)))
        return add(z, u) if op == 'add' else sub(z, u)
    if op == 'mul':
        return mul(z, w)
    if op == 'div':
        return div(z, w)
    if op == 'neg':
        return r('neg', t, z[0]), r('neg', t, z[1])
    if op == 'conj':
        return z[0], r('neg', t, z[1])
    if op == 'exp':
        return exp(z)
    if op == 'exp2':
        return exp(scale(z, math.log(2)))
    if op == 'log':
        return log(z)
    if op in ('log2', 'log10'):
        return scale(log(z), 1 / math.log(2 if op == 'log2' else 10))
    if op == 'log1p':
        square = r('add', t, r('mul', t, z[0], r('add', t, c(2), z[0])), r('mul', t, z[1], z[1]))
        return r('mul', t, c(0.5), real('log1p', square)), r('atan2', t, z[1], r('add', t, c(1), z[0]))
    if op == 'expm1':
        half = r('sin', t, r('mul', t, c(0.5), z[1]))
        return (r('sub', t, r('mul', t, real('expm1', z[0]), r('cos', t, z[1])), r('mul', t, c(2), r('mul', t, half, half))),
                r('mul', t, r('exp', t, z[0]), r('sin', t, z[1])))
    if op == 'sqrt':
        return sqrt(z)
    if op == 'rsqrt':
        return div(one, sqrt(z))
    if op == 'reciprocal':
        return div(one, z)
    if op in ('sigmoid', 'silu'):
        return div(one if op == 'sigmoid' else z, add(one, exp((r('neg', t, z[0]), r('neg', t, z[1])))))
    if op.startswith('logical_'):
        def truth(x):
            return r('or', BOOL, r('ne', BOOL, x[0], c(0)), r('ne', BOOL, x[1], c(0)))
        if op == 'logical_not':
            return r('not', BOOL, truth(z))
        return r('ne' if op == 'logical_xor' else op[8:], BOOL, truth(z), truth(w))
    if op == 'pow':
        if exponent is not None and complex(exponent) == 2:
            return mul(z, z)
        power = exp(mul(w, log(z)))
        nothing = r('and', BOOL, r('eq', BOOL, z[0], c(0)), r('eq', BOOL, z[1], c(0)))
        unit = r('and', BOOL, r('eq', BOOL, w[0], c(0)), r('eq', BOOL, w[1], c(0)))
        zero = r('select', t, unit, c(1), c(0))
        return r('select', t, nothing, zero, power[0]), r('select', t, nothing, c(0), power[1])
    if op == 'sin':
        return sin(z)
    if op == 'cos':
        return cos(z)
    if op == 'tan':
        return over_i(tanh(times_i(z)))
    if op == 'sinh':
        return sinh(z)
    if op == 'cosh':
        return cosh(z)
    if op == 'tanh':
        return tanh(z)
    if op == 'asinh':
        return asinh(z)
    if op == 'acosh':
        return acosh(z)
    if op == 'atanh':
        return atanh(z)
    if op == 'asin':
        return over_i(asinh(times_i(z)))
    if op == 'acos':
        a = over_i(asinh(times_i(z)))
        return r('sub', t, c(math.pi / 2), a[0]), r('neg', t, a[1])
    if op == 'atan':
        return over_i(atanh(times_i(z)))
    if op == 'sgn':
        m = modulus(z)
        zero = r('eq', BOOL, m, c(0))
        safe = r('select', t, zero, c(1), m)
        return (r('select', t, zero, c(0), r('div', t, z[0], safe)), r('select', t, zero, c(0), r('div', t, z[1], safe)))
    if op == 'abs':
        return modulus(z)
    if op == 'abs1':
        return r('add', t, r('abs', t, z[0]), r('abs', t, z[1]))
    if op == 'angle':
        return r('atan2', t, z[1], z[0])
    if op in ('eq', 'ne'):
        same = r('and', BOOL, r('eq', BOOL, z[0], w[0]), r('eq', BOOL, z[1], w[1]))
        return same if op == 'eq' else r('not', BOOL, same)
    if op == 'isnan':
        return r('or', BOOL, r('isnan', BOOL, z[0]), r('isnan', BOOL, z[1]))
    if op == 'isinf':
        return r('or', BOOL, r('eq', BOOL, r('abs', t, z[0]), c(math.inf)), r('eq', BOOL, r('abs', t, z[1]), c(math.inf)))
    if op == 'polar':
        return r('mul', t, z[0], r('cos', t, w[0])), r('mul', t, z[0], r('sin', t, w[0]))
    if op == 'complex':
        return z[0], w[0]
    if op == 'copy':
        return z
    raise NotImplementedError('%s of complex operands: PyTorch defines no complex form of it' % op)

COMPLEX = {A.polar: 'polar', A.complex: 'complex', A.angle: 'angle', A.conj_physical: 'conj',
           A._conj_physical: 'conj'}


def _complexed(target, given, outs):
    packet = target.overloadpacket
    if packet in (A.polar, A.complex):
        first, second = (given['abs'], given['angle']) if packet is A.polar else (given['real'], given['imag'])
        return _complex(COMPLEX[packet], outs[0], first, second)
    if packet in (A.conj_physical, A._conj_physical) and not given['self'].is_complex():
        return _assign(outs[0], given['self'])
    return _complex(COMPLEX[packet], outs[0], given['self'])


# J. W. Cooley and J. W. Tukey, An algorithm for the machine calculation of complex Fourier series, Math. Comp. 19,
# 1965 (with the four-step index split of N = n1 n2: n = n2 a + c, k = k1 + n1 k2, X_k = W_N^(c k1) W_n2^(c k2)
# sum_a x_(n2 a + c) W_n1^(a k1)), n1 the smallest prime factor at each level (P. N. Swarztrauber, FFTPACK, Netlib, and
# M. Reinecke's pocketfft the same factorization); L. I. Bluestein, A linear filtering approach to the computation of
# the discrete Fourier transform, IEEE Trans. Audio Electroacoust. 18(4), 1970, for a prime length above DIRECT: the
# transform the chirp c_n = exp(s pi i n^2 / N) times the convolution of x c with conj(c), by power-of-two transforms.
# Complex values are their real (re, im) views; each level one structured operation over strided views
DIRECT = 16


def _twiddled(r, t, angle):
    return r('cos', t, angle), r('sin', t, angle)


def _complex_rows(rows, loops, c):
    return tuple(rows) + ((0,) * loops + (c,),)


# dst[k] = scale exp(s 2 pi i (k n) / N) summed over n of src[n] along their last dimension (src real or complex,
# dst complex), with the smallest prime factor's radix and a twiddle, the rest a transform of the factor's quotient
def _fft(src, dst, n, sign, scale, t):
    lead = src.dim() - 1
    if n == 1:
        return _copy(dst, src) if scale == 1 else _scaled(src, dst, scale, t)
    p = next(q for q in range(2, n + 1) if n % q == 0)
    if p == n and n > DIRECT:
        return _bluestein(src, dst, n, sign, scale, t)
    m = n // p
    loops = [*src.shape[:-1], m, p, 2, p]
    r, inputs = _Region(), []
    view = src.as_strided((*src.shape[:-1], p, m), (*src.stride()[:-1], src.stride(-1) * m, src.stride(-1)),
                          src.storage_offset())
    rows = _rows(len(loops), [*range(lead), lead + 3, lead])
    if view.is_complex():
        pairs = torch.view_as_real(view)
        for c in (0, 1):
            inputs.append((pairs, _complex_rows(rows, len(loops), c)))
        xr, xi = _load(r, 0, pairs, t), _load(r, 1, pairs, t)
    else:
        inputs.append((view, rows))
        xr, xi = _load(r, 0, view, t), r.const(0, t)
    c, k1, a = r.index(lead), r.index(lead + 1), r.index(lead + 3)
    inner = r('mul', t, _convert(r, r('rem', I32, r('mul', I32, a, k1), r.const(p, I32)), I32, t),
              r.const(sign * 2 * math.pi / p, t))
    outer = r('mul', t, _convert(r, r('mul', I32, c, k1), I32, t), r.const(sign * 2 * math.pi / n, t))
    cos, sin = _twiddled(r, t, r('add', t, inner, outer))
    re = r('sub', t, r('mul', t, xr, cos), r('mul', t, xi, sin))
    im = r('add', t, r('mul', t, xr, sin), r('mul', t, xi, cos))
    r.done(r('mul', t, r('select', t, r('eq', BOOL, r.index(lead + 2), r.const(0, I32)), re, im), r.const(scale, t)))
    y = _temporary([*src.shape[:-1], m, p], torch.complex128 if t == F64 else torch.complex64)
    ops = [_Generic(loops, inputs, [(torch.view_as_real(y), _same(len(loops), lead + 3))], r,
                    [PARALLEL] * (lead + 3) + [REDUCTION], [SUM], [t], precision=REASSOCIATE | CONTRACT)]
    rest = y.permute(*range(lead), lead + 1, lead)
    out = dst.as_strided((*dst.shape[:-1], p, m), (*dst.stride()[:-1], dst.stride(-1), dst.stride(-1) * p),
                         dst.storage_offset())
    return ops + _fft(rest, out, m, sign, 1.0, t)


def _scaled(src, dst, scale, t):
    return _complex('mul', dst, src, complex(scale)) if src.is_complex() else \
        _complex('copy', dst, src) if scale == 1 else _complex('mul', dst, src, scale)


def _bluestein(src, dst, n, sign, scale, t):
    size = 1 << (2 * n - 2).bit_length()
    kind = torch.complex128 if t == F64 else torch.complex64
    lead = src.dim() - 1
    lines = list(src.shape[:-1])
    a, b = _temporary([*lines, size], kind), _temporary([size], kind)
    fa, fb, product, conv = (_temporary([*lines, size], kind), _temporary([size], kind), _temporary([*lines, size], kind),
                             _temporary([*lines, size], kind))

    def chirp(r, j, negate):
        square = r('rem', I64, r('mul', I64, j, j), r.const(2 * n, I64))
        angle = r('mul', t, _convert(r, square, I64, t), r.const((-sign if negate else sign) * math.pi / n, t))
        return _twiddled(r, t, angle)
    r, inputs = _Region(), []
    j = r('cast', I64, r.index(lead))
    inside, at = _bounded(r, r.index(lead), n)
    if src.is_complex():
        pairs = torch.view_as_real(src)
        xr = _convert(r, r('extract', _class(pairs), *[r.index(q) for q in range(lead)], at, r.const(0, I32), integer=0),
                      _class(pairs), t)
        xi = _convert(r, r('extract', _class(pairs), *[r.index(q) for q in range(lead)], at, r.const(1, I32), integer=0),
                      _class(pairs), t)
        inputs.append((pairs, None))
    else:
        xr = _convert(r, r('extract', _class(src), *[r.index(q) for q in range(lead)], at, integer=0), _class(src), t)
        xi = r.const(0, t)
        inputs.append((src, None))
    cos, sin = chirp(r, j, False)
    re = r('sub', t, r('mul', t, xr, cos), r('mul', t, xi, sin))
    im = r('add', t, r('mul', t, xr, sin), r('mul', t, xi, cos))
    zero = r.const(0, t)
    r.done(r('select', t, inside, r('select', t, r('eq', BOOL, r.index(lead + 1), r.const(0, I32)), re, im), zero))
    ops = [_Generic([*lines, size, 2], inputs, [(torch.view_as_real(a), _same(lead + 2))], r)]
    q = _Region()
    position = q.index(0)
    mirrored = q('sub', I32, q.const(size, I32), position)
    near = q('lt', BOOL, position, q.const(n, I32))
    far = q('and', BOOL, q('gt', BOOL, mirrored, q.const(0, I32)), q('lt', BOOL, mirrored, q.const(n, I32)))
    used = q('cast', I64, q('select', I32, near, position, mirrored))
    cos, sin = chirp(q, used, True)
    value = q('select', t, q('eq', BOOL, q.index(1), q.const(0, I32)), cos, sin)
    q.done(q('select', t, q('or', BOOL, near, far), value, q.const(0, t)))
    ops.append(_Generic([size, 2], [], [(torch.view_as_real(b), _same(2))], q))
    ops += _fft(a, fa, size, sign, 1.0, t) + _fft(b, fb, size, sign, 1.0, t)
    ops += _complex('mul', product, fa, fb.expand(*lines, size))
    ops += _fft(product, conv, size, -sign, 1.0 / size, t)
    e = _Region()
    cr, ci = e.arg(0, torch.view_as_real(conv)), e.arg(1, torch.view_as_real(conv))
    cos, sin = chirp(e, e('cast', I64, e.index(lead)), False)
    re = e('sub', t, e('mul', t, cr, cos), e('mul', t, ci, sin))
    im = e('add', t, e('mul', t, cr, sin), e('mul', t, ci, cos))
    e.done(e('mul', t, e('select', t, e('eq', BOOL, e.index(lead + 1), e.const(0, I32)), re, im), e.const(scale, t)))
    loops = [*lines, n, 2]
    crows = _rows(len(loops), [*range(lead + 1)])
    return ops + [_Generic(loops, [(torch.view_as_real(conv), _complex_rows(crows, len(loops), 0)),
                                   (torch.view_as_real(conv), _complex_rows(crows, len(loops), 1))],
                           [(torch.view_as_real(dst), _same(len(loops)))], e)]


def _scale(normalization, n):
    return 1.0 if normalization == 0 else 1 / math.sqrt(n) if normalization == 1 else 1.0 / n


def _moved(value, d):
    order = [j for j in range(value.dim()) if j != d] + [d]
    return value.permute(*order)


def _transformed(target, given, outs):
    packet, x, (out,) = target.overloadpacket, given['self'], outs
    dims = [d % x.dim() for d in given['dim']]
    normalization = given['normalization']
    real = x.dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16)
    t = F64 if x.dtype in (torch.float64, torch.complex128) else F32
    kind = torch.complex128 if t == F64 else torch.complex64
    ops = []
    if packet is A._fft_c2r:
        current = x
        for d in dims[:-1]:
            step = _temporary(current.shape, kind)
            ops += _fft(_moved(current, d), _moved(step, d), x.shape[d], 1, _scale(normalization, x.shape[d]), t)
            current = step
        last, length = dims[-1], given['last_dim_size']
        shape = list(current.shape)
        shape[last] = length
        full = _temporary(shape, kind)
        ops += _hermitian(current, full, last, length, t)
        whole = _temporary(shape, kind)
        ops += _fft(_moved(full, last), _moved(whole, last), length, 1, _scale(normalization, length), t)
        return ops + _copy(out, whole)
    forward = True if packet is A._fft_r2c else given['forward']
    sign = -1 if forward else 1
    shape = list(x.shape)
    current = x
    for j, d in enumerate(dims):
        step = _temporary(shape, kind)
        ops += _fft(_moved(current, d), _moved(step, d), x.shape[d], sign, _scale(normalization, x.shape[d]), t)
        current = step
    if packet is A._fft_r2c and given['onesided']:
        half = [slice(None)] * x.dim()
        half[dims[-1]] = slice(0, out.shape[dims[-1]])
        return ops + _copy(out, current[tuple(half)])
    return ops + _copy(out, current)


# the full spectrum of a real signal's half spectrum along d: X_k for k below the half's extent, conj(X_(L - k)) above
def _hermitian(half, full, d, length, t):
    n, m = full.dim(), half.shape[d]
    r = _Region()
    k = r.index(d)
    mirrored = r('sub', I32, r.const(length, I32), k)
    own = r('lt', BOOL, k, r.const(m, I32))
    at = r('select', I32, own, k, r('select', I32, r('lt', BOOL, mirrored, r.const(m, I32)), mirrored, r.const(0, I32)))
    pairs = torch.view_as_real(half)
    index = [r.index(j) if j != d else at for j in range(n)]
    re = _convert(r, r('extract', _class(pairs), *index, r.const(0, I32), integer=0), _class(pairs), t)
    im = _convert(r, r('extract', _class(pairs), *index, r.const(1, I32), integer=0), _class(pairs), t)
    r.done(r('select', t, r('eq', BOOL, r.index(n), r.const(0, I32)), re, r('select', t, own, im, r('neg', t, im))))
    return [_Generic([*full.shape, 2], [(pairs, None)], [(torch.view_as_real(full), _same(n + 1))], r)]

# torch 2.14.0 torch/_tensor_docs.py, fill_; PyTorch 2.14 documentation, torch.arange
def _fill(out, value, step=0):
    if out.is_complex():
        return [g for k in (0, 1) for g in _fill(torch.view_as_real(out)[..., k], _component(value, k),
                                                  _component(step, k))]
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


# an extremum's (value, index) states, joined in any association to the sequential result: the larger (smaller) value,
# NaN above every number, equal values to the earlier index; `nan_later` the NaN rule of a scan where each NaN replaces
# the kept one (max pooling), so of two NaNs the later; an index below 0 an absent state
def _ordered(c, a, b, t, larger, nan_later=False):
    (va, ia), (vb, ib) = a, b
    earlier = c('lt', BOOL, ib, ia)
    better = c('or', BOOL, c('gt' if larger else 'lt', BOOL, vb, va), c('and', BOOL, c('eq', BOOL, vb, va), earlier))
    if _floating(t):
        na, nb = c('isnan', BOOL, va), c('isnan', BOOL, vb)
        tie = c('gt', BOOL, ib, ia) if nan_later else earlier
        better = c('select', BOOL, c('or', BOOL, na, nb), c('and', BOOL, nb, c('or', BOOL, c('not', BOOL, na), tie)), better)
    zero = c.const(0, I32)
    right = c('and', BOOL, c('ge', BOOL, ib, zero), c('or', BOOL, c('lt', BOOL, ia, zero), better))
    return [c('select', t, right, vb, va), c('select', I32, right, ib, ia)]

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
    if out.is_complex():
        return _complex_contract(out, operands, labels, output, sizes, summed, addend, alpha, beta)
    t = _compute(out.dtype, *[value.dtype for value in operands], *[torch.float32] * out.dtype.is_floating_point)
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


# sum_k (ar_k br_k - ai_k bi_k, ar_k bi_k + ai_k br_k)
def _complex_contract(out, operands, labels, output, sizes, summed, addend, alpha, beta):
    domain = list(output) + summed
    at = {label: d for d, label in enumerate(domain)}
    n = len(domain)
    t = F64 if any(v.dtype in (torch.complex128, torch.float64) for v in (out, *operands)) else F32
    direct = alpha == 1 and (addend is None or beta == 0)
    acc = out if direct else _temporary(out.shape, out.dtype)
    r, inputs, product = _Region(), [], None
    for value, own in zip(operands, labels):
        rows = _rows(n, [None if k == 1 and sizes[label] != 1 else at[label] for k, label in zip(value.shape, own)])
        if value.is_complex():
            parts = _pair(r, inputs, value, rows, n, t)
            product = parts if product is None else _complex_value(r, t, 'mul', product, parts, 1, None)
        else:
            x = _load(r, len(inputs), value, t)
            inputs.append((value, rows))
            product = (x, r.const(0, t)) if product is None else (r('mul', t, product[0], x), r('mul', t, product[1], x))
    r.done(*product)
    rows = _rows(n, [at[label] for label in output])
    made = [_Generic([sizes[label] for label in domain], inputs,
                     [(acc.real, rows), (acc.imag, rows)], r,
                     [PARALLEL] * len(output) + [REDUCTION] * len(summed), [SUM, SUM], [t, t],
                     precision=REASSOCIATE | CONTRACT)]
    if direct:
        return made
    if alpha != 1:
        scaled = _temporary(out.shape, out.dtype)
        made += _complex('mul', scaled, acc, alpha)
        acc = scaled
    return made + (_complex('add', out, acc, addend, beta) if addend is not None and beta != 0 else
                   _complex('copy', out, acc))


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
        _typed(target, [], outs, CONTRACTED)
        return _contract(out, [a, b], [('n', 'k'), ('k', 'm')], ('n', 'm'), c if beta != 0 else None, alpha, beta)
    if packet is A.linear:
        x, w, bias = given['input'], given['weight'], given['bias']
        _typed(target, [], outs, CONTRACTED)
        rows, folded = _folded(x), _folded(out)
        if w.dim() == 2 and rows is not None and folded is not None:
            return _matrix(folded, rows, w.t(), bias)
        lead = tuple(('batch', i) for i in range(x.dim() - 1))
        return _contract(out, [x, w], [lead + ('k',), ('m', 'k')[2 - w.dim():]], lead + ('m',) * (w.dim() == 2), bias,
                         1, int(bias is not None))
    a, b = given['self'], given['mat2'] if packet in (A.mm, A.bmm) else given['other']
    _typed(target, [], outs, CONTRACTED)
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
    _typed(target, [], outs, CONTRACTED)
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


# Reference-LAPACK SRC/dgetf2.f, zgetf2.f, dgetrs.f, zgetrs.f, dpotf2.f and zpotf2.f;
# torch 2.14.0 aten/src/ATen/native/BatchLinearAlgebra.cpp
def _linalg_transpose(x, ops, transpose=True, conjugate=False):
    x = x.transpose(-2, -1) if transpose else x
    if conjugate and x.is_complex():
        made = _temporary(x.shape, x.dtype)
        ops += _complex('conj', made, x)
        return made
    return x


def _linalg_identity(out):
    r = _Region()
    target = torch.view_as_real(out) if out.is_complex() else out
    row = out.dim() - 2
    equal = r('eq', BOOL, r.index(row), r.index(row + 1))
    if out.is_complex():
        equal = r('and', BOOL, equal, r('eq', BOOL, r.index(out.dim()), r.const(0, I32)))
    r.done(r('cast', _class(target), equal))
    return [_Generic(target.shape, [], [(target, _same(target.dim()))], r)]


def _linalg_swap(x, pivot, row, ops):
    made, r = _temporary(x.shape, x.dtype), _Region()
    source = torch.view_as_real(x) if x.is_complex() else x
    target = torch.view_as_real(made) if made.is_complex() else made
    axis, t = x.dim() - 2, _class(source)
    index = pivot.reshape(*pivot.shape, *([1] * (source.dim() - pivot.dim())))
    selected = r('sub', I32, _load(r, 1, index, I32), r.const(1, I32))
    inputs = [(source, None), (index, _aligned(index, target.shape))]
    if isinstance(row, torch.Tensor):
        row = row.reshape(*row.shape, *([1] * (source.dim() - row.dim())))
        inputs.append((row, _aligned(row, target.shape)))
        origin = _load(r, 2, row, I32)
    else:
        origin = r.const(row, I32)
    at, own = [r.index(d) for d in range(source.dim())], r.index(axis)
    at[axis] = r('select', I32, r('eq', BOOL, own, origin), selected,
                 r('select', I32, r('eq', BOOL, own, selected), origin, own))
    r.done(r('extract', t, *at, integer=0))
    ops.append(_Generic(target.shape, inputs, [(target, _same(target.dim()))], r))
    return made


def _linalg_pivot_magnitude(x, ops):
    scores = _temporary(x.shape, x.real.dtype)
    ops += _elementwise('abs1' if x.is_complex() else 'abs', scores, x)
    return scores


def _linalg_lu(a, lu, pivots, info, pivot=True):
    ops, current = [], a
    ops += _fill(info, 0)
    m, n = a.shape[-2:]
    for j in range(min(m, n)):
        column, p = current[..., j:, j], pivots[..., j]
        if pivot:
            scores = _linalg_pivot_magnitude(column, ops)
            ops += _extreme(A.argmax.default, {'self': scores, 'dim': -1, 'keepdim': False}, [p])
            ops += _elementwise('add', p, p, j + 1)
            current = _linalg_swap(current, p, j, ops)
        else:
            ops += _fill(p, j + 1)
            if current is a:
                current = _temporary(a.shape, a.dtype)
                ops += _copy(current, a)
        diagonal = current[..., j, j]
        zero, first = _temporary(info.shape, torch.bool), _temporary(info.shape, torch.bool)
        ops += _elementwise('eq', zero, diagonal, 0)
        ops += _elementwise('eq', first, info, 0)
        ops += _elementwise('logical_and', first, first, zero)
        ops += _elementwise('where', info, first, j + 1, info)
        if j + 1 < m:
            safe = _temporary(diagonal.shape, diagonal.dtype)
            ops += _elementwise('where', safe, zero, 1, diagonal)
            below = current[..., j + 1:, j]
            ops += _elementwise('div', below, below, safe.unsqueeze(-1))
            if j + 1 < n:
                block = current[..., j + 1:, j + 1:]
                change = _temporary(block.shape, block.dtype)
                ops += _product(A.matmul.default, {'self': below.unsqueeze(-1),
                                                    'other': current[..., j, j + 1:].unsqueeze(-2)}, [change])
                ops += _elementwise('sub', block, block, change)
    return ops + _copy(lu, current)


def _linalg_component_maximum(x, dimensions, ops):
    dimensions = [d % x.dim() for d in dimensions]
    source = torch.view_as_real(x) if x.is_complex() else x
    magnitude = _temporary(source.shape, source.dtype)
    maximum = _temporary([s for d, s in enumerate(x.shape) if d not in dimensions], source.dtype)
    axes = dimensions + ([x.dim()] if x.is_complex() else [])
    ops += _elementwise('abs', magnitude, source)
    ops += _reduced(A.amax.default, {'self': magnitude, 'dim': axes, 'keepdim': False}, [maximum])
    return maximum


def _linalg_rescale(b, out, scale, ratio, ops):
    ops += _elementwise('mul', b, b, ratio.unsqueeze(-2))
    ops += _elementwise('mul', out, out, ratio.unsqueeze(-2))
    ops += _elementwise('mul', scale, scale, ratio)


def _linalg_triangular(a, b, out, upper, unit=False, scale=None):
    ops = []
    n = a.shape[-1]
    if scale is not None:
        rhs = _temporary(out.shape, out.dtype)
        ops += _assign(rhs, b) + _fill(out, 0) + _fill(scale, 1)
        b = rhs
    for j in range(n - 1, -1, -1) if upper else range(n):
        start, end = (j + 1, n) if upper else (0, j)
        result = out[..., j:j + 1, :]
        if scale is not None:
            limits = torch.finfo(out.real.dtype)
            big = limits.eps / limits.tiny
            magnitude = _linalg_component_maximum(b, [-2], ops)
            safe, ratio = (_temporary(scale.shape, out.real.dtype) for _ in range(2))
            zero = _temporary(scale.shape, torch.bool)
            ops += _elementwise('eq', zero, magnitude, 0)
            ops += _elementwise('where', safe, zero, 1, magnitude)
            ops += _elementwise('div', ratio, big, safe)
            ops += _elementwise('minimum', ratio, ratio, 1)
            if start != end:
                row = _linalg_component_maximum(a[..., j, start:end], [-1], ops).unsqueeze(-1)
                solved = _linalg_component_maximum(out[..., start:end, :], [-2], ops)
                bound, row_safe = (_temporary(scale.shape, out.real.dtype) for _ in range(2))
                row_zero = _temporary(scale.shape, torch.bool)
                ops += _elementwise('eq', row_zero, row, 0)
                ops += _elementwise('where', row_safe, row_zero, 1, row)
                ops += _elementwise('eq', zero, solved, 0)
                ops += _elementwise('where', safe, zero, 1, solved)
                ops += _elementwise('div', bound, big, row_safe)
                ops += _elementwise('div', bound, bound, safe)
                ops += _elementwise('div', bound, bound, (2 if out.is_complex() else 1) * (end - start))
                ops += _elementwise('logical_or', zero, zero, row_zero)
                ops += _elementwise('where', bound, zero, 1, bound)
                ops += _elementwise('minimum', ratio, ratio, bound)
            _linalg_rescale(b, out, scale, ratio, ops)
        if start != end:
            dot = _temporary(result.shape, result.dtype)
            ops += _product(A.matmul.default, {'self': a[..., j:j + 1, start:end],
                                                'other': out[..., start:end, :]}, [dot])
            ops += _elementwise('sub', result, b[..., j:j + 1, :], dot)
        else:
            ops += _assign(result, b[..., j:j + 1, :])
        if not unit:
            if scale is not None:
                magnitude, diagonal, ratio = (_temporary(scale.shape, out.real.dtype) for _ in range(3))
                large, zero = _temporary(scale.shape, torch.bool), _temporary(scale.shape, torch.bool)
                ops += _elementwise('abs', magnitude, result.squeeze(-2))
                ops += _elementwise('abs', diagonal, a[..., j, j].unsqueeze(-1))
                ops += _elementwise('gt', large, magnitude, diagonal)
                ops += _elementwise('eq', zero, magnitude, 0)
                ops += _elementwise('where', magnitude, zero, 1, magnitude)
                ops += _elementwise('div', ratio, diagonal, magnitude)
                ops += _elementwise('where', ratio, large, ratio, 1)
                _linalg_rescale(b, out, scale, ratio, ops)
            ops += _elementwise('div', result, result, a[..., j:j + 1, j:j + 1])
    return ops


def _linalg_lu_solve(lu, pivots, b, out, left=True, adjoint=False):
    ops = []
    target = out if left else out.transpose(-2, -1)
    rhs = (b if left else b.transpose(-2, -1)).expand(target.shape)
    transposed = left == adjoint
    factor = _linalg_transpose(lu, ops, transposed, adjoint)
    if not transposed:
        for j in range(pivots.shape[-1]):
            rhs = _linalg_swap(rhs, pivots[..., j], j, ops)
        middle = _temporary(target.shape, target.dtype)
        ops += _linalg_triangular(factor, rhs, middle, False, True)
        ops += _linalg_triangular(factor, middle, target, True)
    else:
        middle, result = _temporary(target.shape, target.dtype), _temporary(target.shape, target.dtype)
        ops += _linalg_triangular(factor, rhs, middle, False)
        ops += _linalg_triangular(factor, middle, result, True, True)
        for j in reversed(range(pivots.shape[-1])):
            result = _linalg_swap(result, pivots[..., j], j, ops)
        ops += _copy(target, result)
    return ops


def _linalg_cholesky(a, out, info, upper=False):
    ops, n = [], a.shape[-1]
    source = _linalg_transpose(a, ops, upper, upper)
    work, factor = _temporary(a.shape, a.dtype), _temporary(a.shape, a.dtype)
    ops += _copy(work, source) + _fill(factor, 0) + _fill(info, 0)
    for j in range(n):
        diagonal = work[..., j, j].real
        bad, first = _temporary(info.shape, torch.bool), _temporary(info.shape, torch.bool)
        root = _temporary(diagonal.shape, diagonal.dtype)
        ops += _elementwise('gt', bad, diagonal, 0)
        ops += _elementwise('logical_not', bad, bad)
        ops += _elementwise('eq', first, info, 0)
        ops += _elementwise('logical_and', first, first, bad)
        ops += _elementwise('where', info, first, j + 1, info)
        ops += _elementwise('sqrt', root, diagonal)
        ops += _elementwise('where', root, bad, diagonal, root)
        ops += _copy(factor[..., j, j], root)
        if j + 1 < n:
            column = factor[..., j + 1:, j:j + 1]
            ops += _elementwise('div', column, work[..., j + 1:, j:j + 1], root[..., None, None])
            row = _linalg_transpose(column, ops, True, True)
            change = _temporary(work[..., j + 1:, j + 1:].shape, a.dtype)
            ops += _product(A.matmul.default, {'self': column, 'other': row}, [change])
            ops += _elementwise('sub', work[..., j + 1:, j + 1:], work[..., j + 1:, j + 1:], change)
    return ops + _copy(out, _linalg_transpose(factor, ops, upper, upper))


def _linalg_symmetric(a, ops, hermitian=True, upper=False):
    matrix, other = _temporary(a.shape, a.dtype), _temporary(a.shape, a.dtype)
    opposite = _linalg_transpose(a, ops, True, hermitian)
    ops += _triangular_part(A.triu.default if upper else A.tril.default, {'self': a}, [matrix])
    ops += _triangular_part(A.tril.default if upper else A.triu.default,
                            {'self': opposite, 'diagonal': -1 if upper else 1}, [other])
    ops += _elementwise('add', matrix, matrix, other)
    if hermitian and a.is_complex():
        ops += _fill(matrix.diagonal(dim1=-2, dim2=-1).imag, 0)
    return matrix


# Reference-LAPACK SRC/dsytf2.f, zsytf2.f, zhetf2.f, dsytrs.f and zhetrs.f
def _linalg_diagonal_block(a, j, pair, hermitian, ops, active=None):
    width = min(2, a.shape[-1] - j)
    block = _temporary((*a.shape[:-2], width, width), a.dtype)
    ops += _fill(block, 0)
    if active is None:
        ops += _copy(block[..., 0, 0], a[..., j, j])
    else:
        ops += _elementwise('where', block[..., 0, 0], active, a[..., j, j], 1)
    if width == 2:
        ops += _elementwise('where', block[..., 1, 1], pair, a[..., j + 1, j + 1], 1)
        ops += _elementwise('where', block[..., 1, 0], pair, a[..., j + 1, j], 0)
        opposite = _linalg_transpose(block[..., 1:2, :1], ops, True, hermitian)
        ops += _copy(block[..., :1, 1:2], opposite)
    return block


def _linalg_block_solve(a, b, out, ops, left=True):
    lu, pivots = _temporary(a.shape, a.dtype), _temporary(a.shape[:-1], torch.int32)
    info = _temporary(a.shape[:-2], torch.int32)
    ops += _linalg_lu(a, lu, pivots, info)
    ops += _linalg_lu_solve(lu, pivots, b, out, left)


def _linalg_block_columns(a, j, pair, ops):
    columns = _temporary((*a.shape[:-2], a.shape[-1] - j - 1, 2), a.dtype)
    ops += _copy(columns[..., 0], a[..., j + 1:, j])
    ops += _elementwise('where', columns[..., 1], pair.unsqueeze(-1), a[..., j + 1:, j + 1], 0)
    ops += _elementwise('where', columns[..., 0, 0], pair, 0, columns[..., 0, 0])
    ops += _fill(columns[..., 0, 1], 0)
    return columns


def _linalg_ldl(a, ld, pivots, info, hermitian):
    ops, batch, n = [], a.shape[:-2], a.shape[-1]
    work = _linalg_symmetric(a, ops, hermitian)
    skip = _temporary(batch, torch.bool)
    ops += _fill(info, 0) + _fill(pivots, 0) + _fill(skip, False)
    alpha = (1 + math.sqrt(17)) / 8
    for j in range(n):
        active, bad, pair = (_temporary(batch, torch.bool) for _ in range(3))
        pivot = _temporary(batch, torch.int32)
        diagonal = _linalg_pivot_magnitude(work[..., j, j], ops)
        ops += _elementwise('logical_not', active, skip) + _fill(pair, False) + _fill(pivot, j + 1)
        if j + 1 < n:
            scores = _linalg_pivot_magnitude(work[..., j + 1:, j], ops)
            colmax, rowmax = (_temporary(batch, a.real.dtype) for _ in range(2))
            imax = _temporary(batch, torch.int32)
            ops += _extreme(A.max.dim, {'self': scores, 'dim': -1}, [colmax, imax])
            ops += _elementwise('add', imax, imax, 1)
            row = _temporary((*batch, n - j, 1), a.dtype)
            ops += _linalg_columns(work[..., j:, j:].transpose(-2, -1), imax.unsqueeze(-1), row)
            row = row[..., 0]
            row_scores = _linalg_pivot_magnitude(row, ops)
            selected = imax.unsqueeze(-1)
            r, t = _Region(), _class(row_scores)
            r.done(r('select', t, r('eq', BOOL, r.index(row_scores.dim() - 1),
                                   _load(r, 1, selected, I32)), r.const(0, t), _load(r, 0, row_scores, t)))
            ops.append(_Generic(row_scores.shape, [(row_scores, _same(row_scores.dim())),
                                (selected, _aligned(selected, row_scores.shape))],
                                [(row_scores, _same(row_scores.dim()))], r))
            ops += _reduced(A.amax.default, {'self': row_scores, 'dim': [-1], 'keepdim': False}, [rowmax])
            at = _temporary((*batch, 1, 1), a.dtype)
            ops += _linalg_columns(row.unsqueeze(-2), imax.unsqueeze(-1), at)
            alternate = _linalg_pivot_magnitude(at[..., 0, 0], ops)
            threshold, safe = (_temporary(batch, a.real.dtype) for _ in range(2))
            stay, again, other, zero = (_temporary(batch, torch.bool) for _ in range(4))
            ops += _elementwise('mul', threshold, colmax, alpha)
            ops += _elementwise('ge', stay, diagonal, threshold)
            ops += _elementwise('eq', zero, rowmax, 0)
            ops += _elementwise('where', safe, zero, 1, rowmax)
            ops += _elementwise('div', safe, colmax, safe)
            ops += _elementwise('mul', threshold, threshold, safe)
            ops += _elementwise('ge', again, diagonal, threshold)
            ops += _elementwise('logical_or', stay, stay, again)
            ops += _elementwise('mul', threshold, rowmax, alpha)
            ops += _elementwise('ge', other, alternate, threshold)
            ops += _elementwise('logical_or', pair, stay, other)
            ops += _elementwise('logical_not', pair, pair)
            ops += _elementwise('add', imax, imax, j + 1)
            ops += _elementwise('where', pivot, stay, j + 1, imax)
            ops += _elementwise('maximum', threshold, diagonal, colmax)
            ops += _elementwise('eq', bad, threshold, 0)
        else:
            ops += _elementwise('eq', bad, diagonal, 0)
        nan, usable, first = (_temporary(batch, torch.bool) for _ in range(3))
        ops += _elementwise('isnan', nan, diagonal)
        ops += _elementwise('logical_or', bad, bad, nan)
        ops += _elementwise('logical_and', bad, bad, active)
        ops += _elementwise('eq', first, info, 0)
        ops += _elementwise('logical_and', first, first, bad)
        ops += _elementwise('where', info, first, j + 1, info)
        ops += _elementwise('logical_not', usable, bad)
        ops += _elementwise('logical_and', usable, usable, active)
        ops += _elementwise('logical_and', pair, pair, usable)
        ops += _elementwise('where', pivot, usable, pivot, j + 1)
        signed = _temporary(batch, torch.int32)
        ops += _elementwise('neg', signed, pivot)
        ops += _elementwise('where', signed, pair, signed, pivot)
        ops += _elementwise('where', pivots[..., j], active, signed, pivots[..., j])
        if j + 1 < n:
            ops += _elementwise('where', pivots[..., j + 1], pair, signed, pivots[..., j + 1])
            local, target = _temporary(batch, torch.int32), _temporary(batch, torch.int32)
            ops += _elementwise('sub', local, pivot, j) + _copy(target, pair)
            trailing = _linalg_swap(work[..., j:, j:], local, target, ops)
            trailing = _linalg_swap(trailing.transpose(-2, -1), local, target, ops).transpose(-2, -1)
            block = _linalg_diagonal_block(trailing, 0, pair, hermitian, ops, usable)
            columns = _linalg_block_columns(trailing, 0, pair, ops)
            lower = _temporary(columns.shape, columns.dtype)
            _linalg_block_solve(block, columns, lower, ops, False)
            change = _temporary(trailing[..., 1:, 1:].shape, a.dtype)
            opposite = _linalg_transpose(columns, ops, True, hermitian)
            ops += _product(A.matmul.default, {'self': lower, 'other': opposite}, [change])
            ops += _elementwise('sub', trailing[..., 1:, 1:], trailing[..., 1:, 1:], change)
            ops += _elementwise('where', lower[..., 0, 0], pair, trailing[..., 1, 0], lower[..., 0, 0])
            ops += _copy(trailing[..., 1:, 0], lower[..., 0])
            if j + 2 < n:
                ops += _elementwise('where', trailing[..., 2:, 1], pair.unsqueeze(-1),
                                     lower[..., 1:, 1], trailing[..., 2:, 1])
            if hermitian and a.is_complex():
                ops += _fill(trailing.diagonal(dim1=-2, dim2=-1).imag, 0)
            ops += _elementwise('where', work[..., j:, j:], usable[..., None, None], trailing, work[..., j:, j:])
        ops += _copy(skip, pair)
    return ops + _triangular_part(A.tril.default, {'self': work}, [ld])


def _linalg_ldl_solve(ld, pivots, b, out, hermitian):
    batch, n, ops = out.shape[:-2], out.shape[-2], []
    factor = ld.expand(*batch, n, n)
    pivots = pivots.expand(*batch, n)
    work = _temporary(out.shape, out.dtype)
    skip = _temporary(batch, torch.bool)
    ops += _assign(work, b) + _fill(skip, False)
    for j in range(n):
        active, pair = _temporary(batch, torch.bool), _temporary(batch, torch.bool)
        pivot, target = _temporary(batch, torch.int32), _temporary(batch, torch.int32)
        ops += _elementwise('logical_not', active, skip)
        ops += _elementwise('lt', pair, pivots[..., j], 0)
        ops += _elementwise('logical_and', pair, pair, active)
        ops += _elementwise('abs', pivot, pivots[..., j])
        ops += _copy(target, pair) + _elementwise('add', target, target, j)
        moved = _linalg_swap(work, pivot, target, ops)
        block = _linalg_diagonal_block(factor, j, pair, hermitian, ops, active)
        width = block.shape[-1]
        rhs = _temporary((*batch, width, out.shape[-1]), out.dtype)
        ops += _copy(rhs, moved[..., j:j + width, :])
        if j + 1 < n:
            columns = _linalg_block_columns(factor, j, pair, ops)
            change = _temporary(moved[..., j + 1:, :].shape, out.dtype)
            ops += _product(A.matmul.default, {'self': columns, 'other': rhs}, [change])
            ops += _elementwise('sub', moved[..., j + 1:, :], moved[..., j + 1:, :], change)
        solution = _temporary(rhs.shape, rhs.dtype)
        _linalg_block_solve(block, rhs, solution, ops)
        ops += _copy(moved[..., j, :], solution[..., 0, :])
        if width == 2:
            ops += _elementwise('where', moved[..., j + 1, :], pair.unsqueeze(-1),
                                 solution[..., 1, :], moved[..., j + 1, :])
        ops += _elementwise('where', work, active[..., None, None], moved, work)
        ops += _copy(skip, pair)
    ops += _fill(skip, False)
    for j in reversed(range(n)):
        active, pair = _temporary(batch, torch.bool), _temporary(batch, torch.bool)
        pivot = _temporary(batch, torch.int32)
        ops += _elementwise('logical_not', active, skip)
        ops += _elementwise('lt', pair, pivots[..., j], 0)
        ops += _elementwise('logical_and', pair, pair, active)
        ops += _elementwise('abs', pivot, pivots[..., j])
        start = max(j - 1, 0)
        width = j - start + 1
        made = _temporary(work.shape, work.dtype)
        ops += _copy(made, work)
        if j + 1 < n:
            lower = factor[..., j + 1:, start:j + 1]
            transposed = _linalg_transpose(lower, ops, True, hermitian)
            change = _temporary((*batch, width, out.shape[-1]), out.dtype)
            ops += _product(A.matmul.default, {'self': transposed, 'other': work[..., j + 1:, :]}, [change])
            ops += _elementwise('sub', made[..., j, :], work[..., j, :], change[..., -1, :])
            if j:
                ops += _elementwise('where', change[..., 0, :], pair.unsqueeze(-1), change[..., 0, :], 0)
                ops += _elementwise('sub', made[..., j - 1, :], work[..., j - 1, :], change[..., 0, :])
        made = _linalg_swap(made, pivot, j, ops)
        ops += _elementwise('where', work, active[..., None, None], made, work)
        ops += _copy(skip, pair)
    return ops + _copy(out, work)


FACTORIZATIONS = (A.linalg_lu_factor_ex, A._linalg_solve_ex, A.linalg_solve_ex, A.linalg_inv_ex,
                  A.linalg_cholesky_ex, A.linalg_solve_triangular, A.triangular_solve, A.linalg_lu_solve,
                  A.cholesky_solve, A.cholesky_inverse, A.linalg_ldl_factor_ex, A.linalg_ldl_solve)


def _linalg_vector_rhs(a, b):
    return b.dim() == 1 or b.dim() == a.dim() - 1 and b.shape == a.shape[:-1]


def _factorization(target, given, outs):
    packet, ops = target.overloadpacket, []
    a = given.get('A', given.get('self'))
    if packet is A.linalg_lu_factor_ex:
        return _linalg_lu(a, *outs, given['pivot'])
    if packet is A.linalg_cholesky_ex:
        return _linalg_cholesky(a, *outs, given['upper'])
    if packet is A.linalg_ldl_factor_ex:
        return _linalg_ldl(a, *outs, given['hermitian'])
    if packet is A.linalg_ldl_solve:
        return _linalg_ldl_solve(given['LD'], given['pivots'], given['B'], outs[0], given['hermitian'])
    if packet is A.linalg_lu_solve:
        return _linalg_lu_solve(given['LU'], given['pivots'], given['B'], outs[0], given['left'], given['adjoint'])
    if packet in (A.linalg_solve_triangular, A.triangular_solve):
        b = given['B'] if packet is A.linalg_solve_triangular else given['self']
        left, upper = given.get('left', True), given['upper']
        transpose = given.get('transpose', False) != (not left)
        factor = a.transpose(-2, -1) if transpose else a
        ops += _linalg_triangular(factor, b if left else b.transpose(-2, -1),
                                  outs[0] if left else outs[0].transpose(-2, -1), upper != transpose,
                                  given['unitriangular'])
        return ops + (_copy(outs[1], a) if len(outs) > 1 else [])
    if packet in (A.cholesky_solve, A.cholesky_inverse):
        factor = given['input2'] if packet is A.cholesky_solve else a
        b = given['self'] if packet is A.cholesky_solve else _temporary(outs[0].shape, outs[0].dtype)
        if packet is A.cholesky_inverse:
            ops += _linalg_identity(b)
        lower = _linalg_transpose(factor, ops, given['upper'], given['upper'])
        middle = _temporary(outs[0].shape, outs[0].dtype)
        ops += _linalg_triangular(lower, b, middle, False)
        ops += _linalg_triangular(_linalg_transpose(lower, ops, True, True), middle, outs[0], True)
        return ops
    result, info = outs[0], outs[-1]
    lu, pivots = outs[1:3] if packet is A._linalg_solve_ex else \
        (_temporary(a.shape, a.dtype), _temporary(a.shape[:-1], torch.int32))
    ops += _linalg_lu(a, lu, pivots, info)
    b = given.get('B')
    if b is None:
        b = _temporary(result.shape, result.dtype)
        ops += _linalg_identity(b)
    vector = _linalg_vector_rhs(a, b)
    ops += _linalg_lu_solve(lu, pivots, b.unsqueeze(-1) if vector else b,
                           result.unsqueeze(-1) if vector else result, given.get('left', True))
    return ops


class _Check:
    def __init__(self, info, code, message=''):
        self.info, self.code, self.place, self.node = info, code, CPU, 0
        self.message = message


def _linalg_nonsingular(diagonal, code, message):
    if not diagonal.shape[-1]:
        return []
    zero = _temporary(diagonal.shape, torch.bool)
    scores = _temporary(diagonal.shape, torch.int32)
    any_zero, info = _temporary(diagonal.shape[:-1], torch.bool), _temporary(diagonal.shape[:-1], torch.int32)
    ops = _elementwise('eq', zero, diagonal, 0) + _copy(scores, zero)
    ops += _extreme(A.argmax.default, {'self': scores, 'dim': -1, 'keepdim': False}, [info])
    ops += _elementwise('add', info, info, 1)
    ops += _fold(A.any.dim, {'self': zero, 'dim': -1, 'keepdim': False}, [any_zero])
    ops += _elementwise('where', info, any_zero, info, 0)
    return ops + [_Check(info, code, message)]


def _factorized(target, given, outs):
    ops = _factorization(target, given, outs)
    if target.overloadpacket is A.cholesky_inverse:
        ops = _linalg_nonsingular(given['self'].diagonal(dim1=-2, dim2=-1), 2, 'cholesky_inverse') + ops
    if given.get('check_errors'):
        ops.append(_Check(outs[-1], 2 if target.overloadpacket is A.linalg_cholesky_ex else 1))
    return ops


def _checked(target, given, outs):
    code = 2 if given['api_name'] in ('linalg.cholesky', 'linalg.cholesky_ex', 'cholesky') else 1
    return [_Check(given['info'], code, given['api_name'])]


def _determinant(target, given, outs, transpose=False):
    a, logarithm = given['A'], target.overloadpacket is A._linalg_slogdet
    value, lu, pivots = outs[0], outs[-2], outs[-1]
    info = _temporary(a.shape[:-2], torch.int32)
    ops = _linalg_lu(a.transpose(-2, -1) if transpose else a, lu, pivots, info)
    ops += _fill(value, 1)
    if logarithm:
        ops += _fill(outs[1], 0)
    for j in range(a.shape[-1]):
        diagonal = lu[..., j, j]
        swap, negative = _temporary(value.shape, torch.bool), _temporary(value.shape, value.dtype)
        if logarithm:
            magnitude, safe = _temporary(value.shape, a.real.dtype), _temporary(value.shape, a.real.dtype)
            zero, phase = _temporary(value.shape, torch.bool), _temporary(value.shape, a.dtype)
            ops += _elementwise('abs', magnitude, diagonal)
            ops += _elementwise('eq', zero, magnitude, 0)
            ops += _elementwise('where', safe, zero, 1, magnitude)
            ops += _elementwise('div', phase, diagonal, safe)
            ops += _elementwise('mul', value, value, phase)
            ops += _elementwise('log', magnitude, magnitude)
            ops += _elementwise('add', outs[1], outs[1], magnitude)
        else:
            ops += _elementwise('mul', value, value, diagonal)
        ops += _elementwise('ne', swap, pivots[..., j], j + 1)
        ops += _elementwise('neg', negative, value)
        ops += _elementwise('where', value, swap, negative, value)
    return ops


def _unpacked_lu(target, given, outs):
    a, (permutation, lower, upper) = given['A'], outs
    m, n = a.shape[-2:]
    packed, pivots = _temporary(a.shape, a.dtype), _temporary((*a.shape[:-2], min(m, n)), torch.int32)
    info = _temporary(a.shape[:-2], torch.int32)
    ops = _linalg_lu(a, packed, pivots, info, given['pivot'])
    ops += _triangular_part(A.tril.default, {'self': packed[..., :, :min(m, n)], 'diagonal': -1}, [lower])
    ops += _fill(lower.diagonal(dim1=-2, dim2=-1), 1)
    ops += _triangular_part(A.triu.default, {'self': packed[..., :min(m, n), :]}, [upper])
    if permutation.numel():
        ops += _linalg_identity(permutation)
        current = permutation
        for j in reversed(range(min(m, n))):
            current = _linalg_swap(current, pivots[..., j], j, ops)
        ops += _copy(permutation, current)
    return ops


ASSERTIONS = (A._assert_async, A._assert_scalar, A._assert_tensor_metadata,
              A.sym_constrain_range, A.sym_constrain_range_for_size)


def _asserted(target, given, outs):
    if target.overloadpacket is A._assert_tensor_metadata:
        return []
    value = given.get('self', given.get('size'))
    if not isinstance(value, torch.Tensor):
        target(**given)
        return []
    predicate = _temporary(value.shape, torch.bool)
    if target.overloadpacket in (A.sym_constrain_range, A.sym_constrain_range_for_size):
        ops = _fill(predicate, True)
        for name, comparison in (('min', 'ge'), ('max', 'le')):
            if given.get(name) is not None:
                bound = _temporary(value.shape, torch.bool)
                ops += _elementwise(comparison, bound, value, given[name])
                ops += _elementwise('logical_and', predicate, predicate, bound)
    else:
        ops = _elementwise('ne', predicate, value, 0)
    info = _temporary(value.shape, torch.int32)
    ops += _elementwise('where', info, predicate, 0, 1)
    return ops + [_Check(info, 4, given.get('assert_msg', str(target)))]


# Reference-LAPACK SRC/dlarfg.f, zlarfg.f, dgeqr2.f, zgeqr2.f, dorg2r.f and zung2r.f
def _linalg_reflector(column, tau, beta, ops):
    real, batch = column.real.dtype, column.shape[:-1]
    magnitudes, scale = _temporary(column.shape, real), _temporary(batch, real)
    ops += _elementwise('abs', magnitudes, column)
    ops += _reduced(A.amax.default, {'self': magnitudes, 'dim': [-1], 'keepdim': False}, [scale])
    zero, safe = _temporary(batch, torch.bool), _temporary(batch, real)
    ops += _elementwise('eq', zero, scale, 0)
    ops += _elementwise('where', safe, zero, 1, scale)
    normalized = _temporary(column.shape, column.dtype)
    ops += _elementwise('div', normalized, column, safe.unsqueeze(-1))
    ops += _elementwise('div', magnitudes, magnitudes, safe.unsqueeze(-1))
    squares, norm = _temporary(column.shape, real), _temporary(batch, real)
    ops += _elementwise('mul', squares, magnitudes, magnitudes)
    ops += _reduced(A.sum.dim_IntList, {'self': squares, 'dim': [-1], 'keepdim': False}, [norm])
    ops += _elementwise('sqrt', norm, norm)
    alpha, signed, negative = normalized[..., 0], _temporary(batch, real), _temporary(batch, torch.bool)
    ops += _elementwise('lt', negative, alpha.real, 0)
    ops += _elementwise('neg', signed, norm)
    ops += _elementwise('where', signed, negative, norm, signed)
    inactive = _temporary(batch, torch.bool)
    if column.shape[-1] > 1:
        tail = _temporary(batch, real)
        ops += _reduced(A.amax.default, {'self': magnitudes[..., 1:], 'dim': [-1], 'keepdim': False}, [tail])
        ops += _elementwise('eq', inactive, tail, 0)
    else:
        ops += _fill(inactive, True)
    if column.is_complex():
        real_alpha = _temporary(batch, torch.bool)
        ops += _elementwise('eq', real_alpha, alpha.imag, 0)
        ops += _elementwise('logical_and', inactive, inactive, real_alpha)
    denominator = _temporary(batch, column.dtype)
    ops += _elementwise('sub', denominator, alpha, signed)
    ops += _elementwise('where', denominator, inactive, 1, denominator)
    v = _temporary(column.shape, column.dtype)
    ops += _elementwise('div', v, normalized, denominator.unsqueeze(-1))
    ops += _fill(v[..., 0], 1)
    ops += _elementwise('where', safe, inactive, 1, signed)
    ops += _elementwise('sub', tau, signed, alpha)
    ops += _elementwise('div', tau, tau, safe)
    ops += _elementwise('where', tau, inactive, 0, tau)
    ops += _elementwise('mul', beta, signed, scale)
    ops += _elementwise('where', beta, inactive, column[..., 0].real, beta)
    return v.unsqueeze(-1)


def _linalg_reflect(block, v, tau, ops, left=True, adjoint=False):
    vh = _linalg_transpose(v, ops, True, True)
    scalar = _linalg_transpose(tau[..., None, None], ops, False, adjoint)
    dot = _temporary((*block.shape[:-2], 1, block.shape[-1]) if left else
                     (*block.shape[:-2], block.shape[-2], 1), block.dtype)
    ops += _product(A.matmul.default, {'self': vh if left else block, 'other': block if left else v}, [dot])
    ops += _elementwise('mul', dot, dot, scalar)
    change = _temporary(block.shape, block.dtype)
    ops += _product(A.matmul.default, {'self': v if left else dot, 'other': dot if left else vh}, [change])
    ops += _elementwise('sub', block, block, change)


def _linalg_geqrf(a, packed, tau, pivots=None):
    ops = _copy(packed, a)
    m, n = a.shape[-2:]
    for j in range(min(m, n)):
        if pivots is not None:
            norms = _temporary((*a.shape[:-2], n - j), a.real.dtype)
            _linalg_lengths(packed[..., j:, j:], norms, ops, normalize=False)
            ops += _extreme(A.argmax.default, {'self': norms, 'dim': -1, 'keepdim': False}, [pivots[..., j]])
            ops += _elementwise('add', pivots[..., j], pivots[..., j], j + 1)
            moved = _linalg_swap(packed.transpose(-2, -1), pivots[..., j], j, ops)
            ops += _copy(packed, moved.transpose(-2, -1))
        beta = _temporary(a.shape[:-2], a.real.dtype)
        v = _linalg_reflector(packed[..., j:, j], tau[..., j], beta, ops)
        if j + 1 < n:
            _linalg_reflect(packed[..., j:, j + 1:], v, tau[..., j], ops, adjoint=True)
        ops += _copy(packed[..., j, j], beta)
        if j + 1 < m:
            ops += _copy(packed[..., j + 1:, j], v[..., 1:, 0])
    return ops


def _linalg_householder(packed, tau, out, b=None, left=True, adjoint=False):
    ops = _linalg_identity(out) if b is None else _assign(out, b)
    order = range(tau.shape[-1]) if left == adjoint else reversed(range(tau.shape[-1]))
    for j in order:
        v = _temporary((*packed.shape[:-2], packed.shape[-2] - j, 1), packed.dtype)
        ops += _fill(v[..., 0, 0], 1)
        if v.shape[-2] > 1:
            ops += _copy(v[..., 1:, 0], packed[..., j + 1:, j])
        _linalg_reflect(out[..., j:, :] if left else out[..., :, j:], v, tau[..., j], ops, left, adjoint)
    return ops


def _triangular_part(target, given, outs):
    x, (out,), r = given['self'], outs, _Region()
    source = torch.view_as_real(x) if x.is_complex() else x
    made = torch.view_as_real(out) if out.is_complex() else out
    diagonal, t = given.get('diagonal', 0), _class(source)
    distance = r('sub', I32, r.index(out.dim() - 1), r.index(out.dim() - 2))
    keep = r('ge' if target.overloadpacket is A.triu else 'le', BOOL, distance, r.const(diagonal, I32))
    r.done(r('select', t, keep, r.arg(0, source), r.const(0, t)))
    return [_Generic(made.shape, [(source, _same(source.dim()))], [(made, _same(made.dim()))], r)]


ORTHOGONAL = (A.linalg_qr, A.geqrf, A.linalg_householder_product, A.ormqr)


# PyTorch v2.14.0 aten/src/ATen/native/BatchLinearAlgebra.cpp, linalg_lstsq_out_info and get_default_lstsq_driver;
# PyTorch 2.14 torch.library.register_fake
def _lstsq_driver(a, driver):
    return driver.lower() if driver is not None else ('gels' if a.device.type == 'cuda' else 'gelsy')


@torch.library.register_fake('aten::geqrf')
def _geqrf_metadata(self):
    torch._check(self.dim() >= 2, lambda: 'torch.geqrf: input must have at least 2 dimensions')
    m, n = self.shape[-2:]
    return (self.new_empty((*self.shape[:-2], n, m)).transpose(-2, -1),
            self.new_empty((*self.shape[:-2], min(m, n))))


@torch.library.register_fake('aten::linalg_lstsq')
def _lstsq_metadata(self, b, rcond=None, *, driver=None):
    torch._check(self.dim() >= 2 and b.dim() >= 1,
                 lambda: 'torch.linalg.lstsq: input must have at least 2 dimensions and other at least 1')
    torch._check(0 <= self.dim() - b.dim() <= 1,
                 lambda: 'torch.linalg.lstsq: input.dim() - other.dim() must be 0 or 1')
    torch._check(self.dtype == b.dtype and self.device == b.device,
                 lambda: 'torch.linalg.lstsq: input and other must have the same dtype and device')
    selected = _lstsq_driver(self, driver)
    torch._check(selected in ('gels', 'gelsy', 'gelsd', 'gelss'),
                 lambda: 'torch.linalg.lstsq: driver must be gels, gelsy, gelsd or gelss')
    torch._check(self.device.type != 'cuda' or selected == 'gels',
                 lambda: 'torch.linalg.lstsq: driver other than gels is not supported on CUDA')
    vector = _linalg_vector_rhs(self, b)
    rhs = b.unsqueeze(-1) if vector else b
    m, n = self.shape[-2:]
    torch._check(m == rhs.shape[-2], lambda: 'torch.linalg.lstsq: input and other row sizes must match')
    batch = torch.broadcast_shapes(self.shape[:-2], rhs.shape[:-2])
    nrhs = rhs.shape[-1]
    solution = self.new_empty((*batch, nrhs, max(m, n))).transpose(-2, -1)[..., :n, :]
    if vector:
        solution = solution.squeeze(-1)
    residual = self.real.new_empty((*batch, nrhs) if m > n and (selected == 'gels' or n == 0 and selected != 'gelsy')
                                   else (0,))
    if m > n > 0 and selected in ('gelsd', 'gelss'):
        from torch._subclasses.fake_tensor import DynamicOutputShapeException
        raise DynamicOutputShapeException(A.linalg_lstsq.default)
    rank = self.new_empty(self.shape[:-2] if selected != 'gels' else (0,), dtype=torch.int64)
    singular = self.real.new_empty((*self.shape[:-2], min(m, n)) if selected in ('gelsd', 'gelss') else (0,))
    return solution, residual, rank, singular


# design/torch-source-lowering-change-request.md item 5; metal-microbench metal_ir.h metal_ir_lapack
GESDD, HEEVD, GEEV, GELSD, GELSS = range(5)


class _Lapack:
    def __init__(self, routine, job, batch, inputs, outputs, rcond=0.0):
        self.routine, self.job, self.batch, self.rcond = routine, job, tuple(batch), rcond
        self.inputs, self.outputs = list(inputs), list(outputs)
        self.place, self.node = CPU, 0


def _singular(target, given, outs):
    a, (u, singular, vh) = given['A'], outs
    batch = a.shape[:-2]
    if not a.numel():
        return (_linalg_identity(u) if u.numel() else []) + (_linalg_identity(vh) if vh.numel() else [])
    info = _temporary(batch, torch.int64)
    vectors = [(u, 2), (vh, 2)] if given['compute_uv'] else []
    job = b'N' if not vectors else b'A' if given['full_matrices'] else b'S'
    return [_Lapack(GESDD, job, batch, [(a, 2)], [(singular, 1), *vectors, (info, 0)]), _Check(info, 3, 'linalg.svd')]


def _eigen_hermitian(target, given, outs):
    a, (values, vectors) = given['A'], outs
    batch = a.shape[:-2]
    if not a.numel():
        return []
    info = _temporary(batch, torch.int64)
    job = (b'V' if given['compute_v'] else b'N') + given['UPLO'].upper().encode()
    return [_Lapack(HEEVD, job, batch, [(a, 2)],
                    [(values, 1), *([(vectors, 2)] if given['compute_v'] else []), (info, 0)]),
            _Check(info, 3, 'linalg.eigh')]


# Reference-LAPACK SRC/dgeev.f
def _eigen(target, given, outs):
    a, values = given['self'], outs[0]
    batch, n = a.shape[:-2], a.shape[-1]
    if not a.numel():
        return []
    info = _temporary(batch, torch.int64)
    vectors = outs[1] if len(outs) > 1 else None
    job = b'N' if vectors is None else b'V'
    if a.is_complex():
        return [_Lapack(GEEV, job, batch, [(a, 2)],
                        [(values, 1), *([] if vectors is None else [(vectors, 2)]), (info, 0)]),
                _Check(info, 3, 'linalg.eig')]
    wr, wi = _temporary((*batch, n), a.dtype), _temporary((*batch, n), a.dtype)
    packed = None if vectors is None else _temporary(a.shape, a.dtype)
    ops = [_Lapack(GEEV, job, batch, [(a, 2)], [(wr, 1), (wi, 1), *([] if packed is None else [(packed, 2)]), (info, 0)]),
           _Check(info, 3, 'linalg.eig')]
    ops += _complex('complex', values, wr, wi)
    if packed is not None:
        before, after = _temporary(a.shape, a.dtype), _temporary(a.shape, a.dtype)
        ops += _fill(before, 0) + _fill(after, 0)
        if n > 1:
            ops += _copy(before[..., :, 1:], packed[..., :, :-1]) + _copy(after[..., :, :-1], packed[..., :, 1:])
        first, second = _temporary((*batch, 1, n), torch.bool), _temporary((*batch, 1, n), torch.bool)
        ops += _elementwise('gt', first, wi.unsqueeze(-2), 0) + _elementwise('lt', second, wi.unsqueeze(-2), 0)
        real, imaginary = _temporary(a.shape, a.dtype), _temporary(a.shape, a.dtype)
        ops += _elementwise('where', real, second, before, packed)
        ops += _elementwise('neg', imaginary, packed) + _elementwise('where', imaginary, second, imaginary, 0)
        ops += _elementwise('where', imaginary, first, after, imaginary)
        ops += _complex('complex', vectors, real, imaginary)
    return ops


# PyTorch 2.15 documentation, torch.linalg.pinv
def _pseudoinverse(target, given, outs, default_tolerance_dtype=torch.float64):
    a, (out,), ops = given['self'], outs, []
    if not a.numel():
        return _copy(out, a.transpose(-2, -1))
    batch, (m, n), k = a.shape[:-2], a.shape[-2:], min(a.shape[-2:])
    values = _temporary((*batch, k), a.real.dtype)
    if given['hermitian']:
        vectors = _temporary(a.shape, a.dtype)
        ops += _eigen_hermitian(A._linalg_eigh.default, {'A': a, 'UPLO': 'L', 'compute_v': True}, [values, vectors])
        left, right = vectors, _linalg_transpose(vectors, ops, True, True)
    else:
        u, vh = _temporary((*batch, m, k), a.dtype), _temporary((*batch, k, n), a.dtype)
        ops += _singular(A._linalg_svd.default, {'A': a, 'full_matrices': False, 'compute_uv': True}, [u, values, vh])
        left, right = _linalg_transpose(vh, ops, True, True), _linalg_transpose(u, ops, True, True)
    absolute, maximum = _temporary(values.shape, values.dtype), _temporary(batch, values.dtype)
    ops += _elementwise('abs', absolute, values)
    ops += _reduced(A.amax.default, {'self': absolute, 'dim': [-1], 'keepdim': False}, [maximum])
    batch = out.shape[:-2]
    atol, rtol = given.get('atol'), given.get('rtol', given.get('rcond'))
    tolerance_dtype = values.dtype
    for value in (atol, rtol):
        tolerance_dtype = torch.promote_types(tolerance_dtype, value.dtype if isinstance(value, torch.Tensor)
                                               else default_tolerance_dtype)
    atol = 0 if atol is None else atol
    if rtol is None:
        default = max(m, n) * torch.finfo(a.real.dtype).eps
        positive = _temporary(batch, torch.bool)
        rtol = _temporary(batch, tolerance_dtype)
        ops += _elementwise('gt', positive, atol, 0)
        ops += _elementwise('where', rtol, positive, 0, default)
    tolerance = _temporary(batch, tolerance_dtype)
    ops += _elementwise('mul', tolerance, maximum, rtol)
    ops += _elementwise('maximum', tolerance, tolerance, atol)
    keep, reciprocal = _temporary((*batch, k), torch.bool), _temporary((*batch, k), values.dtype)
    ops += _elementwise('gt', keep, absolute, tolerance.unsqueeze(-1))
    ops += _elementwise('where', reciprocal, keep, values, 1)
    ops += _elementwise('reciprocal', reciprocal, reciprocal)
    ops += _elementwise('where', reciprocal, keep, reciprocal, 0)
    scaled = _temporary((*batch, *left.shape[-2:]), left.dtype)
    ops += _elementwise('mul', scaled, left, reciprocal.unsqueeze(-2))
    return ops + _product(A.matmul.default, {'self': scaled, 'other': right}, [out])


def _orthogonal(target, given, outs):
    packet = target.overloadpacket
    if packet is A.geqrf:
        return _linalg_geqrf(given['self'], *outs)
    if packet is A.linalg_householder_product:
        return _linalg_householder(given['input'], given['tau'], outs[0])
    if packet is A.ormqr:
        return _linalg_householder(given['self'], given['input2'], outs[0], given['input3'], given['left'],
                                   given['transpose'])
    a, (q, r) = given['A'], outs
    packed, tau = _temporary(a.shape, a.dtype), _temporary((*a.shape[:-2], min(a.shape[-2:])), a.dtype)
    ops = _linalg_geqrf(a, packed, tau)
    if q.numel():
        ops += _linalg_householder(packed, tau, q)
    return ops + _triangular_part(A.triu.default, {'self': packed[..., :r.shape[-2], :]}, [r])


# G. H. Golub and C. F. Van Loan, Matrix Computations, 3rd ed., section 8.4, Algorithms 8.4.1 and 8.4.3;
# Z. Drmac and K. Veselic, LAPACK Working Note 196, sections 2 and 3; Reference-LAPACK SRC/dgesvj.f
def _linalg_active(app, aqq, magnitude, tolerance, ops):
    left, right = (_temporary(magnitude.shape, magnitude.dtype) for _ in range(2))
    active = _temporary(magnitude.shape, torch.bool)
    ops += _elementwise('abs', left, app) + _elementwise('sqrt', left, left)
    ops += _elementwise('abs', right, aqq) + _elementwise('sqrt', right, right)
    ops += _elementwise('mul', left, left, tolerance)
    ops += _elementwise('mul', left, left, right)
    ops += _elementwise('gt', active, magnitude, left)
    return active


def _linalg_rotation(app, aqq, apq, ops, tolerance=None, active=None):
    batch, real = app.shape, app.dtype
    magnitude, delta = _temporary(batch, real), _temporary(batch, real)
    d, g, scale = (_temporary(batch, real) for _ in range(3))
    ops += _elementwise('abs', magnitude, apq)
    ops += _elementwise('mul', d, aqq, 0.5)
    ops += _elementwise('mul', g, app, 0.5)
    ops += _elementwise('sub', delta, d, g)
    ops += _elementwise('abs', scale, delta)
    ops += _elementwise('maximum', scale, scale, magnitude)
    zero = _temporary(batch, torch.bool)
    ops += _elementwise('eq', zero, scale, 0)
    ops += _elementwise('where', scale, zero, 1, scale)
    ops += _elementwise('div', d, delta, scale)
    ops += _elementwise('div', g, magnitude, scale)
    square, tangent, cosine = (_temporary(batch, real) for _ in range(3))
    ops += _elementwise('mul', square, d, d)
    ops += _elementwise('mul', tangent, g, g)
    ops += _elementwise('add', square, square, tangent)
    ops += _elementwise('sqrt', square, square)
    ops += _elementwise('abs', tangent, d)
    ops += _elementwise('add', square, square, tangent)
    ops += _elementwise('where', square, zero, 1, square)
    ops += _elementwise('div', tangent, g, square)
    ops += _elementwise('lt', zero, delta, 0)
    ops += _elementwise('neg', square, tangent)
    ops += _elementwise('where', tangent, zero, square, tangent)
    if active is None:
        active = _linalg_active(app, aqq, magnitude, tolerance or torch.finfo(real).eps, ops)
    ops += _elementwise('where', tangent, active, tangent, 0)
    ops += _elementwise('mul', cosine, tangent, tangent)
    ops += _elementwise('add', cosine, cosine, 1)
    ops += _elementwise('rsqrt', cosine, cosine)
    sine = _temporary(batch, apq.dtype)
    ops += _elementwise('eq', zero, magnitude, 0)
    ops += _elementwise('where', scale, zero, 1, magnitude)
    ops += _elementwise('div', sine, apq, scale)
    ops += _elementwise('mul', sine, sine, tangent)
    ops += _elementwise('mul', sine, sine, cosine)
    return cosine, sine, tangent, magnitude, active


def _linalg_columns(source, indices, out):
    given = {'self': source, 'dim': -1, 'index': indices.unsqueeze(-2).expand(out.shape)}
    return _lifted(A.gather.default, _taken, given, [out]) if source.is_complex() else \
        _taken(A.gather.default, given, [out])


def _linalg_range_scale(a, small, ops, maximum=None):
    if not min(a.shape[-2:]):
        scale = _temporary(a.shape[:-2], a.real.dtype)
        ops += _fill(scale, 1)
        return scale
    if maximum is None:
        maximum = _linalg_component_maximum(a, [-2, -1], ops)
    scale, desired, safe = (_temporary(a.shape[:-2], a.real.dtype) for _ in range(3))
    zero = _temporary(a.shape[:-2], torch.bool)
    ops += _elementwise('eq', zero, maximum, 0)
    ops += _elementwise('where', safe, zero, 1, maximum)
    ops += _elementwise('maximum', desired, safe, small)
    ops += _elementwise('minimum', desired, desired, 1 / small)
    ops += _elementwise('div', scale, desired, safe)
    return scale


def _linalg_lengths(matrix, lengths, ops, normalize=True):
    real = matrix.real.dtype
    magnitude, maximum = _temporary(matrix.shape, real), _temporary(lengths.shape, real)
    ops += _elementwise('abs', magnitude, matrix)
    ops += _reduced(A.amax.default, {'self': magnitude, 'dim': [-2], 'keepdim': False}, [maximum])
    zero, safe = _temporary(lengths.shape, torch.bool), _temporary(lengths.shape, real)
    ops += _elementwise('eq', zero, maximum, 0)
    ops += _elementwise('where', safe, zero, 1, maximum)
    ops += _elementwise('div', magnitude, magnitude, safe.unsqueeze(-2))
    ops += _elementwise('mul', magnitude, magnitude, magnitude)
    ops += _reduced(A.sum.dim_IntList, {'self': magnitude, 'dim': [-2], 'keepdim': False}, [lengths])
    ops += _elementwise('sqrt', lengths, lengths)
    ops += _elementwise('mul', lengths, lengths, maximum)
    if not normalize:
        return None
    ops += _elementwise('eq', zero, lengths, 0)
    ops += _elementwise('where', safe, zero, 1, lengths)
    normalized = _temporary(matrix.shape, matrix.dtype)
    ops += _elementwise('div', normalized, matrix, safe.unsqueeze(-2))
    return normalized


# Reference-LAPACK SRC/dgels.f, zgels.f, dgelsy.f and zgelsy.f
def _linalg_least_scaled(a, b, ops):
    limits = torch.finfo(a.real.dtype)
    maximum = _linalg_component_maximum(a, [-2, -1], ops)
    scales = [_linalg_range_scale(a, limits.tiny / limits.eps, ops, maximum),
              _linalg_range_scale(b, limits.tiny / limits.eps, ops)]
    values = [_temporary(x.shape, x.dtype) for x in (a, b)]
    for source, scale, value in zip((a, b), scales, values):
        ops += _elementwise('mul', value, source, scale[..., None, None])
    return *values, *scales, maximum


def _linalg_least_unscale(out, numerator, denominator, ops):
    batch = out.shape[:-2]
    first, second, same = (_temporary(batch, torch.bool) for _ in range(3))
    ratio, multiply, divide = (_temporary(batch, out.real.dtype) for _ in range(3))
    ops += _elementwise('ge', first, numerator, 1) + _elementwise('ge', second, denominator, 1)
    ops += _elementwise('eq', same, first, second)
    ops += _elementwise('div', ratio, numerator, denominator)
    ops += _elementwise('where', multiply, same, ratio, numerator)
    ops += _elementwise('where', divide, same, 1, denominator)
    ops += _elementwise('mul', out, out, multiply[..., None, None])
    ops += _elementwise('div', out, out, divide[..., None, None])


def _linalg_least_qr(a, b, out, residual, maximum, ops):
    m, n = a.shape[-2:]
    source = _linalg_transpose(a, ops, m < n, m < n)
    packed = _temporary(source.shape, source.dtype)
    tau = _temporary((*a.shape[:-2], min(m, n)), a.dtype)
    ops += _linalg_geqrf(source, packed, tau)
    triangle = packed[..., :min(m, n), :]
    zero = _temporary(a.shape[:-2], torch.bool)
    ops += _elementwise('eq', zero, maximum, 0)
    diagonal = triangle.diagonal(dim1=-2, dim2=-1)
    ops += _elementwise('where', diagonal, zero.unsqueeze(-1), 1, diagonal)
    ops += _linalg_nonsingular(diagonal, 1, 'torch.linalg.lstsq (gels)')
    if m >= n:
        transformed = _temporary((*out.shape[:-2], m, out.shape[-1]), out.dtype)
        ops += _linalg_householder(packed, tau, transformed, b, adjoint=True)
        ops += _linalg_triangular(triangle, transformed[..., :n, :], out, True)
        if residual is not None:
            ops += _copy(residual, transformed[..., n:, :])
    else:
        middle = _temporary(out.shape, out.dtype)
        ops += _fill(middle, 0)
        ops += _linalg_triangular(_linalg_transpose(triangle, ops, True, True), b, middle[..., :m, :], False)
        ops += _linalg_householder(packed, tau, out, middle)
    ops += _elementwise('where', out, zero[..., None, None], 0, out)
    if residual is not None:
        ops += _elementwise('where', residual, zero[..., None, None], 0, residual)


# Reference-LAPACK SRC/dlaic1.f and zlaic1.f: the two-dimensional incremental singular-vector problem
def _linalg_condition(packed, j, estimate, vector, largest, ops):
    batch, real = estimate.shape, estimate.dtype
    dot = _temporary((*batch, 1, 1), packed.dtype)
    row = _linalg_transpose(vector[..., :j].unsqueeze(-1), ops, True, True)
    ops += _product(A.matmul.default, {'self': row, 'other': packed[..., :j, j:j + 1]}, [dot])
    alpha, gamma = dot[..., 0, 0], packed[..., j, j]
    absa, absg, scale = (_temporary(batch, real) for _ in range(3))
    zero = _temporary(batch, torch.bool)
    ops += _elementwise('abs', absa, alpha) + _elementwise('abs', absg, gamma)
    ops += _elementwise('maximum', scale, absa, absg) + _elementwise('maximum', scale, scale, estimate)
    ops += _elementwise('eq', zero, scale, 0) + _elementwise('where', scale, zero, 1, scale)
    app, aqq, sest = (_temporary(batch, real) for _ in range(3))
    apq = _temporary(batch, packed.dtype)
    ops += _elementwise('div', absa, absa, scale) + _elementwise('div', absg, absg, scale)
    ops += _elementwise('div', sest, estimate, scale)
    ops += _elementwise('mul', app, sest, sest) + _elementwise('mul', aqq, absa, absa)
    ops += _elementwise('add', app, app, aqq) + _elementwise('mul', aqq, absg, absg)
    ops += _elementwise('div', apq, alpha, scale)
    diagonal = _temporary(batch, packed.dtype)
    ops += _elementwise('div', diagonal, gamma, scale)
    ops += _elementwise('mul', apq, apq, _linalg_transpose(diagonal, ops, False, True))
    active, first = _temporary(batch, torch.bool), _temporary(batch, torch.bool)
    ops += _elementwise('ne', active, apq, 0) + _elementwise('ge', first, app, aqq)
    cosine, sine, tangent, magnitude, _ = _linalg_rotation(app, aqq, apq, ops, active=active)
    delta, upper, lower = (_temporary(batch, real) for _ in range(3))
    ops += _elementwise('mul', delta, tangent, magnitude)
    ops += _elementwise('sub', upper, app, delta) + _elementwise('add', lower, aqq, delta)
    ops += _elementwise('where', upper, first, upper, lower) + _elementwise('sqrt', upper, upper)
    if largest:
        ops += _elementwise('mul', estimate, scale, upper)
    else:
        ops += _elementwise('eq', zero, upper, 0) + _elementwise('where', upper, zero, 1, upper)
        ops += _elementwise('div', lower, absg, upper)
        ops += _elementwise('mul', estimate, estimate, lower)
        ops += _elementwise('logical_not', first, first)
    negative = _temporary(batch, packed.dtype)
    ops += _elementwise('neg', negative, _linalg_transpose(sine, ops, False, True))
    coefficient, last = _temporary(batch, packed.dtype), _temporary(batch, packed.dtype)
    ops += _elementwise('where', coefficient, first, cosine, sine)
    ops += _elementwise('where', last, first, negative, cosine)
    ops += _elementwise('mul', vector[..., :j], vector[..., :j], coefficient.unsqueeze(-1))
    ops += _copy(vector[..., j], last)


def _linalg_least_rank(packed, rank, rcond, ops):
    batch, k = packed.shape[:-2], min(packed.shape[-2:])
    estimates = [_temporary(batch, packed.real.dtype) for _ in range(2)]
    vectors = [_temporary((*batch, k), packed.dtype) for _ in range(2)]
    active = _temporary(batch, torch.bool)
    for estimate, vector in zip(estimates, vectors):
        ops += _elementwise('abs', estimate, packed[..., 0, 0]) + _fill(vector, 0) + _fill(vector[..., 0], 1)
    ops += _elementwise('ne', active, estimates[0], 0) + _copy(rank, active)
    for j in range(1, k):
        for largest, (estimate, vector) in enumerate(zip(estimates, vectors)):
            _linalg_condition(packed, j, estimate, vector, largest, ops)
        threshold, admitted = _temporary(batch, packed.real.dtype), _temporary(batch, torch.bool)
        ops += _elementwise('mul', threshold, estimates[1], rcond)
        ops += _elementwise('le', admitted, threshold, estimates[0])
        ops += _elementwise('logical_and', active, active, admitted)
        increment = _temporary(batch, rank.dtype)
        ops += _copy(increment, active) + _elementwise('add', rank, rank, increment)


# Reference-LAPACK SRC/ztzrzf.f, zlatrz.f and zgelsy.f: RZ followed by the retained reflectors and pivot permutation
def _linalg_least_complete(packed, tau, pivots, rank, b, out, ops):
    batch, (m, n), k = packed.shape[:-2], packed.shape[-2:], min(packed.shape[-2:])
    matrix = _temporary((*batch, k, n), packed.dtype)
    ops += _triangular_part(A.triu.default, {'self': packed[..., :k, :]}, [matrix])
    indices = _temporary((n,), torch.int32)
    ops += _fill(indices, 0, 1)
    reflectors = []
    for j in reversed(range(k)):
        active, tail = _temporary(batch, torch.bool), _temporary((*batch, n - j), torch.bool)
        row, scalar = _temporary((*batch, n - j), packed.dtype), _temporary(batch, packed.dtype)
        beta = _temporary(batch, packed.real.dtype)
        ops += _elementwise('lt', active, j, rank)
        ops += _elementwise('ge', tail, indices[j:], rank.unsqueeze(-1)) + _fill(tail[..., 0], True)
        ops += _elementwise('where', row, tail, matrix[..., j, j:], 0)
        row = _linalg_transpose(row, ops, False, True)
        v = _linalg_reflector(row, scalar, beta, ops)
        ops += _elementwise('where', scalar, active, scalar, 0)
        if j:
            _linalg_reflect(matrix[..., :j, j:], v, scalar, ops, left=False)
        ops += _elementwise('where', matrix[..., j, j], active, beta, matrix[..., j, j])
        reflectors.append((j, v, scalar))
    keep = _temporary((*batch, k), torch.bool)
    square = _temporary((*batch, k, k), torch.bool)
    triangle = _temporary((*batch, k, k), packed.dtype)
    ops += _elementwise('lt', keep, indices[:k], rank.unsqueeze(-1))
    ops += _elementwise('logical_and', square, keep.unsqueeze(-1), keep.unsqueeze(-2))
    ops += _elementwise('where', triangle, square, matrix[..., :k], 0)
    diagonal = triangle.diagonal(dim1=-2, dim2=-1)
    ops += _elementwise('where', diagonal, keep, diagonal, 1)
    transformed = _temporary((*out.shape[:-2], m, out.shape[-1]), out.dtype)
    ops += _linalg_householder(packed, tau, transformed, b, adjoint=True)
    rhs, work = _temporary((*out.shape[:-2], k, out.shape[-1]), out.dtype), _temporary(out.shape, out.dtype)
    ops += _elementwise('where', rhs, keep.unsqueeze(-1), transformed[..., :k, :], 0)
    ops += _fill(work, 0) + _linalg_triangular(triangle, rhs, work[..., :k, :], True)
    for j, v, scalar in reversed(reflectors):
        _linalg_reflect(work[..., j:, :], v, scalar, ops)
    for j in reversed(range(k)):
        work = _linalg_swap(work, pivots[..., j], j, ops)
    ops += _copy(out, work)


def _least_squares(target, given, outs):
    a, b, (solution, residual, rank, singular), ops = given['self'], given['b'], outs, []
    vector = _linalg_vector_rhs(a, b)
    b, out = (b.unsqueeze(-1), solution.unsqueeze(-1)) if vector else (b, solution)
    m, n = a.shape[-2:]
    driver = _lstsq_driver(a, given.get('driver'))
    if not min(m, n) or not b.shape[-1] and driver in ('gels', 'gelsy'):
        ops += _fill(out, 0) + _fill(rank, 0) + _fill(singular, 0)
        if driver == 'gels':
            return ops + _fill(residual, 0)
        raw = b if n == 0 else None
    elif driver in ('gelsd', 'gelss'):
        batch, own = out.shape[:-2], a.shape[:-2]
        if not math.prod(batch):
            return ops
        rcond = given.get('rcond')
        if rcond is None:
            rcond = torch.finfo(a.real.dtype).eps * max(m, n)
        info = _temporary(batch, torch.int64)
        ranks, values = (rank, singular) if own == batch else \
            (_temporary(batch, rank.dtype), _temporary((*batch, min(m, n)), singular.dtype))
        ops.append(_Lapack(GELSD if driver == 'gelsd' else GELSS, b'', batch, [(a, 2), (b, 2)],
                           [(out, 2), (values, 1), (ranks, 0), (info, 0)], rcond))
        ops.append(_Check(info, 3, 'linalg.lstsq'))
        if own != batch:
            first = (0,) * (len(batch) - len(own)) + tuple(slice(0, 1) if size == 1 else slice(None) for size in own)
            ops += _copy(rank, ranks[first]) + _copy(singular, values[first])
        return ops
    else:
        scaled, rhs, numerator, denominator, maximum = _linalg_least_scaled(a, b, ops)
        raw = _temporary((*out.shape[:-2], m - n, out.shape[-1]), out.dtype) if residual.numel() else None
        rcond = given.get('rcond')
        if rcond is None:
            rcond = torch.finfo(a.real.dtype).eps * max(m, n)
        if driver == 'gels':
            _linalg_least_qr(scaled, rhs, out, raw, maximum, ops)
        else:
            packed = _temporary(a.shape, a.dtype)
            tau, pivots = _temporary((*a.shape[:-2], min(m, n)), a.dtype), \
                _temporary((*a.shape[:-2], min(m, n)), torch.int32)
            ops += _linalg_geqrf(scaled, packed, tau, pivots)
            _linalg_least_rank(packed, rank, rcond, ops)
            _linalg_least_complete(packed, tau, pivots, rank, rhs, out, ops)
        _linalg_least_unscale(out, numerator, denominator, ops)
        if raw is not None:
            ops += _elementwise('div', raw, raw, denominator[..., None, None])
    if residual.numel():
        magnitude = _temporary(raw.shape, raw.real.dtype)
        ops += _elementwise('abs', magnitude, raw) + _elementwise('mul', magnitude, magnitude, magnitude)
        ops += _reduced(A.sum.dim_IntList, {'self': magnitude, 'dim': [-2], 'keepdim': False}, [residual])
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
# offsets affine; an input position outside the padded image contributes nothing; complex operands (torch.view_as_real)
# two more loops, the output's component c PARALLEL and the input's q REDUCTION, the weight's component selected by
# (c, q) and negated for (0, 1): (x_r + i x_i)(w_r + i w_i) by its four real products in the one SUM
def _convolving(target, given, outs):
    packet, (out,) = target.overloadpacket, outs
    x, w, bias = given['input'], given['weight'], given['bias']
    _typed(target, [], outs, FLOATING + COMPLEXES)
    transposed = given['transposed'] if packet is A.convolution else packet in TRANSPOSED
    k = w.dim() - 2
    stride, dilation, padding = _ints(given['stride'], k), _ints(given['dilation'], k), given['padding']
    lows = (0,) * k if padding == 'valid' else tuple(d * (n - 1) // 2 for d, n in zip(dilation, w.shape[2:])) \
        if padding == 'same' else _ints(padding, k)
    if x.dim() == k + 1:
        x, out = x.unsqueeze(0), out.unsqueeze(0)
    x, w, out = _spatial(x, k), _spatial(w, k), _spatial(out, k)
    stride, dilation, lows = (1,) * (3 - k) + stride, (1,) * (3 - k) + dilation, (0,) * (3 - k) + lows
    groups, wide = given['groups'], out.is_complex()
    loops = 12 if wide else 10
    outputs, inputs = out.shape[1] // groups, x.shape[1] // groups
    sizes = [x.shape[0], groups, outputs, *out.shape[2:], inputs, *w.shape[2:]] + [2, 2] * wide
    t = (F64 if any(v.dtype in (torch.complex128, torch.float64) for v in (out, x, w)) else F32) if wide else \
        _compute(out.dtype, x.dtype, w.dtype, torch.float32)
    acc = out if bias is None else _temporary(out.shape, out.dtype if wide else STORED[t])
    r = _Region()
    o, kk = [r.index(3 + d) for d in range(3)], [r.index(7 + d) for d in range(3)]
    xs = torch.view_as_real(x) if x.is_complex() else x
    part = [{11: 1}] if x.is_complex() else []
    if not transposed and not any(lows):
        xmap = _rows(loops, [0, {1: inputs, 6: 1}, *[{3 + d: stride[d], 7 + d: dilation[d]} for d in range(3)], *part])
        xv, tests = _load(r, 0, xs, t), []
    else:
        tests, at = [], []
        for d in range(3):
            if transposed:
                shifted = _affine(r, [(o[d], 1), (kk[d], -dilation[d])], lows[d])
                tests += [r('ge', BOOL, shifted, r.const(0, I32)),
                          r('eq', BOOL, r('rem', I32, shifted, r.const(stride[d], I32)), r.const(0, I32))]
                inside, clamped = _bounded(r, r('div', I32, shifted, r.const(stride[d], I32)), x.shape[2 + d])
            else:
                inside, clamped = _window(r, o[d], kk[d], stride[d], lows[d], dilation[d], x.shape[2 + d])
            tests.append(inside)
            at.append(clamped)
        ci = _affine(r, [(r.index(1), inputs), (r.index(6), 1)])
        xv = _convert(r, r('extract', _class(xs), r.index(0), ci, *at, *[r.index(11)] * bool(part), integer=0),
                      _class(xs), t)
        xmap = None
    wmap = [{1: inputs, 6: 1}, 2, 7, 8, 9] if transposed else [{1: outputs, 2: 1}, 6, 7, 8, 9]
    operands = [(xs, xmap)]
    if not wide:
        term = r('mul', t, xv, _load(r, 1, w, t))
        operands.append((w, _rows(loops, wmap)))
    else:
        c, q = r.index(10), r.index(11)
        if not x.is_complex():
            xv = r('select', t, r('eq', BOOL, q, r.const(0, I32)), xv, r.const(0, t))
        if w.is_complex():
            parts = _pair(r, operands, w, _rows(loops, wmap), loops, t)
            product = r('mul', t, xv, r('select', t, r('eq', BOOL, c, q), parts[0], parts[1]))
            term = r('select', t, r('lt', BOOL, c, q), r('neg', t, product), product)
        else:
            operands.append((w, _rows(loops, wmap)))
            term = r('select', t, r('eq', BOOL, c, q), r('mul', t, xv, _load(r, 1, w, t)), r.const(0, t))
    r.done(r('select', t, _all(r, tests), term, r.const(0, t)) if tests else term)
    target = torch.view_as_real(acc) if wide else acc
    made = [_Generic(sizes, operands, [(target, _rows(loops, [0, {1: outputs, 2: 1}, 3, 4, 5] + [10] * wide))], r,
                     [PARALLEL] * 6 + [REDUCTION] * 4 + [PARALLEL, REDUCTION] * wide, [SUM], [t],
                     precision=REASSOCIATE | CONTRACT)]
    if bias is None:
        return made
    if wide:
        return made + _complex('add', out, acc, bias.view(bias.shape[0], 1, 1, 1))
    return made + _epilogue(out, acc, t, [(bias, _rows(5, [1]))], lambda r, a, b: r('add', t, a, b))


# an input position o s - low + k dilation of a strided window, its in-range test and clamped index (_bounded)
def _window(r, o, k, stride, low, dilation, size):
    return _bounded(r, _affine(r, [(o, stride), (k, dilation)], -low), size)


# PyTorch v2.11.0 aten/src/ATen/native/Convolution.cpp, convolution_backward: the input's gradient the dual
# convolution of the output's gradient with the same weight (a transposed convolution's dual is the direct one), the
# weight's the reduction over (n, output position) of the output gradient times the input at the window position, the
# bias's the output gradient's sum over (n, spatial)
def _convolution_backward(target, given, outs):
    gy, x, w = given['grad_output'], given['input'], given['weight']
    mask, transposed, groups = given['output_mask'], given['transposed'], given['groups']
    k = w.dim() - 2
    stride, padding, dilation = _ints(given['stride'], k), _ints(given['padding'], k), _ints(given['dilation'], k)
    ops = []
    if mask[0] and outs[0] is not None:
        ops += _convolving(A.convolution.default, {'input': gy, 'weight': w, 'bias': None, 'stride': stride,
                                                   'padding': padding, 'dilation': dilation, 'transposed': not transposed,
                                                   'groups': groups}, [outs[0]])
    if mask[1] and outs[1] is not None:
        xs, ys, ws = (x.unsqueeze(0), gy.unsqueeze(0), w) if x.dim() == k + 1 else (x, gy, w)
        xs, ys, ws = _spatial(xs, k), _spatial(ys, k), _spatial(ws, k)
        s3, d3, l3 = (1,) * (3 - k) + stride, (1,) * (3 - k) + dilation, (0,) * (3 - k) + padding
        # the operand whose positions the window reaches: the input of a direct convolution, the output gradient of a
        # transposed one; the other read at the reduction's own positions
        near, far = (ys, xs) if not transposed else (xs, ys)
        a, b = near.shape[1] // groups, far.shape[1] // groups
        t = _compute(outs[1].dtype, x.dtype, gy.dtype, torch.float32)
        r = _Region()
        tests, at = [], []
        for d in range(3):
            inside, clamped = _window(r, r.index(7 + d), r.index(3 + d), s3[d], l3[d], d3[d], far.shape[2 + d])
            tests.append(inside)
            at.append(clamped)
        channel = _affine(r, [(r.index(0), b), (r.index(2), 1)])
        v = _convert(r, r('extract', _class(far), r.index(6), channel, *at, integer=1), _class(far), t)
        r.done(r('select', t, _all(r, tests), r('mul', t, _load(r, 0, near, t), v), r.const(0, t)))
        sizes = [groups, a, b, *ws.shape[2:], near.shape[0], *near.shape[2:]]
        nmap = _rows(10, [6, {0: a, 1: 1}, 7, 8, 9])
        wmap = _rows(10, [{0: a, 1: 1}, 2, 3, 4, 5])
        ops.append(_Generic(sizes, [(near, nmap), (far, None)], [(ws, wmap)], r, [PARALLEL] * 6 + [REDUCTION] * 4,
                            [SUM], [t], precision=REASSOCIATE | CONTRACT))
    if mask[2] and outs[2] is not None:
        t = _compute(outs[2].dtype, gy.dtype, torch.float32)
        ops.append(_reduction(A.sum, gy, [0] + list(range(2, gy.dim())) if gy.dim() > k + 1 else list(range(1, gy.dim())),
                              outs[2], t))
    return ops


# PyTorch v2.11.0 aten/src/ATen/native/cpu/AvgPoolKernel.cpp, cpu_avg_pool2d_backward: each input position the sum
# over the kernel offsets of the windows reaching it, the output gradient divided by that window's divisor
def _pooled_backward(target, given, outs):
    gy, x, (gx,) = given['grad_output'], given['self'], outs
    _typed(target, [gy, x], outs)
    kernel, stride, lows = _ints(given['kernel_size'], 2), _ints(given['stride'] or given['kernel_size'], 2), \
        _ints(given['padding'], 2)
    override, include = given.get('divisor_override'), given.get('count_include_pad', True)
    lead = x.dim() - 2
    t = _compute(gx.dtype, gy.dtype, torch.float32)
    r = _Region()
    tests, o = [], []
    for d in range(2):
        shifted = _affine(r, [(r.index(lead + d), 1), (r.index(x.dim() + d), -1)], lows[d])
        tests += [r('ge', BOOL, shifted, r.const(0, I32)),
                  r('eq', BOOL, r('rem', I32, shifted, r.const(stride[d], I32)), r.const(0, I32))]
        inside, clamped = _bounded(r, r('div', I32, shifted, r.const(stride[d], I32)), gy.shape[lead + d])
        tests.append(inside)
        o.append(clamped)
    pool, count = r.const(1, I32), r.const(1, I32)
    for d in range(2):
        lo = _affine(r, [(o[d], stride[d])], -lows[d])
        hi = r('add', I32, lo, r.const(kernel[d], I32))
        edge = r.const(x.shape[lead + d] + lows[d], I32)
        hi = r('select', I32, r('lt', BOOL, edge, hi), edge, hi)
        pool = r('mul', I32, pool, r('sub', I32, hi, lo))
        zero, size = r.const(0, I32), r.const(x.shape[lead + d], I32)
        span = r('sub', I32, r('select', I32, r('lt', BOOL, size, hi), size, hi), r('select', I32, r('lt', BOOL, lo, zero), zero, lo))
        count = r('mul', I32, count, r('select', I32, r('lt', BOOL, span, zero), zero, span))
    divisor = r.const(override, t) if override else _convert(r, pool if include else count, I32, t)
    g = _convert(r, r('extract', _class(gy), *[r.index(j) for j in range(lead)], *o, integer=0), _class(gy), t)
    r.done(r('select', t, _all(r, tests), r('div', t, g, divisor), r.const(0, t)))
    return [_Generic([*x.shape, *kernel], [(gy, None)], [(gx, _rows(x.dim() + 2, range(x.dim())))], r,
                     [PARALLEL] * x.dim() + [REDUCTION] * 2, [SUM], [t], precision=REASSOCIATE)]


# PyTorch 2.14 documentation, torch.nn.AdaptiveMaxPool2d and AdaptiveMaxPool3d (aten/src/ATen/native/cpu
# AdaptiveMaxPoolKernel.cpp: `(val > maxval) || std::isnan(val)`, the index flat over the spatial plane): the
# adaptive windows of _adaptive, REGION (value, index) states joined by max pooling's order (_ordered, NaN later), a
# position past its window's end an absent state
def _adaptive_max(target, given, outs):
    (out, indices), x = outs, given['self']
    _typed(target, [x], outs[:1])
    sizes = list(given['output_size'])
    k, t = len(sizes), _class(x)
    lead = x.dim() - k
    spans = [-(-x.shape[lead + d] // sizes[d]) + 1 for d in range(k)]
    r, tests, at = _Region(), [], []
    for d in range(k):
        o, n, m = r.index(lead + d), x.shape[lead + d], sizes[d]
        start = r('div', I32, r('mul', I32, o, r.const(n, I32)), r.const(m, I32))
        end = r('div', I32, r('add', I32, r('mul', I32, r('add', I32, o, r.const(1, I32)), r.const(n, I32)),
                                r.const(m - 1, I32)), r.const(m, I32))
        p = r('add', I32, start, r.index(x.dim() + d))
        tests.append(r('lt', BOOL, p, end))
        at.append(r('select', I32, tests[-1], p, start))
    flat = r.const(0, I32)
    for d in range(k):
        flat = r('add', I32, r('mul', I32, flat, r.const(x.shape[lead + d], I32)), at[d])
    v = r('extract', t, *[r.index(j) for j in range(lead)], *at, integer=0)
    r.done(v, r('select', I32, _all(r, tests), flat, r.const(-1, I32)))
    lowest = -math.inf if _floating(t) else torch.iinfo(x.dtype).min
    states = [(t, lowest), (I32, -1)]
    omap = _rows(x.dim() + k, range(x.dim()))
    return [_Generic([*out.shape, *spans], [(x, None)], [(out, omap), (indices, omap)], r,
                     [PARALLEL] * x.dim() + [REDUCTION] * k, [REGION] * 2, [t, I32], None, states,
                     _joined(states, lambda c, a, b: _ordered(c, a, b, t, True, True)), _identities(states),
                     REASSOCIATE)]


# PyTorch 2.14 documentation, torch.nn.AdaptiveAvgPool3d (aten/src/ATen/native/AdaptiveAveragePooling3d.cpp,
# start_index and end_index): output position o of extent O over input extent I averages [floor(o I / O),
# ceil((o + 1) I / O)), a reduction over the longest such window with the positions past its end left out
def _adaptive(target, given, outs):
    (out,), x = outs, given['self']
    _typed(target, [x], outs)
    sizes = _ints(given['output_size'], 3)
    lead = x.dim() - 3
    t = _compute(out.dtype, x.dtype, torch.float32)
    spans = [-(-x.shape[lead + d] // sizes[d]) + 1 for d in range(3)]

    def bounds(r, d):
        o, n, m = r.index(lead + d), x.shape[lead + d], sizes[d]
        start = r('div', I32, r('mul', I32, o, r.const(n, I32)), r.const(m, I32))
        end = r('div', I32, r('add', I32, r('mul', I32, r('add', I32, o, r.const(1, I32)), r.const(n, I32)),
                                r.const(m - 1, I32)), r.const(m, I32))
        return start, end
    r, tests, at = _Region(), [], []
    for d in range(3):
        start, end = bounds(r, d)
        p = r('add', I32, start, r.index(x.dim() + d))
        tests.append(r('lt', BOOL, p, end))
        at.append(r('select', I32, r('lt', BOOL, p, end), p, start))
    v = _convert(r, r('extract', _class(x), *[r.index(j) for j in range(lead)], *at, integer=0), _class(x), t)
    r.done(r('select', t, _all(r, tests), v, r.const(0, t)))
    acc = _temporary(out.shape, STORED[t])
    made = [_Generic([*out.shape, *spans], [(x, None)], [(acc, _rows(x.dim() + 3, range(x.dim())))], r,
                     [PARALLEL] * x.dim() + [REDUCTION] * 3, [SUM], [t], precision=REASSOCIATE)]

    def averaged(r, total):
        count = r.const(1, I32)
        for d in range(3):
            start, end = bounds(r, d)
            count = r('mul', I32, count, r('sub', I32, end, start))
        return r('div', t, total, _convert(r, count, I32, t))
    return made + _epilogue(out, acc, t, (), averaged)


# aten/src/ATen/native/AdaptiveAveragePooling.cpp, adaptive_avg_pool2d_backward: input position p of extent I receives
# from the outputs o of extent O whose windows contain it, o in [floor(p O / I), ceil((p + 1) O / I)), each its
# gradient over its window's size
def _adaptive_backward(target, given, outs):
    gy, x, (gx,) = given['grad_output'], given['self'], outs
    _typed(target, [gy, x], outs)
    lead = x.dim() - 2
    t = _compute(gx.dtype, gy.dtype, torch.float32)
    spans = [-(-gy.shape[lead + d] // x.shape[lead + d]) + 1 for d in range(2)]
    r, tests, at = _Region(), [], []
    size = r.const(1, I32)
    for d in range(2):
        n, m, p = x.shape[lead + d], gy.shape[lead + d], r.index(lead + d)
        first = r('div', I32, r('mul', I32, p, r.const(m, I32)), r.const(n, I32))
        last = r('div', I32, r('add', I32, r('mul', I32, r('add', I32, p, r.const(1, I32)), r.const(m, I32)),
                                 r.const(n - 1, I32)), r.const(n, I32))
        o = r('add', I32, first, r.index(x.dim() + d))
        inside = r('lt', BOOL, o, last)
        o = r('select', I32, inside, o, first)
        start = r('div', I32, r('mul', I32, o, r.const(n, I32)), r.const(m, I32))
        end = r('div', I32, r('add', I32, r('mul', I32, r('add', I32, o, r.const(1, I32)), r.const(n, I32)),
                                r.const(m - 1, I32)), r.const(m, I32))
        size = r('mul', I32, size, r('sub', I32, end, start))
        tests.append(inside)
        at.append(o)
    g = _convert(r, r('extract', _class(gy), *[r.index(j) for j in range(lead)], *at, integer=0), _class(gy), t)
    r.done(r('select', t, _all(r, tests), r('div', t, g, _convert(r, size, I32, t)), r.const(0, t)))
    return [_Generic([*x.shape, *spans], [(gy, None)], [(gx, _rows(x.dim() + 2, range(x.dim())))], r,
                     [PARALLEL] * x.dim() + [REDUCTION] * 2, [SUM], [t], precision=REASSOCIATE)]


# PyTorch 2.14 documentation, torch.nn.functional.pdist: element t of the condensed distances the pair i < j of the
# upper triangle in row order, t = i n - i (i + 1) / 2 + j - i - 1, i from t by the triangle's root and corrected by
# the row starts; each pair's p-norm distance as cdist's
def _paired(target, given, outs):
    (out,), x, p = outs, given['self'], float(given['p'])
    _typed(target, [x], outs)
    n, t = x.shape[0], _compute(out.dtype, x.dtype, torch.float32)
    r = _Region()
    k = r.index(0)

    def start(i):
        return r('sub', I32, r('mul', I32, i, r.const(n, I32)),
                 r('div', I32, r('mul', I32, i, r('add', I32, i, r.const(1, I32))), r.const(2, I32)))
    root = r('sqrt', F32, r('cast', F32, r('add', I32, r('mul', I32, k, r.const(-8, I32)), r.const(4 * n * (n - 1) - 7, I32))))
    guess = r('sub', I32, r.const(n - 2, I32),
              r('cast', I32, r('floor', F32, r('sub', F32, r('mul', F32, root, r.const(0.5, F32)), r.const(0.5, F32)))))
    guess = r('select', I32, r('lt', BOOL, k, start(guess)), r('sub', I32, guess, r.const(1, I32)), guess)
    nxt = r('add', I32, guess, r.const(1, I32))
    i = r('select', I32, r('ge', BOOL, k, start(nxt)), nxt, guess)
    j = r('add', I32, r('sub', I32, k, start(i)), r('add', I32, i, r.const(1, I32)))
    c = r.index(1)
    diff = r('abs', t, r('sub', t, _convert(r, r('extract', _class(x), i, c, integer=0), _class(x), t),
                         _convert(r, r('extract', _class(x), j, c, integer=0), _class(x), t)))
    states, join, f = _norm(r, diff, p, t)
    return [_Generic([out.shape[0], x.shape[1]], [(x, None)], [(out, _rows(2, [0]))], r, [PARALLEL, REDUCTION],
                     [REGION], [t], None, states, join, f, REASSOCIATE)]


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
        inside, clamped = _window(r, r.index(2 + d), r.index(5 + d), stride[d], lows[d], 1 if average else dilation[d],
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
            if index is None:
                return [c('select', t, c('or', BOOL, c('gt', BOOL, b[0], a[0]), c('isnan', BOOL, b[0])), b[0], a[0])]
            return _ordered(c, a, b, t, True, nan_later=True)
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


# PyTorch 2.14 documentation, torch.nn.ReflectionPad1d to 3d and torch.nn.ReplicationPad2d and 3d: the padding of
# their last dimensions in the mode they name
PADS = {A.reflection_pad1d: 'reflect', A.reflection_pad2d: 'reflect', A.reflection_pad3d: 'reflect',
        A.replication_pad1d: 'replicate', A.replication_pad2d: 'replicate', A.replication_pad3d: 'replicate'}


def _padding(target, given, outs):
    return _padded(target, {'self': given['self'], 'pad': given['padding'], 'mode': PADS[target.overloadpacket]}, outs)


# PyTorch 2.14 documentation, torch.flip, Tensor.repeat, torch.slice_scatter and torch.select_scatter: a flipped
# dimension's map size - 1 - i; a repeated source's index the result's modulo its extent; a scatter into a slice or a
# selection the base's copy, then the source's into that view of the result
def _flipped(target, given, outs):
    (out,), x = outs, given['self']
    _typed(target, [x], outs, REAL)
    dims = {d % max(x.dim(), 1) for d in given['dims']}
    r = _Region()
    r.done(_convert(r, r.arg(0, x), _class(x), _class(out)))
    rows = _rows(x.dim(), [{d: -1, None: x.shape[d] - 1} if d in dims else d for d in range(x.dim())])
    return [_Generic(out.shape, [(x, rows)], [(out, _same(out.dim()))], r)]


def _repeated(target, given, outs):
    (out,), x = outs, given['self']
    _typed(target, [x], outs, REAL)
    lead, r = out.dim() - x.dim(), _Region()
    at = [r('rem', I32, r.index(lead + d), r.const(n, I32)) for d, n in enumerate(x.shape)]
    r.done(_convert(r, r('extract', _class(x), *at, integer=0), _class(x), _class(out)))
    return [_Generic(out.shape, [(x, None)], [(out, _same(out.dim()))], r)]


def _embedded(target, given, outs):
    (out,), packet = outs, target.overloadpacket
    view = out.select(given['dim'], given['index']) if packet is A.select_scatter else \
        A.slice.Tensor(out, given['dim'], given['start'], given['end'], given['step'])
    return _copy(out, given['self']) + _copy(view, given['src'])

# PyTorch 2.14 documentation, torch.full, torch.zeros, torch.ones, torch.empty, torch.arange, torch.scalar_tensor,
# Tensor.new_full, Tensor.new_zeros, Tensor.new_ones, Tensor.new_empty and Tensor.fill_
FILLS = {A.full: 'fill_value', A.full_like: 'fill_value', A.new_full: 'fill_value', A.scalar_tensor: 's',
         A.fill: 'value', A.zeros: 0, A.zeros_like: 0, A.new_zeros: 0, A.zero: 0, A.ones: 1, A.ones_like: 1,
         A.new_ones: 1}
EMPTY = (A.empty, A.empty_like, A.new_empty, A.empty_strided, A.empty_permuted)


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


# the loops of a reduction of x over `dims` (its free dimensions PARALLEL first, then the reduced), x's map and the
# kept result's
def _reducing(x, dims):
    free = [d for d in range(x.dim()) if d not in dims]
    domain = free + list(dims)
    return ([x.shape[d] for d in domain], [PARALLEL] * len(free) + [REDUCTION] * len(dims),
            _rows(len(domain), [domain.index(d) for d in range(x.dim())]), _rows(len(domain), range(len(free))))


def _reduction(packet, x, dims, kept, t, start=IDENTITY):
    r = _Region()
    r.done(_load(r, 0, x, t))
    loops, iterators, xmap, omap = _reducing(x, dims)
    if packet in (A.sum, A.mean):
        return _Generic(loops, [(x, xmap)], [(kept, omap)], r, iterators, [SUM], [t], [start], precision=REASSOCIATE)
    states = [(t, _identity(packet, x.dtype))]
    return _Generic(loops, [(x, xmap)], [(kept, omap)], r, iterators, [REGION], [t], [start], states,
                    _joined(states, lambda c, a, b: [_extremum(c, a[0], b[0], t, packet is A.amax)]),
                    _identities(states), REASSOCIATE)


def _dims(given, x):
    dims = given.get('dim')
    dims = () if dims is None else _ints(dims, 1)
    return sorted({d % max(x.dim(), 1) for d in dims}) if dims else list(range(x.dim()))


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


# PyTorch 2.14 documentation, torch.prod, torch.any, torch.all, torch.argmax, torch.argmin, torch.max, torch.min and
# torch.var: the product and the boolean folds as states with their identities; an extremum with its first index, NaN
# the greatest (torch 2.14.0 aten/src/ATen/native/SharedReduceOps.h, LessOrNan/GreaterOrNan); the variance about the
# mean over the reduced count less its correction
FOLDS = (A.prod, A.any, A.all)
EXTREMA = (A.argmax, A.argmin, A.max, A.min)


def _fold(target, given, outs):
    (out,), x, packet = outs[:1], given['self'], target.overloadpacket
    if x.is_complex():
        return _complex_fold(packet, x, out, _dims(given, x), given.get('keepdim'))
    _typed(target, [x], outs, REAL)
    dims = _dims(given, x)
    t = BOOL if packet is not A.prod else _compute(out.dtype, x.dtype)
    kept = out.squeeze(tuple(dims)) if given.get('keepdim') and dims else out
    loops, iterators, xmap, omap = _reducing(x, dims)
    r = _Region()
    r.done(_load(r, 0, x, t))
    states = [(t, 1 if packet is not A.any else 0)]
    name = 'mul' if packet is A.prod else 'or' if packet is A.any else 'and'
    return [_Generic(loops, [(x, xmap)], [(kept, omap)], r, iterators, [REGION], [t], None, states,
                     _joined(states, lambda c, a, b: [c(name, t, a[0], b[0])]), _identities(states), REASSOCIATE)]


# PyTorch 2.14 documentation, torch.prod, torch.any and torch.all of complex input: the product the REGION monoid of
# the (re, im) states under the complex product, each element read at both components by constant rows; a truth an
# element with either component nonzero
def _complex_fold(packet, x, out, dims, keepdim):
    kept = out.squeeze(tuple(dims)) if keepdim and dims else out
    loops, iterators, xmap, omap = _reducing(x, dims)
    t, inputs, r = F64 if x.dtype is torch.complex128 else F32, [], _Region()
    re, im = _pair(r, inputs, x, xmap, len(loops), t)
    if packet is not A.prod:
        r.done(r('or', BOOL, r('ne', BOOL, re, r.const(0, t)), r('ne', BOOL, im, r.const(0, t))))
        states = [(BOOL, int(packet is A.all))]
        name = 'or' if packet is A.any else 'and'
        return [_Generic(loops, inputs, [(kept, omap)], r, iterators, [REGION], [BOOL], None, states,
                         _joined(states, lambda c, a, b: [c(name, BOOL, a[0], b[0])]), _identities(states), REASSOCIATE)]
    r.done(re, im)
    states = [(t, 1), (t, 0)]
    made = torch.view_as_real(kept)
    return [_Generic(loops, inputs, [(made[..., 0], omap), (made[..., 1], omap)], r, iterators, [REGION] * 2, [t, t],
                     None, states, _joined(states, lambda c, a, b: list(_complex_value(c, t, 'mul', a, b, 1, None))),
                     _identities(states), REASSOCIATE)]


def _extreme(target, given, outs):
    packet, x = target.overloadpacket, given['self']
    _typed(target, [x], outs[-1:] if packet in (A.argmax, A.argmin) else outs[:1], REAL)
    if packet in (A.argmax, A.argmin) and x.dtype is torch.bool:
        raise NotImplementedError('%s of torch.bool: PyTorch 2.14 defines no bool form of it ("does not support bool '
                                  'input")' % target)
    larger, dim = packet in (A.argmax, A.max), given.get('dim')
    if not x.dim():
        x = x.view(1)
    d = None if dim is None else dim % x.dim()
    order = list(range(x.dim())) if d is None else _lines(x, d)
    free = 0 if d is None else x.dim() - 1
    t = I32 if x.dtype is torch.bool else _class(x)
    r = _Region()
    r.done(_load(r, 0, x, t), _linear(r, x.shape) if d is None else r.index(x.dim() - 1))
    lowest = -math.inf if _floating(t) else 0 if x.dtype is torch.bool else torch.iinfo(x.dtype).min
    highest = math.inf if _floating(t) else 1 if x.dtype is torch.bool else torch.iinfo(x.dtype).max
    states = [(t, lowest if larger else highest), (I32, -1)]

    def join(c, a, b):
        return _ordered(c, a, b, t, larger)
    omap = _rows(x.dim(), range(free))

    def shaped(o):
        return o.squeeze(d) if given.get('keepdim') and d is not None else o.reshape(()) if d is None else o
    f = _Region()
    value, index = f('arg', t, integer=0), f('arg', I32, integer=1)
    if packet in (A.argmax, A.argmin):
        f.done(index)
        outputs, accumulators = [(shaped(outs[0]), omap)], [I32]
    elif len(outs) == 2:
        f.done(value, index)
        outputs, accumulators = [(shaped(outs[0]), omap), (shaped(outs[1]), omap)], [t, I32]
    else:
        f.done(value)
        outputs, accumulators = [(shaped(outs[0]), omap)], [t]
    return [_Generic([x.shape[j] for j in order], [(x, _rows(x.dim(), [order.index(j) for j in range(x.dim())]))],
                     outputs, r, [PARALLEL] * free + [REDUCTION] * (x.dim() - free), [REGION] * len(outputs),
                     accumulators, None, states, _joined(states, join), f, REASSOCIATE)]

def _variance(target, given, outs):
    (out,), x = outs[:1], given['self']
    _typed(target, [] if x.is_complex() else [x], outs)
    dims, t = _dims(given, x), _compute(out.dtype, *([] if x.is_complex() else [x.dtype]), torch.float32)
    correction = given.get('correction')
    correction = (1 if given.get('unbiased', True) else 0) if correction is None else correction
    total = math.prod(x.shape[d] for d in dims)
    groups = [x.shape[d] for d in range(x.dim()) if d not in dims]
    loops, iterators, xmap, gmap = _reducing(x, dims)
    parts = [torch.view_as_real(x)[..., c] for c in (0, 1)] if x.is_complex() else [x]
    sums, squares = [_temporary(groups, STORED[t]) for _ in parts], _temporary(groups, STORED[t])
    ops = []
    for part, total_ in zip(parts, sums):
        r = _Region()
        r.done(_load(r, 0, part, t))
        ops.append(_Generic(loops, [(part, xmap)], [(total_, gmap)], r, iterators, [SUM], [t], precision=REASSOCIATE))
    r, square = _Region(), None
    for c, part in enumerate(parts):
        deviation = r('sub', t, _load(r, 2 * c, part, t), r('div', t, r.arg(2 * c + 1, sums[c]), r.const(total, t)))
        made = r('mul', t, deviation, deviation)
        square = made if square is None else r('add', t, square, made)
    r.done(square)
    ops.append(_Generic(loops, [pair for part, total_ in zip(parts, sums) for pair in ((part, xmap), (total_, gmap))],
                        [(squares, gmap)], r, iterators, [SUM], [t], precision=REASSOCIATE))
    kept = out.squeeze(tuple(dims)) if given.get('keepdim') and dims else out
    return ops + _epilogue(kept, squares, t, (), lambda r, q: r('div', t, q, r.const(max(total - correction, 0), t)))

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


# G. E. Blelloch, Prefix sums and their applications, CMU-CS-90-190, 1990, §1.3, Figures 1.4 and 1.5 (upsweep and
# downsweep): the exclusive scan by an associative operator (+) with identity e of each line of the component
# buffers (lines..., P = 2^m), in place, by the work-efficient tree: for d < m, b[i 2^(d+1) + 2^(d+1) - 1] =
# b[i 2^(d+1) + 2^d - 1] (+) b[i 2^(d+1) + 2^(d+1) - 1]; the root e; for d from m - 1 down, (left, right) =
# (right, right (+) left) of the same pairs (the prefix before the left subtree, then its total); each level one
# structured operation over strided views
def _blelloch(buffers, t, identities, combine):
    buffer, k = buffers[0], len(buffers)
    kinds = list(t) if isinstance(t, (list, tuple)) else [t] * k
    lead, size = buffer.dim() - 1, buffer.shape[-1]
    levels, ops = size.bit_length() - 1, []

    def pair(d, at):
        return _rows(lead + 1, [*range(lead), {lead: 2 << d, None: at}])

    def sweep(d):
        r = _Region()
        left, right = pair(d, (1 << d) - 1), pair(d, (2 << d) - 1)
        inputs = [(b, left) for b in buffers] + [(b, right) for b in buffers]
        return r, [r.arg(q, b) for q, b in enumerate(buffers)], [r.arg(k + q, b) for q, b in enumerate(buffers)], \
            inputs, left, right
    for d in range(levels):
        r, a, b, inputs, left, right = sweep(d)
        r.done(*combine(r, a, b))
        ops.append(_Generic([*buffer.shape[:-1], size >> (d + 1)], inputs, [(x, right) for x in buffers], r))
    r = _Region()
    r.done(*[r.const(v, kind) for v, kind in zip(identities, kinds)])
    ops.append(_Generic([*buffer.shape[:-1]], [], [(x, _rows(lead, [*range(lead), {None: size - 1}])) for x in buffers],
                        r))
    for d in reversed(range(levels)):
        r, a, b, inputs, left, right = sweep(d)
        r.done(*b, *combine(r, b, a))
        ops.append(_Generic([*buffer.shape[:-1], size >> (d + 1)], inputs,
                            [(x, left) for x in buffers] + [(x, right) for x in buffers], r))
    return ops


# the inclusive scan of the components xs of x's lines along d into outs: each component padded with its identity to
# a power-of-two buffer, the exclusive tree scan, each element combined after its prefix; a component None the
# element's position along d (an extremum's index), `t` one kind or each component's
def _scan(xs, d, outs, t, identities, combine):
    x = next(v for v in xs if v is not None)
    kinds = list(t) if isinstance(t, (list, tuple)) else [t] * len(xs)
    order, n = _lines(x, d), x.dim()
    length = x.shape[d]
    size = 1 << max(length - 1, 0).bit_length()
    buffers = [_temporary([*[x.shape[j] for j in order[:-1]], size], STORED[kind]) for kind in kinds]
    tensors = [v for v in xs if v is not None]
    r = _Region()
    inside, at = _bounded(r, r.index(n - 1), length)
    index = [r.index(order.index(j)) if j != d else at for j in range(n)]

    slots = iter(range(len(tensors)))

    def element(v, kind):
        return at if v is None else _convert(r, r('extract', _class(v), *index, integer=next(slots)), _class(v), kind)
    r.done(*[r('select', kind, inside, element(v, kind), r.const(e, kind)) for v, e, kind in zip(xs, identities, kinds)])
    ops = [_Generic(buffers[0].shape, [(v, None) for v in tensors], [(b, _same(n)) for b in buffers], r)]
    ops += _blelloch(buffers, kinds, identities, combine)
    e = _Region()
    own = iter(range(len(buffers), len(buffers) + len(tensors)))
    made = combine(e, [e.arg(q, b) for q, b in enumerate(buffers)],
                   [e.index(n - 1) if v is None else _load(e, next(own), v, kind) for v, kind in zip(xs, kinds)])
    e.done(*[_store(e, m, kind, o) for m, kind, o in zip(made, kinds, outs)])
    xmap = _rows(n, [order.index(j) for j in range(n)])
    return ops + [_Generic([x.shape[j] for j in order], [(b, _same(n)) for b in buffers] + [(v, xmap) for v in tensors],
                           [(o, xmap) for o in outs], e)]


# torch 2.14.0 aten/src/ATen/native/ReduceOpsUtils.h and LogcumsumexpKernel.cpp _log_add_exp_helper: log(e^a + e^b)
# as the larger plus log1p of e^(smaller - larger); equal infinities and NaN by their own cases
def _log_add_exp(r, t, a, b):
    big = r('select', t, r('ge', BOOL, a, b), a, b)
    small = r('select', t, r('ge', BOOL, a, b), b, a)
    sum_ = r('add', t, big, _apply(r, 'log1p', r('exp', t, r('sub', t, small, big)), None, None, 1, t))
    infinite = r('and', BOOL, r('eq', BOOL, r('abs', t, big), r.const(math.inf, t)), r('eq', BOOL, small, small))
    return r('select', t, infinite, big, sum_)


def _complex_log_add_exp(r, t, a, b):
    def c(op, z, w=None):
        return _complex_value(r, t, op, z, w, 1, None)
    wide = r('ge', BOOL, a[0], b[0])
    big = tuple(r('select', t, wide, p, q) for p, q in zip(a, b))
    small = tuple(r('select', t, wide, q, p) for p, q in zip(a, b))
    sum_ = c('add', c('log1p', c('exp', c('sub', small, big))), big)
    infinite = c('log', c('add', c('exp', small), c('exp', big)))
    nan = r('or', BOOL, c('isnan', a), c('isnan', b))
    equal = r('and', BOOL, r('eq', BOOL, small[0], big[0]), r('eq', BOOL, r('abs', t, big[0]), r.const(math.inf, t)))
    below = r('lt', BOOL, big[0], r.const(0, t))
    return tuple(r('select', t, nan, r.const(math.nan, t), r('select', t, equal, r('select', t, below, s, f), m))
                 for s, f, m in zip(small, infinite, sum_))


# G. E. Blelloch, Prefix sums and their applications, CMU-CS-90-190, 1990 (the scan of any associative operator with
# its identity): PyTorch 2.14 documentation, torch.cumsum, torch.cumprod and torch.logcumsumexp; a complex scan the
# monoid of its (re, im) components
SCANS = {A.cumsum: (0, 'add'), A.cumprod: (1, 'mul'), A.logcumsumexp: (-math.inf, None)}


def _scanned(target, given, outs):
    packet, (out,), x = target.overloadpacket, outs, given['self']
    d = given['dim'] % max(x.dim(), 1)
    if not x.dim():
        x, out, d = x.view(1), out.view(1), 0
    identity, op = SCANS[packet]
    if out.is_complex():
        ops = []
        if not x.is_complex():
            wide = _temporary(x.shape, out.dtype)
            ops, x = _copy(wide, x), wide
        t = F64 if out.dtype is torch.complex128 else F32
        pairs, made = torch.view_as_real(x), torch.view_as_real(out)

        def combine(r, a, b):
            return _complex_log_add_exp(r, t, a, b) if op is None else _complex_value(r, t, op, a, b, 1, None)
        return ops + _scan([pairs[..., 0], pairs[..., 1]], d, [made[..., 0], made[..., 1]], t, [identity, 0], combine)
    _typed(target, [x], outs, REAL)
    t = _compute(out.dtype, x.dtype)
    return _scan([x], d, [out], t, [identity],
                 lambda r, a, b: [_log_add_exp(r, t, a[0], b[0]) if op is None else r(op, t, a[0], b[0])])


# PyTorch 2.14 documentation, torch.cummax and torch.cummin: the running extremum and its index, the scan of the
# (value, index) monoid of the later winning ties and a NaN (torch 2.14.0 aten ReduceOps.cpp cummax_cummin_helper:
# `x >= out || isnan(x)` for cummax, `<=` for cummin), the max of one total order and so associative (Blelloch)
def _cumulative(target, given, outs):
    packet, x, (values, indices) = target.overloadpacket, given['self'], outs
    _typed(target, [x], outs[:1], REAL)
    d = given['dim'] % max(x.dim(), 1)
    if not x.dim():
        x, values, indices, d = x.view(1), values.view(1), indices.view(1), 0
    t, larger = _compute(values.dtype, x.dtype), packet is A.cummax
    worst = (-math.inf if larger else math.inf) if _floating(t) else \
        (torch.iinfo(STORED[t]).min if larger else torch.iinfo(STORED[t]).max) if t != BOOL else int(not larger)

    def combine(c, a, b):
        win = c('ge' if larger else 'le', BOOL, b[0], a[0])
        if _floating(t):
            win = c('or', BOOL, c('isnan', BOOL, b[0]), c('and', BOOL, c('not', BOOL, c('isnan', BOOL, a[0])), win))
        return [c('select', t, win, b[0], a[0]), c('select', I32, win, b[1], a[1])]
    return _scan([x, None], d, [values, indices], [t, I32], [worst, -1], combine)


# PyTorch 2.14 documentation, torch.searchsorted (and torch.bucketize, its decomposition): each value's insertion
# point in its sorted line by binary search (D. E. Knuth, TAOCP vol. 3, 6.2.1 Algorithm B), ceil(log2(m + 1)) halvings
# unrolled in the region, each probe an extraction at the computed midpoint (through `sorter` where given); NaN ordered
# above every number, as torch's sort orders it (a NaN value's point m)
def _searched(target, given, outs):
    (out,), line, x = outs, given['sorted_sequence'], given['self']
    sorter, right = given.get('sorter'), given.get('right', False) or given.get('side') == 'right'
    m = line.shape[-1]
    t = _compute(line.dtype, *([x.dtype] if isinstance(x, torch.Tensor) else []))
    r, inputs = _Region(), [(line, None)]
    if sorter is not None:
        inputs.append((_indexing(sorter), None))
    lead = [r.index(j) for j in range(out.dim() - 1)] if line.dim() > 1 else []
    if isinstance(x, torch.Tensor):
        v = _load(r, len(inputs), x, t)
        inputs.append((x, _same(out.dim())))
    else:
        v = r.const(x, t)

    def less(p, q):
        if not _floating(t):
            return r('lt', BOOL, p, q)
        return r('and', BOOL, r('not', BOOL, r('isnan', BOOL, p)), r('or', BOOL, r('isnan', BOOL, q), r('lt', BOOL, p, q)))
    lo, hi = r.const(0, I32), r.const(m, I32)
    for _ in range(m.bit_length()):
        active = r('lt', BOOL, lo, hi)
        mid = r('select', I32, active, r('div', I32, r('add', I32, lo, hi), r.const(2, I32)), r.const(0, I32))
        at = r('extract', I32, *lead, mid, integer=1) if sorter is not None else mid
        probe = _convert(r, r('extract', _class(line), *lead, at, integer=0), _class(line), t)
        go = r('not', BOOL, less(v, probe)) if right else less(probe, v)
        lo = r('select', I32, r('and', BOOL, active, go), r('add', I32, mid, r.const(1, I32)), lo)
        hi = r('select', I32, r('and', BOOL, active, r('not', BOOL, go)), mid, hi)
    r.done(_store(r, lo, I32, out))
    return [_Generic(out.shape, inputs, [(out, _same(out.dim()))], r)]


# PyTorch 2.14 documentation, torch.sort and torch.topk: the compiler's one sort expansion (metal_ir.h metal_ir_sort,
# K. E. Batcher's bitonic network; NaN last ascending, first descending, ties by index), the first k of the order
class _Sort:
    def __init__(self, x, axis, k, descending, values, indices):
        self.x, self.axis, self.k, self.descending, self.values, self.indices = x, axis, k, descending, values, indices
        self.place, self.node = GPU, 0


def _ranked(target, given, outs):
    (values, indices), x = outs, given['self']
    _typed(target, [x], [values], REAL)
    d = given['dim'] % max(x.dim(), 1)
    if not x.dim():
        x, values, indices, d = x.view(1), values.view(1), indices.view(1), 0
    k, descending = (given['k'], given['largest']) if target.overloadpacket is A.topk else \
        (x.shape[d], given['descending'])
    return [_Sort(x, d, k, bool(descending), values, indices)]


# PyTorch 2.14 documentation, torch.kthvalue, torch.median and torch.mode: order statistics of each line from its
# stable ascending sort (_Sort, NaN last): the k-th entry; the median the entry (n - 1) // 2, a line holding a NaN its
# NaN and the first NaN's index (the max extremum, _extreme); the mode the longest run of equal sorted values (ties the
# smaller value, the first run) and its last occurrence: run starts by a max scan of the boundary marks (_scan),
# lengths at the run ends, their first-index argmax (_extreme), the sorted entry there by extraction
SELECTIONS = (A.kthvalue, A.median, A.nanmedian, A.mode)


def _selected(target, given, outs):
    packet, x, ops = target.overloadpacket, given['self'], []
    _typed(target, [x], outs[:1], REAL)
    if packet in (A.median, A.nanmedian) and 'dim' not in given:
        flat = _temporary([x.numel()], x.dtype)
        ops, x, d, outs = _copy(flat, x), flat, 0, [outs[0].reshape(()), None]
    else:
        d = given.get('dim', -1) % max(x.dim(), 1)
        x = x.view(1) if not x.dim() else x
    shaped = [o.squeeze(d) if o is not None and given.get('keepdim') and o.dim() else o for o in outs]
    t, n = _class(x), x.shape[d]
    sv, si = _temporary(x.shape, x.dtype), _temporary(x.shape, torch.int64)
    ops += [_Sort(x, d, n, False, sv, si)]

    def picked(values, indices, nan=None):
        r, inputs = _Region(), [(values, _same(values.dim())), (indices, _same(values.dim()))]
        made = [_load(r, 0, values, t), _load(r, 1, indices, I64)]
        if nan is not None:
            inputs += [(nan[0], _same(values.dim())), (nan[1], _same(values.dim()))]
            top, where = _load(r, 2, nan[0], t), _load(r, 3, nan[1], I64)
            hit = r('isnan', BOOL, top)
            made = [r('select', t, hit, top, made[0]), r('select', I64, hit, where, made[1])]
        results = [_store(r, made[0], t, shaped[0])] + ([_store(r, made[1], I64, shaped[1])] if shaped[1] is not None
                                                        else [])
        outputs = [(o, _same(values.dim())) for o in shaped if o is not None]
        r.done(*results)
        return [_Generic(values.shape, inputs, outputs, r)]
    if packet is A.kthvalue:
        return ops + picked(sv.select(d, given['k'] - 1), si.select(d, given['k'] - 1))
    if packet is A.nanmedian and _floating(t):
        return ops + _ranked_at(x, d, sv, si, shaped, t)
    if packet in (A.median, A.nanmedian):
        k = (n - 1) // 2
        if not _floating(t):
            return ops + picked(sv.select(d, k), si.select(d, k))
        lines = list(sv.select(d, k).shape)
        top, where = _temporary(lines, x.dtype), _temporary(lines, torch.int64)
        ops += _extreme(A.max.dim, {'self': x, 'dim': d, 'keepdim': False}, [top, where])
        return ops + picked(sv.select(d, k), si.select(d, k), (top, where))
    order, m = _lines(x, d), x.dim()
    marks, starts, counts = (_temporary(x.shape, torch.int32) for _ in range(3))
    smap = _rows(m, [order.index(j) for j in range(m)])

    def neighbour(r, shift):
        inside, at = _bounded(r, r('add', I32, r.index(m - 1), r.const(shift, I32)), n)
        other = r('extract', t, *[r.index(order.index(j)) if j != d else at for j in range(m)], integer=0)
        return r('and', BOOL, inside, r('eq', BOOL, other, r.arg(1, sv)))
    r = _Region()
    r.done(r('select', I32, neighbour(r, -1), r.const(-1, I32), r.index(m - 1)))
    ops.append(_Generic([x.shape[j] for j in order], [(sv, None), (sv, smap)], [(marks, smap)], r))
    ops += _scan([marks], d, [starts], I32, [-1], lambda c, a, b: [c('select', I32, c('ge', BOOL, b[0], a[0]), b[0], a[0])])
    r = _Region()
    length = r('add', I32, r('sub', I32, r.index(m - 1), r.arg(2, starts)), r.const(1, I32))
    r.done(r('select', I32, neighbour(r, 1), r.const(0, I32), length))
    ops.append(_Generic([x.shape[j] for j in order], [(sv, None), (sv, smap), (starts, smap)], [(counts, smap)], r))
    lines = [x.shape[j] for j in order[:-1]]
    best, end = _temporary(lines, torch.int32), _temporary(lines, torch.int64)
    ops += _extreme(A.max.dim, {'self': counts, 'dim': d, 'keepdim': False}, [best, end])
    r = _Region()
    at = _convert(r, r.arg(2, end), I64, I32)
    coords = [r.index(j if j < d else j - 1) if j != d else at for j in range(m)]
    r.done(_store(r, r('extract', t, *coords, integer=0), t, shaped[0]),
           _store(r, _convert(r, r('extract', I32, *coords, integer=1), I32, I64), I64, shaped[1]))
    return ops + [_Generic(shaped[0].shape, [(sv, None), (_indexing(si), None), (end, _same(len(lines)))],
                           [(shaped[0], _same(len(lines))), (shaped[1], _same(len(lines)))], r)]


# PyTorch 2.14 documentation, torch.nanmedian: the lower median of a line's c numbers, the sorted entry
# max((c - 1) // 2, 0) (NaN last, so an all-NaN line its first NaN), c one SUM of the line's numbers, the entry by
# extraction at that computed position
def _ranked_at(x, d, sv, si, shaped, t):
    order, m = _lines(x, d), x.dim()
    lines = [x.shape[j] for j in order[:-1]]
    count = _temporary(lines, torch.int32)
    loops, iterators, xmap, omap = _reducing(x, [d])
    r = _Region()
    r.done(_convert(r, r('not', BOOL, r('isnan', BOOL, _load(r, 0, x, t))), BOOL, I32))
    ops = [_Generic(loops, [(x, xmap)], [(count, omap)], r, iterators, [SUM], [I32])]
    r = _Region()
    half = r('div', I32, r('sub', I32, r.arg(2, count), r.const(1, I32)), r.const(2, I32))
    at = r('select', I32, r('lt', BOOL, half, r.const(0, I32)), r.const(0, I32), half)
    coords = [r.index(j if j < d else j - 1) if j != d else at for j in range(m)]
    results = [_store(r, r('extract', t, *coords, integer=0), t, shaped[0])]
    outputs = [(shaped[0], _same(len(lines)))]
    if shaped[1] is not None:
        results.append(_store(r, _convert(r, r('extract', I32, *coords, integer=1), I32, I64), I64, shaped[1]))
        outputs.append((shaped[1], _same(len(lines))))
    r.done(*results)
    return ops + [_Generic(lines, [(sv, None), (_indexing(si), None), (count, _same(len(lines)))], outputs, r)]


# PyTorch 2.14 documentation, torch.histc (torch 2.14.0 aten SummaryOps.cpp histogramdd and histc: min == max takes
# the data's extremes, still equal widens them by 1; pos = (x - min) / (max - min) bins, max into the last bin, NaN and
# outside [min, max] left out): each element's bin, then the counts by the scatter-add reduction (_scatter), an element
# left out at bin `bins`, which no output position matches
def _histogram(target, given, outs):
    (out,), x, bins = outs, given['self'], given['bins']
    _typed(target, [x], outs)
    t = _compute(out.dtype, x.dtype, torch.float32)
    low, high, ops = float(given['min']), float(given['max']), []
    r, inputs = _Region(), [(x, _same(x.dim()))]
    v = _load(r, 0, x, t)
    if low == high:
        extremes = [_temporary([], STORED[t]) for _ in range(2)]
        every = list(range(x.dim()))
        ops += [_reduction(A.amin, x, every, extremes[0], t), _reduction(A.amax, x, every, extremes[1], t)]
        lo, hi = (_load(r, 1 + j, e, t) for j, e in enumerate(extremes))
        inputs += [(e, _rows(x.dim(), [])) for e in extremes]
    else:
        lo, hi = r.const(low, t), r.const(high, t)
    same = r('eq', BOOL, lo, hi)
    lo = r('select', t, same, r('sub', t, lo, r.const(1, t)), lo)
    hi = r('select', t, same, r('add', t, hi, r.const(1, t)), hi)
    scaled = r('mul', t, r('div', t, r('sub', t, v, lo), r('sub', t, hi, lo)), r.const(bins, t))
    position = _convert(r, r('trunc', t, scaled), t, I32)
    position = r('select', I32, r('eq', BOOL, position, r.const(bins, I32)), r.const(bins - 1, I32), position)
    inside = r('and', BOOL, r('ge', BOOL, v, lo), r('le', BOOL, v, hi))
    r.done(r('select', I32, inside, position, r.const(bins, I32)))
    index = _temporary([x.numel()], torch.int32)
    ops.append(_Generic(x.shape, inputs, [(index.view(x.shape), _same(x.dim()))], r))
    zeros = _temporary([bins], out.dtype)
    ops += _fill(zeros, 0)
    targets = [(index.as_strided([bins, x.numel()], [0, 1], 0), 0, bins, [0], [1])]
    return ops + _scatter(out, zeros, 1.0, targets, [x.numel()], [bins], [0], [0], SCATTERS['sum'], True, checked=False)


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


# docs/provider.md#sorted-indexed-updates
SCATTERS = {'last': 0, 'sum': 1, 'add': 1, 'prod': 2, 'multiply': 2, 'mean': 3, 'amax': 4, 'amin': 5}


def _scatter(out, base, src, targets, candidates, limits, near, far, op, include, alpha=1,
             negative=False, checked=True, message='indexed update: index out of bounds'):
    integral, parts = out.dtype in INTEGRAL, 2 if out.is_complex() else 1
    t = _compute(out.dtype, *([src.dtype] if isinstance(src, torch.Tensor) else [])) if integral else \
        F64 if out.dtype in (torch.float64, torch.complex128) else F32

    def split(x):
        return [torch.view_as_real(x)[..., k] for k in range(2)] if x.is_complex() else [x]

    def combined(r, a, b):
        if op in (1, 3):
            return [r('add', t, x, y) for x, y in zip(a, b)]
        if op == 2:
            return list(_complex_value(r, t, 'mul', a, b, 1, None)) if parts == 2 else [r('mul', t, a[0], b[0])]
        return [_extremum(r, a[0], b[0], t, op == 4)] if op in (4, 5) else b

    n, count = out.dim(), math.prod(candidates)
    indexed = sorted(d for _, d, _, _, _ in targets)
    batches = [d for d in range(n) if d not in indexed and any(across[d] for _, _, _, across, _ in targets)]
    payload = [d for d in range(n) if d not in indexed and d not in batches]
    bs, ps = [limits[d] for d in batches], [limits[d] for d in payload]
    if not count or not math.prod(bs + ps):
        return _assign(out, base)
    keyshape, shape = bs + [count], bs + ps + [count]
    keydims, dims, volume = len(bs), len(shape), math.prod(out.shape[d] for d in indexed)
    keys, valid = _temporary(bs + list(candidates), torch.int64), _temporary(bs + list(candidates), torch.bool)
    r, inputs, terms, tests = _Region(), [], {}, []
    for index, d, extent, across, along in targets:
        view = index.as_strided(keys.shape, [across[j] for j in batches] + list(along), index.storage_offset())
        raw = _load(r, len(inputs), view, I64)
        inputs.append((view, _same(keys.dim())))
        inside = r('and', BOOL, r('ge', BOOL, raw, r.const(-extent if negative else 0, I64)),
                   r('lt', BOOL, raw, r.const(extent, I64)))
        at = r('select', I64, r('lt', BOOL, raw, r.const(0, I64)), r('add', I64, raw, r.const(extent, I64)), raw) \
            if negative else raw
        terms[d], tests = r('select', I64, inside, at, r.const(0, I64)), tests + [inside]
    inside, linear = _all(r, tests), r.const(0, I64)
    for d in indexed:
        linear = r('add', I64, r('mul', I64, linear, r.const(out.shape[d], I64)), terms[d])
    r.done(r('select', I64, inside, linear, r.const(volume, I64)), inside)
    made = [_Generic(keys.shape, inputs, [(keys, _same(keys.dim())), (valid, _same(keys.dim()))], r)]
    if checked:
        every = _temporary([], torch.bool)
        made += _fold(A.all.default, {'self': valid}, [every])
        made += _asserted(A._assert_async.msg, {'self': every, 'assert_msg': message}, [])
    if not out.numel():
        return made + _assign(out, base)
    keys = keys.reshape(keyshape)
    ordered, permutation = _temporary(keyshape, torch.int64), _temporary(keyshape, torch.int64)
    made.append(_Sort(keys, keydims, count, False, ordered, permutation))
    values = [_temporary(shape, STORED[t]) for _ in range(parts)]
    if op == 3:
        values.append(_temporary(shape, torch.int64))
    r = _Region()
    keymap = _rows(dims, [*range(keydims), dims - 1])
    inputs = [(permutation, keymap)]
    coords = _coordinates(r, r.arg(0, permutation), candidates)
    if isinstance(src, torch.Tensor):
        view = src.as_strided(bs + ps + list(candidates), [near[d] for d in batches + payload] + list(far),
                              src.storage_offset())
        results = []
        for component in split(view):
            x = r('extract', _class(component), *[r.index(d) for d in range(dims - 1)], *coords, integer=len(inputs))
            results.append(_convert(r, x, _class(component), t))
            inputs.append((component, None))
        results += [r.const(0, t)] * (parts - len(results))
    else:
        results = [r.const(_component(src, k), t) for k in range(parts)]
    if op == 1 and alpha != 1:
        results = list(_complex_value(r, t, 'mul', results,
                       [r.const(_component(alpha, q), t) for q in (0, 1)], 1, None)) if parts == 2 else \
            [r('mul', t, r.const(alpha, t), results[0])]
    r.done(*results, *([r.const(1, I64)] if op == 3 else []))
    made.append(_Generic(shape, inputs, [(v, _same(dims)) for v in values], r))
    if op:
        flags = _temporary(keyshape, torch.bool)
        r = _Region()
        at = r.index(keydims)
        previous = r('select', I32, r('eq', BOOL, at, r.const(0, I32)), at, r('sub', I32, at, r.const(1, I32)))
        prior = r('extract', I64, *[r.index(d) for d in range(keydims)], previous, integer=0)
        r.done(r('or', BOOL, r('eq', BOOL, at, r.const(0, I32)), r('ne', BOOL, r.arg(0, ordered), prior)))
        made.append(_Generic(keyshape, [(ordered, _same(len(keyshape)))], [(flags, _same(len(keyshape)))], r))
        flags = flags.reshape(bs + [1] * len(ps) + [count]).expand(shape)
        kinds = [t] * parts + ([I64] if op == 3 else []) + [BOOL]
        identity = _identity(A.amax if op == 4 else A.amin, STORED[t]) if op in (4, 5) else int(op == 2)
        identities = [identity] + [0] * (parts - 1) + ([0] if op == 3 else []) + [False]
        scanned = [_temporary(shape, STORED[k]) for k in kinds]

        def join(c, a, b):
            joined = combined(c, a[:parts], b[:parts])
            if op == 3:
                joined += [c('add', I64, a[parts], b[parts])]
            return [c('select', kind, b[-1], y, z) for kind, y, z in zip(kinds[:-1], b[:-1], joined)] + \
                [c('or', BOOL, a[-1], b[-1])]
        made += _scan(values + [flags], dims - 1, scanned, kinds, identities, join)
        values = scanned[:-1]
    made += _assign(out, base)
    e, inputs = _Region(), [(v, _same(dims)) for v in values] + [(ordered, keymap)]
    linear = e.arg(len(values), ordered)
    next_at = e('select', I32, e('eq', BOOL, e.index(dims - 1), e.const(count - 1, I32)), e.index(dims - 1),
                e('add', I32, e.index(dims - 1), e.const(1, I32)))
    next_key = e('extract', I64, *[e.index(d) for d in range(keydims)], next_at, integer=len(values))
    last = e('or', BOOL, e('eq', BOOL, e.index(dims - 1), e.const(count - 1, I32)), e('ne', BOOL, linear, next_key))
    last = e('and', BOOL, last, e('lt', BOOL, linear, e.const(volume, I64)))
    at = {d: e.index(k) for k, d in enumerate(batches + payload)}
    position = e('select', I64, last, linear, e.const(0, I64))
    at.update(zip(indexed, _coordinates(e, position, [out.shape[d] for d in indexed])))
    coords = [at[d] for d in range(n)]
    a, b = [e.arg(k, v) for k, v in enumerate(values[:parts])], []
    for component in split(base):
        b.append(_convert(e, e('extract', _class(component), *coords, integer=len(inputs)), _class(component), t))
        inputs.append((component, None))
    own = include and op != 0
    result = combined(e, b, a) if own else a
    if op == 3:
        total = e('add', I64, e.arg(parts, values[parts]), e.const(int(own), I64))
        result = [e('floordiv' if integral else 'div', t, x, _convert(e, total, I64, t)) for x in result]
    outputs = split(out)
    result = [_store(e, x, t, v) for x, v in zip(result, outputs)]
    for k, x in enumerate(result):
        e.scatter(k, coords, x, last)
    e.done(*result)
    return made + [_Generic(shape, inputs, [(v, None) for v in outputs], e)]


def _scattered(target, given, outs):
    packet, (out,), x = target.overloadpacket, outs, given['self']
    _typed(target, [], outs, REAL + COMPLEXES)
    if packet is A.index_put:
        tensors, shape, start, places = _advanced(x, list(given['indices']))
        made = [x.shape[j] for j in range(x.dim()) if j in places and places[j] < start] + list(shape) + \
            [x.shape[j] for j in range(x.dim()) if j in places and places[j] >= start + len(shape)]
        values = given['values'].expand(made)
        near = [values.stride(places[j]) if j in places else 0 for j in range(x.dim())]
        far = [values.stride(start + i) for i in range(len(shape))]
        targets = [(t, j, x.shape[j], [0] * x.dim(), list(t.expand(shape).stride())) for j, t in tensors]
        return _scatter(out, x, values, targets, shape, list(x.shape), near, far,
                        SCATTERS['sum' if given['accumulate'] else 'last'], True, negative=True, message=str(target))
    d, index = given['dim'] % max(x.dim(), 1), given['index']
    src = given.get('src', given.get('source', given.get('value')))
    if not x.dim():
        x, out, index = x.view(1), out.view(1), index.reshape(-1)
        if isinstance(src, torch.Tensor):
            src = src.reshape(-1)
    if packet in (A.index_add, A.index_copy, A.index_fill, A.index_reduce):
        far = [src.stride(d) if src.dim() else 0] if isinstance(src, torch.Tensor) else [0]
        near = [src.stride(j) if isinstance(src, torch.Tensor) and src.dim() and j != d else 0 for j in range(x.dim())]
        targets = [(index, d, x.shape[d], [0] * x.dim(), [index.stride(0) if index.dim() else 0])]
        op = given['reduce'] if packet is A.index_reduce else 'sum' if packet is A.index_add else 'last'
        return _scatter(out, x, src, targets, [index.numel()], list(x.shape), near, far, SCATTERS[op],
                        given.get('include_self', True), given.get('alpha', 1), negative=packet is A.index_fill,
                        message=str(target))
    reduce = given.get('reduce')
    op = SCATTERS['sum' if packet is A.scatter_add else reduce if reduce is not None else 'last']
    near = [src.stride(j) if isinstance(src, torch.Tensor) and j != d else 0 for j in range(x.dim())]
    far = [src.stride(d) if isinstance(src, torch.Tensor) else 0]
    targets = [(index, d, x.shape[d], [index.stride(j) if j != d else 0 for j in range(x.dim())], [index.stride(d)])]
    return _scatter(out, x, src, targets, [index.shape[d]],
                    [index.shape[j] if j != d else x.shape[j] for j in range(x.dim())], near, far, op,
                    given.get('include_self', True), message=str(target))


# docs/provider.md#fragment-coded-weights
def _fragment(target, given, outs):
    (out,), codes, tiles = outs, given['codes'], given['tiles']
    if not codes.numel():
        return _fill(out, 0)
    fields = [given[k] for k in ('rows', 'columns', 'tiles', 'bands', 'row_scale', 'col_scale')]
    fields += [tiles.view(torch.float16)[..., 1::2], codes]
    r, t = _Region(), _compute(out.dtype, torch.float32)

    def c(value):
        return r.const(value, U32)

    def op(name, a, b):
        return r(name, U32, a, b)

    def read(k, *indices):
        return r('extract', _class(fields[k]), *indices, integer=k)
    row, col = (_convert(r, read(k, r.index(k)), _class(fields[k]), U64) for k in (0, 1))
    rr, cc = (r('cast', U32, r('rem', U64, x, r.const(32, U64))) for x in (row, col))
    tr, tc = (r('div', U64, x, r.const(32, U64)) for x in (row, col))
    word = read(2, tr, tc)
    width = op('and', op('shr', word, c(12)), c(15))
    lane = op('or', op('or', op('and', op('div', rr, c(8)), c(1)), op('shl', op('and', cc, c(3)), c(1))),
              op('or', op('shl', op('and', op('div', rr, c(8)), c(2)), c(2)),
                 op('shl', op('and', cc, c(4)), c(2))))
    pair = op('add', op('mul', op('and', rr, c(6)), c(2)), op('div', cc, c(8)))
    bit = op('mul', pair, width)
    shift, half = op('rem', bit, c(16)), op('mul', op('and', rr, c(1)), c(16))
    base = op('add', op('and', read(3, tr), c(0x0fffffff)), op('and', word, c(0xfff)))
    at = r('add', U64, r('mul', U64, r('cast', U64, op('add', base, op('div', bit, c(16)))), r.const(32, U64)),
           r('cast', U64, lane))
    live = r('ne', BOOL, width, c(0))
    crossing = r('gt', BOOL, op('add', shift, width), c(16))
    low = r.extract_if(7, U32, [at], live, c(0))
    high = r.extract_if(7, U32, [r('add', U64, at, r.const(32, U64))], r('and', BOOL, live, crossing), c(0))
    low, high = (op('and', op('shr', x, half), c(0xffff)) for x in (low, high))
    joined = op('or', low, op('shl', high, c(16)))
    power = op('shl', c(1), width)
    symbol = op('and', op('shr', joined, shift), op('sub', power, c(1)))
    odd = r('sub', I32, r('cast', I32, op('add', op('mul', symbol, c(2)), c(1))), r('cast', I32, power))
    decoded = r('mul', t, r('cast', t, odd), r('mul', t, r.const(.5, t), r('cast', t, read(6, tr, tc))))
    value = r('select', t, live, decoded, r.const(0, t))
    value = r('mul', t, r('mul', t, value, _convert(r, read(4, row), _class(fields[4]), t)),
              _convert(r, read(5, col), _class(fields[5]), t))
    r.done(_store(r, value, t, out))
    return [_Generic(out.shape, [(x, None) for x in fields], [(out, _same(out.dim()))], r)]


# T. Dao et al., FlashAttention, NeurIPS 2022, Algorithms 1 and 2;
# H. Liu et al., Ring Attention, arXiv:2310.01889, section 3 and Algorithm 1
def _heads(x, group, kv):
    if x.dim() < 3:
        return x.unsqueeze(0).unsqueeze(0)
    heads = x.shape[-3] if kv else x.shape[-3] // group
    shape = (*x.shape[:-3], heads, 1 if kv else group, *x.shape[-2:])
    stride = (*x.stride()[:-3], x.stride(-3) * (1 if kv else group), 0 if kv else x.stride(-3), *x.stride()[-2:])
    return x.as_strided(shape, stride, x.storage_offset())


def _attention(q, k, v, mask, out, causal, scale, group, dropout=0.0, origin=AttentionOrigin(0)):
    length, size, other, value = q.shape[-2], q.shape[-1], k.shape[-2], v.shape[-1]
    t = _compute(q.dtype, k.dtype, v.dtype, torch.float32)
    qs, ks, vs = _heads(q, group, False), _heads(k, group, True), _heads(v, group, True)
    b = len(qs.shape) - 2
    if not other:
        return _fill(out, 0)
    rows = max(1, origin.page // max(k.element_size() * size, v.element_size() * value, 1)) if origin.page else other
    edges = sorted(set([*range(0, other, rows), *origin.cuts, min(origin.key, other), other]))
    blocks = list(zip(edges, edges[1:]))
    first = next((j for j, (lo, _) in enumerate(blocks) if lo == origin.key), 0)
    blocks = blocks[first:] + blocks[:first]
    shape = [*qs.shape[:-2], length]
    score_storage = _temporary([*shape, min(rows, other)], STORED[t])
    m, s, bm, bs, alpha = [_temporary(shape, STORED[t]) for _ in range(5)]
    acc = _temporary([*shape, value], STORED[t])
    generator = _generator() if dropout else None
    ops = []
    for j, (lo, hi) in enumerate(blocks):
        count = hi - lo
        kb, vb = ks.narrow(-2, lo, count), vs.narrow(-2, lo, count)
        scores = score_storage.narrow(-1, 0, count)
        domain = [*shape, count, size]
        r = _Region()
        r.done(r('mul', t, _load(r, 0, qs, t), _load(r, 1, kb, t)))
        ops.append(_Generic(domain,
            [(qs, _rows(b + 3, [*range(b), b, b + 2])), (kb, _onto(kb, [*range(b), b + 1, b + 2], domain))],
            [(scores, _rows(b + 3, range(b + 2)))], r, [PARALLEL] * (b + 2) + [REDUCTION], [SUM], [t],
            precision=REASSOCIATE | CONTRACT))
        r, inputs = _Region(), [(scores, _same(b + 2))]
        score = r('mul', t, r.arg(0, scores), r.const(scale, t))
        if mask is not None:
            ms = _heads(mask, group, False).narrow(-1, lo, count)
            inputs.append((ms, _onto(ms, list(range(b + 2)), scores.shape)))
            given = r.arg(1, ms)
            score = r('select', t, given, score, r.const(-math.inf, t)) if mask.dtype is torch.bool else \
                r('add', t, score, _convert(r, given, _class(mask), t))
        if causal:
            query = r('add', I32, r.index(b), r.const(origin.query, I32))
            key = r('add', I32, r.index(b + 1), r.const(lo, I32))
            score = r('select', t, r('gt', BOOL, key, query), r.const(-math.inf, t), score)
        r.done(score)
        ops.append(_Generic(scores.shape, inputs, [(scores, _same(b + 2))], r))
        ops.append(_statistics(scores, b + 1, bm if j else m, bs if j else s, t))
        if j:
            r = _Region()
            old, new = [r.arg(d, x) for d, x in enumerate((m, s))], [r.arg(d + 2, x) for d, x in enumerate((bm, bs))]
            top, total = _online(r, old, new, t)
            factor = r('select', t, r('eq', BOOL, old[0], r.const(-math.inf, t)), r.const(0, t),
                       r('exp', t, r('sub', t, old[0], top)))
            r.done(top, total, factor)
            ops.append(_Generic(shape, [(x, _same(b + 1)) for x in (m, s, bm, bs)],
                                [(x, _same(b + 1)) for x in (m, s, alpha)], r))
            ops += _elementwise('mul', acc, acc, alpha.unsqueeze(-1))
        domain = [*shape, value, count]
        inputs = [(scores, _rows(b + 3, [*range(b), b, b + 2])),
                  (vb, _onto(vb, [*range(b), b + 2, b + 1], domain)), (m, _rows(b + 3, range(b + 1)))]
        r = _Region()
        score, top = r.arg(0, scores), r.arg(2, m)
        weight = r('select', t, r('eq', BOOL, top, r.const(-math.inf, t)), r.const(0, t),
                   r('exp', t, r('sub', t, score, top)))
        if dropout:
            at = r.const(0, I32)
            logical = origin.shape or tuple(q.shape)
            offsets = origin.offsets or (0,) * (q.dim() - 2) + (origin.query, 0)
            indices = [r.index(d) for d in range(q.dim() - 2)]
            if q.dim() >= 3:
                indices[-1] = r('add', I32, r('mul', I32, r.index(b - 2), r.const(group, I32)), r.index(b - 1))
            indices += [r.index(b), r.index(b + 2)]
            for index, n, offset in zip(indices, (*logical[:-1], other), (*offsets[:-1], lo)):
                index = r('add', I32, index, r.const(offset, I32))
                at = r('add', I32, r('mul', I32, at, r.const(n, I32)), index)
            keep = r('lt', BOOL, _uniform(r, _draws(r, out, inputs, b + 3, at, generator)[0]), r.const(1 - dropout, F32))
            kept = r('mul', t, weight, r.const(1 / (1 - dropout), t)) if dropout < 1 else r.const(0, t)
            weight = r('select', t, keep, kept, r.const(0, t))
        r.done(r('mul', t, weight, _load(r, 1, vb, t)))
        ops.append(_Generic(domain, inputs, [(acc, _rows(b + 3, range(b + 2)))], r,
                            [PARALLEL] * (b + 2) + [REDUCTION], [SUM], [t], [ACCUMULATE if j else IDENTITY],
                            precision=REASSOCIATE | CONTRACT))
    return ops + _epilogue(_heads(out, group, False), acc, t, [(s, _rows(b + 2, range(b + 1)))],
        lambda r, numerator, total: r('select', t, r('eq', BOOL, total, r.const(0, t)), numerator,
                                      r('div', t, numerator, total)))


def _attending(given):
    q, k, v, mask = given['query'], given['key'], given['value'], given['attn_mask']
    lead = tuple(q.shape[:-2])
    group = q.shape[-3] // k.shape[-3] if given['enable_gqa'] and lead else 1
    kv = (*lead[:-1], lead[-1] // group) if lead else ()
    scale = given['scale'] if given['scale'] is not None else 1 / math.sqrt(q.shape[-1])
    return (q, k.expand(*kv, *k.shape[-2:]), v.expand(*kv, *v.shape[-2:]),
            mask.expand(*lead, q.shape[-2], k.shape[-2]) if mask is not None else None), given['is_causal'], scale, group


def _attended(target, given, outs, origin=AttentionOrigin(0)):
    _typed(target, [given['query'], given['key'], given['value'], given['attn_mask']], outs)
    operands, causal, scale, group = _attending(given)
    return _attention(*operands, outs[0], causal, scale, group, float(given['dropout_p']), origin)


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
def _draws(r, out, inputs, loops=None, element=None, generator=None):
    state, site = _generator() if generator is None else generator
    for word in range(3):
        inputs.append((state, _rows(out.dim() if loops is None else loops, [{None: word}])))
    k0, k1, execution = [r('bitcast', U32, r.arg(len(inputs) - 3 + word, state)) for word in range(3)]
    element = r('bitcast', U32, _linear(r, out.shape) if element is None else element)
    return _philox(r, (element, execution, r.const(site, U32), r.const(0, U32)), (k0, k1))


def _uniform(r, word):
    return r('mul', F32, r('cast', F32, r('shr', U32, word, r.const(8, U32))), r.const(2.0 ** -24, F32))


# PyTorch 2.14 documentation, torch.rand, torch.randn, torch.randint, torch.bernoulli, Tensor.uniform_ and
# torch.nn.functional.dropout (torch 2.14.0 aten native_dropout: mask = U < 1 - p, out = x mask / (1 - p)); the normal
# draws by G. E. P. Box and M. E. Muller, Ann. Math. Statist. 29(2), 1958
NORMAL = (A.randn, A.randn_like, torch.ops.prims.normal)
DRAWS = (A.rand, A.rand_like, *NORMAL, A.randint, A.randint_like, A.bernoulli, A.uniform, A.native_dropout)


def _random(target, given, outs):
    packet, out, scale = target.overloadpacket, outs[0], 1
    if out.is_complex() and packet in (A.rand, A.rand_like, *NORMAL, A.uniform):
        out, scale = torch.view_as_real(out), math.sqrt(0.5) if packet in NORMAL else 1
    _typed(target, [], [out], REAL)
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
    if packet in NORMAL:
        first = r('add', F32, u, r.const(2.0 ** -24, F32))
        radius = r('sqrt', F32, r('mul', F32, r.const(-2, F32), r('log', F32, first)))
        made = r('mul', F32, radius, r('cos', F32, r('mul', F32, r.const(2 * math.pi, F32), _uniform(r, words[1]))))
        made = made if scale == 1 else r('mul', F32, made, r.const(scale, F32))
        if packet is torch.ops.prims.normal:
            made = r('add', F32, r.const(float(given['mean']), F32), r('mul', F32, r.const(float(given['std']), F32), made))
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


# PyTorch 2.14 documentation, torch.nn.RReLU (B. Xu, N. Wang, T. Chen and M. Li, Empirical evaluation of rectified
# activations in convolutional network, arXiv:1505.00853, 2015): training, an element x <= 0 its slope a Philox draw
# a ~ U(lower, upper), out = a x and noise = a, other elements x and 1; evaluation, the slope (lower + upper) / 2, the
# noise passed through (torch 2.14.0 aten rrelu_with_noise_functional)
def _rrelu(target, given, outs):
    x, noise, lower, upper = given['self'], given['noise'], float(given['lower']), float(given['upper'])
    (out, kept), same = outs, _same(outs[0].dim())
    _typed(target, [x, noise], outs)
    t = _compute(out.dtype, x.dtype, torch.float32)
    r, inputs = _Region(), [(x, same)]
    v = _load(r, 0, x, t)
    if not given['training']:
        made = r('select', t, r('ge', BOOL, v, r.const(0, t)), v, r('mul', t, v, r.const((lower + upper) / 2, t)))
        r.done(_store(r, made, t, out))
        return [_Generic(out.shape, inputs, [(out, same)], r)] + _copy(kept, noise)
    drawn = _convert(r, _uniform(r, _draws(r, out, inputs)[0]), F32, t)
    slope = r('add', t, r.const(lower, t), r('mul', t, drawn, r.const(upper - lower, t)))
    low = r('le', BOOL, v, r.const(0, t))
    r.done(_store(r, r('select', t, low, r('mul', t, v, slope), v), t, out),
           _store(r, r('select', t, low, slope, r.const(1, t)), t, kept))
    return [_Generic(out.shape, inputs, [(out, same), (kept, same)], r)]


# the generator's advance: its execution word incremented once an execution, after the program's draws
def _stepped(state):
    r = _Region()
    r.done(r('add', I32, r.arg(0, state), r.const(1, I32)))
    return _Generic((1,), [(state, _rows(1, [{None: 2}]))], [(state, _rows(1, [{None: 2}]))], r)


# the inclusive count of the true elements of `mask` at each of its row-major positions, by the tree scan (_blelloch)
def _counted(mask, count):
    n = mask.numel()
    size = 1 << max(n - 1, 0).bit_length()
    buffer = _temporary([size], torch.int32)

    def element(r, j, k):
        return _convert(r, r('extract', _class(mask), *_coordinates(r, j, mask.shape), integer=k), _class(mask), I32)
    r = _Region()
    inside, at = _bounded(r, r.index(0), n)
    r.done(r('select', I32, inside, element(r, at, 0), r.const(0, I32)))
    ops = [_Generic([size], [(mask, None)], [(buffer, _same(1))], r)]
    ops += _blelloch([buffer], I32, [0], lambda r, a, b: [r('add', I32, a[0], b[0])])
    e = _Region()
    e.done(e('add', I32, e.arg(0, buffer), element(e, e.index(0), 1)))
    return ops + [_Generic([n], [(buffer, _same(1)), (mask, None)], [(count, _same(1))], e)]


# PyTorch 2.14 documentation, Tensor.masked_scatter_: position i of the result the source's next element where the
# (broadcast) mask is true, its rank among the true positions the inclusive count less one
def _masked(target, given, outs):
    (out,), x, mask, source = outs, given['self'], given['mask'], given['source']
    _typed(target, [x, source], outs, REAL)
    mask = mask.expand(x.shape)
    count = _temporary([x.numel()], torch.int32)
    ops = _counted(mask, count)
    r = _Region()
    linear = _linear(r, x.shape)
    taken = _convert(r, r.arg(0, mask), _class(mask), BOOL)
    at = r('select', I32, taken, r('sub', I32, r('extract', I32, linear, integer=1), r.const(1, I32)), r.const(0, I32))
    flat = source.reshape(-1) if source.is_contiguous() else source
    moved = _convert(r, r('extract', _class(flat), *(_coordinates(r, at, flat.shape) if flat.dim() > 1 else [at]),
                          integer=3), _class(flat), _class(out))
    r.done(r('select', _class(out), taken, moved, _convert(r, r.arg(2, x), _class(x), _class(out))))
    return ops + [_Generic(out.shape, [(mask, _same(out.dim())), (count, None), (x, _same(out.dim())), (flat, None)],
                           [(out, _same(out.dim()))], r)]


# PyTorch 2.14 documentation, torch.randperm: the ranks of n uniform draws, a uniformly random permutation
def _permuted(target, given, outs):
    (out,) = outs
    keys = _temporary([given['n']])
    ops = _random(A.rand.default, {'size': [given['n']]}, [keys])
    return ops + _ranked(A.sort.default, {'self': keys, 'dim': 0, 'descending': False}, [_temporary([given['n']]), out])


# PyTorch 2.14 documentation, torch.nn.functional.embedding_bag (aten/src/ATen/native/EmbeddingBag.cpp): bag b the
# indices [offsets[b], offsets[b + 1]) (the last to the end), less padding_idx; its sum, mean or maximum (with the
# chosen row's index) of the weight rows, scaled by per-sample weights; offset2bag each index's bag, bag_size each bag's
# count
def _bagged(target, given, outs):
    weight, indices, offsets = given['weight'], given['indices'], given['offsets']
    out, offset2bag, bag_size, max_indices = outs
    mode, last, padding = given['mode'], given['include_last_offset'], given['padding_idx']
    psw = given.get('per_sample_weights')
    bags, n, d = out.shape[0], indices.numel(), weight.shape[1]
    t = _compute(out.dtype, weight.dtype, torch.float32)
    offsets, indices = _indexing(offsets), _indexing(indices.reshape(-1))
    r = _Region()
    b, j = r.index(0), r.index(2)
    lo = _convert(r, r('extract', I32, b, integer=0), I32, I32)
    nxt = r('add', I32, b, r.const(1, I32))
    room = r('lt', BOOL, nxt, r.const(offsets.shape[0], I32))
    hi = r('select', I32, room, r('extract', I32, r('select', I32, room, nxt, b), integer=0), r.const(n, I32))
    row = r('extract', I32, j, integer=1)
    inside = r('and', BOOL, r('and', BOOL, r('le', BOOL, lo, j), r('lt', BOOL, j, hi)),
               r('ne', BOOL, row, r.const(padding, I32)) if padding >= 0 else r.const(True, BOOL))
    v = _convert(r, r('extract', _class(weight), r('select', I32, inside, row, r.const(0, I32)), r.index(1), integer=2),
                 _class(weight), t)
    if psw is not None:
        v = r('mul', t, v, _convert(r, r('extract', _class(psw), j, integer=3), _class(psw), t))
    larger = mode == 2
    r.done(r('select', t, inside, v, r.const(-math.inf if larger else 0, t)), _convert(r, inside, BOOL, I32),
           r('select', I32, inside, row, r.const(-1, I32)), r('select', I32, inside, j, r.const(-1, I32)))
    states = [(t, -math.inf if larger else 0), (I32, 0), (I32, -1), (I32, -1)]

    def join(c, a, x):
        if not larger:
            return [c('add', t, a[0], x[0]), c('add', I32, a[1], x[1]), a[2], a[3]]
        value, position = _ordered(c, (a[0], a[3]), (x[0], x[3]), t, True)
        right = c('ne', BOOL, position, a[3])
        return [value, c('add', I32, a[1], x[1]), c('select', I32, right, x[2], a[2]), position]
    f = _Region()
    total, count, chosen = f('arg', t, integer=0), f('arg', I32, integer=1), f('arg', I32, integer=2)
    value = f('div', t, total, _convert(f, f('select', I32, f('eq', BOOL, count, f.const(0, I32)), f.const(1, I32), count),
                                        I32, t)) if mode == 1 else \
        f('select', t, f('eq', BOOL, count, f.const(0, I32)), f.const(0, t), total)
    sources = [(offsets, None), (indices, None), (weight, None)] + ([(psw.reshape(-1), None)] if psw is not None else [])
    if larger:
        f.done(value, chosen)
        outputs, accumulators = [(out, _rows(3, [0, 1])), (max_indices, _rows(3, [0, 1]))], [t, I32]
    else:
        f.done(value)
        outputs, accumulators = [(out, _rows(3, [0, 1]))], [t]
    ops = [_Generic([bags, d, n], sources, outputs, r, [PARALLEL, PARALLEL, REDUCTION], [REGION] * len(outputs),
                    accumulators, None, states, _joined(states, join), f, REASSOCIATE)]
    q = _Region()
    b, j = q.index(0), q.index(1)
    lo, nxt = q('extract', I32, b, integer=0), q('add', I32, b, q.const(1, I32))
    room = q('lt', BOOL, nxt, q.const(offsets.shape[0], I32))
    hi = q('select', I32, room, q('extract', I32, q('select', I32, room, nxt, b), integer=0), q.const(n, I32))
    row = q('extract', I32, j, integer=1)
    counted = q('and', BOOL, q('le', BOOL, lo, j), q('lt', BOOL, j, hi))
    if padding >= 0:
        counted = q('and', BOOL, counted, q('ne', BOOL, row, q.const(padding, I32)))
    q.done(_convert(q, counted, BOOL, I32))
    ops.append(_Generic([bags, n], [(offsets, None), (indices, None)], [(bag_size, _rows(2, [0]))], q,
                        [PARALLEL, REDUCTION], [SUM], [I32]))
    g = _Region()
    g.done(_convert(g, g('le', BOOL, g('extract', I32, g.index(1), integer=0), g.index(0)), BOOL, I32))
    ops.append(_Generic([n, offsets.shape[0] - int(bool(last))], [(offsets, None)], [(offset2bag, _rows(2, [0]))], g,
                        [PARALLEL, REDUCTION], [SUM], [I32]))
    h = _Region()
    h.done(h('sub', I32, _convert(h, h.arg(0, offset2bag), _class(offset2bag), I32), h.const(1, I32)))
    ops.append(_Generic([n], [(offset2bag, _same(1))], [(offset2bag, _same(1))], h))
    if not larger and max_indices.numel():
        ops += _fill(max_indices, 0)
    return ops


# PyTorch 2.14 documentation, torch.cdist and torch.nn.functional.pdist: a p-norm of coordinate differences, p 0 the
# count of differing coordinates and inf the largest difference; its term, state, join and finalization
def _norm(r, diff, p, t):
    term = _convert(r, r('ne', BOOL, diff, r.const(0, t)), BOOL, t) if p == 0 else diff if p == 1 or math.isinf(p) else \
        r('mul', t, diff, diff) if p == 2 else r('pow', t, diff, r.const(p, t))
    r.done(term)
    states = [(t, 0)]
    f = _Region()
    total = f('arg', t, integer=0)
    f.done(total if p in (0, 1) or math.isinf(p) else f('sqrt', t, total) if p == 2 else
           f('pow', t, total, f.const(1 / p, t)))
    join = (lambda c, a, b: [_extremum(c, a[0], b[0], t, True)]) if math.isinf(p) else \
        (lambda c, a, b: [c('add', t, a[0], b[0])])
    return states, _joined(states, join), f


def _distances(target, given, outs):
    (out,), x1, x2, p = outs, given['x1'], given['x2'], float(given['p'])
    _typed(target, [x1, x2], outs)
    lead, t = out.dim() - 2, _compute(out.dtype, x1.dtype, x2.dtype, torch.float32)
    loops = [*out.shape, x1.shape[-1]]
    r = _Region()
    states, join, f = _norm(r, r('abs', t, r('sub', t, _load(r, 0, x1, t), _load(r, 1, x2, t))), p, t)
    maps = [_onto(x1, [*range(lead), lead, lead + 2], loops), _onto(x2, [*range(lead), lead + 1, lead + 2], loops)]
    return [_Generic(loops, [(x1, maps[0]), (x2, maps[1])], [(out, _same(lead + 3, out.dim()))], r,
                     [PARALLEL] * (lead + 2) + [REDUCTION], [REGION], [t], None, states, join, f, REASSOCIATE)]


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


def _placed(ops, place, node):
    for g in ops:
        g.place, g.node = place, node


# design/torch-source-lowering-change-request.md item 1; design/compilation-composition-change-request.md, "Complete
# replacement workgroup": each call's structured operations, a view's materialization its copy
SOURCES = {mesh_code.fragment.overloadpacket: _fragment, **dict.fromkeys(POINTWISE, _arithmetic),
           **dict.fromkeys((*COPIES, A.copy, A.copy_), _copying),
           **dict.fromkeys((A.mm, A.bmm, A.matmul, A.linear, A.addmm), _product), A.einsum: _summation,
           **dict.fromkeys(FACTORIZATIONS, _factorized),
           A._linalg_det: _determinant, A._linalg_slogdet: _determinant,
           A.linalg_lu: _unpacked_lu, A.linalg_pinv: _pseudoinverse, A.linalg_lstsq: _least_squares,
           A._linalg_svd: _singular, A._linalg_eigh: _eigen_hermitian,
           **dict.fromkeys((A.linalg_eig, A.linalg_eigvals, A._linalg_eigvals), _eigen),
           A._linalg_check_errors: _checked,
           **dict.fromkeys(ASSERTIONS, _asserted),
           **dict.fromkeys(ORTHOGONAL, _orthogonal), A.triu: _triangular_part, A.tril: _triangular_part,
           **dict.fromkeys((*CONVOLUTION, *TRANSPOSED, A.convolution), _convolving),
           **dict.fromkeys((*MAXIMUM, *AVERAGE, *INDEXED), _pooled), **dict.fromkeys(PAD, _padded),
           **dict.fromkeys((*FILLS, *EMPTY, A.arange), _factory), **dict.fromkeys(REDUCTIONS, _reduced),
           **dict.fromkeys(SOFTMAX, _softened), **dict.fromkeys(SCANS, _scanned), A.sort: _ranked, A.topk: _ranked,
           **dict.fromkeys(NORMS, _normalized),
           **dict.fromkeys((A.gather, A.index_select, A.embedding, A.index), _taken),
           **dict.fromkeys((A.scatter, A.scatter_add, A.scatter_reduce, A.index_put, A.index_add, A.index_copy,
                            A.index_fill, A.index_reduce), _scattered),
           A.scaled_dot_product_attention: _attended, A.cat: _concatenated, **dict.fromkeys(DRAWS, _random),
           **dict.fromkeys(PARAMETERIZED, _parameterized), **dict.fromkeys(FOLDS, _fold),
           **dict.fromkeys(EXTREMA, _extreme), A.var: _variance, **dict.fromkeys(PADS, _padding), A.flip: _flipped,
           A.repeat: _repeated, A.slice_scatter: _embedded, A.select_scatter: _embedded,
           A.convolution_backward: _convolution_backward, A.avg_pool2d_backward: _pooled_backward,
           A._adaptive_avg_pool3d: _adaptive, A.masked_scatter: _masked, A.randperm: _permuted,
           A._embedding_bag: _bagged, A._cdist_forward: _distances, A._adaptive_avg_pool2d_backward: _adaptive_backward,
           A._pdist_forward: _paired, **dict.fromkeys(COMPLEX, _complexed),
           **dict.fromkeys((A._fft_r2c, A._fft_c2r, A._fft_c2c), _transformed),
           A.rrelu_with_noise_functional: _rrelu, A.cummax: _cumulative, A.cummin: _cumulative,
           A.searchsorted: _searched, **dict.fromkeys(SELECTIONS, _selected),
           A.adaptive_max_pool2d: _adaptive_max, A.adaptive_max_pool3d: _adaptive_max, A.histc: _histogram}
# PyTorch 2.14 documentation, Tensor Views
VIEWS = {A.as_strided, A.detach, A.diagonal, A.expand, A.expand_as, A.movedim, A.narrow, A.permute, A.select,
         A.squeeze, A.transpose, A.t, A.adjoint, A.real, A.imag, A.view_as_real, A.unflatten, A.unfold, A.unsqueeze,
         A.view, A.view_as, A.unbind, A.split, A.split_with_sizes, A.tensor_split, A.hsplit, A.vsplit, A.swapaxes,
         A.swapdims, A.chunk, A.slice, A.alias, A.view_as_complex, operator.getitem}


# PyTorch 2.14 documentation, Complex Numbers (and torch.nan_to_num, each component replaced): an operation linear in
# its data (its indices, masks and shapes real), or acting on each component alone, the same operation on each of the
# (re, im) components, a real scalar datum the real component's alone and a real
# datum's imaginary component zero
LINEAR = {*PAD, *PADS, A.sum, A.mean, A.gather, A.index_select, A.embedding, A.index, A.cat, A.flip, A.repeat,
          A.slice_scatter, A.select_scatter, A.masked_scatter, A.nan_to_num}
VALUED = ('value', 'fill_value', 'src', 'source', 'values', 'self', 'input', 'tensors', 'weight')


def _complexes(given, outs):
    return any(isinstance(v, torch.Tensor) and v.is_complex() or isinstance(v, complex)
               for v in pytree.tree_leaves([given, outs]))


def _lifted(target, lower, given, outs):
    ops = []
    for k in (0, 1):
        def part(v, datum):
            if isinstance(v, (list, tuple)):
                return type(v)(part(x, datum) for x in v)
            if isinstance(v, torch.Tensor) and v.is_complex():
                return torch.view_as_real(v)[..., k]
            if isinstance(v, torch.Tensor) and datum and k:
                zero = _temporary(v.shape, v.dtype)
                ops.extend(_fill(zero, 0))
                return zero
            if isinstance(v, (int, float, complex)) and not isinstance(v, bool) and datum:
                return _component(v, k)
            return v
        made = {name: part(v, name in VALUED) for name, v in given.items()}
        ops += lower(target, made, [part(o, False) for o in outs])
    return ops


# Python's operator module on a held scalar (torch.fx symbolic arithmetic of SymInt/SymBool/SymFloat): the pointwise
# operation of the same semantics (floor division and modulo as Python's, torch.floor_divide and torch.remainder)
SYMBOLIC = {operator.add: 'add', operator.sub: 'sub', operator.mul: 'mul', operator.truediv: 'div',
            operator.floordiv: 'floordiv', operator.mod: 'remainder', operator.neg: 'neg', operator.lt: 'lt',
            operator.le: 'le', operator.gt: 'gt', operator.ge: 'ge', operator.eq: 'eq', operator.ne: 'ne',
            operator.and_: 'logical_and', operator.or_: 'logical_or', operator.not_: 'logical_not',
            torch.sym_not: 'logical_not'}


def _lowering(local, page):
    packet = getattr(local.target, 'overloadpacket', local.target)
    if packet is A.scaled_dot_product_attention:
        origin = local.annotation if isinstance(local.annotation, AttentionOrigin) else AttentionOrigin(0, 0, page)
        return lambda args, kwargs, outs: _attended(local.target, _operands(local.target, args, kwargs), outs, origin)
    if packet in (A._linalg_det, A._linalg_slogdet):
        metadata = _given(local, _names(local))['A'].meta['val']
        transpose = metadata.is_contiguous() and not metadata.is_complex()
        return lambda args, kwargs, outs: _determinant(local.target, _operands(local.target, args, kwargs), outs,
                                                       transpose)
    if packet is A.linalg_pinv:
        metadata = _given(local, _names(local))['self'].meta['val']
        dtype = torch.float32 if metadata.device.type in ('mps', 'metal') else torch.float64
        return lambda args, kwargs, outs: _pseudoinverse(local.target, _operands(local.target, args, kwargs), outs,
                                                        dtype)
    if local.target in SYMBOLIC:
        return lambda args, kwargs, outs: _elementwise(SYMBOLIC[local.target], outs[0], *args)
    if packet in LINEAR:
        def lifted(args, kwargs, outs):
            given = _operands(local.target, args, kwargs)
            lower = SOURCES[packet]
            return (_lifted(local.target, lower, given, outs) if _complexes(given, outs)
                    else lower(local.target, given, outs))
        return lifted
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
    lower = _lowering(local, ir.page)
    given = iter(operands)
    args, kwargs = map_arg((local.args, local.kwargs), lambda n: next(given))
    mine = placed(key)
    if isinstance(form, tuple):
        outs = [placed(result) for _, result in form[1]]
        return pytree.tree_unflatten(outs, pytree.tree_structure(ir.prototypes[key])), lower(args, kwargs, outs)
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
    outs, spec = _made(ir, key, local)
    return pytree.tree_unflatten(outs, spec), lower(args, kwargs, outs)


def _made(ir, key, local):
    if isinstance(local.target, torch._ops.OpOverload) and not local.target._schema.returns:
        return [None], pytree.tree_structure(None)
    made, spec = pytree.tree_flatten(ir.prototypes[key])
    return [_held(m) if m is not None else None for m in made], spec


# torch/__init__.py SymInt, SymBool and SymFloat (Python int, bool and float semantics): a scalar a higher-order
# operator carries, its 0-d value in the program (int64, bool, float64)
def _held(m):
    if isinstance(m, torch.Tensor):
        return torch.empty_strided(m.shape, m.stride(), dtype=m.dtype, device='meta')
    kind = torch.bool if isinstance(m, (bool, torch.SymBool)) else \
        torch.int64 if isinstance(m, (int, torch.SymInt)) else torch.float64
    return torch.empty((), dtype=kind, device='meta')


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
        def real(v):
            return torch.view_as_real(v) if v.is_complex() else v
        total, first = real(total), real(first)
        _typed(collective.kind, [first], [total])
        flat = total.view(-1)
        whole = [first.reshape(-1)] + [real(placed(received)).view(-1).narrow(0, j * flat.numel(), flat.numel())
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
    if kind == 'pad':
        dim, extent = operand
        padding = [0] * (2 * values[0].dim())
        padding[2 * (values[0].dim() - dim) - 1] = extent - values[0].shape[dim]
        return A.constant_pad_nd.default(values[0], padding, 0)
    if kind == 'slice':
        (partition_dim, m), world_size = operand, ir.shape[operand[1]]
        weight_list = torch.split(values[0], -(-_layout(ir, reads[0])[1][partition_dim] // world_size), dim=partition_dim)
        (piece,) = weight_list[ir.coords[ir.device][m]::world_size]
        return piece
    return values[0].narrow(*_cut(ir, kind, operand, _layout(ir, reads[0])[1], ir.device))


# torch/export/graph_signature.py InputKind.CONSTANT_TENSOR; mesh_code.Matrix._mesh_constants
def _constant_origins(exported, module):
    values = [value for name, value in exported.constants.items()
              if name not in exported.graph_signature.non_persistent_buffers]
    values += [operator.attrgetter(name)(child) for child in module.modules()
               for name in getattr(child, '_mesh_constants', ())]
    written = _storages(module, [*exported.graph_signature.buffers_to_mutate.values(),
                                 *exported.graph_signature.parameters_to_mutate.values()])
    return frozenset(value.untyped_storage().data_ptr() for value in values if isinstance(value, torch.Tensor)) - written


# metal-microbench docs/provider.md#source-constants-before-target-preparation;
# torch/export/graph_signature.py ExportGraphSignature.buffers_to_mutate, parameters_to_mutate
def _initial_values(ir, immutable):
    if ir.initial is not None:
        return ir.initial
    loaded = ir.loads & ir.homes.keys()
    needed = mesh_graph.reach(loaded, lambda key: ir.attributes[key][1])
    order = [key for key in ir.attributes if key in needed]
    host = mesh_graph.forward(order, lambda key: ir.attributes[key][1],
                              lambda key, values: _hosted(ir, ir.attributes[key], values))
    fixed = mesh_graph.forward(order, lambda key: ir.attributes[key][1],
                               lambda key, values: _key(host[key]) in immutable
                               if ir.attributes[key][0] == 'constant' else all(values))
    initial, mutable = {}, set()
    for key in loaded:
        v, offset, sizes, strides = ir.homes[key]
        if v not in initial:
            dtype, numel, _ = ir.layouts[v]
            initial[v] = torch.zeros(numel, dtype=dtype, device='cpu')
        initial[v].as_strided(sizes, strides, offset).copy_(host[key])
        if not fixed[key]:
            mutable.add(v)
    ir.initial, ir.constant_values = initial, set(initial) - mutable
    return initial


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
    if key in ir.attributes:
        return value
    if kind == 'input':
        value = given[operand]
    elif kind == 'constant' and value is None:
        value = operator.attrgetter(operand)(ir.gm)
    elif kind == 'copy':
        ops += _copied(ir, reads[0], key, operand, operands[0], value)
    elif kind == 'pad' and key not in ir.loads:
        source = operands[0]
        ops += _fill(value, 0)
        ops += _assign(value.narrow(operand[0], 0, source.shape[operand[0]]), source)
    elif kind == 'window' and key in ir.inside:
        dim, m, _, length, starts, size, fill = operand
        q = ir.coords[ir.device][m]
        below, above = min(length, max(0, -starts[q])), min(length, max(0, size - starts[q]))
        for start, extent in ((0, below), (above, length - above)):
            ops += _fill(value.narrow(dim, start, extent), fill) if fill is not None and extent else []
    elif kind in ('slice', 'narrow', 'window') and value is None:
        cut = _cut(ir, kind, operand, _layout(ir, reads[0])[1], ir.device)
        value = pytree.tree_map(lambda tensor: tensor.narrow(*cut), operands[0])
    elif kind == 'call':
        operands = list(operands)
        for at, dim, m, size in ir.trims.get(key, ()):
            extent = -(-size // ir.shape[m])
            valid = max(0, min(extent, size - ir.coords[ir.device][m] * extent))
            operands[at] = operands[at].narrow(dim, 0, valid)
        value, made = _called(ir, key, operands, ir.locals[operand], ir.forms.get(key), placed)
        ops += made
    elif kind == 'collective':
        ops += _collected(ir, placed, item, operands)
    return value


def _instantiate(ir, tensors, given, spec, stream=0):
    RANDOM.update(state=None, sites=0, stream=stream)
    ops, values = [], {}
    nodes = {node: index for index, node in enumerate(ir.order)}

    def placed(key):
        return tensors[key][0] if key in tensors else None

    def read(key):
        return values[key] if key in values else placed(key)

    def remember(key, value):
        values[key] = value
        form = ir.forms.get(key)
        if isinstance(form, tuple):
            for (_, leaf), made in zip(form[1], pytree.tree_leaves(value)):
                if leaf != key:
                    remember(leaf, made)
    for item in ir.program:
        kind, reads, key, operand = item
        if kind == 'output':
            node, keys = operand
            result = map_arg(node.args[0], lambda n: read(keys[n]))
            break
        first = len(ops)
        remember(key, _specialized(ir, placed, item, [read(r) for r in reads], ops, given))
        _placed(ops[first:], CPU if _device(ir, (key[0], ir.sharding.get(key[0]))) == 'cpu' else GPU,
                nodes.get(key[0], len(nodes)))
    if RANDOM['state'] is not None:
        ops.append(_stepped(RANDOM['state']))
    return ops, pytree.tree_unflatten(pytree.tree_leaves(result), spec)


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
    failure = _execution_error(replay, status, replay['error'].value.decode()) if status else None
    mesh.LIB.mesh_retire(context.M, context.client)
    _released(state['owned'])
    mapping = state['mapping']
    state.update(replay=None, mapping=None, devices=(), executions=0, stale=True)
    if mapping is not None:
        mapping.retired = True
        mapping.release()
    if status:
        raise failure


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
    def __init__(self, lib, owners, role, row, identity, constants=None):
        self.lib, self.p, self.owners, self.role, self.row = lib, lib.metal_ir_program_new(), owners, role, row
        self.values, self.roots, self.roles, self.planned, self.builtins = {}, {}, {}, {}, {}
        self.constants = {} if constants is None else constants
        self.base = self.identity = identity

    def root(self, t):
        key = _key(t)
        if key in self.roots:
            return self.roots[key]
        base = t.tensor if isinstance(t, _Words) else t
        base = torch.view_as_real(base) if base.is_complex() else base
        numel = base.untyped_storage().nbytes() // base.element_size()
        owner = self.owners.get(key)
        role = CONSTANT if key in self.constants else self.role(owner) if owner is not None else STATE if key in GENERATORS else \
            CONSTANT if base.device.type != 'meta' else TEMPORARY
        v = self.lib.metal_ir_value(self.p, role, SCALAR[base.dtype], 1, _array(mesh.Q, [numel]), 1,
                                    _array(Term, [Term(0, 0, 1, 1, 0)]), 0, self.identity)
        if owner is not None:
            k, index, _, pages, row = owner
            self.lib.metal_ir_value_planned(self.p, v, k, index, 0)
            if pages:
                self.lib.metal_ir_value_paged(self.p, v, row, pages)
            self.planned[v] = owner
        if key in self.constants or owner is None and base.device.type != 'meta':
            base = self.constants.get(key, base)
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
            elif call == 'extract_if':
                extract = lib.metal_ir_extract_if
                extract.restype, extract.argtypes = mesh.U, [C.c_void_p, mesh.U, mesh.U, mesh.U,
                                                             C.POINTER(mesh.U), mesh.U, mesh.U]
                given = _array(mesh.U, [h[x] for x in record[2]])
                h.append(extract(p, r, record[1], len(record[2]), given, h[record[3]], h[record[4]]))
            elif call == 'scatter':
                given = _array(mesh.U, [h[x] for x in record[2]])
                lib.metal_ir_scatter(p, r, record[1], len(record[2]), given, h[record[3]], h[record[4]])
                h.append(None)
            else:
                given = _array(mesh.U, [h[x] for x in record[1]])
                lib.metal_ir_yield(p, r, len(record[1]), given)
                h.append(None)
            if h[-1] == NONE:
                raise NotImplementedError('a region\'s %s of %s: not an instruction of the IR (metal_ir.h, UINT32_MAX)'
                                          % (' '.join(str(x) for x in record[:2]), KINDS.get(region.kinds[len(h) - 1])))

    # metal_ir.h `identity`: the producer's identity of an operation and of the values it first names, (function,
    # call ordinal, FX node position) of the call's graph, kept through every transformation (metal_ir_operation_info)
    def generic(self, g):
        if isinstance(g, _Check):
            self.identity = self.base | g.node
            operation = self.lib.metal_ir_check(self.p, self.value(g.info), g.code, self.identity)
            if g.message:
                self.lib.metal_ir_check_message(self.p, operation, g.message.encode())
            return operation
        if isinstance(g, _Lapack):
            if g.place != CPU:
                raise NotImplementedError('a LAPACK driver\'s node on a processor other than the CPU: its lowering is '
                                          'unfinished (design/torch-source-lowering-change-request.md item 6)')
            self.identity = self.base | g.node
            sizes, picks, matrices, operands = list(g.batch), [], [], []
            for t, rank in g.inputs + g.outputs:
                real = torch.view_as_real(t) if t.is_complex() else t
                own = rank + t.is_complex()
                lead, first = real.dim() - own, len(g.batch) - (real.dim() - own)
                picks.append([None if n == 1 and g.batch[first + d] != 1 else first + d
                              for d, n in enumerate(real.shape[:lead])] + list(range(len(sizes), len(sizes) + own)))
                loops = list(range(len(sizes), len(sizes) + rank)) + [NONE] * (2 - rank)
                matrices.append(Matrix(loops[0], loops[1], len(sizes) + rank if t.is_complex() else NONE))
                sizes += real.shape[lead:]
                operands.append(self.value(real))
            exprs, terms = [], []
            for pick in picks:
                for row in _rows(len(sizes), pick):
                    first = len(terms)
                    terms += [Term(l, 0, c, 1, 0) for l, c in enumerate(row[:len(sizes)]) if c]
                    exprs.append(Expr(first, len(terms) - first, row[len(sizes)]))
            call = Lapack(g.routine, len(g.batch), g.job, g.rcond)
            return self.lib.metal_ir_lapack(self.p, C.byref(call), len(sizes), _array(mesh.Q, sizes), len(g.inputs),
                                            _array(mesh.U, operands[:len(g.inputs)]), len(g.outputs),
                                            _array(mesh.U, operands[len(g.inputs):]), _array(Matrix, matrices),
                                            _array(Expr, exprs), len(terms), _array(Term, terms), self.identity)
        if isinstance(g, _Sort):
            self.identity = self.base | g.node
            sort = self.lib.metal_ir_sort
            sort.restype, sort.argtypes = mesh.U, [C.c_void_p, mesh.U, mesh.U, mesh.Q, mesh.U, mesh.U, mesh.U, mesh.U,
                                                   mesh.Q]
            return sort(self.p, self.value(g.x), g.axis, g.k, int(g.descending), self.value(g.values),
                        self.value(g.indices), g.place, self.identity)
        self.identity = self.base | g.node
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


class Lapack(C.Structure):
    _fields_ = [('routine', mesh.U), ('batch', mesh.U), ('job', C.c_char * 8), ('rcond', C.c_double)]


class Matrix(C.Structure):
    _fields_ = [('row', mesh.U), ('column', mesh.U), ('part', mesh.U)]


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


class ValueInfo(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('role', 'scalar', 'storage', 'function', 'index', 'word', 'of', 'range',
                                          'received', 'sent', 'zeroed', 'sendings')] + [
        (name, mesh.Q) for name in ('identity', 'bytes', 'offset', 'at', 'data_bytes')] + [
        ('data', C.c_void_p), ('buffer', C.c_void_p), ('release', Release), ('page_bytes', mesh.Q),
        ('pages', mesh.U), ('source', mesh.U)]


MACHINES = []


# Roofline (S. Williams, A. Waterman and D. Patterson, CACM 52(4), 2009, §3) operands of this node's resources, the
# configuration's record (rdma/mesh-metal.h mesh_metal_machines, no path: the etc/mesh-capacity.json of the mesh
# checkout whose libmesh the process loaded), read once; ENOENT: no record of this node's model, so no estimate
def _machines():
    if not MACHINES:
        gpu, cpu = mesh_graph.Machine(), mesh_graph.Machine()
        status = _transport().mesh_metal_machines(None, C.byref(gpu), C.byref(cpu))
        if status not in (0, errno.ENOENT):
            mesh.check(status)
        MACHINES.append({GPU: gpu, CPU: cpu} if status == 0 else {})
    return MACHINES[0]


COMPILER = []


def _compiler():
    if not COMPILER:
        lib = C.CDLL(os.path.join(_repository(), '.build', 'libmetal_ir.dylib'))
        P, U, Q = C.c_void_p, mesh.U, mesh.Q
        for name, result, arguments in (
                ('metal_ir_program_new', P, []), ('metal_ir_program_free', None, [P]),
                ('metal_ir_save', C.c_int, [P, C.POINTER(P), C.POINTER(C.c_size_t), C.c_char_p, C.c_size_t]),
                ('metal_ir_load', P, [P, C.c_size_t, P, C.c_char_p, C.c_size_t]),
                ('metal_ir_loaded', C.c_int, [P, C.c_char_p, C.c_size_t]),
                ('metal_ir_value_info', None, [P, U, C.POINTER(ValueInfo)]),
                ('metal_ir_value', U, [P, U, U, U, C.POINTER(Q), U, C.POINTER(Term), C.c_int64, Q]),
                ('metal_ir_value_view', None, [P, U, U]),
                ('metal_ir_value_data', None, [P, U, P, C.c_size_t, P, Q, P, RELEASE]),
                ('metal_ir_value_storage', None, [P, U, P, Q]),
                ('metal_ir_step_value', U, [P]), ('metal_ir_step_state', U, [P]),
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
                ('metal_ir_check', U, [P, U, U, Q]),
                ('metal_ir_check_message', None, [P, U, C.c_char_p]),
                ('metal_ir_lapack', U, [P, C.POINTER(Lapack), U, C.POINTER(Q), U, C.POINTER(U), U, C.POINTER(U),
                                        C.POINTER(Matrix), C.POINTER(Expr), U, C.POINTER(Term), Q]),
                ('metal_ir_region', U, [P, U, U]), ('metal_ir_parameter', U, [P, U, U]),
                ('metal_ir_literal', U, [P, U, U, Q]), ('metal_ir_binary', U, [P, U, U, U, U]),
                ('metal_ir_unary', U, [P, U, U, U]), ('metal_ir_convert', U, [P, U, U, U]),
                ('metal_ir_bitcast', U, [P, U, U, U]), ('metal_ir_builtin', U, [P, U, U, U, C.POINTER(U)]),
                ('metal_ir_extract', U, [P, U, U, U, C.POINTER(U)]), ('metal_ir_builtin_named', U, [C.c_char_p]),
                ('metal_ir_scatter', None, [P, U, U, U, C.POINTER(U), U, U]),
                ('metal_ir_yield', None, [P, U, U, C.POINTER(U)]),
                ('metal_ir_lower', C.c_int, [P, P, C.c_char_p, C.c_size_t]),
                ('metal_ir_finalize', C.c_int, [P, C.c_char_p, C.c_size_t]), ('metal_ir_values', U, [P]),
                ('metal_ir_operations', U, [P]),
                ('metal_ir_machine', None, [P, C.POINTER(mesh_graph.Machine), C.POINTER(mesh_graph.Machine)]),
                ):
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
    compiler, transport, machines = _compiler(), _transport(), _machines()
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
        # docs/provider.md#fragment-coded-weights
        if compiled[k].target is not None:
            for g in ops:
                _placed([g], compiled[k].target, g.node)
        fixed = {_key(storage(k, v, j)): ir.initial[v] for v in ir.constant_values}
        builder = _Builder(compiler, owners, role, row, (k << 48) | (j << 32), fixed)
        if machines:
            compiler.metal_ir_machine(builder.p, C.byref(machines[GPU]), C.byref(machines[CPU]))
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
        builder.source_values = compiler.metal_ir_values(builder.p)
        if compiler.metal_ir_lower(builder.p, device, error, len(error)):
            raise OSError(errno.EINVAL, 'call %s of compiled function %d: its lowering: %s'
                          % (node.name, k, error.value.decode()))
        prepared = _owned(owned, lib.MetalRecorderPrepare(recorder, builder.p, error, len(error)))
        if not prepared:
            raise OSError(errno.EINVAL, 'call %s of compiled function %d: its preparation: %s'
                          % (node.name, k, error.value.decode()))
        rows.append({'node': node, 'k': k, 'j': j, 'result': result, 'builder': builder, 'prepared': prepared,
                     'marks': marks, 'receives': receives, 'sends': sends})
        return result
    _Program(gm, call).run(*[instance(k, 0)[1][i][0] if k is not None else constant
                             for i, (constant, k) in enumerate(zip(constants, holders))])
    return {'lib': lib, 'recorder': recorder, 'owners': owners, 'rated': rated, 'rows': rows, 'compiler': compiler,
            'device': device, 'inputs': [ir.inputs for ir in irs]}


# design/torch-capture-change-request.md, appendix (actor costs); metal_recording.h MetalRecorderActors: each
# function's call at this node part by part (host p its CPU tasks', work p its GPU dispatches' mesh_graph_actor_time on
# this node's machines; the cross-processor edges inside one execution), the host's writes of the block inputs it
# holds added to its first host part (a CPU actor writing their bytes); no machine record: one part of no time
def _actors(programs, irs):
    machines, lib, made = _machines(), programs['lib'], {}
    for row in programs['rows']:
        k = row['k']
        if row['builder'] is None or k in made or not machines:
            continue
        parts, count = mesh.U(), C.c_size_t()
        gpu, cpu = C.byref(machines[GPU]), C.byref(machines[CPU])
        lib.MetalRecorderActors.restype = C.c_int
        lib.MetalRecorderActors.argtypes = [C.c_void_p, C.POINTER(mesh_graph.Machine), C.POINTER(mesh_graph.Machine),
                                            C.POINTER(C.c_double), C.POINTER(C.c_double), mesh.U, C.POINTER(mesh.U),
                                            C.POINTER(mesh_graph.Dependence), C.c_size_t, C.POINTER(C.c_size_t)]
        status = lib.MetalRecorderActors(row['prepared'], gpu, cpu, None, None, 0, C.byref(parts), None, 0, C.byref(count))
        if status not in (0, errno.ERANGE, errno.ENODATA):
            mesh.check(status)
        host, work = (C.c_double * max(parts.value, 1))(), (C.c_double * max(parts.value, 1))()
        edges = (mesh_graph.Dependence * max(count.value, 1))()
        status = lib.MetalRecorderActors(row['prepared'], gpu, cpu, host, work, parts.value, C.byref(parts), edges,
                                        count.value, C.byref(count))
        if status not in (0, errno.ENODATA):
            mesh.check(status)
        written, complete = sum(irs[k].values[v].section.bytes for v, _ in irs[k].inputs.values()), status == 0
        if parts.value and written:
            spent, measured = mesh_graph.actor_time(machines[CPU], mesh_graph.Actor(0.0, 1, 1, written, 0, 0, None))
            complete &= measured
            host[0] += spent
        made[k] = host, work, edges, parts.value, count.value, complete
    return made

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
        status = compiler.metal_ir_loaded(p, error, len(error)) if 'prepared_record' in row else \
            compiler.metal_ir_finalize(p, error, len(error))
        if status:
            raise OSError(errno.EINVAL, 'call %s of compiled function %d: its emission: %s'
                          % (row['node'].name, k, error.value.decode()))
        if 'prepared_record' in row:
            record = row['prepared_record']
            row['prepared'] = _owned(owned, programs['lib'].MetalRecorderLoad(
                programs['recorder'], p, record, len(record), error, len(error)))
            if not row['prepared']:
                raise OSError(errno.EINVAL, 'call %s: its saved preparation: %s'
                              % (row['node'].name, error.value.decode()))
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
def _replays(programs, context, irs, planned, values, addresses, base, buffer, bound, owned, made, steps, every):
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
        if k in steps:
            step = values[k][steps[k]]
            PLACE_STEP(recorder, _SEL(b'place:step:offsets:copies:shares:'), prepared, buffer,
                       offsets(addresses[k][steps[k]]), step.section.count, bool(step.shares))
        STATE_BUFFER(recorder, _SEL(b'state:buffer:status:'), prepared, made.state, entry['status'])
        replay = Replay()
        status = lib.MetalRecorderReplay(recorder, prepared, C.byref(replay), error, len(error))
        if status:
            raise OSError(status, 'call %s of compiled function %d: the replay: %s'
                          % (entry['node'].name, k, error.value.decode()))
        if replay.commands:
            SET(queue, _SEL(b'addResidencySet:'), replay.residency)
        owned[id(replay)] = lambda replay=replay: (SET(queue, _SEL(b'removeResidencySet:'), replay.residency)
                                                   if replay.commands else None, lib.MetalReplayDestroy(C.byref(replay)))
        rows.append(replay)
    calls = (Call * len(rows))(*[Call(C.pointer(r) if r is not None else None, row + 1, None, 0)
                                 for row, r in enumerate(rows)])
    # metal_recording.h MetalRecorderProgram: this node's machines (GPU, CPU), the executor's estimate of a span the plan
    # gives no horizon; none recorded: the plan's horizon alone
    machines = _machines()
    given = (mesh_graph.Machine * 2)(machines[GPU], machines[CPU]) if machines else None
    arguments = (recorder, calls, len(calls), given, C.byref(planned))
    program = _owned(owned, lib.MetalProgramLoad(*arguments, programs['native_record'],
        len(programs['native_record']), error, len(error)) if 'native_record' in programs else
        lib.MetalRecorderProgram(*arguments, error, len(error)))
    if not program:
        raise OSError(errno.EINVAL, 'the program\'s calls: %s' % error.value.decode())
    # metal_recording.h MetalProgramHorizon: the program's lag (the plan's horizon, else its spans' buffer bound), the
    # executions in flight its allocator ring and signature window cover
    horizon = lib.MetalProgramHorizon(program)
    period = _period({k: b[3] for k, b in enumerate(bound)}, horizon)
    # design/synchronization-signature-change-request.md items 1 and 2: the compiled program's one signature over its
    # warm-up and one period of its residues, the plan's composed in by its producer (MetalProgramSignature), kept with
    # the compiled program as the graph module's forms and its record's bytes
    if 'signature' in programs:
        signed = programs['signature']
    else:
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
        written = True
        for suffix, data in (('.record', signed['record']), ('.mtx', signed['matrix_market'].encode()),
                             ('.dot', signed['dot'].encode()), ('.txt', signed['space_time'].encode())):
            written = mesh_disk.write(prefix + suffix, data, 'a MESH_SIGNATURE artifact', 'w') and written
        if written:
            print('prepared signature %s digest %s' % (prefix, signed['digest']), flush=True)
    gpu = OBJECT(queue, _SEL(b'device'))
    allocators = (C.c_void_p * horizon)(*[_owned(owned, OBJECT(gpu, _SEL(b'newCommandAllocator')))
                                          for _ in range(horizon)])
    return {'lib': lib, 'queue': queue, 'calls': calls, 'program': program, 'allocators': allocators, 'horizon': horizon,
            'ended': _owned(owned, OBJECT(gpu, _SEL(b'newSharedEvent'))), 'error': error, 'signature': signed}


# design/aot-compilation-change-request.md, appendix item 6: block execution p's copy of a call's result, the copy of
# its plan's execution p every + ordinal (a block input's: p)
def _period(copies, horizon):
    return math.lcm(horizon, *(len(c) for function in copies.values() for c in function))


# N. D. Jones, C. K. Gomard and P. Sestoft, 1993, Definition 4.1; design/aot-compilation-change-request.md, appendix
# item 6; torch 2.14.0 torch/_inductor/cudagraph_trees.py: the block's outputs specialized once to each block execution
# residue p, each call's results the placed tensors of their stand-ins at the copies of its execution p every + ordinal
# (a block input's at p)
class _TensorLocation:
    __slots__ = ('function', 'value', 'copy', 'shape', 'stride', 'offset', 'dtype')

    def __init__(self, function, value, copy, shape, stride, offset, dtype):
        self.function, self.value, self.copy = function, value, copy
        self.shape, self.stride, self.offset, self.dtype = tuple(shape), tuple(stride), offset, dtype


def _outputs(recipe, bound):
    def tensor(x):
        if not isinstance(x, _TensorLocation):
            return x
        copy = bound[x.function][3][x.value][x.copy]
        if copy.dtype != x.dtype:
            copy = copy.view(x.dtype)
        return copy.as_strided(x.shape, x.stride, copy.storage_offset() + x.offset)
    return pytree.tree_map(tensor, recipe)


def _specialize(programs, horizon, bound, gm, constants, holders, every):
    copies = {k: b[3] for k, b in enumerate(bound)}
    results = {row['node']: row['result'] for row in programs['rows']}
    owners, rated = programs['owners'], programs['rated']

    def leaf(x, p):
        if not isinstance(x, torch.Tensor):
            return x
        if _key(x) not in owners:
            return x
        k, v, j = owners[_key(x)][:3]
        at = copies[k][v]
        return _TensorLocation(k, v, (p if v in rated[k] else p * every[k] + j) % len(at),
                               x.shape, x.stride(), x.storage_offset(), x.dtype)
    def input(i, p):
        k = holders[i]
        if k is None:
            return constants[i]
        value, shape = programs['inputs'][k][i]
        return _TensorLocation(k, value, p % len(copies[k][value]), shape,
                               [math.prod(shape[d + 1:]) for d in range(len(shape))], 0, copies[k][value][0].dtype)
    return [_Program(gm, lambda node, args, kwargs, p=p: pytree.tree_map(lambda x: leaf(x, p), results[node])).run(
        *[input(i, p) for i in range(len(constants))])
        for p in range(_period(copies, horizon))]


# mesh-call.h mesh_transfers_start; MPI 4.1 §3.9, MPI_START; NCCL User Guide, CUDA Stream Semantics
def _execution_error(replay, status, message):
    failure = CheckFailure()
    if status == errno.EDOM and not replay['lib'].MetalProgramCheckFailure(replay['program'], C.byref(failure)):
        kind = torch.linalg.LinAlgError if failure.code in (1, 2, 3) else RuntimeError
        error = OSError(failure.info, message) if failure.code == 5 else kind(message)
        error.operation_identity, error.batch, error.info, error.code = \
            failure.identity, failure.batch, failure.info, failure.code
        return error
    return OSError(status, message)


def _replayed(context, replay, b):
    status = replay['lib'].MetalProgramRun(replay['program'], replay['allocators'], b, 1, replay['ended'],
                                           C.addressof(context), START, STARTED, replay['error'], len(replay['error']))
    if status:
        raise _execution_error(replay, status, 'execution %d of the program: %s' % (b, replay['error'].value.decode()))
    return replay['outputs'][b % len(replay['outputs'])]


# MTLEvent.h `signaledValue`
def _ended(replay):
    return VALUE(replay['ended'], _SEL(b'signaledValue'))


# design/aot-compilation-change-request.md A1 to A3 and A10, "The calls' graph"; P. Barham et al., MLSys 2022, section
# 3; J. Reed, Z. DeVito, H. He, A. Ussery and J. Ansel, MLSys 2022; torch v2.14.0 torch/fx/_symbolic_trace.py,
# Tracer.is_leaf_module; torch/nn/modules/module.py, _forward_unimplemented
class Function(torch.nn.Module):
    def __init__(self, module, placement, weights, device=None, loss=None, source_device=None):
        super().__init__()
        self.module, self.placement, self.weights, self.device = module, placement, weights, device
        self.loss, self.source_device = loss, source_device


# torch 2.14.0 torch/export/experimental/__init__.py _export_forward_backward;
# torch/export/graph_signature.py ExportBackwardSignature; torch/_functorch/aot_autograd.py aot_export_module
TrainingResult = collections.namedtuple('TrainingResult', 'outputs parameters inputs')


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
# `every[f]` the block's calls of function f (_Export)
def _analyse(gm, of, compiled, sources, every):
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
    # A10, "Its bound": "The traced block is finite, so each function has finitely many calls in it", the most of them
    # outstanding at once "the number of its calls in the block, or fewer where Bernstein's conditions order them": the
    # width (Dilworth and Fulkerson, mesh_graph.width) of a function's calls in the block under A10's pairs and each
    # chain's successive calls, that order's transitive closure by mesh_graph.closure. No clause extends this bound
    # across blocks; the block executions in flight and a host-written value's locations are item 2's Bff(e)
    # (design/torch-capture-change-request.md item 2 and its 2026-11-09 source), which mesh_plan derives from each
    # Function's t(v) (_actors), planned.horizon
    members = {}
    for j, h in enumerate(head):
        members.setdefault(h, []).append(j)
    before = {j: {a for a, b in pairs if b == j} for j in range(len(calls))}
    for chain in members.values():
        for a, b in zip(chain, chain[1:]):
            before[b].add(a)
    after = mesh_graph.closure(range(len(calls)), lambda j: before[j])
    outstanding = {}
    for f in every:
        mine = [j for j in range(len(calls)) if callee[calls[j]] == f]
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
Compiled = collections.namedtuple('Compiled', 'function signature target source_device gm examples result spec written read mutated '
                                              'passed immutable')


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
# design/torch-source-lowering-change-request.md item 2; design/aot-compilation-change-request.md A1;
# design/compilation-composition-change-request.md, "Withdrawn, 2026-11-09"
CONTROL = (torch.ops.higher_order.cond, torch.ops.higher_order.while_loop, torch.ops.higher_order.map_impl,
           torch.ops.higher_order.scan, torch.ops.higher_order.switch)
# design/torch-source-lowering-change-request.md items 5 and 6
ITERATED = (A.linalg_eig, A.linalg_eigvals, A._linalg_eigvals)


def _static(exported, target):
    for node in exported.graph_module.graph.nodes:
        if node.op == 'call_function' and node.target in CONTROL:
            raise ValueError('%s (%s): a program with dynamic control flow is not compiled '
                             '(design/torch-source-lowering-change-request.md item 2)' % (node.name, node.target))
        if node.op == 'call_function' and getattr(node.target, 'overloadpacket', None) in ITERATED:
            device = next((value.device.type for value in pytree.tree_leaves(node.meta.get('val'))
                           if isinstance(value, torch.Tensor)), 'cpu')
            if (target if target is not None else CPU if device == 'cpu' else GPU) != CPU:
                raise ValueError('%s (%s): an iteration depth that depends on data is not compiled '
                                 '(design/torch-source-lowering-change-request.md item 5)' % (node.name, node.target))
        for value in pytree.tree_leaves(node.meta.get('val')):
            if isinstance(value, torch.Tensor) and not all(isinstance(size, int) for size in value.shape):
                raise ValueError('%s: sizes %s; shapes are compiled, a dimension known only at run time is not '
                                 '(design/aot-compilation-change-request.md A1)' % (node.name, tuple(value.shape)))


def _metadata_asserted(gm):
    for node in gm.graph.nodes:
        if node.op == 'call_function' and node.target is A._assert_tensor_metadata.default:
            args, kwargs = map_arg((node.args, node.kwargs), lambda n: n.meta['val'])
            node.target(*args, **kwargs)
            node.meta['val'] = None


# torch 2.14.0 torch/export/decomp_utils.py, default_decompositions; torch/_decomp/__init__.py, decomposition_table;
# torch/export/exported_program.py, run_decompositions: export's own (functional) decompositions of the operators the
# structured lowering does not compose itself (SOURCES and VIEWS keep theirs), then PyTorch's decomposition of each
# operator the graph still holds without a lowering, until none remains that PyTorch decomposes
def _lowered(op):
    packet = getattr(op, 'overloadpacket', op)
    return packet in SOURCES or packet in VIEWS


# PyTorch 2.14 documentation, torch.conj and torch.conj_physical (and Tensor.neg's negative-bit view): a functional
# graph's lazy conjugate (negation) the value of the physical one
LAZY = {A._conj.default: A._conj_physical.default, A._neg_view.default: A.neg.default}


def _decomposed(exported):
    from torch._decomp import decomposition_table
    state = {value.untyped_storage()._cdata for value in exported.state_dict.values()
             if isinstance(value, torch.Tensor)}
    state |= {exported.constants[name].untyped_storage()._cdata
              for name in exported.graph_signature.non_persistent_buffers}
    for path, module in exported.graph_module.named_modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if node.op == 'get_attr':
                value = operator.attrgetter(node.target)(module)
                if isinstance(value, torch.Tensor) and value.untyped_storage()._cdata not in state:
                    exported.constants.setdefault('.'.join(p for p in (path, node.target) if p), value)
            if node.op == 'call_function' and node.target is A.linalg_lstsq.default:
                source = node.args[0].meta['val']
                node.kwargs = {**node.kwargs, 'driver': _lstsq_driver(source, node.kwargs.get('driver'))}
        module.recompile()
    table = {op: f for op, f in torch.export.default_decompositions().items()
             if not (isinstance(op, torch._ops.OpOverload) and _lowered(op))}
    known = {**decomposition_table, **LAZY}
    exported, done = exported.run_decompositions({**table, **LAZY}), set()
    while True:
        wanted = {n.target for module in exported.graph_module.modules() if isinstance(module, torch.fx.GraphModule)
                  for n in module.graph.nodes if n.op == 'call_function' and not _lowered(n.target)
                  and n.target in known} - done
        if not wanted:
            return exported
        done |= wanted
        exported = exported.run_decompositions({op: known[op] for op in done})

def _metadata(value):
    return (tuple(value.shape), tuple(value.stride()), value.storage_offset(), value.dtype, value.device,
            value.requires_grad) \
        if isinstance(value, torch.Tensor) else value


# torch 2.14.0 torch/export/experimental/__init__.py _export_forward_backward;
# torch/export/exported_program.py _decompose_and_get_gm_with_new_signature_constants, ModuleCallSignature;
# torch/nn/attention/__init__.py sdpa_kernel, SDPBackend.MATH
def _exported(function, args, kwargs, target):
    exported = torch.export.export(function.module, tuple(args), dict(kwargs), strict=False)
    _static(exported, target)
    forward = exported.call_spec.out_spec
    if function.loss is not None:
        from torch.export.experimental import _export_forward_backward
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.MATH):
            exported = _export_forward_backward(exported, joint_loss_index=function.loss)
    exported = _decomposed(exported)
    if function.loss is not None:
        backward = exported.graph_signature.backward_signature
        outputs = pytree.tree_unflatten([None] * forward.num_leaves, forward)
        result = TrainingResult(outputs, dict.fromkeys(backward.gradients_to_parameters.values()),
                                (None,) * len(backward.gradients_to_user_inputs))
        exported.module_call_graph[0].signature.out_spec = pytree.tree_structure(result)
    return exported


# design/aot-compilation-change-request.md A1, A10 and appendix item 6; torch 2.14.0 torch/export/__init__.py;
# torch/export/exported_program.py, run_decompositions and ExportedProgram.module; torch/export/_trace.py,
# _export_for_training; torch/_export/non_strict_utils.py, fakify; torch/_subclasses/fake_tensor.py;
# torch/_subclasses/fake_impls.py, constructors; torch/_subclasses/meta_utils.py, describe_tensor; torch/_guards.py,
# detect_fake_mode; R. Frostig, M. J.
# Johnson and C. Leary, SysML 2018; mesh-collective.h struct mesh_function
class _Export(torch.fx.Interpreter):
    def __init__(self, gm, mode):
        super().__init__(gm)
        self.mode, self.compiled, self.of, self.origin, self.calls = mode, [], {}, {}, []
        self.marks, self.sources, self.inputs = {}, [], 0
        self.converters = {}

    # each node's value with every flat leaf replaced by the _Produced naming it
    def marked(self, n, produced):
        leaves, spec = pytree.tree_flatten(self.env[n])
        self.marks[n] = pytree.tree_unflatten([produced(o) for o in range(len(leaves))], spec)

    def fake(self, value, device=None):
        if not isinstance(value, torch.Tensor):
            return value
        placed = torch.device(device or value.device)
        converter = self.converters.setdefault(placed, MetaConverter())
        def make(meta, device):
            with no_dispatch():
                return FakeTensor(self.mode, meta(), placed)
        return converter(value, callback=make)

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
        storages = {}
        aliases = tuple(storages.setdefault(v.untyped_storage()._cdata, len(storages))
                        if isinstance(v, torch.Tensor) else None for v in flat)
        device = function.device
        target = None if device is None else ANE if device == 'ane' else CPU if torch.device(device).type == 'cpu' else GPU
        source = function.source_device or ('cpu' if device == 'ane' else device)
        source = torch.device(source) if source is not None else None
        signature = (spec, tuple(_metadata(v) for v in flat), aliases, target, source)
        found = [k for k, c in enumerate(self.compiled) if c.function is function and c.signature == signature]
        if not found:
            if source is not None:
                args, kwargs = pytree.tree_map(lambda v: self.fake(v, source), (args, kwargs))
            exported = _exported(function, args, kwargs, target)
            gm = exported.module(check_guards=False)
            _metadata_asserted(gm)
            out = [a.meta['val'] if isinstance(a, Node) else a for a in gm.graph.output_node().args[0]]
            result = pytree.tree_unflatten([self.fake(v) for v in out], exported.call_spec.out_spec)
            given, inputs = exported.graph_signature, [node for node in gm.graph.nodes if node.op == 'placeholder']
            mutated = set(given.user_inputs_to_mutate.values())
            self.compiled.append(Compiled(function, signature, target, source, gm, pytree.tree_flatten((args, kwargs))[0], result,
                                          exported.call_spec.out_spec,
                                          _storages(function.module, [*given.buffers_to_mutate.values(),
                                                                      *given.parameters_to_mutate.values()]),
                                          _storages(function.module, [*given.inputs_to_buffers.values(),
                                                                      *given.inputs_to_parameters.values()]),
                                          frozenset(q for q, name in enumerate(given.user_inputs) if name in mutated),
                                          {o: inputs.index(a) for o, a in enumerate(gm.graph.output_node().args[0])
                                           if isinstance(a, Node) and a in inputs},
                                          _constant_origins(exported, function.module)))
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


def _bound(context, state, retained, deadline):
    calls, every, given = retained['calls'], retained['every'], retained['given']
    nodes, page, irs, programs = retained['nodes'], retained['page'], retained['irs'], retained['programs']
    steps, arrays, actors, analysed = retained['steps'], retained['arrays'], retained['actors'], retained['analysed']
    functions = (mesh.Function * len(irs))(*[
        mesh.Function(C.cast(d, C.POINTER(mesh.Declared)), C.cast(w, C.POINTER(mesh.Weight)),
                      C.cast(v, C.POINTER(mesh.Value)), C.cast(e, C.POINTER(mesh.Dependence)), len(d), len(v),
                      len(e), 0, analysed.outstanding[k], 0,
                      *((C.cast(actors[k][0], C.POINTER(C.c_double)), C.cast(actors[k][1], C.POINTER(C.c_double)),
                         C.cast(actors[k][2], C.POINTER(mesh_graph.Dependence)), actors[k][3], actors[k][4])
                        if k in actors else (None, None, None, 0, 0)))
        for k, (d, w, v, e) in enumerate(arrays)])
    # design/compilation-composition-change-request.md item 6, unit 6D: the plan's results for native
    # preparation (mesh-collective.h struct mesh_planned), each call's plan, lead, ordinal and every the
    # identities the specialization and the program's rows take, kept until the program is retired
    planned = mesh.Planned()
    arguments = (C.byref(context), _group(), _program(), functions, len(irs),
                 (mesh.U * len(calls))(*calls), len(calls), analysed.orders, analysed.causes,
                 analysed.norders, analysed.chains, deadline)
    if 'saved_plan' in retained:
        record = retained['saved_plan']
        mesh.check(mesh.LIB.mesh_plan_load(*arguments, record, len(record), C.byref(planned)))
    else:
        mesh.check(mesh.LIB.mesh_plan(*arguments, C.byref(planned)))
    retained['functions'], retained['planned'] = functions, planned
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
        # 6's stage E: each row bound to its placed storage and replayed into the one native program, its
        # outputs specialized over the program's horizon
        state['replay'] = _replays(programs, context, irs, planned, listed, addresses, hull.address, buffer,
                                   bound, state['owned'], made, steps, every)
        state['horizon'] = state['replay']['horizon']
        if 'output_recipe' not in retained:
            retained['output_recipe'] = _specialize(programs, state['horizon'], bound, *retained['capture'],
                                                     retained['holders'], every)
        state['replay']['outputs'] = _outputs(retained['output_recipe'], bound)
        retained['bound'], retained['addresses'] = bound, addresses
    finally:
        if buffer:
            PUT(buffer, SELECTOR[b'release'])
    # torch 2.14.0 torch/export/exported_program.py, ExportedProgram; GSPMD section 4.5; Megatron-LM
    # core_r0.12.0 megatron/core/tensor_parallel/layers.py:156–165; design/problems-of-14b9a757-change-
    # request.md, "Sources for the second stops", 8, and "Sources for the third stops", 8;
    # design/torch-source-lowering-change-request.md item 1
    if 'initial' in retained:
        for k, v, copies in retained['initial']:
            value = listed[k][v]
            for at, data in zip(addresses[k][v], copies):
                if isinstance(at, list):
                    for page_index, address in enumerate(at):
                        piece = data[page_index * page:(page_index + 1) * page]
                        C.memmove(address, piece, len(piece))
                else:
                    C.memmove(at, data, value.section.bytes)
        return
    launched = False
    for ir, (_, _, _, copies) in zip(irs, bound):
        for v, data in ir.initial.items():
            for target in copies[v]:
                target.copy_(data)
                launched = launched or target.device.type == 'mps'
    if launched:
        torch.mps.synchronize()
    retained['initial'] = _initial(retained)
    programs['state_data'] = {}
    for row in programs['rows']:
        if row['builder'] is None:
            continue
        for v in range(row['builder'].source_values):
            info = ValueInfo()
            programs['compiler'].metal_ir_value_info(row['builder'].p, v, C.byref(info))
            if info.role == STATE and info.storage == v and (info.data or info.buffer):
                key, address, length = _constant_storage(info)
                programs['state_data'][key] = C.string_at(address, length)


# design/aot-compilation-change-request.md A1 to A3, A8 to A10 and appendix item 7;
# design/disjoint-pages-change-request.md C3.1, D.3.2 and D.3.3; mesh-collective.h mesh_plan, struct mesh_function and
# struct mesh_value; mesh-dataflow.h mesh_nodes_read, mesh_retire and struct mesh_ctx
# Two quantities, two owners: A10's outstanding bound, the width of each function's calls in the one traced block
# (_analyse); the executions in flight, item 2's Bff(e) that mesh_plan derives from each Function's t(v) (planned.horizon,
# the native program's lag through MetalProgramHorizon, sizing its allocators, copies and input waits); neither given
def program(block, *example_inputs):
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

    # design/aot-compilation-change-request.md A1 ("serialized objects we can resume without needing to repeatedly
    # re-compile anything"): the compilation's results at this node (_built), what an artifact retains, and what plan,
    # placement and binding derive from them each time (_bound)
    def _built():
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
            _initial_values(ir, record.immutable - frozenset().union(*(c.written for c in compiled)))
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
        for k in dict.fromkeys(row['k'] for row in programs['rows'] if row['builder'] is not None
                                and programs['compiler'].metal_ir_step_value(row['builder'].p) != NONE
                                and programs['compiler'].metal_ir_step_state(row['builder'].p) == NONE):
            steps[k] = len(irs[k].values) + len(own[k])
            own[k].append(mesh.Value(section=mesh.Section(bytes=8), kind=mesh.EXECUTION, lag=1, run=1))
        arrays = [((mesh.Declared * len(ir.declared))(*ir.declared),
                   (mesh.Weight * len(ir.weights))(*ir.weights),
                   (mesh.Value * (len(ir.values) + len(mine)))(*ir.values, *mine),
                   (mesh.Dependence * len(ir.dependences))(*ir.dependences)) for ir, mine in zip(irs, own)]
        # design/torch-capture-change-request.md item 2 and its 2026-11-09 appendix: each function's t(v) at this
        # node part by part (_actors), mesh_plan's Bff of their edges the horizon and host copies
        # (mesh-graph.h mesh_graph_actor_time ENODATA: a time over the measured operands alone, a lower bound the
        # plan keeps; the functions whose estimate is such a bound reported as `incomplete`, never a zero cost)
        actors = _actors(programs, irs)
        state['incomplete'] = sorted(k for k, actor in actors.items() if not actor[5])
        analysed = _analyse(gm, of, compiled, sources, every)
        return {'nodes': nodes, 'page': page, 'irs': irs, 'holders': holders, 'programs': programs, 'steps': steps,
                'arrays': arrays, 'actors': actors, 'analysed': analysed, 'calls': calls, 'every': every,
                'given': given, 'constants': constants, 'capture': (gm, constants)}

    def make():
        deadline = time.clock_gettime_ns(time.CLOCK_MONOTONIC) + mesh.PLATFORM_NS
        if state['mapping'] is not None or state['owned']:
            _retire(context, state)
        try:
            state['built'] = _built()
            _bound(context, state, state['built'], deadline)
        except BaseException:
            _retire(context, state)
            raise
        state.update(stale=False)

    return _runner(context, state, make)


# design/aot-compilation-change-request.md A1; metal_ir.h metal_ir_save/load; metal_recording.h
# MetalRecorderSave/Load and MetalProgramSave/Load; mesh-collective.h mesh_plan_save/load
_FREE = C.CDLL(None).free
_FREE.restype, _FREE.argtypes = None, [C.c_void_p]
_NEW_BUFFER = _message(C.c_void_p, C.c_ulong, C.c_ulong)


def _saved_bytes(function, *arguments, diagnostic=True):
    data, size, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(4096)
    status = function(*arguments, C.byref(data), C.byref(size), *([error, len(error)] if diagnostic else []))
    try:
        if status:
            raise OSError(status, error.value.decode() if diagnostic else function.__name__)
        return C.string_at(data, size.value)
    finally:
        _FREE(data)


def _records(kind, data):
    return (kind * (len(data) // C.sizeof(kind))).from_buffer_copy(data)


def _declared_record(value):
    record = mesh.Declared.from_buffer_copy(bytes(value))
    pairs = [tuple(view.pairs[:2 * view.dims]) for view in (value.from_, value.to)]
    record.from_.pairs, record.to.pairs = None, None
    return bytes(record), pairs


def _declared_restore(saved):
    values = []
    for data, pairs in saved:
        value = mesh.Declared.from_buffer_copy(data)
        value.from_.pairs = _array(mesh.Q, pairs[0])
        value.to.pairs = _array(mesh.Q, pairs[1])
        values.append(value)
    return (mesh.Declared * len(values))(*values)


def _initial(retained):
    records = []
    for k, ((_, _, values, _), addresses) in enumerate(zip(retained['arrays'], retained['addresses'])):
        for v, (value, copies) in enumerate(zip(values, addresses)):
            if value.kind != mesh.INVARIANT and not value.lag:
                continue
            contents = []
            for at in copies:
                if isinstance(at, list):
                    data = b''.join(C.string_at(address, min(retained['page'], value.section.bytes - j * retained['page']))
                                    for j, address in enumerate(at))
                else:
                    data = C.string_at(at, value.section.bytes)
                contents.append(data)
            records.append((k, v, contents))
    return records


def _artifact_header():
    machines = _machines()
    return (1, sys.byteorder, C.sizeof(C.c_void_p), torch.__version__,
            tuple(C.sizeof(t) for t in (mesh.Declared, mesh.Weight, mesh.Value, mesh.Dependence, Range, Release,
                                       ValueInfo, mesh_graph.Dependence)),
            tuple((k, bytes(v)) for k, v in sorted(machines.items())))


def _constant_storage(info):
    return (('buffer', info.buffer), OBJECT(info.buffer, _SEL(b'contents')), VALUE(info.buffer, _SEL(b'length'))) \
        if info.buffer else (('data', info.data, info.data_bytes), info.data, info.data_bytes)


def _finish(replay):
    error = C.create_string_buffer(4096)
    status = replay['lib'].MetalProgramEnd(replay['program'], replay['ended'], error, len(error))
    if status:
        raise _execution_error(replay, status, error.value.decode())


def _save_program(context, state, path):
    retained, replay = state['built'], state['replay']
    programs, analysed, calls = retained['programs'], retained['analysed'], retained['calls']
    _finish(replay)
    plan = _saved_bytes(mesh.LIB.mesh_plan_save, C.byref(context), _program(), retained['functions'],
                       len(retained['functions']), _array(mesh.U, calls), len(calls), analysed.orders, analysed.causes,
                       analysed.norders, analysed.chains, C.byref(retained['planned']), diagnostic=False)
    arrays = [([_declared_record(v) for v in d], bytes(w), bytes(v), bytes(e))
              for d, w, v, e in retained['arrays']]
    actors = {k: (bytes(host), bytes(work), bytes(edges), parts, count, complete)
              for k, (host, work, edges, parts, count, complete) in retained['actors'].items()}
    rows, storage, unique = [], [], {}
    for row in programs['rows']:
        record = {key: row[key] for key in ('k', 'j')}
        record['node'] = row['node'].name
        if row['builder'] is not None:
            p, constants = row['builder'].p, []
            for v in range(row['builder'].source_values):
                info = ValueInfo()
                programs['compiler'].metal_ir_value_info(p, v, C.byref(info))
                if info.storage != v or (not info.data and not info.buffer):
                    continue
                key, address, size = _constant_storage(info)
                if key not in unique:
                    unique[key] = len(storage)
                    data = programs.get('state_data', {}).get(key)
                    storage.append((bool(info.buffer), data if data is not None else C.string_at(address, size)))
                constants.append((v, unique[key], info.at, info.data_bytes,
                                  info.data - address if info.data else None))
            received = []
            for v, f, value, declared in row['marks']['received']:
                declaration = Range.from_buffer_copy(bytes(declared))
                declaration.field, declaration.input = None, Remote()
                received.append((v, f, value, bytes(declaration)))
            sent = []
            for v, release in row['marks']['sent']:
                declaration = Release.from_buffer_copy(bytes(release))
                declaration.supplied = declaration.status = 0
                sent.append((v, bytes(declaration)))
            record.update(program=_saved_bytes(programs['compiler'].metal_ir_save, p),
                          prepared=_saved_bytes(programs['lib'].MetalRecorderSave, row['prepared']), constants=constants,
                          source_values=row['builder'].source_values,
                          planned=row['builder'].planned, own=row['own'], receives=row['receives'], sends=row['sends'],
                          marks={'received': received, 'sent': sent, 'sending': row['marks']['sending']})
        rows.append(record)
    payload = {key: retained[key] for key in ('page', 'holders', 'steps', 'calls', 'every', 'given', 'constants',
                                             'output_recipe', 'initial')}
    payload.update(header=_artifact_header(), plan=plan, arrays=arrays, actors=actors, rows=rows, storage=storage,
                   irs=[(ir.layouts, ir.inputs, ir.device) for ir in retained['irs']],
                   analysed=(analysed.outstanding, bytes(analysed.orders), bytes(analysed.causes), analysed.norders,
                             bytes(analysed.chains)),
                   native=_saved_bytes(programs['lib'].MetalProgramSave, replay['program']), signature=replay['signature'])
    return mesh_disk.write(path, b'MESH-AOT\0' + pickle.dumps(payload, protocol=5), 'a compiled mesh program', 'w')


def _restore_programs(saved, owned):
    lib, made = _recorder()
    compiler, error, device = _compiler(), C.create_string_buffer(4096), made()
    recorder = _owned(owned, lib.MetalRecorderOpen(device, error, len(error)))
    PUT(device, SELECTOR[b'release'])
    if not recorder:
        raise OSError(errno.ENODEV, error.value.decode())
    storage = []
    for fixed, data in saved['storage']:
        blob = C.create_string_buffer(data, max(len(data), 1))
        if fixed:
            buffer = _owned(owned, _NEW_BUFFER(device, _SEL(b'newBufferWithLength:options:'), max(len(data), 1), SHARED))
            if not buffer:
                raise OSError(errno.ENOMEM, 'saved constant storage')
            address = OBJECT(buffer, _SEL(b'contents'))
            C.memmove(address, data, len(data))
        else:
            owned[id(blob)] = lambda blob=blob: None
            address, buffer = C.addressof(blob), None
        storage.append((address, buffer))
    rows, state_data = [], {}
    for source in saved['rows']:
        row = {key: source[key] for key in ('k', 'j')}
        row.update(node=types.SimpleNamespace(name=source['node']), builder=None)
        if 'program' in source:
            data = source['program']
            p = compiler.metal_ir_load(data, len(data), device, error, len(error))
            if not p:
                raise OSError(errno.EINVAL, 'saved call %s: %s' % (source['node'], error.value.decode()))
            builder = types.SimpleNamespace(p=p, planned=source['planned'], source_values=source['source_values'])
            owned[id(builder)] = lambda p=p: compiler.metal_ir_program_free(p)
            for v, index, at, count, offset in source['constants']:
                info = ValueInfo()
                compiler.metal_ir_value_info(p, v, C.byref(info))
                if info.data:
                    continue
                address, buffer = storage[index]
                if offset is not None:
                    compiler.metal_ir_value_data(p, v, address + offset, count, buffer, at, None, RELEASE())
                else:
                    compiler.metal_ir_value_storage(p, v, buffer, at)
                if info.role == STATE:
                    key = ('buffer', buffer) if buffer else ('data', address + (offset or 0), count)
                    state_data[key] = saved['storage'][index][1]
            row.update(builder=builder, prepared_record=source['prepared'], own=source['own'],
                       receives=source['receives'], sends=source['sends'],
                       marks={'received': [(v, f, value, Range.from_buffer_copy(data))
                                           for v, f, value, data in source['marks']['received']],
                              'sent': [(v, Release.from_buffer_copy(data)) for v, data in source['marks']['sent']],
                              'sending': source['marks']['sending']})
        rows.append(row)
    return {'lib': lib, 'compiler': compiler, 'device': device, 'recorder': recorder, 'rows': rows,
            'rated': {k: {v for v, _ in inputs.values()} for k, (_, inputs, _) in enumerate(saved['irs'])},
            'native_record': saved['native'], 'signature': saved['signature'], 'state_data': state_data}


def load(path):
    with open(path, 'rb') as stream:
        if stream.read(9) != b'MESH-AOT\0':
            raise ValueError('not a mesh compiled artifact')
        saved = pickle.load(stream)
    if saved['header'] != _artifact_header():
        raise ValueError('the saved program has different Torch, record layout, or machine configuration; compile '
                         'for the current configuration')
    context, state = _context(), {'stale': True, 'mapping': None, 'devices': (), 'executions': 0, 'horizon': 1,
                                  'inputs': [None] * len(saved['given']), 'timeout': None, 'replay': None, 'owned': {}}

    def make():
        if state['mapping'] is not None or state['owned']:
            _retire(context, state)
        try:
            nodes = mesh.Nodes()
            mesh.check(mesh.LIB.mesh_nodes_read(C.byref(context), C.byref(nodes)))
            arrays = [(_declared_restore(d), _records(mesh.Weight, w), _records(mesh.Value, v),
                       _records(mesh.Dependence, e)) for d, w, v, e in saved['arrays']]
            actors = {k: (_records(C.c_double, host), _records(C.c_double, work),
                          _records(mesh_graph.Dependence, edges), parts, count, complete)
                      for k, (host, work, edges, parts, count, complete) in saved['actors'].items()}
            outstanding, orders, causes, count, chains = saved['analysed']
            irs = [types.SimpleNamespace(layouts=layout, inputs=inputs, device=device, declared=d, values=v, homes={})
                   for (layout, inputs, device), (d, _, v, _) in zip(saved['irs'], arrays)]
            retained = {key: saved[key] for key in ('page', 'holders', 'steps', 'calls', 'every', 'given', 'constants',
                                                   'output_recipe', 'initial')}
            retained.update(nodes=nodes, irs=irs, arrays=arrays, actors=actors, saved_plan=saved['plan'],
                            analysed=Calls(outstanding, _records(mesh.U, orders), _records(mesh.U, causes), count,
                                           _records(mesh.U, chains)), programs=_restore_programs(saved, state['owned']))
            state['built'] = retained
            _bound(context, state, retained, time.clock_gettime_ns(time.CLOCK_MONOTONIC) + mesh.PLATFORM_NS)
            state.update(stale=False)
        except BaseException:
            _retire(context, state)
            raise
    return _runner(context, state, make)


def _runner(context, state, make):
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
    def save(path):
        if state['stale']:
            timed(make)
        return timed(lambda: _save_program(context, state, path))
    run.save = save
    run.synchronize = lambda: timed(lambda: _finish(state['replay']))
    run.signature = lambda: state['replay']['signature'] if state['replay'] is not None else None
    weakref.finalize(run, _dropped, context, state)
    return run
