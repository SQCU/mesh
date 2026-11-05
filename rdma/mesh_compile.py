# The torch.compile backend of the ahead-of-time compilation (design/aot-compilation-change-request.md, A1 to A4, A8 and
# A9; design/compilation-coordination.md, "The torch backend (mesh_compile.py, T)" and A-T1 to A-T4).  A function's FX
# graph with its fixed shapes (A1) is partitioned by the placement operand (A2), its collectives are lowered to declared
# transfers and dependences (A4), all carried as one IR (A8) and weighed by the weighting operand (A3), then planned by
# mesh_plan and run by torch's own operators on tensors over the placed pages (A9).  It chooses no placement and no
# weight: both are the results of functions the program passes (the operator's ruling 2 of 2026-11-04).
import collections
import ctypes as C
import errno
import math
import operator
import os
import time

import torch
from torch.fx.node import Node, map_arg

import mesh

F = torch.nn.functional
# Y. Xu et al., GSPMD, arXiv 2105.04663, 2021, section 3.5: elementwise operators, every tensor operand of the output's
# shape, each dimension preserved from the inputs to the output and back
ELEMENTWISE = {operator.add, operator.sub, operator.mul, operator.truediv, operator.neg, torch.add, torch.sub,
               torch.mul, torch.div, torch.neg, torch.relu, torch.tanh, torch.sigmoid, torch.exp, F.relu, F.gelu,
               F.silu, 'add', 'sub', 'mul', 'div', 'neg', 'relu', 'tanh', 'sigmoid', 'exp'}
# GSPMD sections 3.2 and 3.5: Dot, a generalized matrix multiply (an Einsum), each with the out= form torch gives it
# (None: none), by which a partial is written over its range's placed pages (A9)
DOT = {operator.matmul: torch.matmul, torch.matmul: torch.matmul, 'matmul': torch.matmul, torch.mm: torch.mm,
       'mm': torch.mm, F.linear: None}
# design/aot-compilation-change-request.md A2, the placement operand placement(gm, example_inputs) returns.  mesh:
# GSPMD's device mesh, "a logical multi-dimensional tensor" of device ids (section 3.1), nested sequences holding the
# ids 0 .. D - 1 once each.  nodes: device d's map index nodes[d] (mesh-dataflow.h mesh_nodes_read), the mapping of the
# logical mesh onto the map's nodes (L. Zheng et al., Alpa, OSDI 2022, section 4.1: "For a set of physical devices,
# there can be multiple logical views").  splits: a placeholder's name to its dims_mapping, GSPMD's
# mesh_split(tensor, device_mesh, dims_mapping) (section 3.1: each tensor dimension's mesh dimension, -1 none)
Placement = collections.namedtuple('Placement', 'mesh nodes splits')
# an inserted collective (GSPMD sections 2.1, 4.2 and 4.5): its kind, the FX node whose local value it carries, the
# mesh dimensions its groups lie along, the tensor dimension it gathers or switches (None for an AllReduce) and its
# place in the FX graph's order
Collective = collections.namedtuple('Collective', 'kind node dims dim position')


# design/aot-compilation-change-request.md A8: one IR carried from partitioning to serialization, as MSCCLang carries
# its instruction DAG from lowering to scheduling (M. Cowan et al., ASPLOS 2023, sections 4 and 5).  It holds the
# placement operand (A2) and the mesh it names, each tensor's sharding (GSPMD section 3.1), this node's partitioned
# graph with its collectives, and the task graph as mesh_plan takes it (design/plan-to-bridge-change-request.md R3 and
# R4; design/compilation-coordination.md, T1): the declared transfers with their kinds, the dependences and this node's
# values; `weights` is the weighting operand once weights(ir) has given it (A3)
class IR:
    def __init__(self, gm, placement, nodes):
        self.gm, self.placement, self.weights = gm, placement, None
        self.order = list(gm.graph.nodes)
        self.position = {node: i for i, node in enumerate(self.order)}
        self.shape, self.coords, self.map = _mesh(placement, nodes.count)
        here = [d for d, index in enumerate(self.map) if index == nodes.self]
        if not here:
            raise ValueError('map index %d, this node, is no device of the placement operand (A2)' % nodes.self)
        self.device, self.index, self.identity = here[0], nodes.self, 0
        self.sharding, self.program, self.reduced, self.roles, self.events = {}, [], {}, {}, []
        self.declared, self.dependences, self.values, self.layouts = [], [], [], []

    # GSPMD section 3.1, a tiled dimension: each device's tile, the dimension's size over its mesh dimension's
    def local(self, node):
        return tuple(s // self.shape[m] if m >= 0 else s for s, m in zip(_value(node).shape, self.sharding[node]))

    # one range (mesh-collective.h struct mesh_declared): `node`'s local value from device `source` to each device of
    # `destinations`, a declared transfer each, of one identity.  MESH_JOIN: the host releases a range whole by one
    # supplied word (mesh_host_publish), so its consumer takes it whole (design/plan-to-bridge-change-request.md R4;
    # design/compilation-coordination.md, P1's reason).  This node's value of it (MESH_EXECUTION, wire) where it is
    # its source or a destination.  The declared transfers' indices
    def transfer(self, node, source, destinations):
        shape, dtype, value = self.local(node), _value(node).dtype, mesh.ABSENT
        size, ends = dtype.itemsize * math.prod(shape), [self.map[d] for d in [source] + destinations]
        if self.index in ends:
            value = len(self.values)
            self.values.append(mesh.Value(section=mesh.Section(bytes=size), kind=mesh.EXECUTION, wire=1))
            self.layouts.append((dtype, shape))
        first, self.identity = len(self.declared), self.identity + 1
        for end in ends[1:]:
            mine = value if self.index in (ends[0], end) else mesh.ABSENT
            self.declared.append(mesh.Declared(ends[0], end, self.identity, mesh.JOIN, mine, 0, size))
        return list(range(first, len(self.declared)))


# J. Ansel et al., PyTorch 2, ASPLOS 2024, sections 3.5 and 5.1: a node's fake tensor, "the metadata of a tensor without
# its actual data", its sizes propagated by meta functions.  A1: shapes are compiled, so a symbolic size is refused;
# A-T4: binding a tensor not on the CPU to the region's pages is open, so it is refused
def _value(node):
    value = node.meta.get('example_value')
    if not isinstance(value, torch.Tensor):
        raise NotImplementedError('%s (%s): no tensor value (PyTorch 2 section 3.5)' % (node.name, node.target))
    if not all(isinstance(size, int) for size in value.shape):
        raise ValueError('%s: symbolic sizes %s; A1 compiles shapes, so nothing at run time changes one'
                         % (node.name, tuple(value.shape)))
    if value.device.type != 'cpu':
        raise NotImplementedError('%s on %s: binding its tensors to the placed pages is open (A-T4: torch.frombuffer '
                                  'over mesh_section_address wraps the CPU\'s alone)' % (node.name, value.device))
    return value


# GSPMD section 3.1, the device mesh: its shape and each device id's coordinates; Alpa section 4.1, the mapping of its
# devices onto map indices, one device a node
def _mesh(placement, count):
    shape, item, coords = [], placement.mesh, {}
    while isinstance(item, (list, tuple)):
        if not item:
            raise ValueError('the placement operand\'s mesh holds an empty sequence (A2)')
        shape.append(len(item))
        item = item[0]

    def walk(item, at):
        if len(at) == len(shape):
            if not isinstance(item, int) or item in coords:
                raise ValueError('the placement operand\'s mesh at %s: %r is no new device id (A2)' % (at, item))
            coords[item] = at
        elif not isinstance(item, (list, tuple)) or len(item) != shape[len(at)]:
            raise ValueError('the placement operand\'s mesh at %s is not of shape %s (A2)' % (at, shape))
        else:
            for i, inner in enumerate(item):
                walk(inner, at + (i,))
    walk(placement.mesh, ())
    nodes = list(placement.nodes)
    if sorted(coords) != list(range(len(coords))) or len(nodes) != len(coords) or len(set(nodes)) != len(nodes) \
            or not all(isinstance(n, int) and 0 <= n < count for n in nodes):
        raise ValueError('the placement operand: devices %s onto map indices %s of the map\'s %d nodes; each device '
                         '0 .. D - 1 once, each onto its own index (A2)' % (sorted(coords), nodes, count))
    return tuple(shape), coords, nodes


# GSPMD section 3.5, "Merging compatible shardings": a and b are compatible where one sharding gives every device a's
# shard offset on each dimension a shards and b's on each b shards; as dims_mappings over one device mesh, where no
# dimension lies on two mesh dimensions and no mesh dimension on two dimensions.  Their merge, else None
def _merge(a, b):
    merged = tuple(x if y < 0 else y for x, y in zip(a, b))
    used = [m for m in merged if m >= 0]
    if any(0 <= x != y >= 0 for x, y in zip(a, b)) or len(used) != len(set(used)):
        return None
    return merged


# GSPMD section 3.5, "Preserved dimensions in operators": a dimension's sharding carried to the dimension of the same
# label, from inputs to outputs and back; a dimension of no carried label -1
def _carry(sharding, source, target):
    return tuple(sharding[source.index(label)] if label in source else -1 for label in target)


# an operator's propagation level, tensor operands, their labels and its output's (GSPMD section 3.5): an elementwise
# operator's every operand of its output's shape and labels; a Dot's Einsum labels (section 3.2: batch, contracting
# and non-contracting dimensions).  Any other operator, and these where GSPMD gives no rule here, refused
# (design/compilation-coordination.md, A-T2)
def _operator(node):
    operands, out = [a for a in node.args if isinstance(a, Node)], _value(node).shape
    if set(operands) != set(node.all_input_nodes) or node.kwargs.get('inplace'):
        raise NotImplementedError('%s (%s): a tensor operand outside its positional arguments, or in place: refused '
                                  '(A-T2)' % (node.name, node.target))
    if node.target in ELEMENTWISE:
        if any(_value(o).shape != out for o in operands):
            raise NotImplementedError(
                '%s (%s): operands of shapes %s broadcast to %s, an XLA Broadcast and an elementwise operator in one '
                'node; GSPMD section 3.5 gives Broadcast\'s backward propagation a higher priority than its forward, '
                'and no priority against Dot\'s: refused (A-T2)'
                % (node.name, node.target, [tuple(_value(o).shape) for o in operands], tuple(out)))
        labels = tuple(range(len(out)))
        return 0, operands, [labels] * len(operands), labels
    if node.target in DOT and len(operands) == 2:
        (a, b), r = (tuple(_value(o).shape) for o in operands), len(out)
        if node.target is F.linear and len(a) >= 2 and len(b) == 2:
            return 1, operands, [tuple(range(r)), (r, r - 1)], tuple(range(r - 1)) + (r,)
        if node.target is not F.linear and len(a) >= 2 and len(b) == 2:
            return 1, operands, [tuple(range(r)), (r - 1, r)], tuple(range(r - 1)) + (r,)
        if node.target is not F.linear and len(a) == len(b) >= 3 and a[:-2] == b[:-2]:
            return 1, operands, [tuple(range(r)), tuple(range(r - 2)) + (r - 1, r)], tuple(range(r - 1)) + (r,)
    raise NotImplementedError('%s (%s): GSPMD gives no rule here for it; refused (A-T2)' % (node.name, node.target))


# GSPMD section 3.5, sharding completion: it "runs the propagation pass on the entire graph in multiple iterations, and
# alternates between forward propagation (input to output) and backward propagation (output to input)"; it "preserves
# initial user annotations" and "changes the sharding on a tensor only when it finds a more fine-grained sharding
# (possibily by combining with existing compatible sharding)", so it reaches a fixed point.  Priorities: "Elementwise
# operators are assigned the highest priority to propagate through in both directions" and "Operators like Dot ... are
# assigned lower priority": level 0 propagates through elementwise operators alone to its fixed point, level 1 through
# Dot as well (design/compilation-coordination.md, A-T2).  An operator's operands are taken in argument order
def _complete(ir, operators, annotated):
    def refine(node, candidate):
        merged = _merge(ir.sharding[node], candidate)
        if node in annotated or merged is None or merged == ir.sharding[node]:
            return False
        ir.sharding[node] = merged
        return True
    for level in (0, 1):
        changed = True
        while changed:
            changed = False
            for forward, nodes in ((True, ir.order), (False, ir.order[::-1])):
                for node in nodes:
                    if node in operators and operators[node][0] <= level:
                        _, operands, labels, out = operators[node]
                        for operand, own in zip(operands, labels):
                            if forward:
                                changed |= refine(node, _carry(ir.sharding[operand], own, out))
                            else:
                                changed |= refine(operand, _carry(ir.sharding[node], out, own))


# GSPMD section 4.5, resharding: it "uses AllGather to replicate data along sharded dimensions, AllToAll to switch
# sharded dimensions, CollectivePermute to change device order, and DynamicSlice to shard replicated dimensions", and
# "could use multiple steps of resharding to reach the desired sharding": each mesh dimension's AllGather or AllToAll,
# then each one's DynamicSlice (a local slice, no communication).  One device mesh orders every tensor's devices alike,
# so no CollectivePermute arises.  The key of `node`'s local value in `want`
def _reshard(ir, node, have, want, position):
    for m in range(len(ir.shape)):
        i, j = (s.index(m) if m in s else -1 for s in (have, want))
        if i >= 0 and i != j:
            after = tuple(-1 if k == i else m if k == j else s for k, s in enumerate(have))
            collective = Collective('all-gather' if j < 0 else 'all-to-all', node, (m,), i if j < 0 else (i, j),
                                    position)
            ir.program.append(('collective', collective, (node, have), (node, after)))
            have = after
    for m in range(len(ir.shape)):
        if m in want and m not in have:
            after = tuple(m if k == want.index(m) else s for k, s in enumerate(have))
            ir.program.append(('slice', (node, have), (node, after), want.index(m), m))
            have = after
    return node, have


# GSPMD sections 3 and 4, per-operator partitioning: this node's program, in the FX graph's order.  A placeholder is
# the function's whole input, sliced to its sharding (section 4.5, DynamicSlice).  An operator's operands are
# resharded to the output's sharding on the dimensions they share with it; a Dot's contracting dimension keeps a mesh
# dimension both operands shard it along and the output does not use, its local results partial sums then combined by
# "AllReduce when the layer sums over a partitioned dimension" (section 2.1; section 4.2), and is replicated where the
# output uses its mesh dimension ("unsharded on demand", section 3.2; Figure 6, "AllGather non-matching Bx").  The
# output's values are resharded to replicated, so the callable returns the graph's whole values (PyTorch 2 section 3.1:
# the callable stands for the graph)
def _partition(ir, operators):
    inputs = 0
    for position, node in enumerate(ir.order):
        if node.op == 'placeholder':
            whole = (-1,) * len(ir.sharding[node])
            ir.program.append(('input', (node, whole), inputs))
            _reshard(ir, node, whole, ir.sharding[node], position)
            inputs += 1
        elif node.op == 'output':
            keys = {}
            for value in node.all_input_nodes:
                if value in ir.reduced:
                    raise NotImplementedError(
                        '%s: the graph returns an AllReduce\'s result, which lies over its plan\'s pages, written '
                        'again by a later execution (mesh-collective.h struct mesh_value: copy t mod count), while '
                        'what the callable returns is its caller\'s; the contract takes no decision between a copy '
                        'and the caller holding the plan\'s pages: stopped' % value.name)
                keys[value] = _reshard(ir, value, ir.sharding[value], (-1,) * len(ir.sharding[value]), position)
            ir.program.append(('output', node, keys))
        else:
            _, operands, labels, out = operators[node]
            used, partial = {m for m in ir.sharding[node] if m >= 0}, []
            wants = [list(_carry(ir.sharding[node], out, own)) for own in labels]
            for label in sorted(set(labels[0]).intersection(*labels[1:]) - set(out)):
                at = [own.index(label) for own in labels]
                kept = [-1 if s in used else s for s in (ir.sharding[o][i] for o, i in zip(operands, at))]
                if kept[0] != kept[1]:
                    raise NotImplementedError(
                        '%s: its contracting dimension on mesh dimensions %s in its two operands: GSPMD gives the '
                        'AllReduce where both shard it alike (section 2.1) and the AllGather where the output takes '
                        'its mesh dimension (section 3.2, Figure 6), and no rule here (A-T2)' % (node.name, kept))
                for want, i in zip(wants, at):
                    want[i] = kept[0]
                if kept[0] >= 0:
                    partial.append(kept[0])
            keys = [_reshard(ir, o, ir.sharding[o], tuple(w), position) for o, w in zip(operands, wants)]
            key = (node, 'partial') if partial else (node, ir.sharding[node])
            ir.program.append(('call', node, keys, key))
            if partial:
                ir.reduced[node] = Collective('all-reduce', node, tuple(partial), None, position)
                ir.program.append(('collective', ir.reduced[node], key, (node, ir.sharding[node])))


# design/aot-compilation-change-request.md A2: GSPMD's two passes by the placement operand, "sharding completion and
# per-operator partitioning" (section 3).  An unannotated tensor that propagation gives no sharding stays replicated
# (section 3.1, "All devices have the same full data"); a mesh dimension of one device tiles nothing, and a dimension
# its mesh dimension does not divide is refused (section 4.1 pads it and masks the padding, which the contract does not
# give)
def partition(gm, placement):
    if not isinstance(placement, Placement):
        raise TypeError('the placement function returned %r, not a Placement (A2)' % (placement,))
    nodes = mesh.Nodes()
    mesh.check(mesh.LIB.mesh_nodes_read(C.byref(_context()), C.byref(nodes)))
    ir, operators, annotated = IR(gm, placement, nodes), {}, set()
    for node in ir.order:
        if node.op in ('call_function', 'call_method'):
            operators[node] = _operator(node)
        elif node.op not in ('placeholder', 'output'):
            raise NotImplementedError('%s: a %s node; refused (A-T2)' % (node.name, node.op))
        if node.op != 'output':
            ir.sharding[node] = (-1,) * _value(node).dim()
    names = {node.name: node for node in ir.order if node.op == 'placeholder'}
    for name, mapping in placement.splits.items():
        node, mapping = names.get(name), tuple(mapping)
        if node is None or len(mapping) != len(ir.sharding[node]) \
                or not all(isinstance(m, int) and -1 <= m < len(ir.shape) for m in mapping) \
                or len([m for m in mapping if m >= 0]) != len({m for m in mapping if m >= 0}):
            raise ValueError('splits[%r] = %s: no placeholder\'s dims_mapping over a mesh of %d dimensions, each at '
                             'most once (GSPMD section 3.1)' % (name, mapping, len(ir.shape)))
        ir.sharding[node] = tuple(m if m >= 0 and ir.shape[m] > 1 else -1 for m in mapping)
        annotated.add(node)
    _complete(ir, operators, annotated)
    for node, sharding in ir.sharding.items():
        for size, m in zip(_value(node).shape, sharding):
            if m >= 0 and size % ir.shape[m]:
                raise NotImplementedError('%s: a dimension of %d over a mesh dimension of %d (GSPMD section 4.1 pads '
                                          'it, which the contract does not give)' % (node.name, size, ir.shape[m]))
    _partition(ir, operators)
    return ir


# GSPMD section 4.4, partition grouping: the devices that differ along `dims` alone, each group in the device mesh's
# order ("GSPMD preserves the order of devices in device_mesh", section 3.1)
def _groups(ir, dims):
    groups = {}
    for device in sorted(ir.coords, key=ir.coords.get):
        groups.setdefault(tuple(c for m, c in enumerate(ir.coords[device]) if m not in dims), []).append(device)
    return list(groups.values())


# design/compilation-coordination.md, A-T3: the AllReduce's tree is mesh_graph_pack's (A4) and its root the tree's.
# Stopped, as A-T3 directs where a lowering needs a decision its texts do not give
def _tree(ir, group):
    raise NotImplementedError(
        'AllReduce over map indices %s: its tree is mesh_graph_pack\'s and its root the tree\'s (A-T3); the contract '
        'gives neither mesh_graph_pack\'s rooted, eps, most and shallow nor which of the trees it returns (each rooted '
        'vertex\'s, heaviest first) the AllReduce takes: G. Wang et al., Blink, MLSys 2020, section 3.3 reduces "to a '
        'chosen root vertex", and section 4 splits the buffer "among all the spanning trees based on their weights"; '
        'stopped' % [ir.map[d] for d in group])


# design/compilation-coordination.md, A-T3: AllReduce as Blink states it (section 3.3: "we first run a reduce operation
# to a chosen root vertex ... and then do a broadcast operation from the root vertex using the same tree" in "the
# reverse direction").  Each tree edge a declared transfer of the child's partial toward the root; the root's combine
# of each one MSCCLang's receive-reduce-copy (M. Cowan et al., ASPLOS 2023, section 4.2: it "receives a chunk, reduces
# it with a source chunk, and locally copies it to the destination"), by the function's own add; then the sum down the
# same edges, one range from the root to its children.  A child's forward of the sum to children of its own is
# stopped: MSCCLang forwards a chunk by a receive and a send of one buffer (section 4.3, rcs), but a received range's
# value is one range's alone (mesh-collective.h struct mesh_declared `value`), and the contract takes no decision
# between a copy and another form.  Events (device, declared transfer, kind, first and last FX positions): a leaf's
# production of its partial and its consumption of the sum; the root's consumption of each partial, its production of
# the sum and its own read of it
def _all_reduce(ir, collective, group, root, parent):
    leaves, node, at = [d for d in group if d != root], collective.node, collective.position
    if any(parent[d] != root for d in leaves):
        raise NotImplementedError('AllReduce of %s: a tree deeper than one edge forwards the root\'s sum at a child; '
                                  'stopped (A-T3)' % node.name)
    if DOT[node.target] is None:
        raise NotImplementedError('AllReduce of %s (%s): torch gives no out= form to write its partial over its '
                                  'range\'s placed pages (A-T4)' % (node.name, node.target))
    readers = [ir.position[user] for user in node.users] or [at]
    up, down = {d: ir.transfer(node, d, [root]) for d in leaves}, ir.transfer(node, root, leaves)
    for k, d in enumerate(leaves):
        ir.events += [(d, up[d][0], 'P', at, at), (d, down[k], 'C', min(readers), max(readers)),
                      (root, up[d][0], 'C', at, at)]
    ir.events += [(root, down[0], 'P', at, at), (root, down[0], 'S', min(readers), max(readers))]
    if ir.device == root:
        ir.roles[collective] = ('root', [ir.declared[up[d][0]].value for d in leaves], ir.declared[down[0]].value)
    elif ir.device in leaves:
        ir.roles[collective] = ('leaf', ir.declared[up[ir.device][0]].value,
                                ir.declared[down[leaves.index(ir.device)]].value)


# the program's own data edges at each node (mesh-collective.h struct mesh_dependence; design/compilation-
# coordination.md, T1: task to task at one node), every rank's, as the program runs them.  An event is in force over
# the FX positions [first, last] where it writes or reads its range: two events of one node in one execution are
# ordered where the one's last position precedes the other's first (lag 0), and a node's every event of execution t
# precedes its every event of execution t + 1 (lag 1), since execution t's call returns before execution t + 1's
# begins.  An end's kinds bit: a production's, a source read's, a consumption's none
def _dependences(ir):
    bits = {'P': (mesh.FROM_PRODUCED, mesh.TO_PRODUCED), 'S': (mesh.FROM_SOURCE_READ, mesh.TO_SOURCE_READ),
            'C': (0, 0)}
    edges = []
    for a in ir.events:
        for b in ir.events:
            if a[0] == b[0]:
                kinds = bits[a[2]][0] | bits[b[2]][1]
                if a[4] < b[3]:
                    edges.append(mesh.Dependence(a[1], b[1], 0, kinds))
                edges.append(mesh.Dependence(a[1], b[1], 1, kinds))
    return edges


# design/aot-compilation-change-request.md A4: each inserted collective lowered to declared transfers and dependences,
# which mesh_plan's demands and trees route jointly (design/compilation-coordination.md, A-T3).  AllReduce is Blink's
# (_all_reduce); the contract gives no lowering of the others, so each is stopped, naming why
STOPPED = {'all-gather': 'Blink lowers AllGather as "AllReduce without using a reduction function" (G. Wang et al., '
                         'MLSys 2020, section 4), whose root\'s concatenation of the pieces is no operator of the '
                         'function\'s and whose forwards send on a received range, which a declared transfer cannot '
                         'name (mesh-collective.h struct mesh_declared `value`)',
           'all-to-all': 'A-T3 gives the lowering of AllReduce alone'}


def lower(ir):
    for item in ir.program:
        if item[0] == 'collective':
            collective = item[1]
            if collective.kind in STOPPED:
                raise NotImplementedError('%s of %s along mesh dimensions %s (GSPMD section 4.5): %s; stopped (A-T3)'
                                          % (collective.kind, collective.node.name, collective.dims,
                                             STOPPED[collective.kind]))
            for group in _groups(ir, collective.dims):
                _all_reduce(ir, collective, group, *_tree(ir, group))
    ir.dependences = _dependences(ir)


# design/compilation-coordination.md, P2: unit weights, every task and every edge 1, a weighting function the program
# passes (A3), never a default of the compilation
def unit(ir):
    return [mesh.Weight(1.0, 1.0, 1.0, 1.0) for _ in ir.declared]


# one context a program, attached to the node's one region (mesh.h MESH_NAME): its `epoch` counts its mesh_plan calls,
# so every rank's k-th plan is (group, k) (mesh-dataflow.h struct mesh_ctx; mesh-collective.h mesh_plan)
CONTEXT = []


def _context():
    if not CONTEXT:
        context = mesh.Context()
        mesh.check(mesh.LIB.mesh_attach(C.byref(context), mesh.NAME))
        CONTEXT.append(context)
    return CONTEXT[0]


# mesh_plan's `group`, the launcher's MESH_GROUP, a whole nonzero number, the same at every rank of a run
# (design/compilation-coordination.md, "The plan call" and "The program (e2b_replay.m, E) and the launcher")
def _group():
    text = os.environ.get('MESH_GROUP', '')
    try:
        group = int(text, 0)
    except ValueError:
        group = 0
    if not 0 < group < 2 ** 64:
        raise ValueError('MESH_GROUP %r: the run\'s group is a whole nonzero number, the same at every rank' % text)
    return group


# mesh-collective.h mesh_host_arrived, waited on for at most MESH_PLATFORM_NS (mesh.h): 1 arrived; FAILED a field ahead
# or the range cancelled; 0 not yet
def _arrived(context, section, t):
    deadline = time.monotonic_ns() + mesh.PLATFORM_NS
    while True:
        status = mesh.LIB.mesh_host_arrived(C.byref(context), section, t)
        if status == 1:
            return
        if status == mesh.FAILED:
            raise OSError(errno.ECANCELED, 'execution %d of a received range: a field ahead, or the range cancelled '
                                           '(mesh-collective.h mesh_host_arrived)' % t)
        if time.monotonic_ns() > deadline:
            raise OSError(errno.ETIMEDOUT, 'execution %d of a received range: not arrived within MESH_PLATFORM_NS' % t)


# one item of this node's program at execution t, on execution t's copies (mesh-collective.h struct mesh_value: copy t
# mod count).  A leaf's Dot writes its partial over its range's pages by torch's out= form and releases it
# (mesh_host_publish); its AllReduce waits for the sum.  The root's AllReduce waits for each partial in turn and adds it
# in, MSCCLang's receive-reduce-copy (section 4.2) by torch.add's out= form, then releases the sum
def _run(ir, context, tensors, item, env, inputs, t):
    if item[0] == 'input':
        env[item[1]] = inputs[item[2]]
    elif item[0] == 'slice':
        _, source, key, dim, m = item
        size = env[source].shape[dim] // ir.shape[m]
        env[key] = env[source].narrow(dim, ir.coords[ir.device][m] * size, size)
    elif item[0] == 'call':
        _, node, keys, key = item
        given = iter(keys)
        args = [env[next(given)] if isinstance(a, Node) else a for a in node.args]
        role = ir.roles.get(ir.reduced.get(node))
        if role and role[0] == 'leaf':
            section, copies = tensors[role[1]]
            env[key] = DOT[node.target](*args, **node.kwargs, out=copies[t % len(copies)])
            mesh.LIB.mesh_host_publish(C.byref(context), section, t)
        elif node.op == 'call_method':
            env[key] = getattr(args[0], node.target)(*args[1:], **node.kwargs)
        else:
            env[key] = node.target(*args, **node.kwargs)
    elif item[0] == 'collective':
        _, collective, source, key = item
        role = ir.roles[collective]
        section, copies = tensors[role[2]]
        if role[0] == 'leaf':
            _arrived(context, section, t)
            env[key] = copies[t % len(copies)]
        else:
            total, partial = copies[t % len(copies)], env[source]
            for value in role[1]:
                received, arrived = tensors[value]
                _arrived(context, received, t)
                partial = torch.add(partial, arrived[t % len(arrived)], out=total)
            mesh.LIB.mesh_host_publish(C.byref(context), section, t)
            env[key] = total
    else:
        env['output'] = map_arg(item[1].args[0], lambda n: env[item[2][n]])


# design/aot-compilation-change-request.md A9; design/compilation-coordination.md, A-T4.  mesh_plan writes the lists
# from the IR (each node's page instructions, the page table's bindings and the plan's state section); the GPU's
# program is the function's own operators, this node's partitioned graph run by torch on tensors over the placed pages,
# each copy of each value wrapped by torch.frombuffer over mesh_section_address (every address a constant of the plan,
# design/disjoint-pages-change-request.md, C3.3).  The callable submits each function execution to this node's bridge
# by mesh_transfers_start (A7; Q1) before it runs it
def serialize(ir):
    context, count = _context(), len(ir.declared)
    if len(ir.weights) != count or not all(isinstance(w, mesh.Weight) for w in ir.weights):
        raise ValueError('the weighting function gave %d entries for %d declared transfers; one mesh.Weight each (A3)'
                         % (len(ir.weights), count))
    values = (mesh.Value * len(ir.values))(*ir.values)
    mesh.check(mesh.LIB.mesh_plan(C.byref(context), _group(), (mesh.Declared * count)(*ir.declared),
                                  (mesh.Weight * count)(*ir.weights), count, values, len(values),
                                  (mesh.Dependence * len(ir.dependences))(*ir.dependences), len(ir.dependences)))
    state, executions, tensors = context.state, [0], []
    for value, (dtype, shape) in zip(values, ir.layouts):
        copies = []
        for c in range(value.section.count):
            address = mesh.LIB.mesh_section_address(C.byref(context), value.section, c)
            if not address:
                raise OSError(errno.EFAULT, 'copy %d of a placed value has no page (mesh-call.h mesh_section_address)'
                              % c)
            buffer = (C.c_char * value.section.bytes).from_address(address)
            copies.append(torch.frombuffer(buffer, dtype=dtype).view(shape))
        tensors.append((value.section, copies))

    def run(*inputs):
        if context.state != state:
            raise RuntimeError('this graph\'s plan no longer stands: a later mesh_plan of this program closed it '
                               '(mesh-collective.h mesh_plan)')
        t = executions[0]
        mesh.check(mesh.LIB.mesh_transfers_start(C.byref(context), t + 1))
        env = {}
        for item in ir.program:
            _run(ir, context, tensors, item, env, inputs, t)
        executions[0] = t + 1
        return env['output']
    return run


# design/aot-compilation-change-request.md A1 (PyTorch 2 section 3.1: "a custom compile function which takes a
# torch.fx.Graph and a list of example inputs and returns a Python callable"), the placement-choosing function and the
# weighting function its parameters (A2, A3; design/compilation-coordination.md, "The torch backend"), neither with a
# default: torch.compile(f, backend=backend(placement, weights))
def backend(placement, weights):
    def compiler(gm, example_inputs):
        ir = partition(gm, placement(gm, example_inputs))
        lower(ir)
        ir.weights = list(weights(ir))
        return serialize(ir)
    return compiler
