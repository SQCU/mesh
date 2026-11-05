# The torch.compile backend of the ahead-of-time compilation (design/aot-compilation-change-request.md, A1 to A4, A8 and
# A9; design/compilation-coordination.md, "The torch backend (mesh_compile.py, T)" and A-T1 to A-T4).  A function's FX
# graph with its fixed shapes (A1) is partitioned by the placement operand (A2), its collectives are lowered to declared
# transfers and dependences (A4), all carried as one IR (A8) and weighed by the weighting operand (A3), then planned by
# mesh_plan and run by torch's own operators on tensors over the placed pages (A9).  It chooses no placement and no
# weight: both are the results of functions the program passes (the operator's ruling 2 of 2026-11-04).  The
# coordinator's rulings on its first form's stops: A-T3' (the collectives as declared transfers mesh_plan routes), A-T5
# (a returned result over the plan's pages cloned) and A-T6 (one standing graph a program).
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
# mesh dimensions its groups lie along, the tensor dimension it gathers or the two it switches (None for an AllReduce)
# and its item's place in this node's program
Collective = collections.namedtuple('Collective', 'kind node dims dim position')


# design/aot-compilation-change-request.md A8: one IR carried from partitioning to serialization, as MSCCLang carries
# its instruction DAG from lowering to scheduling (M. Cowan et al., ASPLOS 2023, sections 4 and 5).  It holds the
# placement operand (A2) and the mesh it names, each tensor's sharding (GSPMD section 3.1), this node's partitioned
# graph with its collectives, and the task graph as mesh_plan takes it (design/plan-to-bridge-change-request.md R3 and
# R4; design/compilation-coordination.md, T1): the declared transfers with their kinds, the dependences and this node's
# values; `weights` is the weighting operand once weights(ir) has given it (A3).  `placed`: the keys of the local values
# that lie over the plan's pages (an AllReduce's result and its slices); `made`: the keys the reshard steps have made
class IR:
    def __init__(self, gm, placement, nodes):
        self.gm, self.placement, self.weights = gm, placement, None
        self.order = list(gm.graph.nodes)
        self.shape, self.coords, self.map = _mesh(placement, nodes.count)
        here = [d for d, index in enumerate(self.map) if index == nodes.self]
        if not here:
            raise ValueError('map index %d, this node, is no device of the placement operand (A2)' % nodes.self)
        self.device, self.index, self.identity = here[0], nodes.self, 0
        self.sharding, self.program, self.reduced, self.roles, self.events = {}, [], {}, {}, []
        self.declared, self.dependences, self.values, self.layouts = [], [], [], []
        self.placed, self.made = set(), set()

    # GSPMD section 3.1, a tiled dimension: each device's tile, the dimension's size over its mesh dimension's
    def local(self, node, sharding):
        return tuple(s // self.shape[m] if m >= 0 else s for s, m in zip(_value(node).shape, sharding))

    # one range (mesh-collective.h struct mesh_declared): a local value of `node` of `shape` from device `source` to
    # each device of `destinations`, a declared transfer each, of one identity.  MESH_JOIN: the host releases a range
    # whole by one supplied word (mesh_host_publish), so its consumer takes it whole (design/plan-to-bridge-change-
    # request.md R4; design/compilation-coordination.md, P1 and A-T3').  This node's value of it (MESH_EXECUTION, wire)
    # where it is its source or a destination.  Each destination's declared transfer's index
    def transfer(self, node, shape, source, destinations):
        dtype, value, transfers = _value(node).dtype, mesh.ABSENT, {}
        size, ends = dtype.itemsize * math.prod(shape), [self.map[d] for d in [source] + destinations]
        if self.index in ends:
            value = len(self.values)
            self.values.append(mesh.Value(section=mesh.Section(bytes=size), kind=mesh.EXECUTION, wire=1))
            self.layouts.append((dtype, shape))
        self.identity += 1
        for d in destinations:
            transfers[d] = len(self.declared)
            mine = value if self.index in (ends[0], self.map[d]) else mesh.ABSENT
            self.declared.append(mesh.Declared(ends[0], self.map[d], self.identity, mesh.JOIN, mine, 0, size))
        return transfers


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
# Dot as well (design/compilation-coordination.md, A-T2).  An operator's operands are taken in argument order.  It is
# the compilation's own pass (A2), here by the coordinator's ruling on C9's remark of d9efd96b
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
# "could use multiple steps of resharding to reach the desired sharding": each mesh dimension's AllGather, then each
# one's AllToAll into a dimension no mesh dimension holds (one whose dimension is held is refused, naming it), then
# each one's DynamicSlice (a local slice, no communication; a slice of a value over the plan's pages lies over them
# too).  One device mesh orders every tensor's devices alike, so no CollectivePermute arises.  Each step an item of this
# node's program: (kind, the keys it reads, the key it makes, its operand), none repeated whose key an earlier one made
# (that value stands through the execution).  The key of `node`'s local value in `want`
def _reshard(ir, node, have, want):
    def step(item):
        if item[2] not in ir.made:
            ir.made.add(item[2])
            ir.program.append(item)
    for m in range(len(ir.shape)):
        if m in have and m not in want:
            after = tuple(-1 if s == m else s for s in have)
            step(('collective', ((node, have),), (node, after),
                  Collective('all-gather', node, (m,), have.index(m), len(ir.program))))
            have = after
    for m in range(len(ir.shape)):
        if m in have and have.index(m) != want.index(m):
            i, j = have.index(m), want.index(m)
            if have[j] >= 0:
                raise NotImplementedError('%s: an AllToAll along mesh dimension %d into dimension %d, which mesh '
                                          'dimension %d holds: refused (GSPMD section 4.5)'
                                          % (node.name, m, j, have[j]))
            after = tuple(-1 if k == i else m if k == j else s for k, s in enumerate(have))
            step(('collective', ((node, have),), (node, after),
                  Collective('all-to-all', node, (m,), (i, j), len(ir.program))))
            have = after
    for m in range(len(ir.shape)):
        if m in want and m not in have:
            after = tuple(m if k == want.index(m) else s for k, s in enumerate(have))
            step(('slice', ((node, have),), (node, after), (want.index(m), m)))
            if (node, have) in ir.placed:
                ir.placed.add((node, after))
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
    for node in ir.order:
        if node.op == 'placeholder':
            whole = (-1,) * len(ir.sharding[node])
            ir.program.append(('input', (), (node, whole), inputs))
            _reshard(ir, node, whole, ir.sharding[node])
            inputs += 1
        elif node.op == 'output':
            keys = {value: _reshard(ir, value, ir.sharding[value], (-1,) * len(ir.sharding[value]))
                    for value in node.all_input_nodes}
            ir.program.append(('output', tuple(keys.values()), None, (node, keys)))
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
            keys = [_reshard(ir, o, ir.sharding[o], tuple(w)) for o, w in zip(operands, wants)]
            key = (node, 'partial') if partial else (node, ir.sharding[node])
            ir.program.append(('call', tuple(keys), key, node))
            if partial:
                ir.reduced[node] = Collective('all-reduce', node, tuple(partial), None, len(ir.program))
                ir.program.append(('collective', (key,), (node, ir.sharding[node]), ir.reduced[node]))
                ir.placed.add((node, ir.sharding[node]))


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


# Positions are half steps of this node's program: item g writes, reads and releases at 2g, and waits for and reads
# what arrives at 2g + 1.  `node`'s readers over the plan's pages: the items reading a key of it that lies over them,
# else `default`.  The span covers every read, so an edge into the event precedes its first and one out of it follows
# its last
def _readers(ir, node, default):
    reads = [2 * g for g, item in enumerate(ir.program) if any(k in ir.placed and k[0] is node for k in item[1])]
    return (min(reads), max(reads)) if reads else (default, default)


# design/compilation-coordination.md, A-T3': AllReduce as Blink states it (G. Wang et al., MLSys 2020, section 3.3: "we
# first run a reduce operation to a chosen root vertex ... and then do a broadcast operation from the root vertex"), the
# root the group's first device in the device mesh's order (GSPMD section 3.1: it "preserves the order of devices in
# device_mesh", so the placement operand's chooser decides it).  The reduce: each other member's partial one declared
# transfer to the root, written over its range's pages by the Dot's out= form; the root's combine of each landed
# partial with its own MSCCLang's receive-reduce-copy (M. Cowan et al., ASPLOS 2023, section 4.2: it "receives a chunk,
# reduces it with a source chunk, and locally copies it to the destination") by the function's own add.  The broadcast:
# the root's sum one range to every other member.  mesh_plan's demands and trees route them, relays forwarding as the
# transport does (A4: "A relay's forward is its fused receive-copy-send").  Events (device, declared transfer, kind,
# first, last): a member's production of its partial (the Dot's item) and consumption of the sum (its readers); the
# root's consumption of each partial and production of the sum (its item, after arrival) and its own read of the sum
def _all_reduce(ir, c, group, have):
    node, root, members = c.node, group[0], group[1:]
    if DOT[node.target] is None:
        raise NotImplementedError('AllReduce of %s (%s): torch gives no out= form to write its partial over its '
                                  'range\'s placed pages (A-T4)' % (node.name, node.target))
    shape, dot, after = ir.local(node, ir.sharding[node]), 2 * c.position - 2, 2 * c.position + 1
    readers = _readers(ir, node, after)
    up, down = [ir.transfer(node, shape, d, [root])[root] for d in members], ir.transfer(node, shape, root, members)
    for d, partial in zip(members, up):
        ir.events += [(d, partial, 'P', dot, dot), (d, down[d], 'C') + readers, (root, partial, 'C', after, after)]
    first = min(down.values())
    ir.events += [(root, first, 'P', after, after), (root, first, 'S') + readers]
    if ir.device == root:
        ir.roles[c] = (0, {None: ir.declared[first].value}, {p + 1: ir.declared[u].value for p, u in enumerate(up)})
    elif ir.device in members:
        me = group.index(ir.device)
        ir.roles[c] = (me, {0: ir.declared[up[me - 1]].value}, {0: ir.declared[down[ir.device]].value})


# design/compilation-coordination.md, A-T3': AllGather (GSPMD section 4.2: it "concatenates tensors from all
# participants following a specified order"): each member's shard one range to every other member, written over its
# range's pages by torch's copy_ (a range lies on its placed pages, design/disjoint-pages-change-request.md C3.2; the
# shard is torch's own, A-T4); each receiver's result torch.cat of the shards in the group's order.  Events: each
# member's production of its range (its item) and each other member's consumption of it (its item, after arrival)
def _all_gather(ir, c, group, have):
    shape, sent, received = ir.local(c.node, have), 2 * c.position, 2 * c.position + 1
    ranges = [ir.transfer(c.node, shape, d, [e for e in group if e != d]) for d in group]
    for d, transfers in zip(group, ranges):
        ir.events.append((d, min(transfers.values()), 'P', sent, sent))
        ir.events += [(e, i, 'C', received, received) for e, i in transfers.items()]
    if ir.device in group:
        me = group.index(ir.device)
        ir.roles[c] = (me, {None: ir.declared[min(ranges[me].values())].value},
                       {p: ir.declared[t[ir.device]].value for p, t in enumerate(ranges) if p != me})


# design/compilation-coordination.md, A-T3': AllToAll (GSPMD section 4.2: it "logically splits the input of each
# participant along one dimension, then sends each piece to a different participant.  On receiving data pieces from
# others, each participant concatenates the pieces to produce its result"): piece q of member p, along the dimension
# the mesh dimension moves to, one declared transfer p -> q, written by torch's copy_ as an AllGather's shard is; each
# receiver's result torch.cat of the pieces in sender order along the dimension it leaves.  Events as an AllGather's
def _all_to_all(ir, c, group, have):
    shape, sent, received = list(ir.local(c.node, have)), 2 * c.position, 2 * c.position + 1
    shape[c.dim[1]] //= len(group)
    pieces = {(p, q): ir.transfer(c.node, tuple(shape), d, [e])[e]
              for p, d in enumerate(group) for q, e in enumerate(group) if p != q}
    for (p, q), i in pieces.items():
        ir.events += [(group[p], i, 'P', sent, sent), (group[q], i, 'C', received, received)]
    if ir.device in group:
        me = group.index(ir.device)
        ir.roles[c] = (me, {q: ir.declared[i].value for (p, q), i in pieces.items() if p == me},
                       {p: ir.declared[i].value for (p, q), i in pieces.items() if q == me})


# the program's own data edges at each node (mesh-collective.h struct mesh_dependence: event `from` before event `to`,
# `to` in execution t + lag where `from` is in t; design/disjoint-pages-change-request.md, C3.5: a stage's reads and
# writes "in the order its command buffer runs them"), every rank's, by the program's order on that rank, whose
# executions run in order (execution t's call returns before execution t + 1's begins).  An event is in force over
# its half steps [first, last]: two events of one node in one execution are ordered where the one's last precedes the
# other's first (lag 0), and each of its events of execution t precedes each of execution t + 1 (lag 1).  An end's
# kinds bit: a production's, a source read's, a consumption's none
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


# design/aot-compilation-change-request.md A4 and design/compilation-coordination.md, A-T3': each inserted collective,
# over each of its device groups, lowered to declared transfers and dependences, which mesh_plan's demands and trees
# route jointly ("rdma/mesh-collective.c's demands and trees, as now, fed by A2's collectives"); any other collective
# (CollectivePermute, ReduceScatter and the rest) refused, naming it
LOWERINGS = {'all-reduce': _all_reduce, 'all-gather': _all_gather, 'all-to-all': _all_to_all}


def lower(ir):
    for item in ir.program:
        if item[0] == 'collective':
            collective = item[3]
            if collective.kind not in LOWERINGS:
                raise NotImplementedError('%s of %s: no lowering (A-T3\')' % (collective.kind, collective.node.name))
            for group in _groups(ir, collective.dims):
                LOWERINGS[collective.kind](ir, collective, group, item[1][0][1])
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


# one item of this node's program, (kind, the keys it reads, the key it makes, its operand), at execution t on
# execution t's copies (mesh-collective.h struct mesh_value: copy t mod count).  A member's Dot writes its partial over
# its range's pages by torch's out= form and releases it (mesh_host_publish); the root's AllReduce waits for each
# partial in turn and adds it in, MSCCLang's receive-reduce-copy (section 4.2) by torch.add's out= form, then releases
# the sum, for which every other member waits (A-T3').  An AllGather or AllToAll copies each of this node's shards or
# pieces over its range's pages (torch's copy_) and releases it, then waits for each other member's and concatenates
# them in the group's order (torch.cat; A-T3').  A returned value over the plan's pages is torch.clone'd: execution
# t + u writes those pages again (C3.3), and a torch.compile callable's outputs outlive its call
# (design/compilation-coordination.md, A-T5)
def _run(ir, context, tensors, item, env, inputs, t):
    kind, reads, key, operand = item

    def copy(value):
        return tensors[value][1][t % len(tensors[value][1])]

    def publish(value):
        mesh.LIB.mesh_host_publish(C.byref(context), tensors[value][0], t)

    def arrived(value):
        _arrived(context, tensors[value][0], t)
        return copy(value)
    if kind == 'input':
        env[key] = inputs[operand]
    elif kind == 'slice':
        (source,), (dim, m) = reads, operand
        size = env[source].shape[dim] // ir.shape[m]
        env[key] = env[source].narrow(dim, ir.coords[ir.device][m] * size, size)
    elif kind == 'call':
        given = iter(reads)
        args = [env[next(given)] if isinstance(a, Node) else a for a in operand.args]
        role = ir.roles.get(ir.reduced.get(operand))
        if role and role[0]:
            env[key] = DOT[operand.target](*args, **operand.kwargs, out=copy(role[1][0]))
            publish(role[1][0])
        elif operand.op == 'call_method':
            env[key] = getattr(args[0], operand.target)(*args[1:], **operand.kwargs)
        else:
            env[key] = operand.target(*args, **operand.kwargs)
    elif kind == 'collective':
        (source,), (me, sent, received) = reads, ir.roles[operand]
        if operand.kind == 'all-reduce' and me:
            env[key] = arrived(received[0])
        elif operand.kind == 'all-reduce':
            total, partial = copy(sent[None]), env[source]
            for p in sorted(received):
                partial = torch.add(partial, arrived(received[p]), out=total)
            publish(sent[None])
            env[key] = total
        else:
            gather, members = operand.kind == 'all-gather', len(received) + 1
            dim = operand.dim if gather else operand.dim[0]
            size = 0 if gather else env[source].shape[operand.dim[1]] // members

            def part(q):
                return env[source] if gather else env[source].narrow(operand.dim[1], q * size, size)
            for q, value in sent.items():
                copy(value).copy_(part(q))
                publish(value)
            env[key] = torch.cat([part(me) if p == me else arrived(received[p]) for p in range(members)], dim)
    else:
        node, keys = operand
        env['output'] = map_arg(node.args[0], lambda n: torch.clone(env[keys[n]]) if keys[n] in ir.placed
                                else env[keys[n]])


# design/aot-compilation-change-request.md A9; design/compilation-coordination.md, A-T4.  mesh_plan writes the lists
# from the IR (each node's page instructions, the page table's bindings and the plan's state section); the GPU's
# program is the function's own operators, this node's partitioned graph run by torch on tensors over the placed pages,
# each copy of each value wrapped by torch.frombuffer over mesh_section_address (every address a constant of the plan,
# design/disjoint-pages-change-request.md, C3.3).  The callable submits each function execution to this node's bridge
# by mesh_transfers_start (A7; Q1) before it runs it.  One compiled graph stands a program at a time: the plan's name
# (group, epoch) is per context, so a second graph while one stands is refused (design/compilation-coordination.md,
# A-T6, open for the operator)
def serialize(ir):
    context, count = _context(), len(ir.declared)
    if context.state:
        raise RuntimeError('several functions of one program each standing as its own plan: the plan\'s name is per '
                           'context, and this program\'s graph stands (design/compilation-coordination.md, A-T6)')
    if len(ir.weights) != count or not all(isinstance(w, mesh.Weight) for w in ir.weights):
        raise ValueError('the weighting function gave %d entries for %d declared transfers; one mesh.Weight each (A3)'
                         % (len(ir.weights), count))
    values = (mesh.Value * len(ir.values))(*ir.values)
    mesh.check(mesh.LIB.mesh_plan(C.byref(context), _group(), (mesh.Declared * count)(*ir.declared),
                                  (mesh.Weight * count)(*ir.weights), count, values, len(values),
                                  (mesh.Dependence * len(ir.dependences))(*ir.dependences), len(ir.dependences)))
    executions, tensors = [0], []
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
