# The torch.compile backend of the ahead-of-time compilation (design/aot-compilation-change-request.md, A1 to A4, A8 and
# A9; design/compilation-coordination.md, "The torch backend (mesh_compile.py, T)" and A-T1 to A-T4).  A function's FX
# graph with its fixed shapes (A1) is partitioned by the placement operand (A2), its collectives are lowered to declared
# transfers and dependences (A4), all carried as one IR (A8) and weighed by the weighting operand (A3), then planned by
# mesh_plan and run by torch's own operators on tensors over the placed pages (A9).  It chooses no placement and no
# weight: both are the results of functions the program passes (the operator's ruling 2 of 2026-11-04).  The
# coordinator's rulings on its first form's stops and on the reviews' findings: A-T3' (the collectives as declared
# transfers mesh_plan routes), A-T5 (a returned result over the plan's pages cloned), A-T7 (a range written after the
# reads that make it) and A-T8 to A-T10, each at its code.  The appendix's revisions
# (design/compilation-coordination.md, "The appendix's revisions, 2026-11-04"): U1 (no D5), N1 to N5 (each compiled
# graph its own function and plan) and K1 (torch's own MPS kernels on tensors over the placed pages), each at its code.
# 2026-11-04, owner T2: A-T6 (one standing graph a program) goes (N5).  What must stay: no placement and no weight
# chosen here
# 2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", TT1 to TT9): A10, a program's calls of
# compiled functions as a dataflow graph analysed at compile time (program), its output carried in each function's IR
import collections
import ctypes as C
import errno
import itertools
import math
import operator
import os
import time
import weakref

import torch
from torch.fx.node import Node, map_arg

import mesh

F = torch.nn.functional
# Y. Xu et al., GSPMD, arXiv 2105.04663, 2021, section 3.5: elementwise operators, every tensor operand of the output's
# shape, each dimension preserved from the inputs to the output and back
ELEMENTWISE = {operator.add, operator.sub, operator.mul, operator.truediv, operator.neg, torch.add, torch.sub,
               torch.mul, torch.div, torch.neg, torch.relu, torch.tanh, torch.sigmoid, torch.exp, F.relu, F.gelu,
               F.silu, 'add', 'sub', 'mul', 'div', 'neg', 'relu', 'tanh', 'sigmoid', 'exp'}
# GSPMD sections 3.2 and 3.5: Dot, a generalized matrix multiply (an Einsum)
DOT = {operator.matmul, torch.matmul, 'matmul', torch.mm, 'mm', F.linear}
# design/aot-compilation-change-request.md A2, the placement operand placement(gm, example_inputs) returns.  mesh:
# GSPMD's device mesh, "a logical multi-dimensional tensor" of device ids (section 3.1), nested sequences holding the
# ids 0 .. D - 1 once each.  nodes: device d's map index nodes[d] (mesh-dataflow.h mesh_nodes_read), the mapping of the
# logical mesh onto the map's nodes (L. Zheng et al., Alpa, OSDI 2022, section 4.1: "For a set of physical devices,
# there can be multiple logical views").  splits: a placeholder's name to its dims_mapping, GSPMD's
# mesh_split(tensor, device_mesh, dims_mapping) (section 3.1: each tensor dimension's mesh dimension, -1 none)
Placement = collections.namedtuple('Placement', 'mesh nodes splits')
# an inserted collective (GSPMD sections 2.1, 3.2, 4.2 and 4.5): its kind, the FX node whose local value it carries,
# the mesh dimensions its groups lie along, the tensor dimension it gathers or scatters or the two it switches (None for
# an AllReduce; a CollectivePermute's its (source, target) places, TC2) and its item's place in this node's program
Collective = collections.namedtuple('Collective', 'kind node dims dim position')
# design/aot-compilation-change-request.md A10, "Its output", for one compiled function: `mine`, the places of its
# calls among the program's calls of compiled functions, in the traced block's order; `outstanding`, "the most calls
# of it the analysis lets be outstanding at once in the program that submits them"; `orders`, "the order constraints
# between calls", each (a, b) call a before call b; `commuting`, the program's sets of calls "whose effects commute",
# which "may run in any order" (Rinard and Diniz), each a tuple of places.  "both carried in A8's IR" (IR `calls`)
Calls = collections.namedtuple('Calls', 'mine outstanding orders commuting')


# design/aot-compilation-change-request.md A8: one IR carried from partitioning to serialization, as MSCCLang carries
# its instruction DAG from lowering to scheduling (M. Cowan et al., ASPLOS 2023, sections 4 and 5).  It holds the
# placement operand (A2) and the mesh it names, the serial of the map read it was built on, each tensor's sharding
# (GSPMD section 3.1), this node's partitioned graph with its collectives, and the task graph as mesh_plan takes it
# (design/plan-to-bridge-change-request.md R3 and R4; design/compilation-coordination.md, T1): the declared transfers
# with their kinds, the dependences and this node's values; `weights` is the weighting operand once weights(ir) has
# given it (A3).  `placed`: the keys of the local values that lie over the plan's pages (an AllReduce's result and its
# slices); `made`: the keys this node's program has made before the step being added; `calls`: A10's output for this
# function (Calls), from the traced program under whose call it was compiled, None where no traced program submits it
# (design/aot-compilation-change-request.md A10: "both carried in A8's IR"; design/compilation-coordination.md, "The
# revisions of f3f7cf46", TT7)
class IR:
    def __init__(self, gm, placement, nodes):
        self.gm, self.placement, self.weights = gm, placement, None
        self.order = list(gm.graph.nodes)
        self.shape, self.coords, self.map = _mesh(placement, nodes.count)
        here = [d for d, index in enumerate(self.map) if index == nodes.self]
        if not here:
            raise ValueError('map index %d, this node, is no device of the placement operand (A2)' % nodes.self)
        self.device, self.index, self.serial, self.identity = here[0], nodes.self, nodes.serial, 0
        self.sharding, self.program, self.reduced, self.roles, self.events = {}, [], {}, {}, []
        self.declared, self.dependences, self.values, self.layouts = [], [], [], []
        self.placed, self.made, self.calls = set(), set(), None

    # GSPMD section 3.1, a tiled dimension: each device's tile, the dimension's size over its mesh dimension's
    def local(self, node, sharding):
        return tuple(s // self.shape[m] if m >= 0 else s for s, m in zip(_value(node).shape, sharding))

    # one range (mesh-collective.h struct mesh_declared): a local value of `node` of `shape` from device `source` to
    # each device of `destinations`, a declared transfer each, of one identity, each of `kind`, its consumer's at the
    # destination (design/plan-to-bridge-change-request.md R4; the coordinator's ruling P3, at _consumer).  This node's
    # value of it (MESH_EXECUTION, wire) where it is its source or a destination, its layout the element type, the shape
    # and the device of `node`'s tensors (the copies' tensors are bound on that device,
    # design/compilation-coordination.md, K1).  Each destination's declared transfer's index
    # 2026-11-04, owner T2 (the coordinator's ruling P3 on review W3's finding 3): every transfer MESH_JOIN (A-T8,
    # the producer's one mesh_host_publish as the reason) goes: the kind is R4's test of the consumer, so a range of
    # function B is a set at the bridge (design/aot-compilation-change-request.md, appendix item 5).  What must stay:
    # each range released whole by one mesh_host_publish, and read only once it is complete
    def transfer(self, node, shape, source, destinations, kind):
        example, value, transfers = _value(node), mesh.ABSENT, {}
        dtype = example.dtype
        size, ends = dtype.itemsize * math.prod(shape), [self.map[d] for d in [source] + destinations]
        if self.index in ends:
            value = len(self.values)
            self.values.append(mesh.Value(section=mesh.Section(bytes=size), kind=mesh.EXECUTION, wire=1))
            self.layouts.append((dtype, shape, example.device.type))
        self.identity += 1
        for d in destinations:
            transfers[d] = len(self.declared)
            mine = value if self.index in (ends[0], self.map[d]) else mesh.ABSENT
            self.declared.append(mesh.Declared(ends[0], self.map[d], self.identity, kind, mine, 0, size))
        return transfers


# J. Ansel et al., PyTorch 2, ASPLOS 2024, sections 3.5 and 5.1: a node's fake tensor, "the metadata of a tensor without
# its actual data", its sizes propagated by meta functions.  A1: shapes are compiled, so a symbolic size is refused.  A
# tensor on the CPU or on torch's MPS device is bound to the placed pages (torch.from_dlpack of a kDLCPU or kDLMetal
# tensor, design/compilation-coordination.md, K1); one on any other device is refused, naming it
# 2026-11-04, owner T2 (K1; design/aot-compilation-change-request.md, appendix item 7): A-T4's refusal of a tensor not
# on the CPU goes for MPS, whose tensors torch v2.14.0 builds over existing memory by DLPack (_dlpack_tensor).  What
# must stay: the refusal of a device with no binding, and of a symbolic size
def _value(node):
    value = node.meta.get('example_value')
    if not isinstance(value, torch.Tensor):
        raise NotImplementedError('%s (%s): no tensor value (PyTorch 2 section 3.5)' % (node.name, node.target))
    if not all(isinstance(size, int) for size in value.shape):
        raise ValueError('%s: symbolic sizes %s; A1 compiles shapes, so nothing at run time changes one'
                         % (node.name, tuple(value.shape)))
    if value.device.type not in ('cpu', 'mps'):
        raise NotImplementedError('%s on %s: no binding of its tensors to the placed pages (a DLPack tensor of kDLCPU '
                                  'or kDLMetal alone: K1)' % (node.name, value.device))
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
# "could use multiple steps of resharding to reach the desired sharding": the steps in the order it names them
# (design/compilation-coordination.md, A-T9), each mesh dimension's AllGather, then each one's AllToAll into a
# dimension no mesh dimension holds (one whose dimension is held is refused, naming it), then each one's DynamicSlice (a
# local slice, no communication; a slice of a value over the plan's pages lies over them too).  One device mesh orders
# every tensor's devices alike, so no CollectivePermute arises.  A placeholder's whole value is held at every node,
# since torch passes the whole input, so its reshard starts from it and is slices alone (the coordinator's ruling on
# review V2's finding 7).  Each step an item of this node's program: (kind, the keys it reads, the key it makes, its
# operand), none repeated whose key an earlier one made (that value stands through the execution).  The key of
# `node`'s local value in `want`
def _reshard(ir, node, have, want):
    def step(item):
        if item[2] not in ir.made:
            ir.made.add(item[2])
            ir.program.append(item)
    if node.op == 'placeholder':
        have = (-1,) * len(have)
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
# the function's whole input, sliced to its sharding (section 4.5, DynamicSlice).  A Dot's contracting dimension keeps
# a mesh dimension both operands shard it along, its local results partial sums along it.  Where the output does not
# use that mesh dimension they are combined by "AllReduce when the layer sums over a partitioned dimension" (section
# 2.1; section 4.2).  Where the output shards a dimension along it they are reduced onto that dimension by a
# ReduceScatter: "gradients will be communicated with ReduceScatter instead of AllReduce" (section 3.2, where bd is
# split [0, -1] and df [0, 1], so df's gradient contracts B along mesh dimension 0, along which df's D lies), and
# "replace the AllReduce on the output BSM with a ReduceScatter" (section 5.1; Figure 7's finalized graph, the einsum of
# bxShy and hyM to bxSM, then RS-y to bxSmy), so the local result has the output's sharding but on that dimension,
# which it holds whole.  A contracting dimension the two operands shard otherwise is replicated where the output uses
# its mesh dimension ("unsharded on demand", section 3.2; Figure 6, "AllGather non-matching Bx").  An operator's
# operands are resharded to its local result's sharding on the dimensions they share with it.  The output's values are resharded to replicated, so the callable returns the
# graph's whole values (PyTorch 2 section 3.1: the callable stands for the graph).  A partial's key carries its
# sharding, the members' partials' dims_mapping `lower` gives a ReduceScatter (TC1).  Each Dot form of _operator
# contracts one dimension, so `partial` holds at most one mesh dimension
# 2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", TA1 to TA3;
# design/aot-compilation-change-request.md A4: "ReduceScatter goes where a reduction's result is sharded"): a
# contracting dimension both operands shard along a mesh dimension the output uses was replicated, each operand
# gathered before the Dot, and is now reduced by a ReduceScatter after it; the partial's key (node, 'partial') becomes
# (node, its sharding, 'partial').  Why: GSPMD sections 3.2 and 5.1 put the ReduceScatter there, and `lower` reads a
# collective's input sharding from its key.  What must stay: the AllReduce where the output leaves the mesh dimension
# unused, the AllGather of a contracting dimension the operands shard unalike, and the refusal of one GSPMD gives no
# rule for
def _partition(ir, operators):
    inputs = 0
    for node in ir.order:
        if node.op == 'placeholder':
            whole = (-1,) * len(ir.sharding[node])
            ir.program.append(('input', (), (node, whole), inputs))
            ir.made.add((node, whole))
            _reshard(ir, node, whole, ir.sharding[node])
            inputs += 1
        elif node.op == 'output':
            keys = {value: _reshard(ir, value, ir.sharding[value], (-1,) * len(ir.sharding[value]))
                    for value in node.all_input_nodes}
            ir.program.append(('output', tuple(keys.values()), None, (node, keys)))
        else:
            _, operands, labels, out = operators[node]
            used, partial, contracted = {m for m in ir.sharding[node] if m >= 0}, [], {}
            for label in sorted(set(labels[0]).intersection(*labels[1:]) - set(out)):
                held = [ir.sharding[o][own.index(label)] for o, own in zip(operands, labels)]
                if held[0] == held[1] >= 0:
                    partial.append(held[0])
                elif any(s >= 0 and s not in used for s in held):
                    raise NotImplementedError(
                        '%s: its contracting dimension on mesh dimensions %s in its two operands: GSPMD gives the '
                        'AllReduce or the ReduceScatter where both shard it alike (sections 2.1, 3.2) and the '
                        'AllGather where the output takes its mesh dimension (section 3.2, Figure 6), and no rule here '
                        '(A-T2)' % (node.name, held))
                contracted[label] = held[0] if held[0] == held[1] else -1
            local = tuple(-1 if s in partial else s for s in ir.sharding[node])
            wants = [list(_carry(local, out, own)) for own in labels]
            for label, m in contracted.items():
                for want, own in zip(wants, labels):
                    want[own.index(label)] = m
            keys = [_reshard(ir, o, ir.sharding[o], tuple(w)) for o, w in zip(operands, wants)]
            key = (node, local, 'partial') if partial else (node, ir.sharding[node])
            ir.program.append(('call', tuple(keys), key, node))
            if any(m in used for m in partial):
                (m,) = partial
                ir.program.append(('collective', (key,), (node, ir.sharding[node]),
                                   Collective('reduce-scatter', node, (m,), ir.sharding[node].index(m),
                                              len(ir.program))))
            elif partial:
                ir.reduced[node] = Collective('all-reduce', node, tuple(partial), None, len(ir.program))
                ir.program.append(('collective', (key,), (node, ir.sharding[node]), ir.reduced[node]))
                ir.placed.add((node, ir.sharding[node]))


# design/aot-compilation-change-request.md A2: GSPMD's two passes by the placement operand, "sharding completion and
# per-operator partitioning" (section 3), on the map as mesh_nodes_read gives it.  An unannotated tensor that
# propagation gives no sharding stays replicated (section 3.1, "All devices have the same full data"); an annotation
# on a mesh dimension of one device reads as -1, a dims_mapping naming no mesh dimension of more being replicated
# (section 3.1; design/compilation-coordination.md, A-T10); a dimension its mesh dimension does not divide is refused
# (section 4.1 pads it and masks the padding, which the contract does not give)
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


# design/compilation-coordination.md, A-T7: positions are thirds of this node's program's items.  At 3g item g reads
# what this node holds, and an AllGather or AllToAll sends (torch's copy_ reads a shard as it writes its range); at
# 3g + 1 it waits for and reads what arrives, torch.cat's read of its own part among them; at 3g + 2 a range it
# computes is written by copy_, after the reads that make it.  `node`'s readers over the plan's pages: the span of the
# items' reads of a key of it that lies over them (an AllGather's or AllToAll's to 3g + 1), else `default`.  The span
# covers every read, so an edge into the event precedes its first and one out of it follows its last
def _readers(ir, node, default):
    spans = [(3 * g, 3 * g + (item[0] == 'collective' and item[3].kind != 'all-reduce'))
             for g, item in enumerate(ir.program) if any(k in ir.placed and k[0] is node for k in item[1])]
    return (min(s for s, _ in spans), max(e for _, e in spans)) if spans else (default, default)


# the coordinator's ruling P3 on review W3's finding 3: the kind of the declared transfers of `node`'s range by R4's
# test of its consumer at each destination (design/plan-to-bridge-change-request.md R4): MESH_PAGES, function B, where
# every reader there of a key of `node` over the plan's pages is an operator of GSPMD's elementwise set (ELEMENTWISE: a
# map over the range's pages, f of a union of pages the union of f of each, a homomorphism as R4 states it: over lists
# with its combine associative, J. Gibbons, J. Functional Programming 1996, and over pages in any order a bag
# homomorphism of the Boom hierarchy, its combine commutative too, L. Meertens, Algorithmics, CWI Monographs 1, 1986;
# the appendix item 5's "a pointwise nonlinearity after a ffn"); MESH_JOIN, function A, otherwise (a Dot, a slice, a
# cat, a collective, the output's clone).  Every node runs the same program items, so every rank computes the same
# kinds (R6)
# 2026-11-05, owner T (design/compilation-coordination.md, "The revisions of 46ae8fdf", TC6): the homomorphism cited
# as R4 cites it, Gibbons for lists and Meertens's bag homomorphisms for any order, where Gibbons alone was cited.
# What must stay: the kinds as before
def _consumer(ir, node):
    readers = [item for item in ir.program if any(k in ir.placed and k[0] is node for k in item[1])]
    if readers and all(item[0] == 'call' and item[3].target in ELEMENTWISE for item in readers):
        return mesh.PAGES
    return mesh.JOIN


# design/compilation-coordination.md, A-T3': AllReduce as a reduce to a chosen root, then a broadcast from it (G. Wang
# et al., Blink, MLSys 2020, section 3.3: "to do an AllReduce operation on the directed graph, we first run a reduce
# operation to a chosen root vertex using the undirected graph and then do a broadcast operation from the root vertex
# using the same tree but with links going in the reverse direction"), over mesh_plan's routed demands, relays
# forwarding as the transport does (A4: "A relay's forward is its fused receive-copy-send").  The recorded difference
# from Blink: the sum's trees are its own demand's, as mesh_plan routes it, not the partials' tree reversed.  The root
# is the group's first device in the device mesh's order (GSPMD section 3.1: it "preserves the order of devices in
# device_mesh", so the placement operand's chooser decides it).  Each other member's partial is one declared transfer
# to the root; the root's combine of each landed partial with its own is MSCCLang's receive-reduce-copy (M. Cowan et
# al., ASPLOS 2023, section 4.2: it "receives a chunk, reduces it with a source chunk, and locally copies it to the
# destination") by the function's own add; its sum is one range to every other member.  Each partial's transfer is
# MESH_PAGES: its consumer, the root's combine, is R4's function B ("reductions' partial sums", their combine the add,
# associative, Gibbons 1996, and commutative, Meertens's bag homomorphisms, CWI Monographs 1, 1986, as R4 cites them;
# 2026-11-05, owner T, TC6: where this said "associative and commutative" uncited); the sum's kind is its readers'
# (_consumer; the coordinator's ruling P3).  A partial and
# the sum are each written over their range's pages after the reads that make them (A-T7).  Events (device, declared
# transfer, kind, first, last): a member's production of its partial (the Dot's item, at its third step) and its
# consumption of the sum (its readers); the root's consumption of each partial (the AllReduce's item, after arrival),
# its production of the sum (its third step) and its own read of the sum (its readers)
def _all_reduce(ir, c, group, have):
    node, root, members = c.node, group[0], group[1:]
    shape, arrive = ir.local(node, ir.sharding[node]), 3 * c.position + 1
    readers = _readers(ir, node, arrive + 1)
    up = [ir.transfer(node, shape, d, [root], mesh.PAGES)[root] for d in members]
    down = ir.transfer(node, shape, root, members, _consumer(ir, node))
    for d, partial in zip(members, up):
        ir.events += [(d, partial, 'P', arrive - 2, arrive - 2), (d, down[d], 'C') + readers,
                      (root, partial, 'C', arrive, arrive)]
    first = min(down.values())
    ir.events += [(root, first, 'P', arrive + 1, arrive + 1), (root, first, 'S') + readers]
    if ir.device == root:
        ir.roles[c] = (0, {None: ir.declared[first].value}, {p + 1: ir.declared[u].value for p, u in enumerate(up)})
    elif ir.device in members:
        me = group.index(ir.device)
        ir.roles[c] = (me, {0: ir.declared[up[me - 1]].value}, {0: ir.declared[down[ir.device]].value})


# design/compilation-coordination.md, A-T3': AllGather (GSPMD section 4.2: it "concatenates tensors from all
# participants following a specified order"): each member's shard one range to every other member, written over its
# range's pages by torch's copy_ (a range lies on its placed pages, design/disjoint-pages-change-request.md C3.2; the
# shard is torch's own, A-T4); each receiver's result torch.cat of the shards in the group's order, so each transfer is
# MESH_JOIN (a cat, R4's function A: the coordinator's ruling P3).  Events: each member's production of its range (its
# item's sends) and each other member's consumption of it (after arrival)
def _all_gather(ir, c, group, have):
    shape, sent, received = ir.local(c.node, have), 3 * c.position, 3 * c.position + 1
    ranges = [ir.transfer(c.node, shape, d, [e for e in group if e != d], mesh.JOIN) for d in group]
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
# receiver's result torch.cat of the pieces in sender order along the dimension it leaves, each transfer MESH_JOIN as an
# AllGather's.  Events as an AllGather's
def _all_to_all(ir, c, group, have):
    shape, sent, received = list(ir.local(c.node, have)), 3 * c.position, 3 * c.position + 1
    shape[c.dim[1]] //= len(group)
    pieces = {(p, q): ir.transfer(c.node, tuple(shape), d, [e], mesh.JOIN)[e]
              for p, d in enumerate(group) for q, e in enumerate(group) if p != q}
    for (p, q), i in pieces.items():
        ir.events += [(group[p], i, 'P', sent, sent), (group[q], i, 'C', received, received)]
    if ir.device in group:
        me = group.index(ir.device)
        ir.roles[c] = (me, {q: ir.declared[i].value for (p, q), i in pieces.items() if p == me},
                       {p: ir.declared[i].value for (p, q), i in pieces.items() if q == me})


# design/aot-compilation-change-request.md A4: ReduceScatter (GSPMD section 4.2: it "is semantically equivalent to an
# AllReduce followed by a DynamicSlice where each partition gets one slice of the fully reduced data.  An efficient
# implementation has only half the cost of AllReduce"), each slice reduced to the member that keeps it, over its packed
# trees: the first half of Blink's AllReduce (G. Wang et al., MLSys 2020, section 3.3: "we first run a reduce operation
# to a chosen root vertex"), with that member as the root.  Slice q of each member's partial, along tensor dimension
# `dim` (the record's, TC1), is the slice member q of the group keeps (its place in the device mesh's order, GSPMD
# section 3.1), one declared transfer p -> q for each other member p, written by torch's copy_ as an AllToAll's piece
# is.  Each is MESH_PAGES: its consumer, q's combine, is R4's function B ("reductions' partial sums", their combine the
# add, as _all_reduce's), MSCCLang's receive-reduce-copy (M. Cowan et al., ASPLOS 2023, section 4.2) by torch.add, q's
# own slice first and then each landed slice in the group's order (TC5, _all_reduce's root's order); q's result is that
# sum, a tensor of its own over no page of the plan.  `have` is the dims_mapping of the members' partials (TC1).
# Events as an AllToAll's: each member's production of its slices (its item's sends), each keeper's consumption (after
# arrival)
# 2026-11-05, owner T (design/compilation-coordination.md, "The revisions of 46ae8fdf", RS): new, where `lower` refused
# a ReduceScatter.  The partitioner here inserts none (_partition replicates a contracting dimension whose mesh
# dimension the output uses, where GSPMD section 5 would "replace the AllReduce on the output BSM with a
# ReduceScatter"), so this is reached once A2's partitioner inserts one.  What must stay: each slice to its keeper
# 2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", TA1): _partition inserts it where a
# Dot's partial sums lie along a mesh dimension the output shards (GSPMD sections 3.2 and 5.1), `dim` the output's
# dimension along it and `have` the partial's sharding, so it is reached.  Why: A4, "A2's partitioner inserts these two
# where GSPMD's does".  What must stay: each slice to its keeper
def _reduce_scatter(ir, c, group, have):
    shape, sent, received = list(ir.local(c.node, have)), 3 * c.position, 3 * c.position + 1
    shape[c.dim] //= len(group)
    slices = {(p, q): ir.transfer(c.node, tuple(shape), d, [e], mesh.PAGES)[e]
              for p, d in enumerate(group) for q, e in enumerate(group) if p != q}
    for (p, q), i in slices.items():
        ir.events += [(group[p], i, 'P', sent, sent), (group[q], i, 'C', received, received)]
    if ir.device in group:
        me = group.index(ir.device)
        ir.roles[c] = (me, {q: ir.declared[i].value for (p, q), i in slices.items() if p == me},
                       {p: ir.declared[i].value for (p, q), i in slices.items() if q == me})


# design/aot-compilation-change-request.md A4: CollectivePermute (GSPMD section 4.2: it "exchanges data among a list of
# source-destination pairs"), one point-to-point demand a pair, routed as AllToAll's are (mesh_plan's demands, routed
# jointly by Garg and Koenemann's rounds).  Its pairs are XLA's source_target_pairs over the group's places in the
# device mesh's order (the record's `dim`, TC2): "For each pair, the operand is sent from source replica to target
# replica"; "Any two pairs should not have the same target replica id, and they should not have the same source replica
# id" (OpenXLA, Operation semantics, CollectivePermute; GSPMD section 4.2: "We use XLA's operators"), refused otherwise
# as the placement operand's errors are (ValueError).  Each pair p -> q, p != q, is one declared transfer of the
# member's whole local value, written by torch's copy_ as an AllToAll's piece is; the destination's result is a tensor
# of its own, torch.clone of its arrival, as an AllToAll's is torch.cat's, so it lies over no page of the plan (TC4) and
# each transfer is MESH_JOIN, its consumer the clone (TC3; the coordinator's ruling P3 names the clone function A).  A
# pair p -> p keeps the member's own value, no transfer; a member no pair targets holds "a tensor consisting of 0(s)"
# of the input's shape (OpenXLA, as above; TC5).  Events as an AllToAll's
# 2026-11-05, owner T (design/compilation-coordination.md, "The revisions of 46ae8fdf", CP): new, where `lower`
# refused a CollectivePermute.  The partitioner here inserts none (_reshard: one device mesh orders every tensor's
# devices alike, "so no CollectivePermute arises"), so this is reached once A2's partitioner inserts one.  What must
# stay: one transfer a pair
# 2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", TA4): still inserted nowhere.  GSPMD
# puts it in halo exchange, "We use the CollectivePermute operator to exchange halo data between partitions" (section
# 4.3), for window-based operators ("e.g., Convolution, ReduceWindow") and data formatting operators (Slice, Pad,
# Reverse, Reshape), and in a pipeline's buffer shifting, "pad_left(state, 1)[:-1]" along a sharded L (section 3.3),
# whose "shifting using Pad and Slice can be done with a single CollectivePermute" (section 4.6); and in resharding "to
# change device order" (section 4.5).  _operator partitions elementwise operators and Dot alone, refusing the rest
# (A-T2), and the placement operand has one device mesh.  What must stay: one transfer a pair
def _collective_permute(ir, c, group, have):
    pairs = tuple(tuple(pair) for pair in c.dim)
    places = [p for pair in pairs for p in pair]
    if not all(isinstance(p, int) and 0 <= p < len(group) for p in places) \
            or len({p for p, _ in pairs}) != len(pairs) or len({q for _, q in pairs}) != len(pairs):
        raise ValueError('%s: a CollectivePermute\'s pairs %s over a group of %d: each a (source, target) place, no two '
                         'with one source or one target (XLA\'s source_target_pairs)' % (c.node.name, pairs, len(group)))
    shape, sent, received = ir.local(c.node, have), 3 * c.position, 3 * c.position + 1
    moved = {(p, q): ir.transfer(c.node, shape, group[p], [group[q]], mesh.JOIN)[group[q]] for p, q in pairs if p != q}
    for (p, q), i in moved.items():
        ir.events += [(group[p], i, 'P', sent, sent), (group[q], i, 'C', received, received)]
    if ir.device in group:
        me = group.index(ir.device)
        ir.roles[c] = (me, {q: ir.declared[i].value for (p, q), i in moved.items() if p == me},
                       {p: ir.declared[i].value for (p, q), i in moved.items() if q == me})


# the program's own data edges at each node (mesh-collective.h struct mesh_dependence: event `from` before event `to`,
# `to` in execution t + lag where `from` is in t; design/disjoint-pages-change-request.md, C3.5: a stage's reads and
# writes "in the order its command buffer runs them"), every rank's, by the program's order on that rank, whose
# executions run in order (execution t's call returns before execution t + 1's begins), at A-T7's positions
# (design/compilation-coordination.md, A-T7).  An event is in force over its positions [first, last], and precedes
# another where its last precedes the other's first.  That relation as a chain, linear in the events (the coordinator's
# ruling on review V2's finding 8): each class (the events of one position) to the next occupied class (lag 0); an
# event over several positions from the class before its first, to the class after its last, and to each such event
# after it (lag 0); every event no lag-0 edge leaves to every event none enters (lag 1), since each event of execution t
# precedes each of t + 1.  An end's kinds bit: a production's, a source read's, a consumption's none
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


# design/aot-compilation-change-request.md A4 and design/compilation-coordination.md, A-T3': each inserted collective,
# over each of its device groups, lowered to declared transfers and dependences, which mesh_plan's demands and trees
# route jointly ("rdma/mesh-collective.c's demands and trees, as now, fed by A2's collectives"): AllReduce, AllGather,
# AllToAll, ReduceScatter and CollectivePermute, GSPMD section 4.2's communication primitives; a collective of another
# kind refused, naming it
# 2026-11-05, owner T (design/compilation-coordination.md, "The revisions of 46ae8fdf", RS and CP): ReduceScatter and
# CollectivePermute lowered, where this refused them.  Why: A4, "rdma/mesh_compile.py refuses CollectivePermute and
# ReduceScatter now; this requirement covers them".  What must stay: every collective A2 inserts lowered
LOWERINGS = {'all-reduce': _all_reduce, 'all-gather': _all_gather, 'all-to-all': _all_to_all,
             'reduce-scatter': _reduce_scatter, 'collective-permute': _collective_permute}


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


# one context a program, attached to the node's one region (mesh.h MESH_NAME; mesh-dataflow.h struct mesh_ctx), which
# holds no plan.  FUNCTIONS: the program's ordinals of its compiled functions, 1, 2, ... in compile order, one taken at
# each call of the backend's compile function (design/compilation-coordination.md, "The appendix's revisions", N1;
# J. Ansel et al., PyTorch 2, ASPLOS 2024, section 3.1: the custom compile function is called with each captured
# graph), the same at every rank since every rank runs the same program
# 2026-11-04, owner T2 (N1, N5; design/aot-compilation-change-request.md, appendix item 6): STANDING, the record of
# the one graph whose plan stood (A-T6), goes with A-T6, and a plan's name is (group, function) in place of (group, the
# context's count of its plans): compiled functions coexist, each its own plan.  What must stay: one context a program
CONTEXT, FUNCTIONS = [], itertools.count(1)


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


# DLPack as torch v2.14.0 bundles it (torch's include/ATen/dlpack.h, DLPACK_MAJOR_VERSION 1, DLPACK_MINOR_VERSION 3):
# DLDevice, DLDataType, DLTensor, DLPackVersion and DLManagedTensorVersioned, each field in the header's order and of
# its type (DLDeviceType an int32_t enum).  The versioned form, as the header directs: DLManagedTensor "is deprecated
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
# dlpack.h DLDeviceType by torch device type: kDLCPU ("CPU device") and kDLMetal ("Metal for Apple GPU"), the two
# devices whose tensors are bound to the placed pages (_value).  DLDataType (code, bits, lanes) by torch dtype as torch
# v2.14.0's aten/src/ATen/DLConvertor.cpp getDLDataType gives it (bits the element size x 8, lanes 1, but
# Float4_e2m1fn_x2's "dtype.lanes = 2" and "dtype.bits = 4"), for every dtype its toScalarType takes back as itself;
# the codes dlpack.h's DLDataTypeCode (kDLInt 0, kDLUInt 1, kDLFloat 2, kDLBfloat 4, kDLComplex 5, kDLBool 6,
# kDLFloat8_e4m3fn 10, kDLFloat8_e4m3fnuz 11, kDLFloat8_e5m2 12, kDLFloat8_e5m2fnuz 13, kDLFloat8_e8m0fnu 14,
# kDLFloat4_e2m1fn 17).  The capsule's name for the versioned form (include/ATen/DLConvertor.h
# DLPackTraits<DLManagedTensorVersioned> `capsule`), made by Python's PyCapsule_New
# 2026-11-04, owner T2 (the coordinator's ruling on review W2's finding 4): the float8 and float4 types are added; a
# dtype is refused only where getDLDataType gives no type that toScalarType takes back as that dtype (the quantized and
# bit types, which getDLDataType refuses; UInt1 to UInt7 and Int1 to Int7, which come back as Byte and Char).  What
# must stay: each entry the pair those two functions give
DEVICES, VERSION, CAPSULE = {'cpu': 1, 'mps': 8}, DLPackVersion(1, 3), b'dltensor_versioned'
CODES = {torch.int8: 0, torch.int16: 0, torch.int32: 0, torch.int64: 0, torch.uint8: 1, torch.uint16: 1,
         torch.uint32: 1, torch.uint64: 1, torch.float16: 2, torch.float32: 2, torch.float64: 2, torch.bfloat16: 4,
         torch.complex32: 5, torch.complex64: 5, torch.complex128: 5, torch.bool: 6, torch.float8_e4m3fn: 10,
         torch.float8_e4m3fnuz: 11, torch.float8_e5m2: 12, torch.float8_e5m2fnuz: 13, torch.float8_e8m0fnu: 14}
TYPES = {dtype: (code, dtype.itemsize * 8, 1) for dtype, code in CODES.items()}
TYPES[torch.float4_e2m1fn_x2] = (17, 4, 2)
CAPSULE_NEW = C.PYFUNCTYPE(C.py_object, C.c_void_p, C.c_char_p, C.c_void_p)(('PyCapsule_New', C.pythonapi))
# the Objective-C runtime's objc_msgSend, called through a pointer of each method's own type (objc/message.h: "These
# functions must be cast to an appropriate function pointer type before being called"): an object-returning method of
# no argument (-device, MTLResource.h; -retain, NSObject), a void one (-release, NSObject), and
# -[MTLDevice newBufferWithBytesNoCopy:length:options:deallocator:] (MTLDevice.h: "Create a buffer by wrapping an
# existing part of the address space"; NSUInteger length and options, the deallocator a block, nil here)
OBJC = C.CDLL('/usr/lib/libobjc.A.dylib')
OBJC.sel_registerName.restype, OBJC.sel_registerName.argtypes = C.c_void_p, [C.c_char_p]
SEND = C.cast(OBJC.objc_msgSend, C.c_void_p).value
GET = C.CFUNCTYPE(C.c_void_p, C.c_void_p, C.c_void_p)(SEND)
PUT = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p)(SEND)
WRAP = C.CFUNCTYPE(C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_ulong, C.c_ulong, C.c_void_p)(SEND)
SELECTOR = {name: OBJC.sel_registerName(name) for name in
            (b'device', b'retain', b'release', b'newBufferWithBytesNoCopy:length:options:deallocator:')}
# MTLResource.h MTLResourceStorageModeShared (MTLStorageModeShared 0 << MTLResourceStorageModeShift), the option
# rdma/mesh-metal.m and metal-microbench tools/e2b_replay.m placed_buffer give theirs
SHARED = 0
# torch's MPS device's MTLDevice (DEVICE once read): the `device` of a buffer torch's MPS allocator made (MTLResource.h:
# "The device this resource was created against.  This resource can only be used with this device"), the allocator's
# being at::mps::MPSDevice::getInstance()->device() (torch v2.14.0 include/ATen/mps/MPSAllocator.h,
# MPSHeapAllocatorImpl's `m_device`, on which it makes its heaps).  That buffer is an MPS tensor's storage data
# (include/ATen/native/mps/OperationUtils.h getMTLBufferStorage: `__builtin_bit_cast(id<MTLBuffer>,
# tensor.storage().data())`; its iter_tensor_offset: "MPS storage data is not the real GPU pointer, but rather a pointer
# to id<MTLBuffer> object"), which torch.UntypedStorage.data_ptr returns (torch/csrc/StorageMethods.cpp
# THPStorage_dataPtr: `self_.mutable_data()`).  ENODEV where it names none
DEVICE = []


def _metal_device():
    if not DEVICE:
        probe = torch.empty(1, device='mps')
        device = GET(probe.untyped_storage().data_ptr(), SELECTOR[b'device'])
        if not device:
            raise OSError(errno.ENODEV, 'torch\'s MPS buffer names no MTLDevice (MTLResource.h `device`)')
        DEVICE.append(device)
    return DEVICE[0]


# dlpack.h DLManagedTensorVersioned `deleter`: "This should be called to destruct manager_ctx which holds the
# DLManagedTensorVersioned"; "The destructor deletes the argument self as well".  torch v2.14.0 calls it once the
# tensor's storage is freed: torch/csrc/utils/tensor_new.cpp tensor_fromDLPackImpl's deleter_maybe_gil
# (`tensor->deleter(tensor)` where it is not NULL), which it passes through at::DLPackTraits<T>::fromDLPack to
# aten/src/ATen/DLConvertor.cpp fromDLPackImpl, which gives it at::from_blob.  It releases `manager_ctx`, the object
# backing the tensor where there is one (the MPS copy's buffer, -release; none for a CPU copy, whose pages are the
# region's), and lets go of the structure, its shape and its strides (HELD, by the structure's address), once whoever
# calls it
# 2026-11-04, owner T2 (the coordinator's ruling on review W2's finding 6): the deleter's caller cited as torch's
# tensor_fromDLPackImpl, not fromDLPackImpl's own default; and (W3's finding 12) the buffer is `manager_ctx`, so CPU
# and MPS copies share the form.  What must stay: one release a tensor
HELD = {}


@DELETER
def _delete(managed):
    held = HELD.pop(C.addressof(managed.contents), None)
    if held and held[0].manager_ctx:
        PUT(held[0].manager_ctx, SELECTOR[b'release'])


# design/compilation-coordination.md, K1: a placed value's copies wrapped by one MTLBuffer with no copy, as
# metal-microbench tools/e2b_replay.m placed_buffer wraps them: over [its lowest copy's first byte, its highest copy's
# end), each copy `pages` pages of the region's page size (mesh-call.h struct mesh_section; mesh.h struct hdr `pgsz`),
# so the buffer starts on a page and is whole pages long, as placed_buffer's and rdma/mesh-metal.m's are; on torch's
# MPS device, shared storage, no deallocator (the pages are the region's).  ENOMEM where Metal makes none, as
# mesh-metal.m's state buffer.  The buffer (one reference, the caller's) and each copy's byte offset in it
def _placed_buffer(context, section, addresses):
    low = min(addresses)
    length = max(addresses) - low + section.pages * mesh.Header.from_address(context.M).pgsz
    buffer = WRAP(_metal_device(), SELECTOR[b'newBufferWithBytesNoCopy:length:options:deallocator:'], low, length,
                  SHARED, None)
    if not buffer:
        raise OSError(errno.ENOMEM, 'Metal made no buffer over a placed value\'s %d copies, %d bytes (MTLDevice.h '
                                    'newBufferWithBytesNoCopy)' % (section.count, length))
    return buffer, [address - low for address in addresses]


# design/compilation-coordination.md, K1: a copy of a placed value as a tensor of `device` over its pages, by torch's
# own interface for a tensor over existing memory, torch.from_dlpack of a DLPack capsule (torch/utils/dlpack.py: "The
# returned PyTorch tensor will share the memory with the input tensor"), as torch v2.14.0 reads one
# (aten/src/ATen/DLConvertor.cpp fromDLPackImpl): its device by dlDeviceToTorchDevice (kDLCPU to the CPU, kDLMetal to
# `at::Device(DeviceType::MPS, index)`), the index 0 as torchDeviceToDLDevice gives both (its device_id that of a CUDA,
# PrivateUse1 or MTIA device alone, "0" else; dlpack.h: "For vanilla CPU memory ... this is set to 0"); `data` passed
# to at::from_blob unchanged, with the strides and the storage offset toStorageOffset(byte_offset), a multiple of the
# element size.  `data` and `byte_offset` are given as torch's own export sets them for that device (toDLPackImpl): on
# MPS `data` the storage's base, the id<MTLBuffer>, and `byte_offset` "src.storage_offset() *
# c10::elementSize(src.scalar_type())"; on any other device `data` the tensor's data pointer and `byte_offset` 0.
# `owner`, the object backing the tensor (the MPS copy's buffer; None for a CPU copy), is `manager_ctx`, of which the
# tensor holds its own reference (-retain), released by its deleter.  The strides the shape's contiguous ones
# (dlpack.h: strides "can not be NULL if ndim != 0"; fromDLPackImpl's form without them requires byte_offset 0); the
# version the header's, which torch takes where its major is at most DLPACK_MAJOR_VERSION
# (torch/csrc/utils/tensor_new.cpp tensor_fromDLPack); flags 0 ("By default the flags should be set to 0").  Where
# torch makes no tensor, the deleter's work is done here, once whoever calls it
def _dlpack_tensor(device, data, offset, owner, dtype, shape):
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
    HELD[C.addressof(managed)] = (managed, sizes, strides)
    try:
        return torch.from_dlpack(CAPSULE_NEW(C.addressof(managed), CAPSULE, None))
    except BaseException:
        _delete(C.pointer(managed))
        raise


# each copy of a placed value as a tensor of its layout over its pages (every address a constant of the plan,
# design/disjoint-pages-change-request.md, C3.3), every copy by _dlpack_tensor: a CPU copy's `data` its address, its
# offset 0, no owner; an MPS copy's `data` the value's _placed_buffer, its offset the copy's in it, the buffer its
# owner, whose own reference is released once each copy's tensor holds its own (K1).  EFAULT where a copy's row has no
# page (mesh-call.h mesh_section_address)
# 2026-11-04, owner T2 (the coordinator's ruling on review W3's finding 12; design/disjoint-pages-change-request.md,
# C3.8: "no call site handles two forms"): torch.frombuffer for CPU copies goes, and every copy is bound by
# torch.from_dlpack.  What must stay: each device's `data` and `byte_offset` as torch's own export sets them
def _bind(context, section, dtype, shape, device):
    addresses = [mesh.LIB.mesh_section_address(C.byref(context), section, c) for c in range(section.count)]
    if not all(addresses):
        raise OSError(errno.EFAULT, 'copy %d of a placed value has no page (mesh-call.h mesh_section_address)'
                      % addresses.index(None))
    if device == 'cpu':
        return [_dlpack_tensor(device, address, 0, None, dtype, shape) for address in addresses]
    buffer, offsets = _placed_buffer(context, section, addresses)
    try:
        return [_dlpack_tensor(device, buffer, offset, buffer, dtype, shape) for offset in offsets]
    finally:
        PUT(buffer, SELECTOR[b'release'])


# one item of this node's program, (kind, the keys it reads, the key it makes, its operand), at execution t on
# execution t's copies (mesh-collective.h struct mesh_value: copy t mod count).  An operator computes into its own
# fresh output; a member's partial is then written over its range's pages by torch's copy_ and released
# (mesh_host_publish), after the reads that make it (design/compilation-coordination.md, A-T7).  The root's AllReduce
# waits for each partial in turn and adds it in, MSCCLang's receive-reduce-copy (section 4.2) by torch.add, then
# writes the sum over its range's pages by copy_ and releases it, for which every other member waits (A-T3').  An
# AllGather or AllToAll copies each of this node's shards or pieces over its range's pages (torch's copy_) and releases
# it, then waits for each other member's and concatenates them in the group's order (torch.cat; A-T3').  A returned
# value over the plan's pages is torch.clone'd: execution t + u writes those pages again (C3.3), and a torch.compile
# callable's outputs outlive its call (A-T5).
# Every operator is torch's own kernel for its tensors' device, its MPS kernel for MPS tensors
# (design/compilation-coordination.md, K1; design/aot-compilation-change-request.md A9: "No kernel is written by the
# compilation").  torch runs MPS kernels asynchronously, so a range is written (write) between two of torch's own
# synchronizations of its device, its device module's synchronize (torch/__init__.py get_device_module: "Returns the
# module associated with a given device"; torch.mps.synchronize: "Waits for all kernels in all streams on a MPS device
# to complete"; torch.cpu.synchronize: "This function only exists to facilitate device-agnostic code"): after it, so
# the range is published only once the GPU's write to it is complete (K1); before it, so every read the program made
# earlier, of pages the write may reuse, has completed (C3.3: "every read of it in execution t happens before its write
# in execution t + L"), since Metal orders accesses by each resource's own hazard tracking (MTLResource.h
# hazardTrackingMode; torch's buffers tracked, include/ATen/mps/MPSAllocator.h UsageFlags::HAZARD) and two values'
# buffers over one page are two resources.  A received range is read only after mesh_host_arrived reports it
# (arrived), so the kernel reading it is issued after (K1)
# 2026-11-04, owner T2 (the coordinator's ruling on review W3's finding 12; C3.8: "no call site handles two forms"):
# torch.mps.synchronize for an MPS range alone becomes the range's device module's synchronize for every range.  What
# must stay: both synchronizations around every write of a range
# 2026-11-05, owner T (design/compilation-coordination.md, "The revisions of 46ae8fdf", RS and CP): a ReduceScatter
# copies each slice of its partial over its range's pages and releases it, then adds each landed slice into its own,
# in the group's order (torch.add, as the AllReduce's root); a CollectivePermute copies its value over each of its
# ranges' pages and releases it, then takes its arrival by torch.clone, keeps its own value where its pair is its own,
# or holds torch.zeros_like (XLA's zeros) where no pair targets it.  What must stay: as above
def _run(ir, context, tensors, item, env, inputs, t):
    kind, reads, key, operand = item

    def copy(value):
        return tensors[value][1][t % len(tensors[value][1])]

    def publish(value):
        mesh.LIB.mesh_host_publish(C.byref(context), tensors[value][0], t)

    def write(value, source):
        target = copy(value)
        synchronize = torch.get_device_module(target.device).synchronize
        synchronize()
        target.copy_(source)
        synchronize()
        publish(value)
        return target

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
        if operand.op == 'call_method':
            env[key] = getattr(args[0], operand.target)(*args[1:], **operand.kwargs)
        else:
            env[key] = operand.target(*args, **operand.kwargs)
        role = ir.roles.get(ir.reduced.get(operand))
        if role and role[0]:
            write(role[1][0], env[key])
    elif kind == 'collective':
        (source,), (me, sent, received) = reads, ir.roles[operand]
        if operand.kind == 'all-reduce' and me:
            env[key] = arrived(received[0])
        elif operand.kind == 'all-reduce':
            total = env[source]
            for p in sorted(received):
                total = torch.add(total, arrived(received[p]))
            env[key] = write(sent[None], total)
        elif operand.kind == 'reduce-scatter':
            size = env[source].shape[operand.dim] // (len(received) + 1)

            def piece(q):
                return env[source].narrow(operand.dim, q * size, size)
            for q, value in sent.items():
                write(value, piece(q))
            total = piece(me)
            for p in sorted(received):
                total = torch.add(total, arrived(received[p]))
            env[key] = total
        elif operand.kind == 'collective-permute':
            for value in sent.values():
                write(value, env[source])
            if received:
                env[key] = torch.clone(arrived(*received.values()))
            elif any(tuple(pair) == (me, me) for pair in operand.dim):
                env[key] = env[source]
            else:
                env[key] = torch.zeros_like(env[source])
        else:
            gather, members = operand.kind == 'all-gather', len(received) + 1
            dim = operand.dim if gather else operand.dim[0]
            size = 0 if gather else env[source].shape[operand.dim[1]] // members

            def part(q):
                return env[source] if gather else env[source].narrow(operand.dim[1], q * size, size)
            for q, value in sent.items():
                write(value, part(q))
            env[key] = torch.cat([part(me) if p == me else arrived(received[p]) for p in range(members)], dim)
    else:
        node, keys = operand
        env['output'] = map_arg(node.args[0], lambda n: torch.clone(env[keys[n]]) if keys[n] in ir.placed
                                else env[keys[n]])


# design/compilation-coordination.md, N3: the program's retirement of one compiled function's plan.  First every
# device its copies are on is synchronized by torch (its device module's synchronize, as `write` in _run), so no work
# queued on its pages runs after: the bridge releases a closed claim once no port holds it
# (design/disjoint-pages-change-request.md C3.4) and knows nothing of the GPU, as metal-microbench tools/e2b_replay.m
# (`retired:`) keeps its client while its GPU may still write the placed pages.  Then its claim is closed
# (mesh_plan_retire) and its tensors let go.  The retirement's error value, after its tensors are let go
# 2026-11-04, owner T2 (the coordinator's ruling on review W2's finding 1): a plan was retired, and its tensors let go,
# with the GPU's work on its pages possibly queued.  What must stay: the synchronization before the claim is closed
def _retire(context, plan):
    for device in plan['devices']:
        torch.get_device_module(device).synchronize()
    status = mesh.LIB.mesh_plan_retire(C.byref(context), plan['handle'])
    plan.update(handle=0, devices=(), ir=None, tensors=None)
    mesh.check(status)


# the coordinator's ruling on review W2's finding 3: a compiled function is retired when the program drops it, by
# weakref.finalize on the callable returned to torch (the Python documentation, weakref.finalize: "Return a callable
# finalizer object which will be called when obj is garbage collected"; "When the program exits, each remaining live
# finalizer is called unless its atexit attribute has been set to false.  They are called in reverse order of
# creation"; "A finalizer will never invoke its callback during the later part of the interpreter shutdown when module
# globals are liable to have been replaced by None").  Its arguments, the context and the plan, hold no reference to
# the callable (the documentation's note: func, args and kwargs must "not own any references to obj").  At the
# program's exit it runs from the atexit hook weakref registers at its first finalizer (Lib/weakref.py, finalize
# `_registered_with_atexit`), while the context, attached for the program's life (_context), still maps the region.
# A failure here is "shown on the standard error output, but cannot be propagated" (the same documentation); the
# program's own retirement, the bridge's of a gone pid (N3), follows at its exit as before.  A plan already retired (a
# replan that failed before it stood) is nothing to retire
def _dropped(context, plan):
    if plan['handle']:
        _retire(context, plan)


# design/aot-compilation-change-request.md A9; design/compilation-coordination.md, A-T4.  `build` gives the IR on the
# map as it is; mesh_plan writes the lists from it (each node's page instructions, the page table's bindings and the
# plan's state section); the GPU's program is the function's own operators, this node's partitioned graph run by torch
# on tensors over the placed pages, each copy of each value bound by _bind (torch.from_dlpack, a kDLCPU or kDLMetal
# tensor: design/compilation-coordination.md, K1).
# - Each compiled graph is its own function and its own plan (design/compilation-coordination.md, "The appendix's
#   revisions", N1): `function` its ordinal, the plan named (group, function), and `handle` the plan's state section
#   offset, mesh_plan's `*plan` (N2), as an instantiated graph's handle (NVIDIA, CUDA Programming Guide, section 4.2.2:
#   "It may be launched any number of times without repeating the instantiation").  Other functions' plans stand
#   beside it (R. Frostig, M. J. Johnson and C. Leary, SysML 2018; P. Barham et al., Pathways, MLSys 2022, section 3).
# - The plan is ESTALE where its map serial (struct mesh_state `serial`) is not the IR's map read's (the coordinator's
#   ruling on review V2's finding 6).  A plan that stood when the planning fails after it (that test, or a copy's
#   binding) is retired at once (_retire), as mesh_plan closes its own claim on its own failure after its writes (N3;
#   design/plan-to-bridge-change-request.md R8: nothing is held for a program but its plans' claims).
# - Each call launches one execution of its function: call t raises the plan's executions to t + 1, the program's
#   submission of that function execution to this node's bridge (N4; A7, Q1), before it runs it.  An execution is
#   consumed at its submission: once the plan's `executions` word has reached t + 1, the next call is t + 1 whether
#   this call's run then completes or fails, since `executions` and the supplied words only rise
#   (design/disjoint-pages-change-request.md C3.7; mesh-call.h mesh_transfers_start;
#   design/disjoint-pages-revision-coordination.md, D2), so no later call writes the published pages of t again;
#   nothing is retried (the coordinator's ruling on review W2's finding 2).  A start that fails before its raise (a
#   port's refusal, its bounded wait) submitted nothing, and the next call is t again.
# - After an ESTALE or a cancelled range, the next call plans again: the stale plan retired (_retire), then the IR
#   built anew on the map as it is and planned with the same function (N1; mesh-call.h mesh_transfers_start: "plan
#   again"; docs/principles.md section 4: recover across calls; V2's finding 5)
# - The function is retired when the program drops it: its plan retired (_retire) once the callable returned to torch
#   is collected, or at the program's exit, by Python's weakref.finalize (_dropped; the coordinator's ruling on review
#   W2's finding 3)
# 2026-11-04, owner T2 (W2's finding 2): `executions` set to t + 1 only after the run completed, so a call failing after
#   its submission had the next call run execution t again over pages already published, goes.  What must stay: one
#   execution a call
# 2026-11-04, owner T2 (U1; design/aot-compilation-change-request.md, appendix item 1): the ENOMEM message citing D5 and
# a lone AllGather or AllToAll goes, since C3.3 refuses only a value with no least lag; mesh_plan's failure is
# mesh.check's as any other.  (N5; appendix item 6): A-T6's refusal of a graph planned while another's plan stands goes,
# and the plan is found by its handle, not by context.state.  What must stay: the serial test, and a replan's IR built
# anew with the same placement operand
def serialize(build, function):
    context, plan = _context(), {'stale': True, 'handle': 0, 'devices': ()}

    def make():
        if plan['handle']:
            _retire(context, plan)
        ir = build()
        count = len(ir.declared)
        if len(ir.weights) != count or not all(isinstance(w, mesh.Weight) for w in ir.weights):
            raise ValueError('the weighting function gave %d entries for %d declared transfers; one mesh.Weight each '
                             '(A3)' % (len(ir.weights), count))
        values, handle = (mesh.Value * len(ir.values))(*ir.values), mesh.Q()
        mesh.check(mesh.LIB.mesh_plan(C.byref(context), _group(), function, (mesh.Declared * count)(*ir.declared),
                                      (mesh.Weight * count)(*ir.weights), count, values, len(values),
                                      (mesh.Dependence * len(ir.dependences))(*ir.dependences), len(ir.dependences),
                                      C.byref(handle)))
        plan.update(handle=handle.value, devices={layout[2] for layout in ir.layouts})
        try:
            if mesh.State.from_address(context.M + handle.value).serial != ir.serial:
                raise OSError(errno.ESTALE, 'the plan\'s map serial is not the one its IR was built on: plan again')
            tensors = [(value.section, _bind(context, value.section, *layout))
                       for value, layout in zip(values, ir.layouts)]
        except BaseException:
            _retire(context, plan)
            raise
        plan.update(ir=ir, tensors=tensors, executions=0, stale=False)
    make()

    def run(*inputs):
        if plan['stale']:
            make()
        ir, tensors, t = plan['ir'], plan['tensors'], plan['executions']
        status = mesh.LIB.mesh_transfers_start(C.byref(context), plan['handle'], t + 1)
        if mesh.State.from_address(context.M + plan['handle']).executions > t:
            plan['executions'] = t + 1
        try:
            mesh.check(status)
            env = {}
            for item in ir.program:
                _run(ir, context, tensors, item, env, inputs, t)
        except OSError as failure:
            if failure.errno in (errno.ESTALE, errno.ECANCELED):
                plan['stale'] = True
            raise
        return env['output']
    weakref.finalize(run, _dropped, context, plan)
    return run


# design/aot-compilation-change-request.md A10, "The calls' graph": Pathways' program tracer, "a new program tracer
# (Figure 2) that a user can wrap around a block of Python code that calls many compiled functions.  The tracer
# generates a single PATHWAYS program where each compiled function is represented by a computation node in a dataflow
# graph" (P. Barham et al., MLSys 2022, section 3); for torch, torch.fx's symbolic tracer (J. Reed, Z. DeVito, H. He,
# A. Ussery and J. Ansel, MLSys 2022), each compiled function a leaf, "left as calls in the FX trace rather than traced
# through" (torch v2.14.0 torch/fx/_symbolic_trace.py, wrap).  A compiled module is a leaf by Tracer.is_leaf_module
# ("Leaf modules are the atomic units that appear in the IR, referenced by call_module calls"), where the block is a
# module that holds it (path_of_module); a compiled function by torch.fx.wrap, which the program calls on its name at
# its module's top level ("This function can be called at module-level scope to register fn_or_name as a 'leaf
# function'"), Dynamo refusing to be traced through (torch/_dynamo/eval_frame.py: "Detected that you are using FX to
# symbolically trace a dynamo-optimized function").  A compiled function is what torch.compile returns, a function or
# an OptimizedModule, each marked by Dynamo with `_torchdynamo_orig_callable` and `_torchdynamo_wrapper_id` its own id
# (eval_frame.py, innermost_fn's test) (design/compilation-coordination.md, "The revisions of f3f7cf46", TT1)
def _compiled(target):
    return getattr(target, '_torchdynamo_orig_callable', None) is not None \
        and getattr(target, '_torchdynamo_wrapper_id', None) == id(target)


class Tracer(torch.fx.Tracer):
    def is_leaf_module(self, m, module_qualified_name):
        return _compiled(m) or super().is_leaf_module(m, module_qualified_name)


# the compiled function a node of a traced block calls, None where it calls none
def _callee(gm, node):
    target = gm.get_submodule(node.target) if node.op == 'call_module' else node.target
    return target if node.op in ('call_module', 'call_function') and _compiled(target) else None


# M. C. Rinard and P. C. Diniz, Commutativity analysis: a new analysis technique for parallelizing compilers, ACM TOPLAS
# 19(6), 1997, section 4.9.2, the expression simplifier: "simple arithmetic rewrite rules such as ex1-ex2 => ex1+(-ex2),
# -(-ex) => ex and ex1*(ex2+ex3) => (ex1*ex2)+(ex1*ex3)", and "rules such as (ex1+ex2)+ex3 => (ex1+ex2+ex3) that convert
# binary applications of commutative and associative operators to n-ary applications.  It then sorts the operands
# according to an arbitrary order on expressions", here their reprs.  ALGEBRA: torch's elementwise operators these
# rules read, as FX records them (an operator, a torch function or a Tensor method, with no keyword argument); add,
# mul, maximum and minimum are commutative and associative, as footnote 4 "ignore[s] here potential anomalies caused by
# the finite representation of numbers" and A10 sets floating-point non-associativity aside; a subtraction is rewritten
# to an addition and a negation before (TT4)
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
# expressions instead of values".  A body graph's result as an expression of `args`, one for each of its placeholders
# in order: each node of ALGEBRA rewritten by section 4.9.2's rules, any other kept as itself, its operation and its
# arguments' expressions (a get_attr, a value of the body no call writes, is an extent constant, section 3.4.1, named by
# itself)
def _execute(graph, args):
    env, given = {}, iter(args)
    for node in graph.nodes:
        if node.op == 'placeholder':
            env[node] = next(given)
        elif node.op == 'output':
            return map_arg(node.args[0], env.__getitem__)
        else:
            operands, kwargs = tuple(map_arg(node.args, env.__getitem__)), map_arg(node.kwargs, env.__getitem__)
            name = ALGEBRA.get(node.target) if node.op in ('call_function', 'call_method') and not kwargs else None
            env[node] = _rewrite(name, operands) if name and len(operands) == 1 + (name != 'neg') \
                else (node.op, node.target, operands, tuple(sorted(kwargs.items())))


# Rinard and Diniz, section 4.8 and Figure 15: two operations commute where, executed symbolically "in both execution
# orders", the simplified expressions "denote the same value"; "If it cannot symbolically execute the methods, the
# commutativity-testing algorithm conservatively assumes that the methods may not commute".  Here the two are calls of
# compiled function `callee` whose receiver (section 3.1) is their argument k: its body traced by torch.fx (Tracer, so a
# compiled function it calls stays one operation) is executed on the receiver with one call's other arguments, then on
# that result with the other call's, in both orders.  A body FX cannot trace (torch.fx's TraceError, "symbolically
# traced variables cannot be used as inputs to control flow") is one it cannot execute (TT4)
def _commutes(callee, k):
    try:
        graph = Tracer().trace(callee._orig_mod if isinstance(callee, torch.nn.Module)
                               else callee._torchdynamo_orig_callable)
    except torch.fx.proxy.TraceError:
        return False
    count = sum(node.op == 'placeholder' for node in graph.nodes)

    def run(first, second):
        inner = _execute(graph, [('symbol', 'receiver') if i == k else ('symbol', first, i) for i in range(count)])
        return _execute(graph, [inner if i == k else ('symbol', second, i) for i in range(count)])
    return k < count and run(1, 2) == run(2, 1)


# design/aot-compilation-change-request.md A10, "Its bound": "the most calls of a function outstanding at once is the
# number of its calls in the block, or fewer where Bernstein's conditions order them": the most of its calls no two of
# which are ordered, the order's width.  That is the fewest chains covering them (R. P. Dilworth, A decomposition
# theorem for partially ordered sets, Annals of Mathematics 51, 1950), their number less a maximum matching of the
# bipartite graph with an edge from a to b for each a ordered before b (D. R. Fulkerson, Note on Dilworth's
# decomposition theorem for partially ordered sets, Proc. AMS 7, 1956), matched by Hopcroft and Karp's phases: a
# breadth-first search layers the graph from the unmatched vertices to the length of the shortest augmenting paths,
# then depth-first searches augment along vertex-disjoint paths of that length, until none is left (J. E. Hopcroft and
# R. M. Karp, An n^5/2 algorithm for maximum matchings in bipartite graphs, SIAM J. Comput. 2(4), 1973); the three as
# the turns' D.5 names them for chains (design/plan-to-bridge-change-request.md, "The turns").  after[a]: the b ordered
# after a (TT3)
def _width(after):
    count = len(after)
    nil, mate, partner, depth = count, [count] * count, [count] * count, [0] * (count + 1)

    def layer():
        queue = [a for a in range(count) if mate[a] == nil]
        for a in range(count):
            depth[a] = 0 if mate[a] == nil else math.inf
        depth[nil] = math.inf
        for a in queue:
            if depth[a] < depth[nil]:
                for b in after[a]:
                    if depth[partner[b]] == math.inf:
                        depth[partner[b]] = depth[a] + 1
                        if partner[b] != nil:
                            queue.append(partner[b])
        return depth[nil] != math.inf

    def augment(a):
        if a == nil:
            return True
        for b in after[a]:
            if depth[partner[b]] == depth[a] + 1 and augment(partner[b]):
                mate[a], partner[b] = b, a
                return True
        depth[a] = math.inf
        return False
    matched = 0
    while layer():
        matched += sum(augment(a) for a in range(count) if mate[a] == nil)
    return count - matched


# design/aot-compilation-change-request.md A10 over a traced block.  Its nodes, in the block's order, each read its
# argument nodes and write its own result ("Its reads and writes are its nodes' tensor arguments and results, which is
# what Bernstein's conditions take"); FX names every result a node of its own, so of Bernstein's three conditions on an
# earlier node i and a later j, that W(i) and R(j), R(i) and W(j), and W(i) and W(j) be disjoint (A. J. Bernstein,
# Analysis of programs for parallel processing, IEEE Trans. Electronic Computers EC-15(5), 1966), only the first can
# fail, where j reads i (TT2).  A node is ordered after another where a chain of such reads joins them: held[n], the
# nodes ordered before n.  Commuting calls (Rinard and Diniz): a chain of calls of one compiled function, each passing
# its result, read by the next call alone and once, as the next call's argument k, the receiver's place in every call
# of the chain, where _commutes(callee, k); a call follows one call of its chain, its first argument that qualifies.
# The chain's calls "can execute in any order without affecting the overall behavior" (section 2), each updating the
# receiver "atomically" (section 2), not at once: in `orders` they are unordered among themselves, the receiver's
# producer ordered before each and each before the readers of the chain's last result; for the bound they stay ordered
# (TT5).  `orders` is the order between the program's calls whole (TT6), shared by every
# entry (TT8).  A pass of the compilation, here as _complete is (TT9).  Each call's entry, its compiled function's
def _analyse(gm):
    nodes = list(gm.graph.nodes)
    callee = {node: _callee(gm, node) for node in nodes}
    calls = [node for node in nodes if callee[node] is not None]
    place = {node: i for i, node in enumerate(calls)}
    link, follower = {}, {}
    for b in calls:
        reads = []
        map_arg((b.args, b.kwargs), reads.append)
        for k, a in enumerate(b.args):
            if isinstance(a, Node) and a in place and callee[a] is callee[b] and len(a.users) == 1 \
                    and reads.count(a) == 1 and link.get(a, (a, k))[1] == k and _commutes(callee[b], k):
                link[b], follower[a] = (a, k), b
                break
    chains = []
    for a in calls:
        if a in follower and a not in link:
            chains.append([a])
            while chains[-1][-1] in follower:
                chains[-1].append(follower[chains[-1][-1]])
    step = {node: chain for chain in chains for node in chain}

    def ancestors(inputs):
        before = {}
        for node in nodes:
            before[node] = set().union(*({i} | before[i] for i in inputs(node)))
        return before

    def inputs(node):
        found = set(node.all_input_nodes)
        if node in link:
            receiver = step[node][0].args[link[node][1]]
            found = (found - {link[node][0]}) | ({receiver} if isinstance(receiver, Node) else set())
        for i in node.all_input_nodes:
            if i in step and step[i][-1] is i:
                found |= set(step[i])
        return found
    held, ordered = ancestors(lambda node: node.all_input_nodes), ancestors(inputs)
    orders = frozenset((place[a], place[b]) for b in calls for a in ordered[b] if a in place)
    commuting = tuple(tuple(place[node] for node in chain) for chain in chains)
    entries = {}
    for f in dict.fromkeys(callee[node] for node in calls):
        mine = [node for node in calls if callee[node] is f]
        entries[f] = Calls(tuple(place[node] for node in mine),
                           _width([[j for j, b in enumerate(mine) if a in held[b]] for a in mine]), orders, commuting)
    return {node: entries[callee[node]] for node in calls}


# design/aot-compilation-change-request.md A10: the program a block traces to.  It runs as Pathways' traced program
# does ("When the user wants to run a traced program, it calls the PATHWAYS client library which first assigns virtual
# devices to any computations that have not been run before, and registers the computations with the resource
# manager, triggering the servers to compile the computations", section 4.2), by torch.fx's Interpreter, which
# "executes an FX graph Node-by-Node" (torch v2.14.0 torch/fx/interpreter.py), in the block's order, which keeps every
# order constraint.  A call of a compiled function runs with its function's entry current (CALLING), so the backend's
# compile function, which Dynamo calls within that call where it compiles a graph of the function, carries the entry in
# that graph's IR (TT7).  CALLING: the current entries, the innermost last
CALLING = []


class _Program(torch.fx.Interpreter):
    def __init__(self, gm, entries):
        super().__init__(gm)
        self.entries = entries

    def run_node(self, n):
        if n not in self.entries:
            return super().run_node(n)
        CALLING.append(self.entries[n])
        try:
            return super().run_node(n)
        finally:
            CALLING.pop()


def program(block):
    tracer = Tracer()
    graph = tracer.trace(block)
    gm = torch.fx.GraphModule(tracer.root, graph)
    return _Program(gm, _analyse(gm)).run


# design/aot-compilation-change-request.md A1 (PyTorch 2 section 3.1: "a custom compile function which takes a
# torch.fx.Graph and a list of example inputs and returns a Python callable"), the placement-choosing function and the
# weighting function its parameters (A2, A3; design/compilation-coordination.md, "The torch backend"), neither with a
# default: torch.compile(f, backend=backend(placement, weights)).  Each call of the compile function takes the next
# ordinal of the program's compiled functions, the graph's `function` (design/compilation-coordination.md, "The
# appendix's revisions", N1), kept by every replan of that graph.  The placement operand is chosen once; the IR is built
# from it on the map as it is, and built again where the callable plans again.  A10's output is the entry of the traced
# program's call within which Dynamo compiles the graph (CALLING), carried by every IR built for it (TT7)
def backend(placement, weights):
    def compiler(gm, example_inputs):
        function, calls = next(FUNCTIONS), CALLING[-1] if CALLING else None
        operand = placement(gm, example_inputs)

        def build():
            ir = partition(gm, operand)
            ir.calls = calls
            lower(ir)
            ir.weights = list(weights(ir))
            return ir
        return serialize(build, function)
    return compiler
