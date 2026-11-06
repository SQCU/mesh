# The ahead-of-time compilation of a torch program (design/aot-compilation-change-request.md, A1 to A4 and A8 to A10;
# design/compilation-coordination.md, "The torch backend (mesh_compile.py, T)" and A-T1 to A-T4).  A function's FX
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
# 2026-11-05 (design/aot-compilation-change-request.md A4 and A10; design/compilation-coordination.md, "The revisions
# of abecd389", T4-1 to T4-22): every operator partitioned, the windowed and data formatting operators by GSPMD's halo
# exchange, so CollectivePermute is reached; A10's graph operations are rdma/mesh_graph.py's and a body's symbolic
# execution runs on torch.fx.Interpreter
# 2026-11-05 (design/aot-compilation-change-request.md, appendix item 7 at mesh 9c8c8865, and A4; design/compilation-
# coordination.md, "The revisions of e4a7398e"): every operator of an out variant runs it into its result's pages,
# every value this node makes is placed by mesh_plan's allocator, a sent value is made in its sent pages or copied
# there once, a publish waits for nothing (a torch MPSEvent's notification, rdma/mesh-torch.mm), no output is copied,
# and a node the placement maps no device returns meta tensors
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", points 1, 2 and 4 to 8,
# and "The revisions of 4e78f6f4"), reversing the notes above where they name a torch.compile backend, a function
# compiled at its first call, a plan call for each function and meta tensors at a node with no section: a program is
# traced whole (program, A10), each of its compiled functions (Function) captured by torch.export at its calls in
# program order before the program runs, the same at every rank, with nothing run eagerly between its calls (A1,
# corrected: "every function of a torch program is captured and compiled before the program runs, the same at every
# rank, and nothing runs eagerly between its compiled calls"); the program planned by one mesh_plan call (point 5:
# "there is one IR of the program and one serialized operand of it"); every value this node binds typed one run and
# bound over the program's one mapping (points 6 and 8); a collective's parts placed at their places in its values
# (point 7); and at a node with no section a call starts its plan and computes nothing (point 2).  What must stay: no
# placement and no weight chosen here
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

A = torch.ops.aten
# Y. Xu et al., GSPMD, arXiv 2105.04663, 2021, section 3.5: elementwise operators, each dimension preserved from the
# inputs to the output and back; an operand of fewer dimensions, or of size 1 on one the output's is larger on, is
# broadcast there (design/compilation-coordination.md, "The revisions of abecd389", T4-15)
ELEMENTWISE = {A.add, A.sub, A.mul, A.div, A.neg, A.relu, A.tanh, A.sigmoid, A.exp, A.gelu, A.silu}
# GSPMD sections 3.2 and 3.5: Dot, a generalized matrix multiply (an Einsum): torch's matrix products, linear, and
# einsum, whose equation is section 3.2's "string equation, e.g., ABC, ACD -> ABD" (T4-21)
DOT = {A.matmul, A.mm, A.bmm}
# GSPMD section 4.3, "Windowed operators. The most typical use case of halo exchange is for partitinoning window-based
# operators (e.g., Convolution, ReduceWindow)": torch's convolutions and transposed convolutions by their spatial
# dimensions, and its max and average pooling, ReduceWindow by max and by sum; section 4.3, "Halo exchange for data
# formatting ... after a Slice or Pad operator": torch's constant Pad (pad of mode "constant", and constant_pad_nd) and
# its Slice (slice, select and unsqueeze, the ATen operators a tensor's indexing by slices, ints and None is)
CONVOLUTION = {A.conv1d: 1, A.conv2d: 2, A.conv3d: 3}
TRANSPOSED = {A.conv_transpose1d: 1, A.conv_transpose2d: 2, A.conv_transpose3d: 3}
MAXIMUM = {A.max_pool1d: 1, A.max_pool2d: 2, A.max_pool3d: 3}
AVERAGE = {A.avg_pool1d: 1, A.avg_pool2d: 2, A.avg_pool3d: 3}
PAD = {A.pad, A.constant_pad_nd}
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 1, and "The
# revisions of 4e78f6f4"), reversing the notes above where they name torch's and torch.nn.functional's Python
# functions as Dynamo recorded them: each table holds ATen operators, the packets of torch.ops.aten, since a function
# is captured by torch.export, which "produces normalized operators in the functional ATen operator set" (torch 2.14.0
# torch/export/__init__.py, export).  A node's target is an OpOverload, of which "Each OpOverload object contains
# pointer to a specific operator overload, a pointer to the parent `OpOverloadPacket` object" (torch/_ops.py), so a
# node is matched by its target's `overloadpacket`.  PARAMETERS goes: an operator's parameters are its schema's
# arguments' names (torch._C.FunctionSchema `arguments`, each an Argument with its `name`, torch/_C/__init__.pyi;
# _names), as native_functions.yaml declares them (conv2d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2]
# stride=1, ...), max_pool2d(Tensor self, int[2] kernel_size, ...), torchgen/packaged/ATen/native).  INPLACE goes with
# _functional: the exported graph's operators are functional (partition's note).  narrow goes: an aliasing operator
# is no CompositeImplicitAutograd operator export keeps (torch/_export/utils.py _check_valid_to_preserve:
# "is_mutating_or_aliasing"), so a narrow reaches the graph as slice.  What must stay: GSPMD's rule for each operator
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
# one operator as this node's program calls it: FX's op, its target, and its arguments with each tensor operand the FX
# node whose value it takes (the node's own, its functional equivalent's, or its local configuration's, GSPMD section
# 4.1, "Static operator configurations")
Local = collections.namedtuple('Local', 'op target args kwargs')
# an operator's rule for GSPMD's two passes (sections 3.5 and 4): its kind; its propagation level (0 elementwise, 1
# lower, None no propagation, the general rule); its tensor operands in the order torch.fx's map_arg visits its
# arguments and their dimensions' labels, and its output's (None for the general rule); the operands propagation goes
# through; the contracting labels, whose sum a sharding alike in every operand leaves partial (section 3.2); its
# windows; `fused`, an operand added after the contraction (a bias, T4-13); `merged`, a Pad merged into this Slice
# (section 4.6, T4-17); `configure`, the local operators given the windowed labels sharded, each with its local output
# size
Rule = collections.namedtuple('Rule', 'kind level operands labels out propagate contracting windows fused merged '
                                      'configure')
# a windowed dimension (GSPMD section 4.3 and Appendix A.2): its label, the operand and the operand's dimension it
# windows, the window's extent (its dilated size) and stride, its low padding, for a transposed convolution its
# padding and its kernel's dilated reach less one (A.2's base dilation, T4-10), and the value outside the operand
# (section 4.1's masking by "the identity value of the reduction", T4-18)
Window = collections.namedtuple('Window', 'label operand dim extent stride low transposed fill')


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
# revisions of f3f7cf46", TT7).  `locals`: each operator's Local as this node's program calls it; `shapes`: the local
# shapes of the values a halo exchange makes (its halos, their arrivals, its regions), which no node's sharding gives
# (design/aot-compilation-change-request.md A4; design/compilation-coordination.md, "The revisions of abecd389");
# `reduced`: each AllReduce by the key of the partial sums it reduces
# 2026-11-05 (design/aot-compilation-change-request.md A4: "A node of the program to which the placement operand maps
# no device (A2) compiles and plans each function as every node does, with no section of its own.  It declares the
# function's transfers as every rank does, and its part of each call is the relaying its routes give it"; NCCL:
# "Collective operations have to be called for each rank (hence CUDA device), using the same count and the same
# datatype, to form a complete collective operation.  Failure to do so will result in undefined behavior, including
# hangs, crashes, or data corruption" (NCCL User Guide, "Collective Operations"); design/compilation-coordination.md,
# "The revisions of b5a3c6e6", ND): the refusal of a node that is no device of the placement operand ("map index %d,
# this node, is no device of the placement operand (A2)") goes; `device` is None there.  Its program, declared
# transfers, weights and dependences are built as at every rank: _partition reads no device, `transfer` gives this node
# a value only where its map index is an end, which no device's is, and the lowerings and _dependences declare every
# device's transfers and events, so mesh_plan's operands are every rank's; no lowering gives it a role (each tests
# ir.device against its group), and serialize runs none of its items.  What must stay: at most one device a node
# (_mesh), every operand of mesh_plan the same at every rank
# 2026-11-05 (design/aot-compilation-change-request.md, appendix item 7: "place every value of the node's partitioned
# graph by the one allocator.  A view (a narrow, a slice) is its base's pages, and its base lives as long as it does";
# "run each operator as its out variant into its result's placed pages"; "have each sent value's producer make it in its
# sent pages, so nothing copies it there"; design/compilation-coordination.md, "The revisions of e4a7398e", OV and PV):
# `placed` (the keys over the plan's pages, an AllReduce's result and the slices, pieces and regions _reshard and _region
# added) goes, for `over`: each key's set of the keys whose pages it may lie over (_over), a key that makes pages of its
# own over itself.  `metas`: each operator's result on PyTorch's meta device (_forms); `forms`: each operator's out
# variant, a torch OpOverload, or 'functional' where it makes a result of its own and has none, absent where it may be a
# view of its operands (_forms); `producer`: the item that makes each key; `inplace`: the AllGather operands made in
# their sent pages by their producer, every rank's (_over).  This node's: `homes`, the index in `values` of the pages
# each key it makes is written into (_place); `copies` and `publishes`, by item, the sent values that item copies once
# into their pages and publishes, and those it made there and publishes (_place).  What must stay: every operand of
# mesh_plan the same at every rank
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", points 1, 5 and 7, and
# "The revisions of 4e78f6f4"): `gm` is a compiled function's exported graph (its ExportedProgram's module, _Export),
# `calls` its A10 entry, never None, and the IR one struct mesh_function of the program's one plan call (program).
# `inplace`, `copies` and `reduced` go, for each range's place (point 7: "name, in each declared transfer, the pages
# of its destination value that it lands in"): `blockbytes`, a block's bytes (block x pgsz, mesh.h struct hdr), at
# which a part of a value begins; `inside`, the AllGather operands made in place in their results (_all_gather),
# every rank's; `blocks`, by key, the blocks of its pages its ranges hold, every rank's (_part, _landing).  This
# node's: `at`, each declared transfer's place, the key whose value it is sent from or lands in and the element of that
# value it begins at (the lowerings); `added`, the values of no key of the program (a part copied once, a reduction's
# received parts); `sends`, by collective item, the ranges it releases before it waits, each with the copy that makes
# it where one does; `publishes`, by item, the ranges released after it; `homes`, each placed key's value, element
# offset and shape (_place).  What must stay: every operand of mesh_plan the same at every rank
class IR:
    def __init__(self, gm, placement, nodes, blockbytes):
        self.gm, self.placement, self.weights, self.blockbytes = gm, placement, None, blockbytes
        self.order = list(gm.graph.nodes)
        self.shape, self.coords, self.map = _mesh(placement, nodes.count)
        here = [d for d, index in enumerate(self.map) if index == nodes.self]
        self.device, self.index, self.serial, self.identity = here[0] if here else None, nodes.self, nodes.serial, 0
        self.sharding, self.program, self.roles, self.events = {}, [], {}, []
        self.declared, self.dependences, self.values, self.layouts = [], [], [], []
        self.made, self.calls, self.locals, self.shapes = set(), None, {}, {}
        self.metas, self.forms, self.over, self.producer = {}, {}, {}, {}
        self.homes, self.at, self.inside, self.blocks, self.added = {}, {}, {}, {}, {}
        self.sends, self.publishes = {}, {}

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
    # 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 7, and "The
    # revisions of 4e78f6f4"): no value is made for a range; each declared transfer's `value` and `first` are this
    # node's place of it, set where its key is placed (_place; mesh-collective.h struct mesh_declared `first`).  Why:
    # design/plan-to-bridge-change-request.md R2, corrected: "A range is not a value of its own".  What must stay: one
    # identity a range, its bytes its local shape's
    def transfer(self, node, shape, source, destinations, kind):
        size, ends = _value(node).dtype.itemsize * math.prod(shape), [self.map[d] for d in [source] + destinations]
        self.identity += 1
        transfers = {}
        for d in destinations:
            transfers[d] = len(self.declared)
            self.declared.append(mesh.Declared(ends[0], self.map[d], self.identity, kind, mesh.ABSENT, 0, size))
        return transfers

    # one value of this node (mesh-collective.h struct mesh_value), MESH_EXECUTION: a routed one in the window (`wire`
    # 1), a local one outside it (0) with its first and last positions on this node's program order, and its layout,
    # the element type, the shape and the device of `node`'s tensors, which its copies' tensors are bound on (K1).  Its
    # index in `values`.  A tensor on the meta device has no binding (_bind), and only a node the placement operand maps
    # no device takes meta tensors (A4), where no value is made: refused, naming it
    # 2026-11-05 (design/aot-compilation-change-request.md, appendix item 7: "place every value of the node's partitioned
    # graph by the one allocator"; A4: "Its calls return tensors on PyTorch's meta device"; design/compilation-
    # coordination.md, "The revisions of e4a7398e", PV-1 and MD-2): new, the routed values' form in `transfer` and the
    # local values' (_place).  What must stay: a routed value MESH_EXECUTION in the window, of its range's bytes
    # 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", points 6 and 8, and
    # "The revisions of 4e78f6f4"), reversing MD-2's refusal of a meta tensor, which no node takes now: `run` 1 on
    # every value, each bound as a tensor (_bind), "a torch MPS operator's operand" seen "in one strided progression of
    # virtual addresses" (design/aot-compilation-change-request.md, appendix item 7, "Which operands are seen as one
    # run": "MPSGraph takes an `MTLBuffer` with a shape, and optionally "rowBytes for the fastest moving dimension""),
    # and a CPU tensor one storage with sizes and strides (torch 2.14.0 include/c10/core/TensorImpl.h, `Storage
    # storage_`); its layout the element type, its elements and its device, every key over it a run of its elements
    # (_place).  Its place is the mapping mesh_plan writes (mesh-collective.h struct mesh_value `view`).  What must
    # stay: a value MESH_EXECUTION, routed in the window, local outside it with its positions
    def value(self, dtype, numel, device, wire, begin, end):
        self.values.append(mesh.Value(section=mesh.Section(bytes=dtype.itemsize * numel), kind=mesh.EXECUTION,
                                      wire=wire, begin=begin, end=end, run=1))
        self.layouts.append((dtype, numel, device))
        return len(self.values) - 1


# J. Ansel et al., PyTorch 2, ASPLOS 2024, sections 3.5 and 5.1: a node's fake tensor, "the metadata of a tensor without
# its actual data", its sizes propagated by meta functions.  A1: shapes are compiled, so a symbolic size is refused.  A
# tensor on the CPU or on torch's MPS device is bound to the placed pages (torch.from_dlpack of a kDLCPU or kDLMetal
# tensor, design/compilation-coordination.md, K1); one on any other device is refused, naming it
# 2026-11-04, owner T2 (K1; design/aot-compilation-change-request.md, appendix item 7): A-T4's refusal of a tensor not
# on the CPU goes for MPS, whose tensors torch v2.14.0 builds over existing memory by DLPack (_dlpack_tensor).  What
# must stay: the refusal of a device with no binding, and of a symbolic size
# 2026-11-05 (design/aot-compilation-change-request.md A4: at a node the placement operand maps no device, "Its calls
# return tensors on PyTorch's meta device ... and its next compiled function is traced with the shapes and types every
# rank's is"; design/compilation-coordination.md, "The revisions of e4a7398e", MD-2): the meta device is taken, where
# it was refused, since that node's next function is traced from meta tensors and builds every rank's IR; its tensors
# are bound nowhere (IR.value refuses one at a device of the placement).  What must stay: the refusal of any other
# device with no binding, and of a symbolic size
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", points 1 and 2, and "The
# revisions of 4e78f6f4"), reversing MD-2: a node's value is its exported graph's fake tensor, `meta['val']` (torch
# 2.14.0 torch/fx/experimental/proxy_tensor.py set_meta: `proxy.node.meta["val"] = extract_val(...)`), its device its
# fake device, the device the call's arguments are on (torch/_subclasses/fake_tensor.py FakeTensor: "FakeTensor extends
# MetaTensors to also carry an additional `fake_device` which tracks devices that would have been used"), where it was
# Dynamo's `example_value`; the meta device goes, since no node computes on meta tensors (point 2).  What must stay:
# the refusal of a device with no binding, and of a symbolic size
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


# the local shape of the value of a key of this node's program: a halo exchange's own (ir.shapes), else its node's
# local value in the sharding the key carries (GSPMD section 3.1's tile)
def _shape(ir, key):
    return ir.shapes[key] if key in ir.shapes else ir.local(key[0], key[1])


# a key's element type and local shape: an operator's result's as it ran on the meta device (_forms), else its node's
# type and the key's local shape (2026-11-05, "The revisions of 4e78f6f4": _place's and _bind's one reading of them)
def _layout(ir, key):
    if key in ir.metas:
        return ir.metas[key].dtype, tuple(ir.metas[key].shape)
    return _value(key[0]).dtype, tuple(_shape(ir, key))


# GSPMD section 3.1, the device mesh: its shape and each device id's coordinates; Alpa section 4.1, the mapping of its
# devices onto map indices, one device a node.  The nested sequences are a tree whose vertices are the places (index
# paths) in it, walked by the graph module's breadth-first walk (rdma/mesh_graph.py walk): the shape along the first
# children, then every place to the shape's depth, each checked in the walk's order, a parent before its children
# 2026-11-05 (design/aot-compilation-change-request.md A10, "Its graph operations are the graph module's": "The same
# holds wherever mesh's code walks a graph"; design/compilation-coordination.md, "The revisions of abecd389", T4-3):
# the recursive walk and the loop down the first children inlined here go, for mesh_graph.walk.  What must stay: the
# shape, the coordinates and each refusal as before
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


# a parameter given as one int or a sequence, as k ints; the parameters an operator's call gives, by name (`names` its
# parameters in order, PARAMETERS); the call with some replaced, each where the call gives it, else by keyword; the
# names of the parameters its tensor operands are given as, in map_arg's order, None for one inside a sequence; an ATen
# operator's parameters, its schema's arguments' names in order (2026-11-05, "The revisions of 4e78f6f4": where
# PARAMETERS held them by hand; torch 2.14.0 torch/_C/__init__.pyi, FunctionSchema `arguments`, Argument `name`)
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


# GSPMD sections 3.2 and 3.5: a Dot's labels, batch, contracting ('k') and non-contracting ('n', 'm') dimensions, as
# torch.matmul's operands give them (a vector's one dimension contracted; an operand's batch dimensions aligned to the
# output's from the last, one of size 1 under a larger one broadcast and of a label of its own, T4-21); propagation
# through an operand with no broadcast dimension (an absent batch dimension is an Einsum operand's absent letter, as a
# matrix's in A-T2's forms; one of size 1 broadcast is an XLA Broadcast, T4-15)
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


# GSPMD section 3.2, the Einstein Summation: "An Einsum can be expressed as a string equation, e.g., ABC, ACD -> ABD,
# where A is an embarrassingly parallel batch dimension in both operands and the output, C is a contracting dimension
# that only exist in the operands and will be sum-reduced, while B and D are non-contracting dimensions that exist in
# one operand and are inherited by the output": each operand's letters its labels, the output's its own (torch.einsum's
# implicit output, without "->", the letters appearing once, in alphabetical order), a letter in every operand and not
# in the output contracting; an equation with "...", a letter repeated in an operand or the output, or a letter of two
# sizes (broadcast) is none of these (None) (T4-21)
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


# GSPMD section 4.3, windowed operators, and section 4.4: Convolution "allows the user to combine spatial partitioning
# and feature partitioning in the same operator".  Labels: the batch ('n', where the input is batched), the input
# feature ('ci', contracted with the weight's, as a Dot's) and output feature ('co', the weight's and the bias's), and
# each spatial dimension ('s', i) a window of the input onto the output's; the kernel's dimensions of labels of their
# own, so a kernel is held whole ("If the RHS is replicated, window dilation only affects the effective window size when
# the operator is partitioned based on its LHS", Appendix A.2; of a partitioned RHS, "We skip the implementation
# details").  A window's extent is its dilated size d(k - 1) + 1, its stride and low padding the operator's (padding
# 'valid' none, 'same' half of d(k - 1)).  A transposed convolution is a convolution of the input dilated by the stride
# (A.2: "Base dilation is the dilation factor of the LHS"), its window as A.2's base dilation gives it (T4-10).  With
# groups the feature dimensions are labels of their own, held whole and the output's computed whole and sliced, and so
# with padding 'same' of an odd d(k - 1) the spatial dimensions (T4-12); a bias, added after the contraction, `fused`
# (T4-13).  Each sharded spatial dimension's padding (and output padding) 0 in the local operator: its halo region
# holds the padding (GSPMD section 4.1, "Static operator configurations"; Appendix A.2)
# 2026-11-05 ("The revisions of 4e78f6f4"): the operator an ATen convolution, its parameters its schema's (_names);
# conv2d.padding's padding the string "valid" or "same", as torch.conv2d's.  What must stay: the rule
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


# GSPMD section 4.3, ReduceWindow: torch's max pooling (ReduceWindow by max, masked by its identity, -inf or the
# integer type's least, section 4.1) and average pooling (by sum, masked by 0); the leading dimensions preserved, each
# spatial dimension ('s', i) a window, of extent d(k - 1) + 1 (max; avg has no dilation, k), its stride the operator's
# (the kernel's where None or empty) and its low padding the operator's.  An average's dimension is a window only where
# its divisor does not depend on the window's place: a divisor_override, or no ceil_mode with count_include_pad or no
# padding there; another of labels of its own, held whole (T4-11).  return_indices gives a pair, no tensor (None).  Each
# sharded window's padding 0 in the local operator
# 2026-11-05 ("The revisions of 4e78f6f4"): the operator ATen's max_pool1d to 3d or avg_pool1d to 3d, whose operand is
# `self` and which have no return_indices (max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], ...),
# native_functions.yaml); avg_pool1d has no divisor_override.  What must stay: the rule
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


# GSPMD section 4.3, "Halo exchange for data formatting ... after a Slice or Pad operator, the shape of the tensor
# changes, and so do the boundaries between partitions.  This requires us to realign the data on different partitions,
# which can be handled as a form of halo exchange (Figure 5b)": a constant Pad's padded dimension d a window of extent
# 1 and stride 1, its low padding the pad's, outside the operand the pad's value; a Slice's sliced dimension d one of
# extent 1, its stride the slice's step and its low padding the slice's start negated (Python's slice.indices).  Every
# other dimension preserved; an int index's dimension, absent from the output, held whole; None's of its own.  A Pad of
# another mode (reflect, replicate, circular), no XLA Pad, holds its padded dimensions whole.  Each sharded window's
# pad 0 in the local Pad, its slice the step alone, its narrow from 0 the local length (a narrow's dimension a window
# even where it keeps the dimension whole, so its local length is its shard's)
# 2026-11-05 ("The revisions of 4e78f6f4"): the Pad ATen's pad(Tensor self, SymInt[] pad, str mode="constant", float?
# value=None) or constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0); the Slice ATen's slice(Tensor(a) self,
# int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1), a window of its dimension, its local operator's start
# and end None (the whole local region); select(Tensor(a) self, int dim, SymInt index), its dimension held whole and
# absent from the output, as an int index's was; unsqueeze(Tensor(a) self, int dim), a dimension of its own, as
# None's was (native_functions.yaml); a narrow reaches the exported graph as slice (the tables' note).  What must stay:
# each dimension's rule as the getitem's was
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


# design/aot-compilation-change-request.md A2 and A4: "The partitioner reaches them by partitioning every operator, as
# A2 states ("propagates them to every operator")".  An operator's rule (Rule), from its Local: GSPMD's for an
# elementwise operator (section 3.5), a Dot or Einsum (sections 3.2 and 3.5), a windowed operator (sections 4.3 and 4.4,
# Appendix A.2) and a constant Pad or a Slice (sections 4.3 and 4.6); for any other operator, and these where their
# operands or arguments are of no form those rules read, the general rule: "GSPMD always produces a valid partitioned
# graph regardless of what sharding annotations are provided.  If the provided shardings are not the typical supported
# cases, the partitioner will perform resharding on the inputs and/or outputs" (section 4.5), and every operator
# supports the replicated sharding, in which "All devices have the same full data" (section 3.1), so its tensor
# operands are resharded to replicated, it runs whole at every node, and its output is resharded from replicated to its
# sharding (section 4.5's DynamicSlice); since which of its dimensions it preserves is not known, nothing propagates
# through it (section 3.5, "Preserved dimensions in operators") (T4-5).  A module the program calls (call_module) and
# an operator of a value that is no tensor (a tuple a later getitem takes apart) take the general rule (T4-20)
# 2026-11-05 (design/aot-compilation-change-request.md A4: "The partitioner's refusal of an operator, "GSPMD gives no
# rule here for it" (rdma/mesh_compile.py `_operator`), is false for these"; design/compilation-coordination.md, "The
# revisions of f3f7cf46", stop 2): the refusal of every operator but elementwise ones and three Dot forms goes, and so
# do the refusals of a broadcast elementwise operator and of a tensor operand outside the positional arguments (A-T2):
# the windowed and data formatting operators take GSPMD's rules, the rest the general rule (design/compilation-
# coordination.md, "The revisions of abecd389", T4-5, T4-15, T4-21).  What must stay: the elementwise and Dot
# operators' rules as before (their labels renamed, their partitions the same)
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 1, and "The
# revisions of 4e78f6f4"): an operator's rule is chosen by its ATen packet (the tables' note); a node whose target is
# no OpOverload (an operator.getitem of a result of several tensors, a higher-order operator) takes the general rule;
# the call_module form goes, since the exported graph holds none ("call_module is not valid", torch 2.14.0
# torch/_export/verifier.py).  What must stay: each operator's rule as before
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


# GSPMD section 4.6, "Pre-processing": "Similar optimizations include merging a sequence of Pad and Slice operators,
# which is important to partition the pipelined program in Section 3.3 where the shifting using Pad and Slice can be
# done with a single CollectivePermute" (section 3.3's "pad_left(state, 1)[:-1]").  A constant Pad whose one reader is a
# Slice: the Slice's rule over the Pad's operand, each dimension's window the two composed (the Slice's start in the
# padded tensor less the Pad's low padding, outside the operand the Pad's value), the Pad's local operator run before
# the Slice's, its value read by the Slice alone (T4-17)
def _merged(pad, padding, slicing):
    pads, fill = {w.label: w for w in padding.windows}, padding.windows[0].fill if padding.windows else 0
    windows = [w._replace(low=w.low + (pads.pop(w.label).low if w.label in pads else 0), fill=fill)
               for w in slicing.windows]
    windows += [w for label, w in pads.items() if label in slicing.out]

    def configure(sharded):
        return tuple(padding.configure(sharded)) + tuple(slicing.configure(sharded))
    return slicing._replace(operands=padding.operands, windows=tuple(windows), merged=pad, configure=configure)


# GSPMD section 3.5, sharding completion: it "runs the propagation pass on the entire graph in multiple iterations, and
# alternates between forward propagation (input to output) and backward propagation (output to input)"; it "preserves
# initial user annotations" and "changes the sharding on a tensor only when it finds a more fine-grained sharding
# (possibily by combining with existing compatible sharding)", so it reaches a fixed point.  Priorities: "Elementwise
# operators are assigned the highest priority to propagate through in both directions" and "Operators like Dot ... are
# assigned lower priority": level 0 propagates through elementwise operators alone to its fixed point, level 1 through
# Dot as well (design/compilation-coordination.md, A-T2), and through the windowed and data formatting operators, whose
# consistent shardings still exchange halos (section 4.3), so not of the priority section 3.5 gives because "there is
# no communication if their inputs/outputs are sharded consistently" (T4-6).  An operator's operands are taken in its
# rule's order, those it propagates through alone (Rule `propagate`).  A sharding is carried onto no dimension its mesh
# dimension does not divide: section 4.1 pads such a dimension, which partition refuses (T4-9).  Its rounds are the
# graph module's (rdma/mesh_graph.py propagate), each operator's edges the operands its rule propagates through at the
# level, its visits the rule's carry (_carry) and refinement (_merge), forward into the operator's output and backward
# into the operand
# 2026-11-05 (design/aot-compilation-change-request.md A10: "The same holds wherever mesh's code walks a graph"; the C9
# hook's remark on cc4a3e12, "a fixed-point traversal of the dataflow graph's operator edges ... it moves into the
# graph module"; design/compilation-coordination.md, "The revisions of abecd389", T4-3): the rounds' loop over the
# graph's order and its reverse goes, for mesh_graph.propagate, where the coordinator's ruling on C9's remark of
# d9efd96b kept the pass whole here (A2's own pass: its rules stay here).  What must stay: the levels, the order of the
# visits and each refinement as before
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


# GSPMD section 4.5, resharding: it "uses AllGather to replicate data along sharded dimensions, AllToAll to switch
# sharded dimensions, CollectivePermute to change device order, and DynamicSlice to shard replicated dimensions", and
# "could use multiple steps of resharding to reach the desired sharding": the steps in the order it names them
# (design/compilation-coordination.md, A-T9), each mesh dimension's AllGather, then each one's AllToAll into a
# dimension no mesh dimension holds (one whose dimension is held is refused, naming it), then each one's DynamicSlice (a
# local slice, no communication; a slice of a value over the plan's pages lies over them too).  One device mesh orders
# every tensor's devices alike, so no CollectivePermute arises.  A placeholder's whole value is held at every node,
# since torch passes the whole input, so its reshard starts from it and is slices alone (the coordinator's ruling on
# review V2's finding 7); so is a get_attr's, the module's attribute at every node (T4-20).  Each step an item of this
# node's program: (kind, the keys it reads, the key it makes, its operand), none repeated whose key an earlier one made
# (that value stands through the execution).  The key of `node`'s local value in `want`
# 2026-11-05 (design/aot-compilation-change-request.md, appendix item 7: "A view (a narrow, a slice) is its base's
# pages, and its base lives as long as it does"; design/compilation-coordination.md, "The revisions of e4a7398e", OV):
# a slice no longer adds its key to `placed`; _over gives every view its base's pages.  What must stay: the steps
def _reshard(ir, node, have, want):
    def step(item):
        if item[2] not in ir.made:
            ir.made.add(item[2])
            ir.program.append(item)
    if node.op in ('placeholder', 'get_attr'):
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
            have = after
    return node, have


# GSPMD Appendix A.2, the windows of `parts` partitions of a windowed dimension, each n of the operand and o of the
# output: for partition q, the operand's region it reads, from starts[q] for `length` (in the operand's coordinates, so
# before 0 or from its size on outside it), where its local output begins in its local operator's (after[q]), and the
# largest left and right halos ("we calculate the maximum sizes of left and right halos across partitions and perform
# the halo exchange of the maximum size", Figure 9b's steps 1 and 2).  A window of stride s, extent e and low padding
# p: output j reads the operand from j s - p for e, so partition q from q o s - p for (o - 1) s + e, and its local
# operator, of no padding there, gives its o outputs from 0; the halo sizes, linear in q, are A.2's "linear function of
# the partition ID".  A transposed convolution, A.2's base dilation (window stride 1 over the operand dilated by s, low
# padding d(k - 1) - p): output j takes operand i where i s + t d = j + p, so partition q reads from
# floor((q o + p - d(k - 1)) / s), and its local operator, of no padding, gives global output j at j + p - start s;
# "we can use the maximum low padding on all partitions to ensure that each partition calculates all required windows,
# then perform a DynamicSlice on the output of the partitioned operator to remove unnecessary data" (A.2, its second
# case; T4-10): `length` the most any partition reads and the least that holds its o outputs
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


# GSPMD section 4.3, halo exchange with dynamic bounds: "We use the CollectivePermute operator to exchange halo data
# between partitions", Figure 9b's sequence along one windowed dimension of the value of `key`, for `consumer`, along
# mesh dimension m: (1) each partition's left halo the right end of its left neighbours' shards and its right halo the
# left end of its right neighbours', each piece a narrow of the sender's local value sent by one CollectivePermute whose
# source_target_pairs are every partition and the one j places from it (a halo wider than a shard one CollectivePermute
# a distance j, its piece within that shard, T4-7), a partition no pair targets holding XLA's zeros; (2) "Concatenate
# exchanged left and right halos" with the value; (3) "DynamicSlice on the region actually needed" (section 4.3:
# "GSPMD uses maximum halo size across partitions, then uses DynamicSlice to remove excessive data in the halos"), at
# the partition's own offset; (4) "Mask out invalid regions with the identity value": "some partitions may include
# garbage values (e.g., halos from out-of-range input data), so we apply masking as described in Section 4.1", "a
# Select according to a mask calculated by comparing Iota and an offset based on PartitionId", the masked value the
# window's (T4-18).  So the operator's padding is the masked region's (Figure 5a: Padding, DynamicSlice,
# CollectivePermute, Concat, Convolution).  Every operand's halos are exchanged so, a placeholder's after its reshard
# (T4-8).  Each step an item: ('narrow', (key,), the piece, (dim, start, length)), the CollectivePermute's, and
# ('window', the pieces and the value in order, the region, (dim, m, offsets by partition, length, starts by
# partition, the operand's size, the fill or None)).  A piece or region of a value over the plan's pages lies over
# them as long as it is a view (as _reshard's slices).  The region's key and the partitions' local output offsets
# 2026-11-05 (design/aot-compilation-change-request.md, appendix item 7; design/compilation-coordination.md, "The
# revisions of e4a7398e", OV): a piece or a region no longer adds its key to `placed`; _over gives a view its base's
# pages, and a region made by its concatenation pages of its own.  What must stay: the steps
def _region(ir, consumer, key, window, m, o):
    node, shape = key[0], _shape(ir, key)
    dim, parts, size = window.dim, ir.shape[m], _value(key[0]).shape[window.dim]
    n = shape[dim]
    starts, length, after, left, right = _extent(window, parts, n, o)
    reads = []
    for side, width in (('left', left), ('right', right)):
        pieces = []
        for j in range(1, -(-width // n) + 1):
            part = min(n, width - (j - 1) * n)
            piece, arrival = (node, key[1], 'halo', consumer, dim, side, j), (node, key[1], 'arrival', consumer, dim,
                                                                               side, j)
            ir.shapes[piece] = ir.shapes[arrival] = tuple(part if i == dim else s for i, s in enumerate(shape))
            ir.program.append(('narrow', (key,), piece, (dim, n - part if side == 'left' else 0, part)))
            pairs = tuple((p, p + j) if side == 'left' else (p + j, p) for p in range(parts - j))
            ir.program.append(('collective', (piece,), arrival,
                               Collective('collective-permute', node, (m,), pairs, len(ir.program))))
            pieces.append(arrival)
        reads += pieces[::-1] + [key] if side == 'left' else pieces
    region = (node, key[1], 'region', consumer, dim)
    fill = window.fill if any(a < 0 or a + length > size for a in starts) else None
    ir.shapes[region] = tuple(length if i == dim else s for i, s in enumerate(shape))
    ir.program.append(('window', tuple(reads), region, (dim, m, tuple(a - q * n + left for q, a in enumerate(starts)),
                                                        length, tuple(starts), size, fill)))
    return region, after


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
# 2026-11-05 (design/aot-compilation-change-request.md A4: "The partitioner reaches them by partitioning every
# operator"; design/compilation-coordination.md, "The revisions of abecd389", T4-5, T4-13, T4-14, T4-17, T4-20, T4-21,
# T4-22): every operator is partitioned by its rule (_operator), the general rule's operands replicated and its output
# sliced; a windowed dimension a mesh dimension shards takes its halos by CollectivePermute (_region) before the local
# operator, configured by its rule, and a transposed convolution's output its DynamicSlice after; a contracting
# dimension the operands shard unalike, where the output leaves its mesh dimension unused, is gathered as Figure 6's
# (GSPMD section 4.5's AllGather), where it was refused ("no rule here"), and so is one whose operator adds a bias
# after it, and an Einsum's second contracting dimension sharded alike, so `partial` holds one mesh dimension at most
# (TA3); an output dimension no operand gives (a label of its own) is computed whole and sliced after (section 4.5's
# DynamicSlice); a get_attr's value is the module's attribute; ir.reduced is keyed by the partial sums' key, which the
# AllReduce reads.  What must stay: the AllReduce, the ReduceScatter and the AllGather of a contracting dimension as
# before
# 2026-11-05 (design/aot-compilation-change-request.md, appendix item 7; design/compilation-coordination.md, "The
# revisions of e4a7398e", OV): an AllReduce's result no longer added to `placed`; _over makes it a key over the plan's
# pages, its own.  What must stay: the program's items as before
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 1, and "The
# revisions of 4e78f6f4"): an input or attribute that is no tensor (an exported graph's placeholder of an int, a
# higher-order operator's subgraph) has no sharding and is taken whole; an output is its node's value, since the
# exported graph's operators are functional (partition's note), where it was the last in-place writer's; an
# AllReduce's record is its item's alone (`reduced` goes: _place reads the item).  What must stay: the program's items
# as before
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
            keys = [_reshard(ir, o, ir.sharding[o], tuple(w)) for o, w in zip(rule.operands, wants)]
            sharded, cuts = {}, []
            for window in rule.windows:
                at = out.index(window.label)
                if local[at] >= 0:
                    sharded[window.label] = ir.local(node, local)[at]
                    keys[window.operand], after = _region(ir, node, keys[window.operand], window, local[at],
                                                          sharded[window.label])
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


# design/aot-compilation-change-request.md A2: GSPMD's two passes by the placement operand, "sharding completion and
# per-operator partitioning" (section 3), on the map as mesh_nodes_read gives it.  An unannotated tensor that
# propagation gives no sharding stays replicated (section 3.1, "All devices have the same full data"); an annotation
# on a mesh dimension of one device reads as -1, a dims_mapping naming no mesh dimension of more being replicated
# (section 3.1; design/compilation-coordination.md, A-T10); a dimension its mesh dimension does not divide is refused
# (section 4.1 pads it and masks the padding, which the contract does not give).  Each operator's Local is its
# functional equivalent's (_functional), each read after an in-place write reading the writer's result, as AOTAutograd's
# functionalization replaces "operations that perform mutation" (PyTorch 2 section 3.9; T4-16); a Pad its Slice alone
# reads is merged into it (section 4.6, _merged).  An operator's value that is no tensor (a tuple, a size) has no
# sharding (None): it is computed whole at every node by the general rule (T4-20)
# 2026-11-05 (design/aot-compilation-change-request.md A4, "partitioning every operator"; design/compilation-
# coordination.md, "The revisions of abecd389", T4-16, T4-17, T4-20): the refusal of a get_attr or call_module node
# (A-T2) goes, and the in-place refusal goes but for a write PyTorch 2 section 3.9's functionalization is not taken for
# here.  What must stay: the placement operand's checks and the refusal of an undivided dimension
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 1, and "The
# revisions of 4e78f6f4"), reversing the notes above where they take each operator's functional equivalent: the graph
# is a compiled function's exported graph, whose operators are functional, so _functional and the refusal of an
# in-place write go.  torch.export.export gives its training IR, verified by TrainingIRVerifier, which admits a
# mutating operator ("if self.dialect != "TRAINING" and not is_functional(op): raise SpecViolationError(f"operator
# '{op}' is not functional")", torch 2.14.0 torch/_export/verifier.py; torch/export/_trace.py _export_for_training,
# `verifiers=[TrainingIRVerifier]`), and its run_decompositions with an empty table ("If empty, we don't decompose any
# operator", torch/export/exported_program.py) gives the ATen dialect, verified by Verifier, every operator functional
# (_Export).  Every node is a placeholder, a get_attr (the module's parameters and buffers, "a self contained
# GraphModule with all the parameters/buffers inlined", ExportedProgram.module), a call_function or the output.  The
# map read is the program's one (program's `make`), so every function of one plan call is partitioned on one map, and
# the block's bytes are its region's (mesh.h struct hdr).  What must stay: the placement operand's checks and the
# refusal of an undivided dimension
def partition(gm, placement, nodes, blockbytes):
    if not isinstance(placement, Placement):
        raise TypeError('the placement function returned %r, not a Placement (A2)' % (placement,))
    ir, operators, annotated = IR(gm, placement, nodes, blockbytes), {}, set()
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
    _complete(ir, operators, annotated)
    for node, sharding in ir.sharding.items():
        for size, m in zip(_value(node).shape if sharding is not None else (), sharding or ()):
            if m >= 0 and size % ir.shape[m]:
                raise NotImplementedError('%s: a dimension of %d over a mesh dimension of %d (GSPMD section 4.1 pads '
                                          'it, which the contract does not give)' % (node.name, size, ir.shape[m]))
    _partition(ir, operators)
    _forms(ir)
    _over(ir)
    return ir


# an operator as this node's program calls it (Local): a Tensor method on its first operand, a module the program
# calls by its GraphModule's submodule, any other by its target
# 2026-11-05 (design/compilation-coordination.md, "The revisions of e4a7398e", OV-1): new, _run's three forms of the call,
# which _forms makes too, on meta tensors.  What must stay: each form as _run made it
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 1, and "The
# revisions of 4e78f6f4"): an exported graph's operator is called by its target, an OpOverload or operator.getitem,
# the Tensor method's and the module's forms going (the exported graph holds neither: torch 2.14.0
# torch/_export/verifier.py); _overload, OPERATORS and BINDINGS go, since the target is the ATen overload itself.  What
# must stay: _run's and _forms' one form of the call
def _call(local, args, kwargs):
    return local.target(*args, **kwargs)


# design/aot-compilation-change-request.md, appendix item 7: "run each operator as its out variant into its result's
# placed pages", ExecuTorch's out variant ("Instead of allocating returned tensors in kernel implementations, an
# operator's out variant will take in a pre-allocated tensor to its out kwarg, and store the result there", ExecuTorch,
# "Concepts").  Each operator item ('call') of this node's program run on PyTorch's meta device (A4: "Most operations
# can be performed on meta tensors, producing new meta tensors that describe what the result would have been if you
# performed the operation on a real tensor"), by _call on its operands' meta tensors (a key's local shape and element
# type, _shape; an earlier operator's result, `metas`; a value that is no tensor, its example value as Dynamo recorded
# it, each tensor in it a meta tensor of its metadata): its result, `metas`, and its form, `forms`:
# - the out variant of its overload (_overload) by torch's own to_out_variant ("Given a functional operator overload,
#   return its corresponding out variant", torch v2.14.0 torch/_library/_out_variant.py, by "Compar[ing] two schemas by
#   their non-mutable arguments"; inductor lowers an operator to it so: "Try to lower single output functional custom
#   ops to their out-variant", torch/_inductor/ir.py), where the overload is functional (_out_variant's _is_functional:
#   "no argument is written to and the name doesn't end with '_'"), its schema returns one tensor aliasing no argument
#   (a return of no alias set), the result has elements, and the out variant has one out argument (get_out_arg_names);
# - 'functional' where its overload is functional and returns no alias but has no such out variant: "an operator with
#   no out variant, whose result is copied once into its placed pages where it must be there" (appendix item 7);
# - none where its overload's results alias an argument (a view), or it takes no overload, or it runs on no meta kernel
#   (NotImplementedError, or RuntimeError: "you cannot perform data-dependent operations like torch.nonzero() or item()",
#   PyTorch, "Meta device"), or it is a module the program calls: its result may lie over any operand's pages (_over).
# Every rank computes the same forms, the meta device being every rank's (A4)
# 2026-11-05 (design/compilation-coordination.md, "The revisions of e4a7398e", OV-1 to OV-3): new.  What must stay:
# every rank's forms the same
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 1, and "The
# revisions of 4e78f6f4"): the overload is the node's target (_call's note), where _overload resolved a torch-level
# call; a value that is no tensor is its exported node's meta['val']; a form is its out variant with its one out
# argument's name (get_out_arg_names), so _run passes the result's place by that name.  What must stay: every rank's
# forms the same
def _forms(ir):
    unknown = object()

    def meta(key):
        if key in ir.metas:
            return ir.metas[key]
        if key in ir.shapes or len(key) == 2 and isinstance(key[1], tuple):
            return torch.empty(_shape(ir, key), dtype=_value(key[0]).dtype, device='meta')
        if len(key) == 2 and key[1] is None:
            return pytree.tree_map(
                lambda v: torch.empty_strided(v.shape, v.stride(), dtype=v.dtype, device='meta')
                if isinstance(v, torch.Tensor) else v, key[0].meta.get('val'))
        return unknown
    for kind, reads, key, operand in ir.program:
        if kind != 'call':
            continue
        local, operands = ir.locals[operand], [meta(k) for k in reads]
        if any(o is unknown for o in operands):
            continue
        given = iter(operands)
        args, kwargs = map_arg((local.args, local.kwargs), lambda n: next(given))
        try:
            result = _call(local, args, kwargs)
        except (NotImplementedError, RuntimeError):
            continue
        if isinstance(result, torch.Tensor):
            ir.metas[key] = result
        schema = local.target._schema if isinstance(local.target, torch._ops.OpOverload) else None
        if schema is None or not _is_functional(schema) or any(r.alias_info is not None for r in schema.returns):
            continue
        out = to_out_variant(local.target) if len(schema.returns) == 1 and isinstance(result, torch.Tensor) \
            and result.numel() else None
        names = get_out_arg_names(out) if out is not None else ()
        ir.forms[key] = (out, names[0]) if len(names) == 1 else 'functional'


# design/aot-compilation-change-request.md, appendix item 7: "A view (a narrow, a slice) is its base's pages, and its
# base lives as long as it does".  `over`: each key's set of the keys whose pages it may lie over, in the program's
# order: a slice, a halo's piece and a region that is one narrow of one value (_viewed) its read's; an operator's result
# its own where it runs as its out variant (`forms`), none where torch makes it ('functional'), else every operand's
# (_forms: it may be a view); a region its concatenation makes, and a collective's result, its own (a CollectivePermute's
# its own and its operand's, which a member keeps where its pair is its own); an input or a module attribute none, the
# caller's and the module's own.  `producer`: the item making each key.  `inplace`: "have each sent value's producer make
# it in its sent pages, so nothing copies it there" (appendix item 7), for an AllGather's operand, the one range of its
# member's (_all_gather), where its producer writes it into pages of its own (_fresh) and no other collective reads it;
# every rank's, since every rank runs the same program.  The other sent values are parts of their operand (AllToAll's
# pieces, ReduceScatter's slices, CollectivePermute's halo pieces), an operand no out variant makes, or one read by
# several collectives, each copied once into its sent pages (_run; design/compilation-coordination.md, "The revisions
# of e4a7398e", where the work stopped)
# 2026-11-05 (design/compilation-coordination.md, "The revisions of e4a7398e", OV and IP): new, where `placed` held the
# keys over an AllReduce's result alone.  What must stay: every rank's `over` and `inplace` the same
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 7, and "The
# revisions of 4e78f6f4"): `inplace` and _fresh go, for each part's place (_part, _directed).  What must stay: every
# rank's `over` the same
def _viewed(item):
    return item[0] == 'window' and len(item[1]) == 1 and item[3][6] is None


# 2026-11-05 (the C9 hook's remark on 63638057: "a topological pass over the program DAG ... it moves into the graph
# module"; design/compilation-coordination.md, "The revisions of e4a7398e", C9): the pass is mesh_graph.forward over the
# keys in the program's order, each key's predecessors the keys its item reads, its transfer the rule above.  What must
# stay: each key's set as above
def _over(ir):
    items = {item[2]: item for item in ir.program if item[0] != 'output'}
    ir.producer = {item[2]: g for g, item in enumerate(ir.program) if item[0] != 'output'}

    def rule(key, sets):
        kind, reads, _, operand = items[key]
        joined = frozenset().union(*sets)
        if kind in ('slice', 'narrow') or _viewed(items[key]):
            return joined
        if kind == 'call':
            form = ir.forms.get(key)
            return joined if form is None else frozenset() if form == 'functional' else frozenset((key,))
        if kind in ('window', 'collective'):
            return frozenset((key,)) | (joined if kind == 'collective' and operand.kind == 'collective-permute'
                                        else frozenset())
        return frozenset()
    ir.over = mesh_graph.forward(list(items), lambda key: items[key][1], rule)


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
# 2026-11-05 (design/aot-compilation-change-request.md, appendix item 7: "A view (a narrow, a slice) is its base's pages,
# and its base lives as long as it does"; design/compilation-coordination.md, "The revisions of e4a7398e", OV): the
# readers of `roots`, the keys whose pages are the range's, are the items reading a key over any of them (`over`), so a
# view an operator makes of them (_forms) is read over them too, where they were the reads of a key of `node` in
# `placed`.  What must stay: the positions as before
def _readers(ir, roots, default):
    spans = [(3 * g, 3 * g + (item[0] == 'collective' and item[3].kind != 'all-reduce'))
             for g, item in enumerate(ir.program) if any(ir.over.get(k, frozenset()) & roots for k in item[1])]
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
# 2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", T4-19): an elementwise reader is the
# operator its Local calls, and reads the key over the plan's pages page by page only where that key's local shape is
# its result's, since a broadcast operand (T4-15) is read whole by every page of the result (R4's function A: "set of
# pages operand_a to be contiguous and complete").  What must stay: the kinds as before for operands of one shape
# 2026-11-05 (design/compilation-coordination.md, "The revisions of e4a7398e", OV): the readers of the keys over
# `roots` (_readers), where they were those of `node`'s keys in `placed`.  What must stay: the kinds as before
# 2026-11-05 ("The revisions of 4e78f6f4"): an elementwise reader is an ATen operator of ELEMENTWISE, by its packet
# (the tables' note).  What must stay: the kinds as before
def _consumer(ir, roots):
    readers = [(item, k) for item in ir.program for k in item[1] if ir.over.get(k, frozenset()) & roots]
    if readers and all(item[0] == 'call' and isinstance(ir.locals[item[3]].target, torch._ops.OpOverload)
                       and ir.locals[item[3]].target.overloadpacket in ELEMENTWISE
                       and _shape(ir, k) == _shape(ir, item[2]) for item, k in readers):
        return mesh.PAGES
    return mesh.JOIN


# A range's place (2026-11-05, design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 7,
# and "The revisions of 4e78f6f4"): "name, in each declared transfer, the pages of its destination value that it
# lands in, as NCCL's receives land at offsets of one result ... A collective's result is then the pages its parts land
# in, so no concatenation and no copy of a part remains"; design/plan-to-bridge-change-request.md R2, corrected: "A
# collective's receive lands in the pages of the value it belongs to, at its place in that value: "receiving data from
# rank `i` at offset `i*sendcount`" (NCCL, ncclAllGather); "the `recvbuff` on rank `i` will contain the i-th block of
# the result" (NCCL, ncclReduceScatter)".  A part of a key's local value is a run of its elements in row-major order:
# [start, start + length) along `dim` is one where every dimension before `dim` is of size 1, from element start x (the
# product of the dimensions after it).  A range's place is a block of its value (mesh-collective.h struct
# mesh_declared `first`: "the range's P blocks are rows [first, first + P) of each copy of that value"), so a part's
# place begins a block, and "no two ranges at this node share a block of it": `blocks` holds, by key, the blocks its
# ranges hold, every rank's, the program being every rank's.
# - _landing: a value `parts` parts land in along `dim` (a collective's result; a reduction's received parts, "the
#   received slices landing at their places in one value", the coordinator's brief for point 7), each a run beginning
#   at a block, else refused at compile time naming point 7 ("A part whose bytes are not whole blocks (block x pgsz)
#   cannot begin a block of the result: refuse it at compile time naming point 7", the same brief); its parts' blocks
#   held; a part's elements.
# - _based: the place of a key whose pages are one run of a value its producer makes (an operator's out variant, a
#   region's concatenation, a collective's result but a CollectivePermute's, whose result is its operand where its pair
#   is its own), or of a narrow that is a run of such a key: (that value's key, element offset); None else.
# - _part: where a collective sends a part from.  From its place (R1: "The code must move every page from the page
#   that holds it to the page it lands in, with nothing copied on the way"), where its key is _based, the part is a run
#   beginning at a block and no range holds a block of it; else from a value of its own, which the collective's item
#   copies it into once: the part is then a slice, "an operator with no out variant, whose result is copied once into
#   its placed pages where it must be there" (design/aot-compilation-change-request.md, appendix item 7, "What stays"),
#   of a value no out variant makes or of an input (the caller's tensor, PV-3).  The choice reads the program alone, so
#   every rank makes it alike, and with it the range's events.  (The copy where the part is no run of a placed value
#   is the implementation's choice, recorded in design/compilation-coordination.md, "The revisions of 4e78f6f4".)
# - _directed: an AllGather's operand its producer can make in its place in the result ("the source's own part made in
#   place by its producer's out variant into its narrow of the result and sent from there", the brief): an operator's
#   out variant, a region's concatenation or a ReduceScatter's sum, of a key not yet placed in another value.
# - _produced: a sent range's events, its production at its producer's third step where it is made in its place, and
#   the source's own reads of that value its source read (mesh-collective.h struct mesh_dependence,
#   MESH_FROM_SOURCE_READ, "the source's own read of the range it sends", as _all_gather's IP had it), else its
#   production at the item's sends, where it is copied (design/compilation-coordination.md, A-T7).
# - _send: this node's release of a sent range, after its producer where it is made in place (`publishes`), else at the
#   item's sends after its copy (`sends`).
def _span(ir, key, first, count):
    size = _value(key[0]).dtype.itemsize
    return first * size // ir.blockbytes, -(-((first + count) * size) // ir.blockbytes)


def _landing(ir, c, base, dim, parts):
    shape, size = _shape(ir, base), _value(base[0]).dtype.itemsize
    part = math.prod(shape) // parts
    if parts > 1 and (math.prod(shape[:dim]) != 1 or part * size % ir.blockbytes):
        raise NotImplementedError(
            '%s of %s: its %d parts along dimension %d of %s, %d bytes each, are not runs of its pages that each '
            'begin a block of %d bytes, so a part cannot land at its place in the value (design/compilation-'
            'coordination.md, '
            '"The documents\' corrections of 2026-11-05", point 7; mesh-collective.h struct mesh_declared `first`)'
            % (c.kind, c.node.name, parts, dim, tuple(shape), part * size, ir.blockbytes))
    for p in range(parts):
        ir.blocks.setdefault(base, []).append(_span(ir, base, p * part, part))
    return part


def _based(ir, key):
    g = ir.producer.get(key)
    if g is None or key in ir.inside:
        return None
    kind, reads, _, operand = ir.program[g]
    if kind == 'narrow':
        base, (dim, start, length), shape = _based(ir, reads[0]), operand, _shape(ir, reads[0])
        if base is None or math.prod(shape[:dim]) != 1:
            return None
        return base[0], base[1] + start * math.prod(shape[dim + 1:])
    if kind == 'call' and ir.forms.get(key) not in (None, 'functional') \
            or kind == 'window' and not _viewed(ir.program[g]) \
            or kind == 'collective' and operand.kind != 'collective-permute':
        return key, 0
    return None


def _directed(ir, key):
    g = ir.producer.get(key)
    if g is None or key in ir.inside or key in ir.blocks:
        return False
    kind, _, _, operand = ir.program[g]
    return kind == 'call' and ir.forms.get(key) not in (None, 'functional') \
        or kind == 'window' and not _viewed(ir.program[g]) \
        or kind == 'collective' and operand.kind == 'reduce-scatter'


def _part(ir, c, key, at):
    shape, size = _shape(ir, key), _value(key[0]).dtype.itemsize
    count = math.prod(shape) if at is None else math.prod(shape[:at[0]]) * at[2] * math.prod(shape[at[0] + 1:])
    offset = 0 if at is None else at[1] * math.prod(shape[at[0] + 1:])
    base = _based(ir, key) if at is None or math.prod(shape[:at[0]]) == 1 else None
    if base is not None:
        span = _span(ir, base[0], base[1] + offset, count)
        if (base[1] + offset) * size % ir.blockbytes == 0 \
                and not any(a < span[1] and span[0] < b for a, b in ir.blocks.get(base[0], ())):
            ir.blocks.setdefault(base[0], []).append(span)
            return (base[0], base[1] + offset), True
    own = (key[0], 'sent', c.position, key, at)
    ir.shapes[own] = tuple(shape) if at is None else tuple(at[2] if d == at[0] else s for d, s in enumerate(shape))
    return (own, 0), False


def _produced(ir, d, i, place, own, sent, received):
    if not own:
        return [(d, i, 'P', sent, sent)]
    at = 3 * ir.producer[place[0]] + 2
    return [(d, i, 'P', at, at), (d, i, 'S') + _readers(ir, {place[0]}, received)]


def _send(ir, c, i, place, own, key, at):
    if own:
        ir.publishes.setdefault(ir.producer[place[0]], []).append((i, None))
    else:
        ir.added[place[0]] = None
        ir.sends.setdefault(c.position, []).append((i, ((place[0], None), (key, at))))


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
# 2026-11-05 (design/aot-compilation-change-request.md, appendix item 7; design/compilation-coordination.md, "The
# revisions of e4a7398e", OV): the sum's readers and kind those of its key, `made` (_readers, _consumer).  A member's
# partial is made in its sent pages by its producer, the Dot's out variant, and the root's sum in its sent pages by the
# combine's (_place, _run); their positions as before.  What must stay: the events as before
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 7, and "The
# revisions of 4e78f6f4"): every group lowered by one call, so each part's place is chosen once (_part); a member's
# partial sent from its place, made there by its producer's out variant (its production at that producer's third
# step, its reads there its source read), else copied once into a value of its own at the item's sends (_part, where
# `copies` copied it after its producer); the root's received partials landing at their places in one value
# (`received`, one part a member in the group's order: "the received slices landing at their places in one value", as
# ReduceScatter's), where each was a value of its own; the sum sent from the result's value and every member's sum
# landing in its result's.  What must stay: the events as before but the member's production where its partial is copied
def _all_reduce(ir, c, groups, key, made):
    shape, arrive = _shape(ir, key), 3 * c.position + 1
    readers, (place, own) = _readers(ir, {made}, arrive + 1), _part(ir, c, key, None)
    received, k, n = (c.node, 'received', c.position), len(groups[0]), math.prod(shape)
    ir.shapes[received] = (k - 1,) + tuple(shape)
    _landing(ir, c, received, 0, k - 1)
    _landing(ir, c, made, 0, 1)
    for group in groups:
        root, members = group[0], group[1:]
        up = [ir.transfer(c.node, shape, d, [root], mesh.PAGES)[root] for d in members]
        down = ir.transfer(c.node, shape, root, members, _consumer(ir, {made}))
        for d, partial in zip(members, up):
            ir.events += _produced(ir, d, partial, place, own, arrive - 1, arrive) \
                + [(d, down[d], 'C') + readers, (root, partial, 'C', arrive, arrive)]
        first = min(down.values())
        ir.events += [(root, first, 'P', arrive + 1, arrive + 1), (root, first, 'S') + readers]
        if ir.device == root:
            ir.roles[c] = (0, {None: first}, {j + 1: i for j, i in enumerate(up)}, received)
            ir.at.update({i: (made, 0) for i in down.values()})
            ir.at.update({i: (received, j * n) for j, i in enumerate(up)})
            ir.added[received] = None
            ir.publishes.setdefault(c.position, []).append((first, None))
        elif ir.device in members:
            me = group.index(ir.device)
            ir.roles[c] = (me, {0: up[me - 1]}, {0: down[ir.device]}, None)
            ir.at.update({up[me - 1]: place, down[ir.device]: (made, 0)})
            _send(ir, c, up[me - 1], place, own, key, None)


# design/compilation-coordination.md, A-T3': AllGather (GSPMD section 4.2: it "concatenates tensors from all
# participants following a specified order"): each member's shard one range to every other member, written over its
# range's pages by torch's copy_ (a range lies on its placed pages, design/disjoint-pages-change-request.md C3.2; the
# shard is torch's own, A-T4); each receiver's result torch.cat of the shards in the group's order, so each transfer is
# MESH_JOIN (a cat, R4's function A: the coordinator's ruling P3).  Events: each member's production of its range (its
# item's sends) and each other member's consumption of it (after arrival)
# 2026-11-05 (design/aot-compilation-change-request.md, appendix item 7: "have each sent value's producer make it in its
# sent pages, so nothing copies it there"; design/compilation-coordination.md, "The revisions of e4a7398e", IP): where
# the operand is made in its sent pages (`inplace`, _over), each member's production of its range is its producer's
# third step, and the member's own reads of it there are its source read (mesh-collective.h struct mesh_dependence,
# MESH_FROM_SOURCE_READ: "the source's own read of the range it sends"), the span of its readers (_readers), its
# result's concatenation among them; else as before, the range written at the item's sends.  What must stay: each
# member's range to every other member, MESH_JOIN
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 7, and "The
# revisions of 4e78f6f4"), reversing the notes above where they concatenate (the stop 1 of "The revisions of
# e4a7398e"): the result is the value the parts land in, member p's part at its place p along the gathered dimension
# (NCCL: "receiving data from rank `i` at offset `i*sendcount`"), each a run beginning at a block (_landing); the
# member's own part made in place by its producer's out variant into its narrow of the result and sent from there
# (`inside`, _directed), else copied once into that narrow at the item's sends; its reads there, the result's readers
# and, made in place, the operand's, its source read; each other member's consumption of its part spanning the result's
# readers, as a CollectivePermute's arrival's (CP-1).  What must stay: each member's range to every other member,
# MESH_JOIN
def _all_gather(ir, c, groups, key, made):
    shape, sent, received = _shape(ir, key), 3 * c.position, 3 * c.position + 1
    n = _landing(ir, c, made, c.dim, len(groups[0]))
    own = _directed(ir, key)
    if own:
        ir.inside[key] = (made, None)
    produced, roots = 3 * ir.producer[key] + 2 if own else sent, {made} | ({key} if own else set())
    last = max(received, _readers(ir, {made}, received)[1])
    for group in groups:
        ranges = [ir.transfer(c.node, shape, d, [e for e in group if e != d], mesh.JOIN) for d in group]
        for d, transfers in zip(group, ranges):
            i = min(transfers.values())
            ir.events += [(d, i, 'P', produced, produced), (d, i, 'S') + _readers(ir, roots, received)]
            ir.events += [(e, j, 'C', received, last) for e, j in transfers.items()]
        if ir.device in group:
            me = group.index(ir.device)
            for p, transfers in enumerate(ranges):
                ir.at.update({j: (made, p * n) for e, j in transfers.items() if p == me or e == ir.device})
            mine = min(ranges[me].values())
            ir.roles[c] = (me, {None: mine}, {p: t[ir.device] for p, t in enumerate(ranges) if p != me}, None)
            if own:
                ir.inside[key] = (made, me * n)
                ir.publishes.setdefault(ir.producer[key], []).append((mine, None))
            else:
                ir.sends.setdefault(c.position, []).append(
                    (mine, ((made, (c.dim, me * shape[c.dim], shape[c.dim])), (key, None))))


# design/compilation-coordination.md, A-T3': AllToAll (GSPMD section 4.2: it "logically splits the input of each
# participant along one dimension, then sends each piece to a different participant.  On receiving data pieces from
# others, each participant concatenates the pieces to produce its result"): piece q of member p, along the dimension
# the mesh dimension moves to, one declared transfer p -> q, written by torch's copy_ as an AllGather's shard is; each
# receiver's result torch.cat of the pieces in sender order along the dimension it leaves, each transfer MESH_JOIN as an
# AllGather's.  Events as an AllGather's
# 2026-11-05 (design/compilation-coordination.md, "The revisions of e4a7398e"): `made`, the item's result, taken as
# every lowering takes it (lower).  Each piece is copied once into its sent pages, and the result concatenated into its
# placed pages (torch.cat's out variant), the stop recorded there.  What must stay: as above
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 7, and "The
# revisions of 4e78f6f4"), reversing the note above (the stops of "The revisions of e4a7398e"): piece q sent from its
# place in the operand's value where it is one (_part), else copied once; the piece from member p landing at its place
# p in the result along the dimension it leaves (_landing), its consumption spanning the result's readers; the result
# is the pages the pieces land in, so torch.cat goes.  The member's own piece, which no range carries, is copied once
# from its place in the operand into its place in the result at the item's sends: the operand is one tensor its
# producer's out variant writes, of which the other pieces are sent, so its own piece is in the result's pages only by
# a copy, as NCCL moves a rank's own block by its send and receive to itself (the implementation's choice, recorded in
# design/compilation-coordination.md, "The revisions of 4e78f6f4", against point 7's "no copy of a part remains").
# What must stay: one declared transfer p -> q a piece, MESH_JOIN
def _all_to_all(ir, c, groups, key, made):
    (i, j), k = c.dim, len(groups[0])
    shape, result = _shape(ir, key), _shape(ir, made)
    sent, received, nj, ni = 3 * c.position, 3 * c.position + 1, shape[j] // k, result[i] // k
    piece = tuple(nj if d == j else size for d, size in enumerate(shape))
    part = _landing(ir, c, made, i, k)
    places = [_part(ir, c, key, (j, q * nj, nj)) for q in range(k)]
    last = max(received, _readers(ir, {made}, received)[1])
    for group in groups:
        pieces = {(p, q): ir.transfer(c.node, piece, d, [e], mesh.JOIN)[e]
                  for p, d in enumerate(group) for q, e in enumerate(group) if p != q}
        for (p, q), x in pieces.items():
            ir.events += _produced(ir, group[p], x, *places[q], sent, received) + [(group[q], x, 'C', received, last)]
        if ir.device in group:
            me = group.index(ir.device)
            ir.roles[c] = (me, {q: x for (p, q), x in pieces.items() if p == me},
                           {p: x for (p, q), x in pieces.items() if q == me}, None)
            for (p, q), x in pieces.items():
                if p == me:
                    ir.at[x] = places[q][0]
                    _send(ir, c, x, *places[q], key, (j, q * nj, nj))
                elif q == me:
                    ir.at[x] = (made, p * part)
            ir.sends.setdefault(c.position, []).append((None, ((made, (i, me * ni, ni)), (key, (j, me * nj, nj)))))


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
# sum, a tensor of its own over no page of the plan.  `key` is the members' partials' key, whose sharding their
# dims_mapping (TC1), its shape theirs (_shape).  Events as an AllToAll's: each member's production of its slices (its
# item's sends), each keeper's consumption (after arrival)
# 2026-11-05, owner T (design/compilation-coordination.md, "The revisions of 46ae8fdf", RS): new, where `lower` refused
# a ReduceScatter.  The partitioner here inserts none (_partition replicates a contracting dimension whose mesh
# dimension the output uses, where GSPMD section 5 would "replace the AllReduce on the output BSM with a
# ReduceScatter"), so this is reached once A2's partitioner inserts one.  What must stay: each slice to its keeper
# 2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", TA1): _partition inserts it where a
# Dot's partial sums lie along a mesh dimension the output shards (GSPMD sections 3.2 and 5.1), `dim` the output's
# dimension along it and `have` the partial's sharding, so it is reached.  Why: A4, "A2's partitioner inserts these two
# where GSPMD's does".  What must stay: each slice to its keeper
# 2026-11-05 (design/compilation-coordination.md, "The revisions of e4a7398e"): `made`, as every lowering takes it;
# each slice copied once into its sent pages, the keeper's sum made in its placed pages by torch.add's out variant.
# What must stay: as above
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 7, and "The
# revisions of 4e78f6f4"; the coordinator's brief: "ReduceScatter's slices sent from their places in the partial (no
# copy), the received slices landing at their places in one value and summed by torch.add's out variant"): slice q
# sent from its place in the partial's value where it is one (_part), else copied once; the slices member q receives
# landing in one value (`received`), the one from the j-th other member of the group's order at its place j
# (_landing), each a value of its own before.  What must stay: each slice to its keeper, MESH_PAGES
def _reduce_scatter(ir, c, groups, key, made):
    dim, k, shape = c.dim, len(groups[0]), _shape(ir, key)
    sent, received, n = 3 * c.position, 3 * c.position + 1, shape[dim] // k
    piece = tuple(n if d == dim else size for d, size in enumerate(shape))
    places = [_part(ir, c, key, (dim, q * n, n)) for q in range(k)]
    buffer = (c.node, 'received', c.position)
    ir.shapes[buffer] = (k - 1,) + piece
    part = _landing(ir, c, buffer, 0, k - 1)
    for group in groups:
        slices = {(p, q): ir.transfer(c.node, piece, d, [e], mesh.PAGES)[e]
                  for p, d in enumerate(group) for q, e in enumerate(group) if p != q}
        for (p, q), x in slices.items():
            ir.events += _produced(ir, group[p], x, *places[q], sent, received) \
                + [(group[q], x, 'C', received, received)]
        if ir.device in group:
            me = group.index(ir.device)
            others = [p for p in range(k) if p != me]
            ir.roles[c] = (me, {q: x for (p, q), x in slices.items() if p == me},
                           {p: slices[(p, me)] for p in others}, buffer)
            for (p, q), x in slices.items():
                if p == me:
                    ir.at[x] = places[q][0]
                    _send(ir, c, x, *places[q], key, (dim, q * n, n))
            ir.at.update({slices[(p, me)]: (buffer, r * part) for r, p in enumerate(others)})
            ir.added[buffer] = None


# design/aot-compilation-change-request.md A4: CollectivePermute (GSPMD section 4.2: it "exchanges data among a list of
# source-destination pairs"), one point-to-point demand a pair, routed as AllToAll's are (mesh_plan's demands, routed
# jointly by Garg and Koenemann's rounds).  Its pairs are XLA's source_target_pairs over the group's places in the
# device mesh's order (the record's `dim`, TC2): "For each pair, the operand is sent from source replica to target
# replica"; "Any two pairs should not have the same target replica id, and they should not have the same source replica
# id" (OpenXLA, Operation semantics, CollectivePermute; GSPMD section 4.2: "We use XLA's operators"), refused otherwise
# as the placement operand's errors are (ValueError).  Each pair p -> q, p != q, is one declared transfer of the
# member's local value of `key` (a halo's piece, _region; its shape _shape's), written by torch's copy_ as an AllToAll's
# piece is; the destination's result is a tensor
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
# 2026-11-05 (design/aot-compilation-change-request.md A4: "CollectivePermute goes into halo exchange ... and into a
# pipeline's buffer shifting"; "The partitioner reaches them by partitioning every operator"; design/compilation-
# coordination.md, "The revisions of f3f7cf46", stop 2, its answer): reached.  _region inserts one for each halo piece
# of a windowed or data formatting operator's windowed dimension a mesh dimension shards (GSPMD section 4.3, Appendix
# A.2), and a pipeline's "pad_left(state, 1)[:-1]" is one Slice over its Pad merged (section 4.6, _merged), its shift
# one CollectivePermute of one piece a partition; the operand is the halo's piece, its shape _shape's.  Resharding "to
# change device order" (section 4.5) is still not inserted: one device mesh orders every tensor's devices alike.  What
# must stay: one transfer a pair
# 2026-11-05 (design/aot-compilation-change-request.md, appendix item 7: "A collective's result is the pages its parts
# land in, so no concatenation copies them"; "It clones each placed output (`torch.clone`)" is the code's, not torch's;
# design/compilation-coordination.md, "The revisions of e4a7398e", CP-1): the destination's result is its arrival's
# pages, where it was torch.clone of them, so its consumption spans its readers (_readers of `made`, the region's
# concatenation that reads it), where it ended at the arrival; its kind stays MESH_JOIN, its consumer that
# concatenation (R4's function A).  What must stay: one transfer a pair
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 7, and "The
# revisions of 4e78f6f4"): the halo's piece sent from its place where it is a run of its base's value (_part: a
# narrow of a value made by an out variant), else copied once (a slice, which has no out variant); the arrival landing
# in the result's value, held whole.  What must stay: one transfer a pair
def _collective_permute(ir, c, groups, key, made):
    pairs = tuple(tuple(pair) for pair in c.dim)
    places, k = [p for pair in pairs for p in pair], len(groups[0])
    if not all(isinstance(p, int) and 0 <= p < k for p in places) \
            or len({p for p, _ in pairs}) != len(pairs) or len({q for _, q in pairs}) != len(pairs):
        raise ValueError('%s: a CollectivePermute\'s pairs %s over a group of %d: each a (source, target) place, no two '
                         'with one source or one target (XLA\'s source_target_pairs)' % (c.node.name, pairs, k))
    shape, sent, received = _shape(ir, key), 3 * c.position, 3 * c.position + 1
    (place, own), last = _part(ir, c, key, None), max(received, _readers(ir, {made}, received)[1])
    _landing(ir, c, made, 0, 1)
    for group in groups:
        moved = {(p, q): ir.transfer(c.node, shape, group[p], [group[q]], mesh.JOIN)[group[q]]
                 for p, q in pairs if p != q}
        for (p, q), x in moved.items():
            ir.events += _produced(ir, group[p], x, place, own, sent, received) + [(group[q], x, 'C', received, last)]
        if ir.device in group:
            me = group.index(ir.device)
            ir.roles[c] = (me, {q: x for (p, q), x in moved.items() if p == me},
                           {p: x for (p, q), x in moved.items() if q == me}, None)
            for (p, q), x in moved.items():
                if p == me:
                    ir.at[x] = place
                    _send(ir, c, x, place, own, key, None)
                elif q == me:
                    ir.at[x] = (made, 0)


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
# 2026-11-05 (design/aot-compilation-change-request.md A4; design/compilation-coordination.md, "The revisions of
# abecd389"): each lowering is given the key its item reads, where it was given that key's sharding, so its operand's
# local shape is _shape's (a halo's piece has its own).  What must stay: each collective's lowering as before
# 2026-11-05 (design/aot-compilation-change-request.md, appendix item 7; design/compilation-coordination.md, "The
# revisions of e4a7398e", OV and PV): each lowering is given the key its item makes as well, whose readers are its
# result's (_readers), and once every collective is lowered this node's values are placed (_place).  What must stay:
# each collective's lowering as before
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 7, and "The
# revisions of 4e78f6f4"): each lowering is given every group of its collective at once, so the places of its parts,
# one key's at every member, are chosen once (_part), where it was called once a group.  What must stay: each group
# lowered as before
LOWERINGS = {'all-reduce': _all_reduce, 'all-gather': _all_gather, 'all-to-all': _all_to_all,
             'reduce-scatter': _reduce_scatter, 'collective-permute': _collective_permute}


def lower(ir):
    for item in ir.program:
        if item[0] == 'collective':
            collective = item[3]
            if collective.kind not in LOWERINGS:
                raise NotImplementedError('%s of %s: no lowering (A-T3\')' % (collective.kind, collective.node.name))
            LOWERINGS[collective.kind](ir, collective, _groups(ir, collective.dims), item[1][0], item[2])
    ir.dependences = _dependences(ir)
    _place(ir)


# design/aot-compilation-change-request.md, appendix item 7: "place every value of the node's partitioned graph by the
# one allocator" (C3.2; ExecuTorch's memory planning: "This assigns a memory ID to each tensor and an offset in the
# buffer"), "have each sent value's producer make it in its sent pages, so nothing copies it there", and "an operator
# with no out variant, whose result is copied once into its placed pages where it must be there".  At a node the
# placement operand maps a device (one it maps none has "no section of its own", A4), `homes`:
# - a member's AllReduce partial, its sent range's value, where its producer runs as its out variant (_fresh), the
#   range published after it (`publishes`); else that item copies it once into the range's pages and publishes it
#   (`copies`).  The root's sum, its sent range's value, made by its combine (_run);
# - an AllGather's operand made in its sent pages (`inplace`): this member's range's value, published after its producer;
# - every other key that makes pages of its own (`over`) and lies over no range of this node (an AllReduce's result, a
#   CollectivePermute's arrival or a member's own value where its pair is its own): a local value, MESH_EXECUTION
#   outside the window, its positions on this node's program order its producer's item and its last reader's (mesh-
#   collective.h struct mesh_value `begin` and `end`), a reader of a key over it being its reader, so a view's base
#   lives as long as it does; the program's output among its readers, so no value of the execution reuses an output's
#   pages.  Its layout: an operator's result's on the meta device (_forms), else its key's local shape (_shape).
# The positions are item indices, an item's reads and its write one position, so no value it reads shares pages with
# the one it writes (its out variant reads and writes at once).  An output's pages are its value's: "Each output is a
# view of its placed pages and holds until the plan reuses them, u executions later (C3.3)" (appendix item 7), the next
# execution for a local value, its one copy (mesh-collective.h: `count` "1 for every other")
# 2026-11-05 (design/compilation-coordination.md, "The revisions of e4a7398e", PV): new, where a node's values were its
# ranges alone and every operator's result torch's.  What must stay: no value at a node the placement maps no device
# 2026-11-05 (the C9 hook's remark on 63638057: "last-use liveness over the program's read edges ... it moves into the
# graph module"; design/compilation-coordination.md, "The revisions of e4a7398e", C9): each value's last reader is
# mesh_graph.ends over the items in order, each item's uses the keys over which its reads lie.  What must stay: the
# positions as above
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", points 6 to 8, and "The
# revisions of 4e78f6f4"), reversing the notes above where a range is a value of its own and a sent value is copied
# into one: a value is made for every key with pages of its own but an AllGather operand made inside its result
# (`inside`) and a CollectivePermute's result that is its operand, and for every value of no key (`added`); it is
# routed (`wire` 1, MESH_EXECUTION in the window) where a range of this node is placed in it (`at`: "the value is
# MESH_EXECUTION and wire", mesh-collective.h struct mesh_declared), else local with its positions as above, each
# `run` 1 (IR.value); every placed key's home its value, its element offset there and its shape; each declared transfer
# of this node its value and `first`, the block its place begins (every place begins a block: _landing, _part).  The
# placement of the values' pages is mesh_plan's ("Every value's pages are placed anywhere, by the fewest chains",
# design/plan-to-bridge-change-request.md R2, corrected).  What must stay: no value at a node the placement maps no
# device, and the positions as above
def _place(ir):
    if ir.device is None:
        return
    last = mesh_graph.ends(range(len(ir.program)),
                           lambda g: [owner for k in ir.program[g][1] for owner in ir.over.get(k, ())])
    wired, bases = {base for base, _ in ir.at.values()}, []
    for g, (kind, reads, key, operand) in enumerate(ir.program):
        if kind == 'output' or key in ir.inside or key not in ir.over.get(key, ()):
            continue
        if kind == 'collective' and operand.kind == 'collective-permute' and not ir.roles[operand][2] \
                and any(tuple(pair) == (ir.roles[operand][0],) * 2 for pair in operand.dim):
            continue
        bases.append((key, g))
    for key, g in bases + [(key, None) for key in ir.added]:
        dtype, shape = _layout(ir, key)
        wire = int(key in wired)
        ir.homes[key] = (ir.value(dtype, math.prod(shape), _value(key[0]).device.type, wire, 0 if wire else g,
                                  0 if wire else last.get(key, g)), 0, shape)
    for key, (made, offset) in ir.inside.items():
        ir.homes[key] = (ir.homes[made][0], offset, _layout(ir, key)[1])
    for i, (base, offset) in ir.at.items():
        ir.declared[i].value = ir.homes[base][0]
        ir.declared[i].first = offset * _layout(ir, base)[0].itemsize // ir.blockbytes


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
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 1, and "The
# revisions of 4e78f6f4"): FUNCTIONS goes, a compiled function's ordinal being its place in the program's order of
# compilation, its first call's in the traced block (_Export; mesh-collective.h struct mesh_function: "Function k (from
# 0) is named (group, k + 1)"), where Dynamo's compile function took the next.  What must stay: one context a program
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


# mesh_plan's `program`, the nodes the launcher started the program at, which it passes as it passes MESH_GROUP:
# MESH_PROGRAM, the program's nodes' map indices, whole numbers apart by commas, each below 64, the same at every rank
# of a run; as a set of map indices, bit i for map index i (mesh-collective.h mesh_plan)
# 2026-11-05 (design/aot-compilation-change-request.md A7, sender(m): "The program's nodes are the nodes its launcher
# started it at, an operand of the plan call (`program`) that the launcher passes as it passes `group` (MESH_GROUP)";
# design/compilation-coordination.md, "The revisions of 3f106747", PG), in place of the placement operand's nodes (the
# coordinating implementer's PS-1 of "The revisions of abecd389").  Why: the program's nodes are its launcher's, not
# one function's placement, so every plan of one program names one sender ("The program's sender ... the least of the
# program's nodes"), and every run is launched on the whole collective, its relays among its nodes.  The variable's
# name and form are the coordinating implementer's (PG-1).  What must stay: the launcher's operand, refused where it
# is missing or not one
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


# design/aot-compilation-change-request.md, appendix item 7, "A publish waits for nothing": "a torch MPS event is
# recorded after the producer's out-variant kernel, the publish is done in that event's notification, and torch's
# command buffer is committed without waiting"; "On the CPU device a kernel has ended when its call returns, so the
# publish follows the call".  Execution t of each of `values`, sent ranges this node's program has written: a CPU
# value's by mesh_host_publish at once, the MPS values' all in one notification of rdma/libmesh-torch.dylib
# (mesh-torch.mm mesh_torch_publish, built by rdma/mesh-torch.mk against the program's torch, which is loaded first,
# above), its publish function mesh_host_publish of the program's libmesh (mesh.LIB) and its context the program's,
# which lives as long as the program (_context).  Its error value, mesh.check's
# 2026-11-05 (design/compilation-coordination.md, "The revisions of e4a7398e", TP): new, where each sent value was
# written between two synchronizations of its device and published after the second.  What must stay: a publish after
# its range's last write, no wait of the device
TORCH = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(mesh.__file__)), 'libmesh-torch.dylib'))
TORCH.mesh_torch_publish.restype, TORCH.mesh_torch_publish.argtypes = C.c_int, [C.c_void_p, C.c_void_p,
                                                                                C.POINTER(mesh.Section), mesh.U, mesh.Q]
TORCH.mesh_torch_notifying.restype, TORCH.mesh_torch_notifying.argtypes = mesh.Q, []
PUBLISH = C.cast(mesh.LIB.mesh_host_publish, C.c_void_p)


# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 7, and "The
# revisions of 4e78f6f4"): each publish a range's section, its first row the range's first block in its value
# (_bind: "mesh_host_arrived and mesh_host_publish take a section of the range", the coordinator's brief), each with
# its value's device, where it was a value's section.  What must stay: a publish after its range's last write, no wait
def _publish(context, sections, devices, t):
    metal = [section for section, device in zip(sections, devices) if device == 'mps']
    for section, device in zip(sections, devices):
        if device != 'mps':
            mesh.LIB.mesh_host_publish(C.byref(context), section, t)
    if metal:
        mesh.check(TORCH.mesh_torch_publish(PUBLISH, C.addressof(context), (mesh.Section * len(metal))(*metal),
                                            len(metal), t))


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
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 6, and "The
# revisions of 4e78f6f4"; the coordinator's brief: "Release the view (mesh_view_release) only after the program
# retires and no tensor over it remains"): each tensor over the program's mapping counted in its _Mapping, the count
# lowered by its deleter, torch's word that it holds the tensor no more; the mapping released (mesh-dataflow.h
# mesh_view_release) once the program has retired it and the count is 0: reference counting (G. E. Collins, A method
# for overlapping and erasure of lists, CACM 3(12), 1960) over DLPack's own report of each release ("This should be
# called to destruct manager_ctx", dlpack.h).  An output the program keeps (appendix item 7: "Each output is a view of
# its placed pages") holds the mapping until torch frees it.  What must stay: one release a tensor, the mapping
# released after the last
class _Mapping:
    def __init__(self, view):
        self.view, self.tensors, self.retired = view, 0, False

    def release(self):
        if self.retired and not self.tensors and self.view.address:
            mesh.LIB.mesh_view_release(self.view)
            self.view = mesh.View()


HELD = {}


@DELETER
def _delete(managed):
    held = HELD.pop(C.addressof(managed.contents), None)
    if held:
        if held[0].manager_ctx:
            PUT(held[0].manager_ctx, SELECTOR[b'release'])
        held[3].tensors -= 1
        held[3].release()


# design/compilation-coordination.md, K1: a placed value's copies wrapped by one MTLBuffer with no copy, as
# metal-microbench tools/e2b_replay.m placed_buffer wraps them: over [its lowest copy's first byte, its highest copy's
# end), each copy `pages` pages of the region's page size (mesh-call.h struct mesh_section; mesh.h struct hdr `pgsz`),
# so the buffer starts on a page and is whole pages long, as placed_buffer's and rdma/mesh-metal.m's are; on torch's
# MPS device, shared storage, no deallocator (the pages are the region's).  ENOMEM where Metal makes none, as
# mesh-metal.m's state buffer.  The buffer (one reference, the caller's) and each copy's byte offset in it
# 2026-11-05 (design/aot-compilation-change-request.md, appendix item 7: "A publish waits for nothing"; design/
# compilation-coordination.md, "The revisions of e4a7398e", PV-2): one buffer over the copies of every MPS value of
# the plan in one space (the window's, or outside it), [the lowest copy's first byte, the highest copy's end), where it
# was one a value.  Why: the allocator shares a space's pages among values whose lifetimes do not meet (C3.3), so two
# values' buffers over one page were two resources, which Metal orders by each resource's own hazard tracking alone
# (MTLResource.h hazardTrackingMode; this note's earlier reason for the device waits around each write); over one
# buffer, a value's write after another's read of the same pages is ordered by Metal as one resource's, with no wait of
# the device.  The two spaces never share a page (two colourings, mesh-collective.c plan_place), and each is one run of
# the plan's claim (mesh-dataflow.h struct mesh_claim), so the buffer spans that run at most.  torch 2.14.0 reads an MPS
# tensor at its storage offset in place on macOS 15 or later (include/ATen/native/mps/OperationUtils.h needsGather:
# `!is_macOS_15_0_or_newer && (!t.is_contiguous() || t.storage_offset())`).  What must stay: a page-aligned buffer of
# whole pages over the region's pages, no deallocator
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", points 6 and 8, and "The
# revisions of 4e78f6f4"; the coordinator's brief: "MPS over ONE MTLBuffer made over the whole program view
# (newBufferWithBytesNoCopy of `view.address`, `view.bytes`), each copy's tensor at its offset in it"): the buffer is
# over the program's one mapping (mesh-collective.h mesh_plan `*view`), where it was over a space's copies, whose rows
# are now bound each to a block placed anywhere (mesh-section.h), so no run of the region's pages holds a value.  PV-2's
# reason stays: the allocator's chains share pages among values (design/plan-to-bridge-change-request.md R2,
# corrected), and with one buffer over the mapping a write after another value's read of the same physical page stays
# within one resource, which Metal orders by its hazard tracking (MTLResource.h hazardTrackingMode).  The mapping is
# mmap'd (mesh-dataflow.h mesh_view_compose), so it starts on a page, and is whole blocks long.  What must stay: a
# page-aligned buffer of whole pages, no deallocator
def _placed_buffer(view):
    buffer = WRAP(_metal_device(), SELECTOR[b'newBufferWithBytesNoCopy:length:options:deallocator:'], view.address,
                  view.bytes, SHARED, None)
    if not buffer:
        raise OSError(errno.ENOMEM, 'Metal made no buffer over the program\'s mapping, %d bytes (MTLDevice.h '
                                    'newBufferWithBytesNoCopy)' % view.bytes)
    return buffer


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
# 2026-11-05 ("The revisions of 4e78f6f4"): each tensor counted in the mapping it lies over (_Mapping).  What must stay:
# as above
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


# each copy of a placed value as a tensor of its layout over its pages (every address a constant of the plan,
# design/disjoint-pages-change-request.md, C3.3), every copy by _dlpack_tensor: a CPU copy's `data` its address, its
# offset 0, no owner; an MPS copy's `data` the value's _placed_buffer, its offset the copy's in it, the buffer its
# owner, whose own reference is released once each copy's tensor holds its own (K1).  EFAULT where a copy's row has no
# page (mesh-call.h mesh_section_address)
# 2026-11-04, owner T2 (the coordinator's ruling on review W3's finding 12; design/disjoint-pages-change-request.md,
# C3.8: "no call site handles two forms"): torch.frombuffer for CPU copies goes, and every copy is bound by
# torch.from_dlpack.  What must stay: each device's `data` and `byte_offset` as torch's own export sets them
# 2026-11-05 (design/compilation-coordination.md, "The revisions of e4a7398e", PV-2): every value of the plan bound at
# once, its section and its copies' tensors, an MPS value's over its space's one _placed_buffer, where each value was
# bound alone over a buffer of its own.  What must stay: as above
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", points 6 to 8, and "The
# revisions of 4e78f6f4"; the coordinator's brief: "bind every copy's tensor over its view: CPU by DLPack at the copy's
# view address; MPS over ONE MTLBuffer made over the whole program view"), reversing the notes above where a copy is
# found by its rows (mesh_section_address, EFAULT) and an MPS value lies in its space's buffer: copy c of each value of
# one compiled function at its place in the program's mapping, view + c x section.pages x pgsz (mesh-collective.h
# struct mesh_value `view`: "every copy's blocks consecutive there, in row order"), as one tensor of its elements; a
# CPU copy's `data` that address, an MPS copy's the program's one buffer (`buffer`, _placed_buffer) at the copy's
# offset in it, each counted in `mapping`.  Each placed key's tensors, one a copy, a narrow of its value's at its
# element offset viewed in its shape (`homes`); each declared transfer of this node its range's section, its first row
# the range's first block in its value's rows (value.first + first, struct mesh_declared), its blocks x block pages,
# the value's count and stride (the brief: "mesh_host_arrived and mesh_host_publish take a section of the range"), and
# its value's device.  What must stay: each device's `data` and `byte_offset` as torch's own export sets them
def _bind(context, ir, values, buffer, mapping):
    header, copies = mesh.Header.from_address(context.M), []
    for v, (dtype, numel, device) in zip(values, ir.layouts):
        addresses = [v.view + c * v.section.pages * header.pgsz for c in range(v.section.count)]
        copies.append([_dlpack_tensor('cpu', a, 0, None, dtype, (numel,), mapping) if device == 'cpu' else
                       _dlpack_tensor('mps', buffer, a - mapping.view.address, buffer, dtype, (numel,), mapping)
                       for a in addresses])
    places = {key: [copy.narrow(0, offset, math.prod(shape)).view(shape) for copy in copies[v]]
              for key, (v, offset, shape) in ir.homes.items()}
    sections, devices = {}, {}
    for i, d in enumerate(ir.declared):
        if d.value != mesh.ABSENT:
            value = values[d.value].section
            sections[i] = mesh.Section(value.first + d.first, -(-d.bytes // ir.blockbytes) * header.block, d.bytes,
                                       value.count, value.stride)
            devices[i] = ir.layouts[d.value][2]
    return places, sections, devices


# item g of this node's program, (kind, the keys it reads, the key it makes, its operand), at execution t on execution
# t's copies (mesh-collective.h struct mesh_value: copy t mod count).  An operator of an out variant (_forms) writes its
# result into the pages its key is placed in (`homes`: a local value, or a sent range's pages, _place); any other
# computes its own, torch's.  A member's AllReduce partial is made in its sent pages by its producer where that runs as
# its out variant, else copied once into them by that item (`copies`), and published after the item (`publishes`;
# _publish), after the reads that make it (design/compilation-coordination.md, A-T7).  The root's AllReduce waits for
# each partial in turn and adds it in, MSCCLang's receive-reduce-copy (section 4.2), by torch.add's out variant into its
# sum's sent pages, then publishes it, for which every other member waits (A-T3').  An AllGather's operand made in its
# sent pages was published by its producer; any other shard or piece is copied once into its range's pages and
# published, then each other member's is waited for and the parts concatenated in the group's order by torch.cat's out
# variant into the result's placed pages (A-T3').  A returned value is the tensor its key is, over its placed pages
# where it is placed, held until they are reused: "No output is copied.  Each output is a view of its placed pages and
# holds until the plan reuses them, u executions later (C3.3)"; "A program that keeps an output longer clones it
# itself, outside the compiled function" (design/aot-compilation-change-request.md, appendix item 7, as torch's
# CUDAGraph Trees' outputs are: "If a tensor is a user-visible output that must remain live across generations,
# manually clone the tensor outside torch.compile()", torch v2.14.0 torch/_inductor/cudagraph_trees.py).
# Every operator is torch's own kernel for its tensors' device, its MPS kernel for MPS tensors
# (design/compilation-coordination.md, K1; design/aot-compilation-change-request.md A9: "No kernel is written by the
# compilation"), on torch's device stream, which stays torch's (appendix item 7, "What stays, being torch's").  No
# write waits for the device: a value's write after another's read of its pages is ordered by Metal over its space's
# one buffer (_placed_buffer), and a publish follows its range's writes in a torch MPSEvent's notification (_publish).
# A received range is read only after mesh_host_arrived reports it (arrived), so the kernel reading it is issued after
# (K1)
# 2026-11-04, owner T2 (the coordinator's ruling on review W3's finding 12; C3.8: "no call site handles two forms"):
# torch.mps.synchronize for an MPS range alone becomes the range's device module's synchronize for every range.  What
# must stay: both synchronizations around every write of a range
# 2026-11-05, owner T (design/compilation-coordination.md, "The revisions of 46ae8fdf", RS and CP): a ReduceScatter
# copies each slice of its partial over its range's pages and releases it, then adds each landed slice into its own,
# in the group's order (torch.add, as the AllReduce's root); a CollectivePermute copies its value over each of its
# ranges' pages and releases it, then takes its arrival by torch.clone, keeps its own value where its pair is its own,
# or holds torch.zeros_like (XLA's zeros) where no pair targets it.  What must stay: as above
# 2026-11-05 (design/aot-compilation-change-request.md A4; design/compilation-coordination.md, "The revisions of
# abecd389", T4-16, T4-18, T4-20, T4-22): an operator runs as its Local (ir.locals: its functional equivalent, its
# local configuration), each tensor operand wherever its arguments hold it taking the next key read, in map_arg's
# order; a module the program calls by its GraphModule's submodule; a get_attr ('constant') is the module's
# attribute; a halo's piece ('narrow') torch's narrow; a region ('window') torch.cat of its pieces and the value, then
# narrow at this partition's offset (DynamicSlice; ir.coords as `slice`'s), then masked where outside the operand by
# masked_fill, torch's Select, of a mask comparing torch.arange (Iota) from this partition's start with the operand's
# bounds (GSPMD section 4.1); a member's partial is written by the item that makes the AllReduce's partial key (the
# operator's, or a cut's after it).  No kernel is written (A9): each is torch's own operator.  What must stay: as above
# 2026-11-05 (design/aot-compilation-change-request.md, appendix item 7 at mesh 9c8c8865: "run each operator as its out
# variant into its result's placed pages"; "publish with no wait for the device"; "copy no output"; "A collective's
# result is the pages its parts land in"; design/compilation-coordination.md, "The revisions of e4a7398e", OV, PV, TP,
# NC and CP-1), reversing the notes above where they write: each operator of an out variant runs it into its placed
# pages (`forms`, `homes`), where every operator computed its own fresh output; a sent value made in its sent pages by
# its producer is published after it, any other copied once into them, where `write` copied each between two
# synchronizations of its device; every publish is _publish's, with no wait; the AllReduce's and ReduceScatter's sums by
# torch.add's out variant into their placed pages, the AllGather's and AllToAll's concatenation by torch.cat's, the
# CollectivePermute's result its arrival's pages, where torch.clone copied them, and its zeros torch's zero_ of its
# placed pages; a region by torch.cat's out variant of the narrows of its pieces and value that the window covers, then
# torch's fill_ of the parts outside the operand, where it was torch.cat, narrow and masked_fill of a mask from
# torch.arange (GSPMD section 4.1's Select: those parts are "index < 0" and "index >= size" along the window); no output
# torch.clone'd.  What must stay: one execution a call, every read of a received range after its arrival
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", points 1, 6 and 7, and
# "The revisions of 4e78f6f4"), reversing the notes above where they concatenate a collective's parts and copy each
# sent part (`send`, "#6's stop 2. Point 7 removes it", the coordinator): each placed key's tensor at execution t is its
# place in copy t mod count of its value (`places`, _bind); a collective item first releases its sends (`sends`: each
# range sent from its place, or copied once into a value of its own and released; an AllToAll's own piece copied into
# its place in the result), then waits for each range it receives and reads its result where the parts landed: an
# AllGather's and AllToAll's result is that value, torch.cat goes; a ReduceScatter's and an AllReduce root's received
# parts lie at their places in one value, each added in by torch.add's out variant as it arrives; a CollectivePermute's
# result is its arrival's value.  Every item then releases the ranges it made in place (`publishes`).  An operator is
# its exported graph's ATen operator (_call), its out variant called with its result's place by its out argument's
# name.  What must stay: one execution a call, every read of a received range after its arrival, a release after its
# range's writes
def _run(ir, context, run, g, env, inputs, t):
    kind, reads, key, operand = ir.program[g]

    def place(k):
        copies = run['places'][k]
        return copies[t % len(copies)]

    def release(entries):
        for _, copy in entries:
            if copy is not None:
                (target, into), (source, out) = copy
                _narrow(place(target), into).copy_(_narrow(env[source], out))
        published = [i for i, _ in entries if i is not None]
        if published:
            _publish(context, [run['sections'][i] for i in published], [run['devices'][i] for i in published], t)

    def arrived(i):
        _arrived(context, run['sections'][i], t)

    def combine(total, first, parts):
        for i, part in parts:
            arrived(i)
            torch.add(first, part, out=total)
            first = total
        return total
    if kind == 'collective':
        release(ir.sends.get(g, ()))
    if kind == 'input':
        env[key] = inputs[operand]
    elif kind == 'constant':
        env[key] = operator.attrgetter(operand)(ir.gm)
    elif kind == 'slice':
        (source,), (dim, m) = reads, operand
        size = env[source].shape[dim] // ir.shape[m]
        env[key] = env[source].narrow(dim, ir.coords[ir.device][m] * size, size)
    elif kind == 'narrow':
        (source,), (dim, start, length) = reads, operand
        env[key] = env[source].narrow(dim, start, length)
    elif kind == 'window':
        dim, m, offsets, length, starts, size, fill = operand
        q = ir.coords[ir.device][m]
        if key not in ir.homes:
            env[key] = env[reads[0]].narrow(dim, offsets[q], length)
        else:
            target, at, parts = place(key), 0, []
            for k in reads:
                n = env[k].shape[dim]
                low, high = max(offsets[q], at), min(offsets[q] + length, at + n)
                if low < high:
                    parts.append(env[k].narrow(dim, low - at, high - low))
                at += n
            torch.cat(parts, dim, out=target)
            if fill is not None:
                below, above = min(length, max(0, -starts[q])), min(length, max(0, size - starts[q]))
                target.narrow(dim, 0, below).fill_(fill)
                target.narrow(dim, above, length - above).fill_(fill)
            env[key] = target
    elif kind == 'call':
        local, given = ir.locals[operand], iter(reads)
        args, kwargs = map_arg((local.args, local.kwargs), lambda n: env[next(given)])
        if key in ir.homes:
            out, name = ir.forms[key]
            out(*args, **kwargs, **{name: place(key)})
            env[key] = place(key)
        else:
            env[key] = _call(local, args, kwargs)
    elif kind == 'collective':
        (source,), (me, sent, received, buffer) = reads, ir.roles[operand]
        if operand.kind == 'all-reduce' and me:
            arrived(received[0])
            env[key] = place(key)
        elif operand.kind == 'all-reduce':
            parts = place(buffer)
            env[key] = combine(place(key), env[source],
                               [(received[p], parts[j]) for j, p in enumerate(sorted(received))])
        elif operand.kind == 'reduce-scatter':
            size, parts = env[source].shape[operand.dim] // (len(received) + 1), place(buffer)
            env[key] = combine(place(key), env[source].narrow(operand.dim, me * size, size),
                               [(i, parts[j]) for j, i in enumerate(received.values())])
        elif operand.kind == 'collective-permute' and not received:
            own = any(tuple(pair) == (me, me) for pair in operand.dim)
            env[key] = env[source] if own else place(key).zero_()
        else:
            for i in received.values():
                arrived(i)
            env[key] = place(key)
    else:
        node, keys = operand
        env['output'] = map_arg(node.args[0], lambda n: env[keys[n]])
    release(ir.publishes.get(g, ()))


# a tensor's narrow (dim, start, length), the tensor itself where it is None (2026-11-05, "The revisions of 4e78f6f4":
# a copy's target and source, _run)
def _narrow(tensor, at):
    return tensor if at is None else tensor.narrow(*at)


# design/compilation-coordination.md, N3: the program's retirement of one compiled function's plan.  First every
# device its copies are on is synchronized by torch (its device module's synchronize), so no work
# queued on its pages runs after: the bridge releases a closed claim once no port holds it
# (design/disjoint-pages-change-request.md C3.4) and knows nothing of the GPU, as metal-microbench tools/e2b_replay.m
# (`retired:`) keeps its client while its GPU may still write the placed pages.  Then its claim is closed
# (mesh_plan_retire) and its tensors let go.  The retirement's error value, after its tensors are let go
# 2026-11-04, owner T2 (the coordinator's ruling on review W2's finding 1): a plan was retired, and its tensors let go,
# with the GPU's work on its pages possibly queued.  What must stay: the synchronization before the claim is closed
# 2026-11-05 (design/aot-compilation-change-request.md, appendix item 7, "A publish waits for nothing"; design/
# compilation-coordination.md, "The revisions of e4a7398e", TP-4): then the publishes' notifications still to run are
# waited out (rdma/mesh-torch.mm mesh_torch_notifying), for at most MESH_PLATFORM_NS, ETIMEDOUT past it with the claim
# left standing, before the claim is closed.  Why: a notification writes its range's supplied word, in the plan's state
# section, and the claim's pages are the bridge's to release once it is closed: the grace period of deferred
# reclamation, which ends once every reader that may hold a reference has finished (C3.4, P. E. McKenney and J. D.
# Slingwine, Read-copy update, PDCS 1998); the device's synchronization runs every MPSEvent's signal before it.  This is
# a retirement's wait, never a publish's.  What must stay: the synchronization before the claim is closed
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
# revisions of 4e78f6f4"; the coordinator's brief: "The program's plans are retired with the program (mesh_retire(M,
# client) closes its root ...); after ESTALE or a cancelled range the program plans again: retire, release the view
# after torch's device work on it is done (device synchronize, as `_retire` does now, and the notifications waited out,
# TP-4), then plan the whole program again"), reversing the notes above where one compiled function's claim is closed:
# the program is retired whole, its root closed (mesh-dataflow.h mesh_retire: its `closed` 1 where its `owner` is this
# client), which ends every plan of it (mesh.h mesh_program_plan), after every device of its values is synchronized
# and the notifications are waited out; then its tensors let go and its mapping released once none over it remains
# (_Mapping).  The bridge releases the root once no port holds a plan of it (mesh-dataflow.h mesh_retired_release,
# deferred reclamation), and mesh_plan is EBUSY until then.  What must stay: the synchronization and the wait before
# the root is closed, ETIMEDOUT with it standing
def _retire(context, state):
    for device in state['devices']:
        torch.get_device_module(device).synchronize()
    deadline = time.monotonic_ns() + mesh.PLATFORM_NS
    while TORCH.mesh_torch_notifying():
        if time.monotonic_ns() > deadline:
            raise OSError(errno.ETIMEDOUT, 'a publish\'s notification not run within MESH_PLATFORM_NS of its device\'s '
                                           'synchronization: the program\'s root left standing (rdma/mesh-torch.mm)')
    mesh.LIB.mesh_retire(context.M, context.client)
    mapping = state['mapping']
    mapping.retired = True
    state.update(mapping=None, runs=None, devices=(), stale=True)
    mapping.release()


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
# 2026-11-05 ("The revisions of 4e78f6f4"): the program, retired whole when its callable (program) is collected, where
# each compiled function's callable was.  What must stay: as above
def _dropped(context, state):
    if state['mapping'] is not None:
        _retire(context, state)


# One call of a compiled function at this node, the program's j-th (2026-11-05, design/compilation-coordination.md,
# "The documents' corrections of 2026-11-05", points 1, 2 and 5, and "The revisions of 4e78f6f4"; design/aot-
# compilation-change-request.md, appendix item 6: "have each call launch one execution of its function"): its plan's
# next execution started (mesh-call.h mesh_transfers_start: its waits on the plan's synchronization edges, then the
# raise of `executions` to t + 1, the program's submission of it), counted as serialize's was: once `executions` has
# reached t + 1 the next call is t + 1 whether this one then completes or fails, since `executions` and the supplied
# words only rise (design/disjoint-pages-change-request.md C3.7), and a start that fails before its raise (ETIMEDOUT)
# submitted nothing; then, at a node with a section, each item of the function's program run at execution t (_run) on
# the call's arguments flattened as the exported module's own code flattens them (torch 2.14.0 torch/fx/graph.py
# _PyTreeCodeGen process_inputs: `pytree.arg_tree_leaves(*inputs)`, "A slightly faster version of tree_leaves((args,
# kwargs))", torch/utils/_pytree.py), and its outputs unflattened by the exported program's out_spec as its
# process_outputs does (`pytree.tree_unflatten(out, self.pytree_info.out_spec)`).  At a node the placement operand maps no device the call starts its plan and computes
# nothing (point 2: "at such a node, start its plans' executions and relay, and compute nothing.  `_absent` goes"),
# its result no value, None (design/aot-compilation-change-request.md A4, withdrawn text: "it has no value for any
# program code to read"); a call with a section here that reads one is refused when the program is planned (program).
# _absent and serialize go: the program is planned whole by program, and _execute is a call's part of its run.  What
# must stay: one execution a call, the start at every node of the program
def _execute(context, run, args, kwargs):
    ir, t = run['ir'], run['executions']
    status = mesh.LIB.mesh_transfers_start(C.byref(context), run['plan'], t + 1)
    if mesh.State.from_address(context.M + run['plan']).executions > t:
        run['executions'] = t + 1
    mesh.check(status)
    if ir.device is None:
        return None
    env, inputs = {}, pytree.arg_tree_leaves(*args, **kwargs)
    for g in range(len(ir.program)):
        _run(ir, context, run, g, env, inputs, t)
    return pytree.tree_unflatten(list(env['output']), run['spec'])


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
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 1, and "The
# revisions of 4e78f6f4"; design/aot-compilation-change-request.md A1, corrected: "For a torch program: `torch.export`,
# not a `torch.compile` backend"), reversing the note above where it names a function torch.compile returns: a compiled
# function is a Function, the module torch.export captures (A1) with its placement-choosing and weighting functions
# (A2, A3: "The compilation must take placement as an operand"; "The compilation must take every weight its router and
# its scheduler use as an operand"), neither with a default, held by the traced block as its submodule
# (path_of_module), so the Tracer keeps it a leaf.  It has no forward: nothing runs it eagerly ("nothing runs eagerly
# between its compiled calls", A1), and a call of it outside a traced program meets torch's own
# _forward_unimplemented ("Module [Function] is missing the required "forward" function", torch 2.14.0
# torch/nn/modules/module.py).  _compiled's Dynamo markers, _callee and torch.fx.wrap's form go.  What must stay: each
# compiled function a leaf of the block's trace
class Function(torch.nn.Module):
    def __init__(self, module, placement, weights):
        super().__init__()
        self.module, self.placement, self.weights = module, placement, weights


class Tracer(torch.fx.Tracer):
    def is_leaf_module(self, m, module_qualified_name):
        return isinstance(m, Function) or super().is_leaf_module(m, module_qualified_name)


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
# expressions instead of values".  A body's result as an expression of the arguments run gives it, one for each of its
# placeholders in order, by torch.fx's Interpreter, which "executes an FX graph Node-by-Node" (torch v2.14.0
# torch/fx/interpreter.py), each operation computing an expression: an operation of ALGEBRA rewritten by section
# 4.9.2's rules, any other kept as itself, its operation and its arguments' expressions (a get_attr, a value of the body
# no call writes, is an extent constant, section 3.4.1, named by itself; a call_module, a leaf module of the body, an
# operation as any other)
# 2026-11-05 (design/aot-compilation-change-request.md A10: "A body's symbolic execution runs on torch.fx.Interpreter,
# as the program's run already does (`_Program`), not on a second loop over a graph's nodes"): _execute's loop over
# graph.nodes goes, for _Symbolic, an Interpreter whose operations compute the same expressions.  What must stay: the
# expressions and the rewrite rules as before
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


# Rinard and Diniz, section 4.8 and Figure 15: two operations commute where, executed symbolically "in both execution
# orders", the simplified expressions "denote the same value"; "If it cannot symbolically execute the methods, the
# commutativity-testing algorithm conservatively assumes that the methods may not commute".  Here the two are calls of
# compiled function `callee` whose receiver (section 3.1) is their argument k: its body traced by torch.fx (Tracer, so a
# compiled function it calls stays one operation) is executed on the receiver with one call's other arguments, then on
# that result with the other call's, in both orders.  A body FX cannot trace (torch.fx's TraceError, "symbolically
# traced variables cannot be used as inputs to control flow") is one it cannot execute (TT4)
# 2026-11-05 ("The revisions of 4e78f6f4"): the body a Function's module, where it was Dynamo's original callable.  What
# must stay: the test as before
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
# entry (TT8).  A pass of the compilation, here as _complete is (TT9).  Each call's entry, its compiled function's.
# Its graph operations are the graph module's (rdma/mesh_graph.py): the chains are the paths of the linear forest each
# call's follower makes (paths); `held` and the chained order are the transitive closures (closure) of the reads and of
# the reads with each chain's calls taken unordered; each function's bound, "the most calls of a function outstanding
# at once is the number of its calls in the block, or fewer where Bernstein's conditions order them" (A10, "Its
# bound"), is the width of the order on its calls (width: Dilworth's theorem, by Fulkerson's matching in Hopcroft and
# Karp's phases; the three as the turns' D.5 names them for chains, design/plan-to-bridge-change-request.md, "The
# turns"; TT3)
# 2026-11-05 (design/aot-compilation-change-request.md A10, "Its graph operations are the graph module's": "The order's
# transitive closure, the walk of commuting chains, and the width by Fulkerson's matching in Hopcroft and Karp's phases
# are functions of rdma/mesh_graph.py ... rdma/mesh_compile.py calls them and keeps no copy"; the operator's ruling on
# C9a, design/compilation-coordination.md, "The revisions of f3f7cf46"): _analyse's `ancestors` and chain loop, and
# _width, go, for mesh_graph.closure, mesh_graph.paths and mesh_graph.width (design/compilation-coordination.md, "The
# revisions of abecd389", T4-1, T4-2).  What must stay: the entries, `orders` and `commuting` as before
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 1, and "The
# revisions of 4e78f6f4"): a call's compiled function is the one _Export gave it (`of`, its ordinal), a Function called
# with other argument metadata being another (design/aot-compilation-change-request.md, appendix item 6: JAX traces
# anew only "on a trace cache miss"), so each bound is the width of one compiled function's calls; `modules`, each
# compiled function's module, the body _commutes executes.  The entries are each compiled function's, by its ordinal.
# What must stay: the entries, `orders` and `commuting` as before
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


# design/aot-compilation-change-request.md A10: the program a block traces to.  It runs as Pathways' traced program
# does ("When the user wants to run a traced program, it calls the PATHWAYS client library which first assigns virtual
# devices to any computations that have not been run before, and registers the computations with the resource
# manager, triggering the servers to compile the computations", section 4.2), by torch.fx's Interpreter, which
# "executes an FX graph Node-by-Node" (torch v2.14.0 torch/fx/interpreter.py), in the block's order, which keeps every
# order constraint.  A call of a compiled function runs with its function's entry current (CALLING), so the backend's
# compile function, which Dynamo calls within that call where it compiles a graph of the function, carries the entry in
# that graph's IR (TT7).  CALLING: the current entries, the innermost last
# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", points 1 and 2, and "The
# revisions of 4e78f6f4"), reversing the note above where a call runs its function's Dynamo-compiled callable: CALLING
# goes, every function being exported and its entry carried before the program runs (program); a call of a compiled
# function runs its execution (`execute`, _execute), an operator.getitem of a call's result takes its item, and an
# item of no value (a call at a node with no section) is no value.  What must stay: the block's order
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


# A compiled function as exported (2026-11-05, "The revisions of 4e78f6f4"): its Function, its call's argument
# metadata (`signature`), its exported graph (`gm`), its example arguments, its fake result and its result's out_spec
Compiled = collections.namedtuple('Compiled', 'function signature gm examples result spec')


# a value's metadata, as a compiled function's argument: a tensor's shape, strides, element type and device, any other
# value itself
def _metadata(value):
    return (tuple(value.shape), tuple(value.stride()), value.dtype, value.device) \
        if isinstance(value, torch.Tensor) else value


# The capture of a traced block's compiled functions (2026-11-05, design/compilation-coordination.md, "The documents'
# corrections of 2026-11-05", point 1, and "The revisions of 4e78f6f4"; design/aot-compilation-change-request.md A1,
# corrected: "every function of a torch program is captured and compiled before the program runs, the same at every
# rank, and nothing runs eagerly between its compiled calls").  The block's nodes run in its order on torch.fx's
# Interpreter, as A10 has a body's symbolic execution run ("not on a second loop over a graph's nodes"), on fake
# example values:
# - a placeholder takes its example input; an operator.getitem of a call's result that is no tensor (a tuple or list
#   the call returns) takes its item; a call of a Function exports it; the output returns.  Any other node (a module
#   or operator the block runs itself, an attribute, a getitem of a tensor) is refused, an exception before anything
#   is planned.
# - A call's arguments are fake tensors of their metadata, every one of one FakeTensorMode (`mode`), made by
#   torch.empty_strided in it, so nothing is allocated, in the region or by torch: "FakeTensor extends MetaTensors to
#   also carry an additional `fake_device`" (torch 2.14.0 torch/_subclasses/fake_tensor.py), and a factory in that mode
#   runs on the meta device and wraps its result with the device asked for (torch/_subclasses/fake_impls.py
#   constructors: `new_kwargs["device"] = torch.device("meta")`, then from_meta_and_device with `out_device`).
#   torch.export takes them: its non-strict tracing fakifies each input by its own mode's from_tensor
#   (torch/_export/non_strict_utils.py fakify: `fake = mode.from_tensor(t, source=source,
#   symbolic_context=symbolic_context)`), which describes a fake tensor by its fake device (torch/_subclasses/
#   meta_utils.py describe_tensor, `device=t.device`) and its sizes ("FakeTensor from a different ShapeEnv.  Transfer
#   symbols."), and _export_for_training expects them ("If we are tracing with fake inputs, it is expected to see fake
#   tensor constants", torch/export/_trace.py); every fake input of one call must be of one mode (torch/_guards.py
#   detect_fake_mode: "fake mode ... doesn't match mode"), so all are `mode`'s.
# - Each export is torch.export.export(module, args, kwargs, strict=False), as mesh's engines.py captured
#   (`torch.export.export(module, tuple(arguments), strict=False)`, mesh 1f794b8f^ rdma/torch-mesh/torch_mesh/
#   engines.py), which "produces a traced graph representing only the Tensor computation of the function in an
#   Ahead-of-Time (AOT) fashion, which can subsequently be executed with different inputs or serialized"
#   (torch/export/__init__.py); then run_decompositions with an empty table, the functional ATen dialect (partition's
#   note); its module, "a self contained GraphModule with all the parameters/buffers inlined" (ExportedProgram.module,
#   with no guards submodule: check_guards=False), with torch's own pass removing its run-time assertions
#   (torch/_export/passes/remove_runtime_assertions.py _RemoveRuntimeAssertionsPass, _assert_tensor_metadata among
#   them), since every metadata is fixed when the function is compiled ("Compile-time, never runtime checks", the
#   operator).  Its result is its outputs' fake values, each output node's meta['val'] (_value's note), unflattened by
#   its call_spec's out_spec, as fake tensors of `mode`.
# - A Function called again with the same argument metadata (its pytree's spec and each leaf's _metadata) is the same
#   compiled function; with other metadata another, exported anew, as JAX traces anew only "on a trace cache miss"
#   (R. Frostig, M. J. Johnson and C. Leary, SysML 2018; design/aot-compilation-change-request.md, appendix item 6).  A
#   compiled function's ordinal is its place in the order of first calls, the same at every rank whatever its device
#   (mesh-collective.h struct mesh_function: function k named (group, k + 1)).
# `of`: each call node's compiled function; `calls`: the block's calls' compiled functions in program order (mesh_plan's
# `calls`); `reads`: for each call, the calls whose results it reads, through getitems (`origin`).
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
            out = [a.meta['val'] if isinstance(a, Node) else a for a in gm.graph.output_node().args[0]]
            result = pytree.tree_unflatten([self.fake(v) for v in out], exported.call_spec.out_spec)
            self.compiled.append(Compiled(function, signature, gm, flat, result, exported.call_spec.out_spec))
            found = [len(self.compiled) - 1]
        self.of[n], self.origin[n] = found[0], len(self.calls)
        self.reads.append({self.origin[a] for a in n.all_input_nodes if a in self.origin})
        self.calls.append(found[0])
        return self.compiled[found[0]].result


# design/aot-compilation-change-request.md A1, A8 to A10 (2026-11-05, design/compilation-coordination.md, "The
# documents' corrections of 2026-11-05", points 1, 2, 4 and 5, and "The revisions of 4e78f6f4"): program(block,
# *example_inputs), the program's callable, in place of torch.compile(f, backend=backend(placement, weights)) and
# serialize, which go.  `block`, an nn.Module whose forward calls its compiled functions (Function submodules), is
# traced by torch.fx's symbolic tracer with each compiled function a leaf (Tracer; A10, "The calls' graph"), and its
# compiled functions are exported at their calls in program order (_Export); A10's analysis gives each its entry
# (_analyse) and each Function's placement function its operand, once (A2: placement(gm, example_inputs), its exported
# graph and its example arguments).  Then the program is planned (`make`), before it runs:
# - on one map read (mesh_nodes_read), each compiled function partitioned, lowered and weighed (partition, lower,
#   its weighting function, A3), one IR a function, and refused at compile time where a call with a section at this
#   node reads the result of a call with none (A4, withdrawn text: "it has no value for any program code to read");
# - one mesh_plan call for the whole program (point 5: "The plan call's form for a whole program is A8 and A9 ... there
#   is one IR of the program and one serialized operand of it"): every function's declared transfers, weights, values,
#   dependences and A10 bound as its struct mesh_function, the block's calls in program order as `calls`, the
#   program's one mapping out (`*view`); every plan's map serial then compared with the IR's (ESTALE, the coordinator's
#   ruling on review V2's finding 6), every value bound over the mapping (_bind), MPS values over one buffer of it
#   (_placed_buffer); a failure after mesh_plan retires the program;
# - after an ESTALE or a cancelled range (ECANCELED) the next run plans again: the program retired (_retire), then
#   planned whole on the map as it is; mesh_plan is EBUSY until the bridge has released the root (mesh-collective.h:
#   "EBUSY where the root's `owner` is not 0 ... before anything is written"), raised as any failure, and the next
#   run tries again (the coordinator's brief for point 5).
# Each run interprets the block (_Program), each call of a compiled function one execution of its plan (_execute).
# The program is retired when its callable is collected (_dropped).  What must stay: no placement and no weight chosen
# here; every operand of mesh_plan the same at every rank but this node's values
def program(block, *example_inputs):
    tracer = Tracer()
    gm = torch.fx.GraphModule(tracer.root, tracer.trace(block))
    exporter = _Export(gm, FakeTensorMode())
    exporter.run(*[exporter.fake(x) for x in example_inputs])
    compiled, of, calls, reads = exporter.compiled, exporter.of, exporter.calls, exporter.reads
    entries = _analyse(gm, of, [record.function.module for record in compiled])
    operands = [record.function.placement(record.gm, record.examples) for record in compiled]
    context, state = _context(), {'stale': True, 'mapping': None, 'runs': None, 'devices': ()}

    def make():
        if state['mapping'] is not None:
            _retire(context, state)
        nodes, header = mesh.Nodes(), mesh.Header.from_address(context.M)
        mesh.check(mesh.LIB.mesh_nodes_read(C.byref(context), C.byref(nodes)))
        irs = []
        for k, (record, operand) in enumerate(zip(compiled, operands)):
            ir = partition(record.gm, operand, nodes, header.block * header.pgsz)
            ir.calls = entries[k]
            lower(ir)
            ir.weights = list(record.function.weights(ir))
            if len(ir.weights) != len(ir.declared) or not all(isinstance(w, mesh.Weight) for w in ir.weights):
                raise ValueError('the weighting function gave %d entries for %d declared transfers; one mesh.Weight '
                                 'each (A3)' % (len(ir.weights), len(ir.declared)))
            irs.append(ir)
        for j, k in enumerate(calls):
            absent = sorted(r for r in reads[j] if irs[calls[r]].device is None)
            if irs[k].device is not None and absent:
                raise ValueError('call %d of the program, of compiled function %d, has its section at map index %d '
                                 'and reads the result of call %d, whose function has none there: such a node "has '
                                 'no value for any program code to read" (design/aot-compilation-change-request.md '
                                 'A4, withdrawn text)' % (j, k, nodes.self, absent[0]))
        arrays = [((mesh.Declared * len(ir.declared))(*ir.declared), (mesh.Weight * len(ir.weights))(*ir.weights),
                   (mesh.Value * len(ir.values))(*ir.values),
                   (mesh.Dependence * len(ir.dependences))(*ir.dependences)) for ir in irs]
        functions = (mesh.Function * len(irs))(*[
            mesh.Function(C.cast(d, C.POINTER(mesh.Declared)), C.cast(w, C.POINTER(mesh.Weight)),
                          C.cast(v, C.POINTER(mesh.Value)), C.cast(e, C.POINTER(mesh.Dependence)), len(d), len(v),
                          len(e), 0, ir.calls.outstanding, 0) for ir, (d, w, v, e) in zip(irs, arrays)])
        view = mesh.View()
        mesh.check(mesh.LIB.mesh_plan(C.byref(context), _group(), _program(), functions, len(irs),
                                      (mesh.U * len(calls))(*calls), len(calls), C.byref(view)))
        state.update(mapping=_Mapping(view), devices={layout[2] for ir in irs for layout in ir.layouts})
        try:
            if any(mesh.State.from_address(context.M + f.plan).serial != nodes.serial for f in functions):
                raise OSError(errno.ESTALE, 'a plan\'s map serial is not the one its IR was built on: plan again')
            buffer = _placed_buffer(view) if 'mps' in state['devices'] else None
            try:
                bound = [_bind(context, ir, values, buffer, state['mapping']) for ir, (_, _, values, _) in
                         zip(irs, arrays)]
            finally:
                if buffer:
                    PUT(buffer, SELECTOR[b'release'])
        except BaseException:
            _retire(context, state)
            raise
        state.update(stale=False, runs=[{'ir': ir, 'plan': f.plan, 'executions': 0, 'spec': record.spec,
                                         'places': places, 'sections': sections, 'devices': devices}
                                        for ir, f, record, (places, sections, devices) in
                                        zip(irs, functions, compiled, bound)])
    make()

    def run(*inputs):
        if state['stale']:
            make()
        try:
            return _Program(gm, lambda node, args, kwargs: _execute(context, state['runs'][of[node]], args,
                                                                    kwargs)).run(*inputs)
        except OSError as failure:
            if failure.errno in (errno.ESTALE, errno.ECANCELED):
                state['stale'] = True
            raise
    weakref.finalize(run, _dropped, context, state)
    return run
