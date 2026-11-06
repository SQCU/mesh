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
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines" and "Answered in
# the documents (2026-11-05, the coordinator, on this part's record)"; "The revisions of 55b9cbad"): every part sent
# from its producer's pages, each value laid out by PB R2's layout answer (V1, the note above _span); a rank's own part
# shared through the page table (V2, struct mesh_share); a halo's pieces landing in its window's value (V3); every
# parameter and buffer an invariant (V4); every program input placed and written by the caller (V5, program); the
# runner's constants of the plan (K3, _steps); one deadline an execution (B1); mesh_plan waiting for the release (B2);
# the retirement after torch's work in an MPSEvent's notification (B3); one pairwise-exchange body (G2, _exchange); the
# program's own mapping of its rows (G3, _mapped); the program's operand struct mesh_algo (G4).  What must stay: no
# placement and no weight chosen here
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1, R-2,
# R-4, R-7, R-9 and R-10; design/plan-to-bridge-change-request.md R1 at mesh 56ede97e; "The revisions of 14b9a757"),
# reversing the notes above where they lay a value out for a transfer, copy a part, share an own part or pad a layout:
# every declared transfer's ends are views of the values that hold its bytes, as torch has them (Place, _strided), so a
# part is sent from its producer's pages and lands at its bytes in its destination's own layout, and `_laid`, `_claim`,
# the padded layout, `ir.fallbacks`, `ir.shared`, `ir.share`, mesh.Share and AllToAll's buffer go (R-1); an own part
# lies in its result where its producer writes it there, else the compilation stops (_inside); every operator with an
# out variant runs it, one `out` per result (R-2, _forms); each execution is a straight-line sequence of its kernels,
# waits and publishes made once per copy (R-4, _steps); make()'s waits share one deadline and every run's wait ends at
# MESH_PLATFORM_NS without progress, the first ETIMEDOUT ending the run (R-7); the program's mapping is
# mesh_view_values' (R-10).  What must stay: no placement and no weight chosen here
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
# every rank's; `blocks`, by key, the blocks of its pages its ranges hold, every rank's (_part, _landing); `bases`,
# each key's base, the key whose value its pages are one run of, and its element offset there, every rank's (_over).
# This node's: `at`, each declared transfer's place, the key whose value it is sent from or lands in and the element of
# that value it begins at (the lowerings); `added`, the values of no key of the program (a part copied once, a
# reduction's received parts); `sends`, by collective item, the ranges it releases before it waits, each with the copy
# that makes it where one does; `publishes`, by item, the ranges released after it; `homes`, each placed key's value,
# element offset and shape (_place).  What must stay: every operand of mesh_plan the same at every rank
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", V5; "The
# revisions of 55b9cbad"): `inputs`, this node's: by the index of an input of the traced block, the value of `lag`
# HOST_LAG that holds it here and its shape (program).  What must stay: as above
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", V1 to V3; "The
# revisions of 55b9cbad"): every rank's: `form` and `fixed`, each value's layout (the note above _span); `landing`, by
# a window's region and each CollectivePermute's arrival, the value they lie in, every piece of the region landing at its
# place there, and the arrival's first slice along the window's dimension (_region); `regions`, by that landing value,
# the windowed operand, the dimension, the operand's first slice there and the grain of its parts (_region); `halos`,
# by landing value, where its middle comes from (_halo).  This node's: `views`, keys lying in another's value, by key
# (that key, the first element, sizes, strides); `shared`, the own parts shared (the note above _span), and `share`
# their struct mesh_share (_place); `fallbacks`, by item, the copies it makes where a share cannot be (_place);
# `added`, the values of no key of the program, each with its positions where it is local.  What must stay: every
# operand of mesh_plan the same at every rank
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1: "That
# removes `_laid`, `_claim`, `ir.fallbacks`, struct mesh_share with its tests on both sides, the padded layout and
# AllToAll's buffer.  A value's layout is the program's own, torch's strides"; "The revisions of 14b9a757"): `form`,
# `fixed`, `views`, `shared`, `share`, `fallbacks`, `regions`, `halos`, `blocks`, `bases`, `sends` and `blockbytes`
# go.  Every rank's: `places`, by device, each key's Place (its strided view of the value holding its bytes, _places);
# `inside`, by key, the value its producer writes it into (owner, dimension, its start there by device: an AllGather's
# own part, a window's middle, a halo's arrival and region, _reshard, _inside and _region); `results`, by the result of
# an operator of several, that operator's key, and `taken`, by an operator.getitem of one, the result it takes (R-2,
# _forms); `sent`, the keys a collective reads and the reads of the slices, narrows and viewed windows among them;
# `contiguous`, the view nodes for which "it will not be possible to view self tensor as shape without copying it (e.g.,
# via contiguous())" (torch 2.14.0 torch/_tensor_docs.py, view; _view).  This node's: `owned`, each value's index by the
# key that owns it.  What must stay: every operand of mesh_plan the same at every rank
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
        self.publishes, self.inputs = {}, {}
        self.taken, self.results, self.sent, self.contiguous = {}, {}, set(), set()

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
    # 2026-11-05 ("The revisions of 55b9cbad", V1): `size`, the range's bytes, its run's (a run of whole blocks in its
    # value's layout, padding within it: the note above _span), where they were its local shape's.  What must stay:
    # one identity a range, its bytes every rank's
    # 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1:
    # "A transfer's source and destination are each a view (value, offset, sizes, strides)"; mesh-collective.h struct
    # mesh_declared at mesh 31ee5d4a: "`from`: the range's bytes in its source's value, the same in every declared
    # transfer of one range ... `to`: declared[i]'s bytes in its destination's value"; "The revisions of 14b9a757"),
    # reversing `size` and `first`: the range's `part`, its source's mesh.Strided view, and each destination's
    # `landings[d]`, both every rank's (_strided).  What must stay: one identity a range, its views every rank's
    def transfer(self, source, destinations, kind, part, landings):
        self.identity += 1
        transfers = {}
        for d in destinations:
            transfers[d] = len(self.declared)
            self.declared.append(mesh.Declared(self.map[source], self.map[d], self.identity, kind, mesh.ABSENT, 0,
                                               part, landings[d]))
        return transfers

    # one value of this node (mesh-collective.h struct mesh_value), MESH_EXECUTION: a routed one in the window (`wire`
    # 1), a local one outside it (0), each with its first and last positions on this node's program order, its layout,
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
    # 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G3; "The
    # revisions of 55b9cbad"): its place is in the program's own mapping of its rows (_mapped), `view` being gone from
    # struct mesh_value.  What must stay: as above
    # 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", V4 and V5;
    # "The revisions of 55b9cbad"): `kind` and `lag` given, MESH_INVARIANT for a parameter or buffer of the exported
    # module (_place) and `lag` HOST_LAG for a program input (program), each MESH_EXECUTION and lag 0 otherwise as
    # before.  What must stay: as above
    def value(self, dtype, numel, device, wire, begin, end, kind=mesh.EXECUTION, lag=0):
        self.values.append(mesh.Value(section=mesh.Section(bytes=dtype.itemsize * numel), kind=kind, wire=wire,
                                      begin=begin, end=end, lag=lag, run=1))
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


# a key's device type, its fake value's (_value), result j of an operator of several its node's j-th fake value
# (2026-11-06, R-2; "The revisions of 14b9a757": a result of several has a value of its own, _forms)
def _device(ir, key):
    return (key[0].meta['val'][key[3]] if key in ir.results else _value(key[0])).device.type


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
# (that value stands through the execution).  The key of `node`'s local value in `want`.  Before an AllGather and an
# AllToAll, its member's own part ('copy'): MPICH allgather_intra_ring.c, "First, load the "local" version in the
# recvbuf", and alltoall_intra_pairwise.c, "Make local copy first" (MPIR_Localcopy; _copied)
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
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", V3: "The halo
# window concatenates its received pieces, `torch.cat(parts, dim, out=target)` (2192).  The pieces must land at their
# places in the window region's value (point 7), so no concatenation remains"; "The revisions of 55b9cbad"), reversing
# step (2)'s concatenation: where a region is no view of its one read, the pieces and the value lie in one landing
# value, (2)'s concatenation as a layout: `landing`, each arrival's and the region's, the arrival's first slice along
# the dimension (the reads' order, Figure 9b's); `regions`, the landing value's operand, dimension, the operand's first
# slice and the gcd of the reads' extents there (every part's bound a multiple of it, _grain); the region (3) a view of
# it (_halo).  What must stay: the steps (1), (3) and (4) as before, every rank's
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1;
# design/plan-to-bridge-change-request.md R1 at mesh 56ede97e: "it lands at its byte ranges in its destination value's
# layout"; "The revisions of 14b9a757"), reversing `regions`, the grain and the landing value's padded layout: the
# landing value is its reads laid end to end along the dimension in its own row-major layout (ir.shapes, a value of no
# key, `added`), each arrival and the region lying in it (`inside`: the arrival at its start, the region at this
# partition's offset, every rank's by device).  What must stay: the steps (1), (3) and (4) as before, every rank's
# GSPMD section 4.5, "DynamicSlice to shard replicated dimensions": an operand `whole` along the dimension at every
# partition has no halo exchange, its region the DynamicSlice of its value at the partition's start, its landing value
# padded where a window lies outside it; the middle ('copy') in place where its producer writes through its view
# (_inside), one local copy otherwise (MPICH MPIR_Localcopy; _copied)
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
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1; "The
# revisions of 14b9a757"), reversing the block's bytes: no block enters the compilation, since a transfer's ends are
# byte views (_strided), not runs of blocks.  After the partitioning, each key's producer and the keys collectives
# read with the reads of their slices, narrows and viewed windows (`producer`, `sent`), the forms (_forms), the values
# an own part lies in (_inside), every device's places (_places) and the pages each key may lie over (_over), every
# rank's.  What must stay: the placement operand's checks
# and the refusal of an undivided dimension
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
                               lambda key: items[key][1][:1] if items[key][0] in ('slice', 'narrow')
                               or _viewed(items[key]) else ())
    _forms(ir)
    _inside(ir)
    ir.places = {device: _places(ir, device) for device in ir.coords}
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
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-2:
# "Operators with several outputs run their out variants ... one `out` per result"; W4: "Operators with several outputs
# are classed `'functional'` (1124–1127), so torch allocates their results, though they have out variants (for example
# `native_layer_norm.out`)"; "The revisions of 14b9a757"), reversing the one-tensor return: an operator returning k
# tensors, each with elements, takes the out variant to_out_variant finds, whose mutable arguments are as many as the
# returns (torch 2.14.0 torch/_library/_out_variant.py: `if len(mutable_args) != len(schema.returns): continue`), each
# return aliasing its one in order (_has_valid_out_variant_returns: "Each return must alias exactly one mutable arg, in
# order"); torchgen autogenerates it for such an operator, its out arguments `out0` .. `outk-1` and its tag `out`
# (torchgen/native_function_generation.py: `name="out" if len(func.returns) == 1 else f"out{i}"`, `tags.add("out")`;
# native_functions.yaml: "native_layer_norm(...) -> (Tensor, Tensor, Tensor)", "autogen: native_layer_norm.out").  A
# form is (the out variant, its (out argument, result key) pairs in order): a key's own where it returns one tensor,
# else result j's key + ('out', j), a value of no key of the program made at the operator's item (`added`, `results`,
# its fake value metas), which an operator.getitem of it takes (`taken`).  What must stay: every rank's forms the same
# 2026-11-06 (the C9 hook's remark on f2a1efd1: "_forms(): a forward data-flow pass over the program's items in order,
# each operator's result and getitem's taken result propagated from its operands' metas; it moves into the graph
# module"; "The revisions of 14b9a757"): the pass is mesh_graph.forward over the program's keys, each key's value its
# meta value (_formed: an operator's result on the meta device, else _meta's, UNKNOWN where neither is), its forms
# recorded as before.  What must stay: every rank's forms the same
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
# revisions of 4e78f6f4"): `inplace` and _fresh go, for each part's place (_part, _directed); `bases`, each key's base
# and element offset in it, by the same pass (_base, the note above _span).  What must stay: every rank's `over` the
# same
def _viewed(item):
    return item[0] == 'window' and len(item[1]) == 1 and item[3][6] is None


# 2026-11-05 (the C9 hook's remark on 63638057: "a topological pass over the program DAG ... it moves into the graph
# module"; design/compilation-coordination.md, "The revisions of e4a7398e", C9): the pass is mesh_graph.forward over the
# keys in the program's order, each key's predecessors the keys its item reads, its transfer the rule above.  What must
# stay: each key's set as above
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", V4: "The
# exported module's parameters and buffers are read from torch's allocations (2170, `_over`'s empty set for a
# constant).  That fails region-only memory"; "The revisions of 55b9cbad"): a module attribute that is a tensor (a
# parameter, a buffer, a lifted constant) makes pages of its own, an invariant of the plan (_place), where its set was
# empty.  What must stay: each other key's set as above
# 2026-11-05 ("The revisions of 55b9cbad", V3): a CollectivePermute's arrival and a window's region lie over their
# landing value (_region), the region over its reads' pages as well (its middle may be the operand's own); a
# CollectivePermute's own pair, where its result was its operand's, no longer arises apart (no pair of _region's is
# one).  What must stay: each other key's set as above
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1 and
# R-2; "The revisions of 14b9a757"), reversing `bases` and `landing`'s sets: a key its producer writes inside another
# value (`inside`: an AllGather's own part, a window's middle, a halo's arrival and region) lies over that value, a
# region over its reads' pages as well; an operator of an out variant over its results' values (each its own, or the
# value it lies inside); a key whose place is its own value (_places) over that value; any other over none.  `producer`
# is partition's.  What must stay: each other key's set as above
def _over(ir):
    items, places = {item[2]: item for item in ir.program if item[0] != 'output'}, ir.places[min(ir.coords)]

    def rule(key, sets):
        kind, reads, _, operand = items[key]
        joined = frozenset().union(*sets)
        if key in ir.inside:
            return frozenset((ir.inside[key][0],)) | (joined if kind == 'window' else frozenset())
        if places[key] is not None and places[key].owner == key:
            return frozenset((key,))
        if kind in ('slice', 'narrow') or _viewed(items[key]):
            return joined
        if kind == 'call' and isinstance(ir.forms.get(key), tuple):
            return frozenset(ir.inside[result][0] if result in ir.inside else result for _, result in ir.forms[key][1])
        return joined if kind == 'call' and ir.forms.get(key) is None else frozenset()
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


# A part's place (2026-11-06, design/compilation-coordination.md, "The root cause of the copies, and the revised work
# list", R-1: "A transfer's source and destination are each a view (value, offset, sizes, strides).  The compilation
# flattens each into its list of (row, offset, bytes) blocks, adjacent blocks merged, as datatype processing flattens a
# type map"; design/plan-to-bridge-change-request.md R1 at mesh 56ede97e: "A torch tensor is a strided array: one
# storage, with an offset, sizes and strides ... A view (a slice, a narrow, a permutation) is the same storage under a
# new offset, sizes and strides"; "So a part is sent from its producer's pages, whatever its layout, with no copy, and
# it lands at its byte ranges in its destination value's layout"; "The revisions of 14b9a757"), reversing the note of
# 2026-11-05 that stood here (a range a run of whole blocks, a value's layout fixed by its first collective and padded,
# a part laid out into a value of its own, an own part shared, received parts landing in a buffer in their sender's
# layout):
# - A Place is a key's strided view of the value holding its bytes, as torch has it: `owner`, the key whose value it is
#   (one with pages of its own, _over, or a value of no key, `added`); `offset`, `sizes` and `strides` in elements, a
#   torch tensor's storage offset, sizes and strides over that value's row-major storage (torch's own layout,
#   _contiguous); `writer`, the key whose producer writes those bytes (its own; a view's base's; for a result of
#   several, its operator's).
# - _places gives every key's place at every device, every rank's, a forward data-flow problem over the program's keys
#   (rdma/mesh_graph.py forward, as `over`): a key inside another value (`inside`) its narrow there at that device's
#   start; a slice, a narrow and a viewed window its read's view narrowed at that device's offset (_cut; "A view (a
#   narrow, a slice) is its base's pages", design/aot-compilation-change-request.md appendix item 7); a view an
#   operator makes its base's place, at the offset, sizes and strides torch's own view gives it over that place (_view),
#   or its own value where torch copies (`contiguous`); a key with pages of its own (a call's result a collective reads,
#   `sent`: "an operator with no out variant, whose result is copied once into its placed pages where it must be
#   there", appendix item 7), and a value of no key, its own value whole; a local copy's ('copy') its destination, or
#   its source where that is the destination's view (MPI 4.1 section 7.8, in place: "Data sent and received must have
#   the same type map"); any other none (an input, a result torch allocates), never a transfer's end.
# - _strided gives a place's mesh.Strided (mesh-collective.h struct mesh_strided): its offset and strides in bytes,
#   `element` the element's bytes, its dimensions of size 1 dropped and adjacent ones merged where they nest, as torch's
#   own collapse merges them ("The dimensions can be merged if and only if they are all "nested" with each other.  That
#   is, they all have the property that stride[i] = stride[i+1] * shape[i+1]", torch 2.14.0 torch/_prims/__init__.py
#   collapse_view, whose _collapse_view_helper takes `shape[idx] == 1` alike), which keeps the bytes' C order, its
#   `dims` (size, stride) pairs at `pairs` (MPI 4.1 section 6.1.3, MPI_TYPE_CREATE_SUBARRAY's ndims).
# - _produced and _send: a sent range's production at its writer's third step, its source read the span of the reads of
#   its key and its value (_readers), and this node's release of it after that writer (`publishes`); "The plan refuses
#   a transfer whose source is MESH_INVARIANT (EINVAL at the plan)" (design/problems-of-14b9a757-change-request.md,
#   "Sources for the second stops", 8).
# What must stay: every rank's places and views the same; no part copied, laid out or shared for a transfer
Place = collections.namedtuple('Place', 'owner offset sizes strides writer')


def _contiguous(shape):
    return tuple(math.prod(shape[d + 1:]) for d in range(len(shape)))


def _own(ir, key):
    shape = _layout(ir, key)[1]
    return Place(key, 0, shape, _contiguous(shape), ir.results.get(key, key))


def _narrowed(place, dim, start, length):
    return place._replace(offset=place.offset + start * place.strides[dim],
                          sizes=place.sizes[:dim] + (length,) + place.sizes[dim + 1:])


# part j of a value of parts along its first dimension: an AllReduce root's and a ReduceScatter keeper's received
# parts, each landing whole at its place ("the received slices landing at their places in one value")
def _part(ir, key, j):
    shape = _layout(ir, key)[1][1:]
    return Place(key, j * math.prod(shape), shape, _contiguous(shape), key)


# a slice's, a narrow's and a window's view of its read at `device`, (dimension, start, length): a slice's this device's
# shard (GSPMD section 4.5, DynamicSlice; `sizes` its read's), a narrow's its own, a window's this device's offset
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
    if (kind in ('slice', 'narrow') or _viewed(item)) and reads[0] is not None:
        return _narrowed(reads[0], *_cut(ir, kind, operand, reads[0].sizes, device))
    form = ir.forms.get(key)
    if kind == 'call' and form is None and isinstance(ir.locals[operand].target, torch._ops.OpOverload) \
            and _aliases(ir.locals[operand]):
        view = _view(ir, key, ir.locals[operand], item[1], reads)
        if view is not None:
            return view
    if kind == 'collective' or kind == 'constant' and isinstance(key[0].meta.get('val'), torch.Tensor) \
            or isinstance(form, tuple) and any(result == key for _, result in form[1]) \
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


# NCCL User Guide, In-place Operations: "For ncclReduceScatter and ncclAllGather, in place operations are done when the
# per-rank pointer is located at the rank offset of the global buffer"; MPI 4.1 section 7.7, in place: "the input data
# of each MPI process is assumed to be in the area where that MPI process would receive its own contribution to the
# receive buffer".  A whole own part's ('copy' of no cut: an AllGather's, a window's middle) producer writing through
# an `out` tensor (an out variant, a result of an operator of several, a ReduceScatter's torch.add) writes it at its
# view in the result (`inside`), where it lies in no other value already
def _inside(ir):
    for kind, reads, key, operand in ir.program:
        if kind == 'copy' and operand is None:
            target = ir.taken.get(reads[0], reads[0])
            producer, _, _, collective = ir.program[ir.producer[ir.results.get(target, target)]]
            if target not in ir.inside and (target in ir.results or producer == 'call'
                                            and isinstance(ir.forms.get(target), tuple)
                                            or producer == 'collective' and collective.kind == 'reduce-scatter'):
                ir.inside[target] = ir.inside[key]


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
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", V1 and K3; "The
# revisions of 55b9cbad"): a member's partial sent from its run (_sent: its producer's pages where it is a value's own,
# else laid out once into a value of its own), the root's received partials landing in one value each viewed as the
# partial is (_buffer), the sum its result's value whole; `roles` this node's place in the group and its receives in
# the order the runner reads them, each with the key of the part it lands in (K3: "the received parts' landing order,
# which lowering records in `ir.at` and the runner re-derives", now the runner's constant).  What must stay: the
# events as before, a member's production where its partial is laid out at the item's sends
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1; "The
# revisions of 14b9a757"), reversing _sent, _buffer and the sender's layout at the root: a member's partial is sent from
# its place, whatever its layout (a Dot's result, or a transposed convolution's cut of it), its view `from`; it lands at
# its part of the root's received value (`received`, its parts row-major in the partial's shape, a value of no key) in
# that value's own layout, `to`; the sum goes from the root's result whole to each member's result whole.  `roles` and
# the events as before; `at` each end's value.  What must stay: the events as before; no partial laid out or copied
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
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", V1 and V2, and
# "Answered in the documents", "A part that is not whole blocks"; "The revisions of 55b9cbad"), reversing the note
# above where a part must be a row-major run beginning a block (_landing) and the own shard not made in place is copied:
# the result laid out with the gathered dimension most-major, each part padded to whole blocks (PB R2, corrected: "So a
# collective's result takes the layout with the collective's dimension most-major, each part padded to whole blocks"),
# its stride the shard's run's where the shard is sent from a run, so the result's part p is member p's shard as laid
# out, and the result one strided view of its logical shape, read by its consumers as it is ("The consumer reads the
# result through its strided view"; Megatron's column-parallel gather along the last dimension, "the common case", is
# such a view).  The own shard: made in place by its producer (`inside`, as before); else sent from its run (_sent) and
# the result's part me those pages (`shared`, V2: "make the result's rows for the own part name the operand's own
# pages"), where it was copied into the result's narrow.  What must stay: each member's range to every other member,
# MESH_JOIN
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1;
# design/plan-to-bridge-change-request.md R2 at mesh 56ede97e, withdrawn: "With transfers as flattened lists of
# blocks, a part lands at its byte ranges in the result's own layout, whatever its dimension.  No layout is chosen for a
# transfer, and no part is padded"; "The revisions of 14b9a757"), reversing the padded most-major layout, _sent and the
# own shard's share: the result is row-major in its logical shape, torch's own; member p's shard (`key`, its 'copy')
# is the result's view at p x its extent along the gathered dimension, sent from there and landing at the same view of
# every other member's result, so `from` and each `to` are one view.  What must stay: each member's range to every
# other member, MESH_JOIN
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


# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G2: "AllToAll's
# and ReduceScatter's one pairwise-exchange body, with CollectivePermute's send branch (mesh_compile.py 1430–1502)", to
# be merged ("Refactor by pattern analysis: merge duplicated patterns found by analysis"); "The revisions of
# 55b9cbad"): the lowerings' one body, a declared transfer for each pair (p, q), p != q, of each group, MESH_JOIN or
# MESH_PAGES by `kind`; its events, the sender's production and source read (_produced) and the receiver's consumption
# from the arrival to `last`; at this node its sends (_send) and its receives, `roles` this node's place and its
# receives in the group's order.  What must stay: one transfer a pair, every rank's
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1; "The
# revisions of 14b9a757"), reversing `runs` and `lands`: `sent(e, q)`, the place at device e of the part of `key` it
# sends to place q, `from`; `landed(f, p, q)`, at device f, place q, the place the part from place p lands at, `to`, its
# value and the key the runner reads it by; both every rank's.  What must stay: one transfer a pair, every rank's
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


# MPI 4.1 section 7.8, MPI_ALLTOALL: "The j-th block sent from MPI process i is received by MPI process j and is placed
# in the i-th block of recvbuf": block q of member p's operand along the dimension the mesh dimension moves to (`j`) to
# block p of member q's result along the one it leaves (`i`), MESH_JOIN as an AllGather's part; block p itself the
# 'copy' before it (_reshard)
def _all_to_all(ir, c, groups, key, made):
    (i, j), k = c.dim, len(groups[0])
    ni, nj, received = _shape(ir, key)[i], _shape(ir, key)[j] // k, 3 * c.position + 1
    _exchange(ir, c, groups, [(p, q) for p in range(k) for q in range(k)], key,
              lambda e, q: _narrowed(ir.places[e][key], j, q * nj, nj),
              lambda f, p, q: (_narrowed(ir.places[f][made], i, p * ni, ni), made, None), mesh.JOIN,
              max(received, _readers(ir, {made}, received)[1]))


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
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", V1 and G2; "The
# revisions of 55b9cbad"): slice q a run of the partial laid out with `dim` most-major (_sent), sent by the exchange
# AllToAll's is (_exchange); the slices from the j-th other member landing at part j of one value, each viewed as its
# sender laid it out (_buffer).  What must stay: each slice to its keeper, MESH_PAGES, the sum in the group's order
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1; "The
# revisions of 14b9a757"), reversing _sent and _buffer: slice q is its view of the partial's place, whatever its
# layout, `from`; it lands whole at part j of the keeper's received value (its parts row-major in the slice's shape, a
# value of no key), `to`.  What must stay: each slice to its keeper, MESH_PAGES, the sum in the group's order
def _reduce_scatter(ir, c, groups, key, made):
    dim, k = c.dim, len(groups[0])
    n, received = _shape(ir, key)[dim] // k, (c.node, 'received', c.position)
    shape = _layout(ir, key)[1]
    ir.shapes[received], ir.added[received] = (k - 1,) + shape[:dim] + (n,) + shape[dim + 1:], None
    _exchange(ir, c, groups, [(p, q) for p in range(k) for q in range(k)], key,
              lambda e, q: _narrowed(ir.places[e][key], dim, q * n, n),
              lambda f, p, q: (_part(ir, received, p - (p > q)), received, (received, p - (p > q))), mesh.PAGES,
              3 * c.position + 1)


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
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", V1, V3 and G2;
# "The revisions of 55b9cbad"), reversing the note above where the arrival is a value of its own and the piece a narrow
# of a row-major run or copied: V3, "The pieces must land at their places in the window region's value (point 7), so no
# concatenation remains": the arrival lands at its place in its region's landing value (`landing`, _region), laid out
# with the window's dimension most-major (_halo); the piece sent from its run in the value its region's middle is
# (_halo: the operand's own value or its one layout change), or, where its blocks are another piece's there (a halo
# wider than a shard, or two halos over a shard), laid out once into a value of its own ("no two ranges at this node
# share a block of it", struct mesh_declared), by the exchange AllToAll's is (_exchange); the arrival's consumption to
# the last reader of the landing value.  A member no pair targets holds XLA's zeros at its arrival's place, written
# there (_steps).  What must stay: one transfer a pair, XLA's pairs refused at compile time where malformed
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1;
# design/plan-to-bridge-change-request.md R2 at mesh 1a2dc4fd, corrected: "At a source, any number of ranges read the
# same blocks.  A halo wider than a shard, or two halos over one shard, is then sent with no copy"; "The revisions of
# 14b9a757"), reversing _halo, its layout change and its share: the piece is sent from its place, its view of the
# window's middle in the landing value, `from`, and lands at its arrival's place there, `to`.  What must stay: one
# transfer a pair, XLA's pairs refused at compile time where malformed
# OpenXLA operation semantics, CollectivePermute: "For each pair, the operand is sent from source replica to target
# replica"; a pair (p, p) the local copy at p (_collected)
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
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", V4: "C3.2:
# "Weights, the KV cache, pooled intermediates, exchanges … are all placed by `mesh_plan`'s allocator".  The code must
# place them as invariants, as e2b does"; "The revisions of 55b9cbad"): a module attribute's key (a parameter or buffer
# of the exported module, _over) is a local MESH_INVARIANT value, "written before the first execution and read by every
# one: placed once for the plan's life, as Lam places a loop invariant outside the kernel" (mesh-collective.h struct
# mesh_value), its positions not read, as tools/e2b_replay.m places every operand no route names and no host writes;
# its producer's one write, the load, is program's (`make`).  What must stay: as above
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", V1 to V3; "The
# revisions of 55b9cbad"), reversing the notes above where a value is its key's elements row-major: each value its
# form's elements (_size: padding within it, the note above _span), each key's home (value, first element, sizes,
# strides) its strided view there, a key that lies in another's value (`inside`, `views`) at its place in it, a value of
# no key local with the positions it was made with where no range names it here (a region's landing value at a
# partition that receives no piece, a layout change that sends none); each own part shared (`shared`) a struct
# mesh_share in blocks where both its values are named by a declared transfer here, as struct mesh_share requires
# ("of the same copies (both named by a declared transfer here, or both of the same `lag`)"), else copied at its
# item (`fallbacks`, the implementer's decision recorded in design/compilation-coordination.md: the pages cannot be one
# item of two values of unlike copies); a region with no CollectivePermute (a fill alone) gets its landing value here
# (_halo with none); a received parts' value named by no declared transfer here (an AllReduce member's) holds no pages.
# A share whose `of` run is another share's run is copied too (mesh-collective.h struct mesh_share at mesh e627f61b,
# sub-agent C's test: "EINVAL also where `of`'s run is itself another share's run ... and where the two values' `wire`
# differ"; every share here is of two values a declared transfer names, so both are `wire`).  What must stay: no value
# at a node the placement maps no device, the positions as above
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1: "A
# value's layout is the program's own, torch's strides"; mesh-collective.h struct mesh_value at mesh 31ee5d4a: "A
# value a declared transfer names here ... is placed in the registered window whatever its `wire`"; "The revisions of
# 14b9a757"), reversing the padded form, `views`, the shares and the fallbacks: a value for each key that owns its place
# at this device (_places, the received values the lowerings added among them) and for each value of no key with a span
# or named here, its elements its row-major shape's; `owned` its index by that key; `wire` 1 where a declared transfer
# names it here, else local with its first position the earliest item writing its bytes (a key whose place it holds and
# whose writer is that key) and its last its last reader's; every placed key's home (value, offset, sizes, strides) its
# place; each declared transfer's `value` its end's value here.  No struct mesh_share is made, and nothing falls back to
# a copy.  What must stay: no value at a node the placement maps no device, the positions as above
# 2026-11-06 (design/problems-of-14b9a757-change-request.md, "Sources for the stops", item 7), reversing the note above
# where a routed value has no positions: every value's positions its lifetime, "the index of the first and last node
# referring that tensor in its inputs/outputs" (ExecuTorch, exir/memory_planning.py, update_tensor_lifetime); a value a
# range lands in here from the call's start, the execution's receives' start (mesh_transfers_start), to the item that
# waits for the landing (MPI 4.1 section 3.7: "The receiver should not access any part of the receive buffer after a
# nonblocking receive operation is called, until the receive completes")
def _place(ir):
    if ir.device is None:
        return
    places = ir.places[ir.device]
    places.update({key: _own(ir, key) for key in ir.added if key not in places})
    waits = {c.position: [ir.at[i] for i, _ in receives] for c, (_, receives) in ir.roles.items()}
    last = mesh_graph.ends(range(len(ir.program)),
                           lambda g: [owner for k in ir.program[g][1] for owner in ir.over.get(k, ())]
                           + waits.get(g, []))
    wired, begins, landed = set(ir.at.values()), {}, {owner for owners in waits.values() for owner in owners}
    for key, place in places.items():
        if place is not None and place.writer == ir.results.get(key, key) and place.writer in ir.producer:
            begins[place.owner] = min(begins.get(place.owner, len(ir.program)), ir.producer[place.writer])
    for key, place in places.items():
        if place is None or place.owner != key or key in ir.added and ir.added[key] is None and key not in wired:
            continue
        dtype, shape = _layout(ir, key)
        g, wire, first = ir.producer.get(key), int(key in wired), begins.get(key, 0)
        ir.owned[key] = ir.value(dtype, math.prod(shape), _device(ir, key), wire, 0 if key in landed else first,
                                 max(first, last.get(key, first)),
                                 kind=mesh.INVARIANT if g is not None and ir.program[g][0] == 'constant'
                                 else mesh.EXECUTION)
    ir.homes = {key: (ir.owned[place.owner], place.offset, place.sizes, place.strides)
                for key, place in places.items() if place is not None and place.owner in ir.owned}
    for i, owner in ir.at.items():
        ir.declared[i].value = ir.owned[owner]


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


# A program input's copies (2026-11-05, design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement
# lines", V5; design/disjoint-pages-change-request.md D.3.3: "The program declares the host's lag, and the allocator
# gives them copies by it ... With the host writing step t + 1 while the GPU runs step t, the lag is 2"; "The revisions
# of 55b9cbad"): `lag` 2 (mesh-collective.h struct mesh_value), as metal-microbench tools/e2b_replay.m HOST_LAG.  What
# must stay: D.3.3's lag
HOST_LAG = 2


# mesh-collective.h mesh_host_arrived, waited on for at most MESH_PLATFORM_NS (mesh.h): 1 arrived; FAILED a field ahead
# or the range cancelled; 0 not yet
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", B1: "An
# execution's waits each take a fresh MESH_PLATFORM_NS deadline: the start's edges, then each `_arrived`.  So one
# execution is bounded by 5 s x (1 + arrivals).  The code must give one execution one absolute deadline, shared by all
# its waits"; "The revisions of 55b9cbad"): the wait ends at `deadline`, the execution's one (_execute), on the clock
# mesh-call.h mesh_transfers_start names, clock_gettime_nsec_np(CLOCK_MONOTONIC), which Python reads as
# time.clock_gettime_ns(time.CLOCK_MONOTONIC), where each wait took time.monotonic_ns() + MESH_PLATFORM_NS of its own.
# The spin is bounded by the deadline even where the range never arrives (a bounded spin-wait).  What must stay: 1,
# FAILED and 0 as mesh_host_arrived gives them, ETIMEDOUT past the deadline
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-7:
# "Every wait ends at MESH_PLATFORM_NS without progress, and the first ETIMEDOUT ends the run ... The operator's bound is
# judged by time without progress"; W10: "B1: a deadline per execution is not the rule"; R-1: mesh-collective.h
# mesh_host_arrived at mesh 31ee5d4a takes (plan, transfer); "The revisions of 14b9a757"), reversing the execution's one
# deadline and the range's section: _wait spins until `ready()`, ending with ETIMEDOUT once `progress()` has not risen
# for MESH_PLATFORM_NS (each rise restarting the bound), so a wait whose event never comes ends in bounded time (a
# bounded spin-wait); _arrived's progress is the range's count word of its copy (`count`, rdma/mesh.py Publication:
# "copy c's at count + 8 c"), which every landing and every cancellation raises (mesh-collective.h mesh_host_arrived).
# What must stay: 1, FAILED and 0 as mesh_host_arrived gives them; ETIMEDOUT at MESH_PLATFORM_NS without progress
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
# 2026-11-06 (R-1; rdma/mesh-torch.mm mesh_torch_publish at mesh f0523b8d; "The revisions of 14b9a757"): the publish
# takes the plan and its declared transfers' indices, where it took sections.  What must stay: the file's declaration
TORCH = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(mesh.__file__)), 'libmesh-torch.dylib'))
TORCH.mesh_torch_publish.restype, TORCH.mesh_torch_publish.argtypes = C.c_int, [C.c_void_p, C.c_void_p, mesh.Q,
                                                                                C.POINTER(mesh.U), mesh.U, mesh.Q]
TORCH.mesh_torch_notifying.restype, TORCH.mesh_torch_notifying.argtypes = mesh.Q, []
PUBLISH = C.cast(mesh.LIB.mesh_host_publish, C.c_void_p)
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", B3 and V5; "The
# revisions of 55b9cbad"): rdma/mesh-torch.mm mesh_torch_retire, the program's retirement after torch's work in an
# MPSEvent's notification, given mesh_retire's address as the publish is given mesh_host_publish's (RETIRE); and
# mesh_torch_end and mesh_torch_ended, an execution's end signalled on torch's stream and read (_Inputs).  What must
# stay: each binding the file's declaration
TORCH.mesh_torch_retire.restype, TORCH.mesh_torch_retire.argtypes = C.c_int, [C.c_void_p, C.c_void_p, mesh.Q]
TORCH.mesh_torch_end.restype, TORCH.mesh_torch_end.argtypes = C.c_int, [mesh.Q]
TORCH.mesh_torch_ended.restype, TORCH.mesh_torch_ended.argtypes = mesh.Q, []
RETIRE = C.cast(mesh.LIB.mesh_retire, C.c_void_p)


# 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 7, and "The
# revisions of 4e78f6f4"): each publish a range's section, its first row the range's first block in its value
# (_bind: "mesh_host_arrived and mesh_host_publish take a section of the range", the coordinator's brief), each with
# its value's device, where it was a value's section.  What must stay: a publish after its range's last write, no wait
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", K3; "The
# revisions of 55b9cbad"): the CPU sections and the MPS sections' array given, constants of the plan (_release), where
# each publish split its sections by device and made the array.  What must stay: a publish after its range's last
# write, no wait
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1 and
# R-4: "each execution's work becomes a straight-line sequence of its kernel calls and publishes, with no walk of the IR
# and no test of a make()-time fact"; W6, the branch at 2156; "The revisions of 14b9a757"), reversing the sections and
# the test of the MPS array: the publishes of `transfers`, declared transfers of plan `plan` this node sends, as steps
# (_steps), one mesh_host_publish a CPU range and one mesh_torch_publish of every MPS range, made only where there are
# some.  What must stay: a publish after its range's last write, no wait
def _publishes(context, plan, devices, transfers):
    metal = [i for i in transfers if devices[i] == 'mps']
    steps = [lambda env, inputs, t, i=i: mesh.LIB.mesh_host_publish(C.byref(context), plan, i, t)
             for i in transfers if devices[i] != 'mps']
    if metal:
        array = (mesh.U * len(metal))(*metal)
        steps.append(lambda env, inputs, t: mesh.check(TORCH.mesh_torch_publish(PUBLISH, C.addressof(context), plan,
                                                                                array, len(array), t)))
    return steps


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


# The program's one mapping of its values (2026-11-05, design/compilation-coordination.md, "Review of 4e78f6f4 against
# the requirement lines", G3: "The planner calls `mach_vm_remap` and writes virtual addresses into the IR, and reads
# `getpid()` ... Deciders see pages.  The plan writes rows, and the program's effector maps them"; "The revisions of
# 55b9cbad"; mesh-dataflow.h mesh_view_compose: "called by the program's effector ... over its `run` values' rows once
# mesh_plan has written them"): every `run` value of every compiled function, function by function and value by value
# in `values`' order, each value's copies in order, block k of copy c its row section.first + c x section.stride + k
# (mesh-section.h: "copy c's rows first + c x P + k, k < P"), as mesh_plan's mapping was ordered (the coordinating
# implementer's MV-1: one mapping a program, one release, one MTLBuffer over it).  Every value here is `run` 1
# (IR.value).  The mapping (mesh.View) and, by function, each value's copies' addresses in it, copy c of a value P
# blocks long at its first block's address + c x section.pages x pgsz; mesh_view_compose's error value raised (ENOMEM,
# EINVAL), nothing mapped.  An effector: it reads the rows the plan wrote and decides nothing.  What must stay: one
# mapping a program, every `run` value's copies consecutive in it in row order
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-10: "the
# mapping's row list in one mesh-dataflow helper over the plan's rows"; W13: "the mapping's row list in two effectors
# (mesh_compile.py 2289; e2b_replay.m 1437), to be one mesh-dataflow helper"; "The revisions of 14b9a757"), reversing
# the row list and the walk of the offsets here: mesh-collective.h mesh_view_values maps every `run` value of the
# program's compiled functions (`functions`, the plan call's array, `count` of them) and gives value i's copy-0 offset
# in the mapping, i over every function's values in order; copy c of value v at that offset + c x section.pages x pgsz
# (its note), each value's copies' addresses by function from `values`, the functions' value arrays.  Every value here
# is `run` 1 (IR.value), so no offset is UINT64_MAX.  What must stay: one mapping a program; an effector deciding
# nothing
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
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G3: "Deciders see
# pages.  The plan writes rows, and the program's effector maps them"; "The revisions of 55b9cbad"): each value's
# copies at the addresses the program's own mapping gives them (`addresses`, _mapped), where they were read from
# struct mesh_value `view`, which the plan no longer writes.  What must stay: each device's `data` and `byte_offset`
# as torch's own export sets them
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", V5; "The
# revisions of 55b9cbad"): `given` as well, for each program input this function's values hold (IR `inputs`), its copies'
# tensors on its device and, for the caller's write, each copy's CPU tensor over the same pages of the mapping (the
# device tensor itself on the CPU), by _dlpack_tensor as every copy is.  What must stay: as above
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", V1; "The
# revisions of 55b9cbad"): each placed key's tensor its home's strided view of its copy (torch's as_strided at the
# copy's storage offset plus the home's first element: "the offset in the underlying storage of the output tensor",
# torch 2.14.0 torch/_torch_docs.py as_strided), where it was a narrow of the copy viewed in its shape.  What must
# stay: as above
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1; "The
# revisions of 14b9a757"), reversing each declared transfer's section: a range is named by its plan and its declared
# transfer's index (mesh-collective.h mesh_host_publish, mesh_host_arrived at mesh 31ee5d4a), so no section is made;
# `devices`, each declared transfer of this node by its value's device, as before.  What must stay: each device's
# `data` and `byte_offset` as torch's own export sets them
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
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", K3: "Values
# recomputed every execution from make()-time facts become constants of the plan (mesh_compile.py 2167, 2220–2221,
# 2320, 2323): `ir.sends`' release lists, `own`, `ir.device`, and the received parts' landing order, which lowering
# records in `ir.at` and the runner re-derives (2215, 2219)"; V1 to V3; "The revisions of 55b9cbad"), reversing _run,
# which read each of these from the IR at every execution: each item of this node's program is a step made once, when
# the plan is bound (_steps), holding as constants its release lists (_release: each copy's target views by copy, the
# CPU sections and the MPS sections' array), its receives' sections and part views in the order the lowering recorded
# them (`roles`), its place's tensors by copy, a CollectivePermute's own pair, a window's offsets and fill views, an
# AllToAll's layout change's views; a node with no section has no step and gives no value (`ir.device` read once).
# The steps of an item: its fallback copies (_place), then a collective's sends (`sends`), then its work, then its
# publishes (`publishes`).  Its work as before but: a window's region is a view of its landing value, where its pieces
# landed (V3), so it fills its parts outside the operand alone, torch.cat goes; an AllToAll's result is laid out from its
# parts' view by torch's copy_ into its strided view (_all_to_all); a CollectivePermute's result is its arrival's place.
# What must stay: one execution a call, every read of a received range after its arrival, a release after its range's
# writes, every wait of the execution by its one deadline
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-4: "At
# make(), each execution's work becomes a straight-line sequence of its kernel calls and publishes, with no walk of the
# IR and no test of a make()-time fact.  That is a graph instantiated once and launched by each call (aot item 6: "CUDA
# Graphs: a graph instantiated once and launched many times").  Its form is the specialization of an interpreter to a
# fixed program, partial evaluation (Y. Futamura, Partial evaluation of computation process: an approach to a
# compiler-compiler, 1971; cited from memory)"; W6: "per-execution branches on make()-time facts remain (mesh_compile.py
# 2156, 2598, 2605, 2606, 2635)"; R-1 and R-7; "The revisions of 14b9a757"), reversing _step, _release, _combine and
# their branches:
# - The program is instantiated once a copy (_instantiate), as C3.3 unrolls the template ("The template is unrolled
#   into u copies, and execution t runs copy t mod u"), here over the period of every placed value's copies (the least
#   common multiple of their counts), so at copy c every placed key's tensor is a constant (`placed`, its home's tensor
#   of copy c mod its count).  Each item is specialized to its steps, closures in program order, each one kernel call,
#   wait or publish with its operands bound, by mesh_graph.forward over the program's keys, each key's value its
#   specialization's (_specialized): a view of constants (a slice, a narrow, a window's view, an operator whose result
#   aliases its operand, an operator.getitem) is evaluated once, when it is instantiated, and is a constant; a key no
#   constant gives (an input, the caller's; a result torch allocates; a view of either) is the execution's (`env`,
#   _Dynamic), set by the step that makes it, and a call reading one sets it into its arguments' leaves by index
#   (_bound): partial evaluation's division of a program's values into static and dynamic.  No step tests a
#   make()-time fact: each such fact chooses which steps are made.
# - An operator of an out variant writes its results' places; one with none, placed because a collective reads it or it
#   lies in another value, is copied once into its place (appendix item 7, "What stays"); a placed view is its place;
#   a view node in `contiguous` is torch's contiguous into its place by copy_, its view of it; any other makes its
#   result.  A window's region is its place in its landing value, filled where it lies outside the operand.  An
#   AllReduce root's and a ReduceScatter keeper's combine waits for each received part in the group's order and adds
#   it in, its own first (MSCCLang's receive-reduce-copy, torch.add's out variant); every other collective waits for
#   its receives; a CollectivePermute member no pair targets writes XLA's zeros at its place, one of a pair (p, p) its
#   local copy.  A 'copy' is torch's copy_ of its source's view into its place, none where that is its source's
#   (_copied).  Each item's publishes follow its work.
# - Each receive's wait (_arrived) ends at MESH_PLATFORM_NS without a rise of its range's count word of copy c (R-7),
#   that word found once from its declared transfer's entry (mesh.h mesh_transfer_at; rdma/mesh.py Publication).
# What must stay: one execution a call, every read of a received range after its arrival, a release after its range's
# writes
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


def _count(context, state, transfer, c):
    entry = mesh.Publication.from_address(context.M + state.transfers + transfer * mesh.PUBLICATION_BYTES)
    return C.c_uint64.from_address(context.M + entry.count + 8 * (c % state.copies))


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
    waits = [lambda env, inputs, t, i=i, count=_count(context, state, i, c): _arrived(context, plan, i, t, count)
             for i, _ in receives]
    if collective.kind == 'reduce-scatter' or collective.kind == 'all-reduce' and me == 0:
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
    return waits


# MPICH MPIR_Localcopy, torch 2.14.0 torch/_tensor_docs.py copy_: "Copies the elements from src into self tensor and
# returns self"; none where the source is the destination's view (MPI_IN_PLACE, _placement)
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


# item `item`'s value at copy c, the transfer of _instantiate's forward pass: its placed tensor (`placed`), a constant
# it evaluates, or the execution's (_Dynamic), its steps appended to `steps` in program order, then its publishes
def _specialized(ir, context, plan, state, devices, placed, c, item, operands, steps):
    kind, reads, key, operand = item
    value = placed(key)
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


# copy c of the program at this node, mesh_graph.forward over its keys in program order with _specialized each key's
# transfer (2026-11-06, the C9 hook's remark on f2a1efd1: "_instantiate(): a forward pass over the program's items in
# order, deciding each key's static or dynamic status from its source's and building the steps per copy; it moves into
# the graph module"): its steps and its result, the output item's values unflattened by the exported program's out_spec
def _instantiate(ir, context, plan, devices, tensors, c, spec):
    state, steps = mesh.State.from_address(context.M + plan), []
    items = {item[2]: item for item in ir.program if item[0] != 'output'}

    def placed(key):
        return tensors[key][c % len(tensors[key])] if key in tensors else None
    values = mesh_graph.forward(list(items), lambda key: items[key][1],
                                lambda key, operands: _specialized(ir, context, plan, state, devices, placed, c,
                                                                   items[key], operands, steps))
    node, keys = next(item[3] for item in ir.program if item[0] == 'output')
    arguments = _bound(node.args[0], lambda n: values[keys[n]])[0]
    return steps, lambda env: pytree.tree_unflatten(list(arguments(env)), spec)


def _steps(ir, context, plan, tensors, devices, spec):
    if ir.device is None:
        return [[]], [lambda env: None]
    made = [_instantiate(ir, context, plan, devices, tensors, c, spec)
            for c in range(math.lcm(1, *(len(x) for x in tensors.values())))]
    return [steps for steps, _ in made], [result for _, result in made]



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
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G4 and B2;
# "The revisions of 55b9cbad"; mesh-dataflow.h mesh_retire at mesh b35234c3), reversing the note above where it names
# a root, its `owner` and `closed`, and EBUSY: mesh_retire retires the program's operand (mesh.h struct mesh_algo)
# where hdr `client` is this context's `client` and its `published` is not 0 (`retired` = `published`, release, then
# hdr `plans` raised and woken); the bridge parent releases it once no port holds a plan of it (mesh_retired_release:
# `released` = `published`), and the next mesh_plan waits for that release, bounded by MESH_PLATFORM_NS, ETIMEDOUT
# past it (B2), where it was EBUSY.  What must stay: the retirement after torch's work on the values, its wait
# bounded, ETIMEDOUT with the operand standing; the mapping released once retired and no tensor over it remains
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", B3: "`_retire`'s
# device synchronization (mesh_compile.py 2268) is an unbounded device wait, and it runs on every replan.  The record's
# release must follow torch's work in an MPSEvent notification, as torch_mesh's allocator.mm did its release (fba41159's
# event form, item 7).  It is then bounded by the notification wait already there"; "The revisions of 55b9cbad"),
# reversing the device synchronization above: where a value of the program is on torch's MPS device, mesh_retire is
# done in the notification of a torch MPSEvent recorded after torch's work so far (rdma/mesh-torch.mm
# mesh_torch_retire: acquireEvent(false, stream), record, notify, SyncType::COMMIT, the publish's form); else, on the
# CPU device, where a kernel has ended when its call returns (appendix item 7: "On the CPU device a kernel has ended
# when its call returns, so the publish follows the call"), mesh_retire follows the calls.  Then every notification
# (the publishes' and the retirement's) is waited out, bounded by MESH_PLATFORM_NS from the wait's start, ETIMEDOUT past
# it with the operand standing (the next plan retires it again: mesh_retire stores the one value `published`), and the
# mapping is released once no tensor over it remains (_Mapping), torch's work over it having ended.  What must stay: no
# device synchronization; the retirement after torch's work; the wait bounded; the mapping released after it
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-7:
# "make()'s waits share one deadline"; mesh-collective.h mesh_plan at mesh 31ee5d4a: "The caller's one deadline covers
# its own earlier wait of the same preparation (rdma/mesh_compile.py's retirement) and this one"; "The revisions of
# 14b9a757"), reversing the wait's own MESH_PLATFORM_NS: the notifications are waited out until `deadline`, the
# caller's (make()'s one, which mesh_plan's release wait then takes; the dropped program's own).  What must stay: no
# device synchronization; the retirement after torch's work; the wait bounded; the mapping released after it
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
# 2026-11-06 (R-7: "the first ETIMEDOUT ends the run: no later wait runs"; "The revisions of 14b9a757"): the
# retirement's wait ends MESH_PLATFORM_NS from its start, or at once where the run has ended at an ETIMEDOUT
# (`timeout`), so no wait follows that one.  What must stay: as above
def _dropped(context, state):
    if state['mapping'] is not None:
        _retire(context, state, time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                + (0 if state['timeout'] is not None else mesh.PLATFORM_NS))


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
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", B1: "The code
# must give one execution one absolute deadline, shared by all its waits"; "The revisions of 55b9cbad"): the
# execution's one deadline, taken once where it begins, clock_gettime_nsec_np(CLOCK_MONOTONIC) + MESH_PLATFORM_NS
# (mesh-call.h mesh_transfers_start: "which the caller takes once where the execution begins (its MESH_PLATFORM_NS
# bound) and gives every wait of that execution, these and its own later ones"), given to mesh_transfers_start and to
# every _arrived of the execution (_run), where the start and each arrival took a deadline of its own.  What must
# stay: one execution a call, the start at every node of the program, one deadline an execution
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", K3; "The
# revisions of 55b9cbad"): the plan's state section read through one ctypes view made when the plan is bound (`state`),
# and the items run as their steps (_steps), a node with no section having none and giving no value, where each
# execution made the view and tested `ir.device`.  What must stay: as above
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-4 and
# R-7; mesh-call.h mesh_transfers_start at mesh 31ee5d4a, "reversing `deadline`: each wait here ends at
# MESH_PLATFORM_NS after the last rise of its count it read"; "The revisions of 14b9a757"), reversing the execution's
# deadline: the start takes none, and each step waits on its own progress (_arrived); execution t runs its copy's
# straight-line steps, t mod their period, and returns that copy's result (_steps).  What must stay: one execution a
# call, the start at every node of the program
def _execute(context, run, args, kwargs):
    t = run['executions']
    status = mesh.LIB.mesh_transfers_start(C.byref(context), run['plan'], t + 1)
    if run['state'].executions > t:
        run['executions'] = t + 1
    mesh.check(status)
    c, env, inputs = t % len(run['steps']), {}, pytree.arg_tree_leaves(*args, **kwargs)
    for step in run['steps'][c]:
        step(env, inputs, t)
    return run['results'][c](env)


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
#   program's one mapping out (`*view`); every plan's view (`serial`, `sequencer`, `boot`) then compared with the map
#   read's, ESTALE where they differ (JGroups ViewId.java: "The comparison is done on the IDs, if they are equal, we use
#   the creator"), every value bound over the mapping (_bind), MPS values over one buffer of it (_placed_buffer); a
#   failure after mesh_plan retires the program;
# - after an ESTALE or a cancelled range (ECANCELED) the next run plans again: the program retired (_retire), then
#   planned whole on the map as it is; mesh_plan is EBUSY until the bridge has released the root (mesh-collective.h:
#   "EBUSY where the root's `owner` is not 0 ... before anything is written"), raised as any failure, and the next
#   run tries again (the coordinator's brief for point 5).
# Each run interprets the block (_Program), each call of a compiled function one execution of its plan (_execute).
# The program is retired when its callable is collected (_dropped).  What must stay: no placement and no weight chosen
# here; every operand of mesh_plan the same at every rank but this node's values
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G3, G4 and B2,
# and "Answered in the documents", EB-1; "The revisions of 55b9cbad"; mesh-collective.h mesh_plan at mesh b35234c3),
# reversing the notes above where they differ:
# - mesh_plan gives no mapping and writes no address (G3: "The plan writes rows, and the program's effector maps
#   them"): once it returns, this program maps its values' rows itself (_mapped, mesh-dataflow.h mesh_view_compose)
#   and binds each value at its place in that mapping (_bind); a mapping Mach refuses is this program's error after a
#   plan that stands ("A mapping Mach refuses is the program's error, after a plan that stands", mesh-collective.h),
#   so the program is retired (_retire) and the error raised, as any failure after mesh_plan.
# - EBUSY goes (B2: "The replan must wait for the release, bounded by MESH_PLATFORM_NS, as RCU's writer waits out a
#   grace period (`synchronize_rcu`), with ETIMEDOUT past it"; EB-1): a plan again after ESTALE or ECANCELED first
#   retires this program's own operand (_retire;
#   mesh-dataflow.h mesh_retire), then mesh_plan waits for the bridge's release of it and returns ETIMEDOUT past
#   MESH_PLATFORM_NS with nothing written, raised as any failure, `stale` left set so the next run plans again.
# - The program's operand is mesh.h struct mesh_algo, A9's serialized operand in MSCCL-IR's shape (G4: "Its name is
#   A9's ... "Root" also names the trees' roots ... and goes from this record"), its holder this process, the
#   context's `client` and `start`, which mesh_attach set (mesh-dataflow.h struct mesh_ctx) and mesh_plan copies.
# What must stay: no placement and no weight chosen here; every operand of mesh_plan the same at every rank but this
# node's values; a failure after mesh_plan retires the program
# 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", V4 and V5; "The
# revisions of 55b9cbad"):
# - V4 ("The code must place them as invariants, as e2b does"): every parameter and buffer of each exported module is a
#   MESH_INVARIANT value (_over, _place), written into its placed pages once, after the plan and before the first
#   execution, by torch's copy_ from the module's own tensor, "For the weights, that write is the load: one copy from
#   the buffer the delegate loaded, at the bind.  C3.7 allows it, because the producer is writing its own value"
#   (design/disjoint-pages-change-request.md D.3.2); every execution reads it there (_run), so after the load nothing
#   reads torch's allocation of it ("After the bind, no command reads a delegate buffer", D.3.2).  On MPS the load is
#   on torch's stream before the first execution's kernels, ordered by the stream's command order (C3.1: "A producer on
#   the same queue comes before its consumer by the command buffer's order"); on the CPU it has ended when it returns.
# - V5 ("Region-only memory makes every program value region pages, so the plan places each input.  The caller writes
#   into the input placed for it, as a host producer, ordered by the event (C3.1: "A host producer is ordered by the
#   event")"): each tensor input of the traced block is a value of `lag` HOST_LAG (MESH_EXECUTION, wire 0, named by no
#   declared transfer: mesh-collective.h struct mesh_value, D.3.3's host input) of the compiled function of the first
#   call reading it whose function has a section at this node (the implementer's decision, recorded in design/
#   compilation-coordination.md: a value is one function's, struct mesh_function `value`; one reading it with no
#   section here holds no value, A4), execution b of the program reading its copy b mod HOST_LAG, every call reading the
#   input given that copy's tensor (_Program), as a call is given an earlier call's result over its pages.  The order
#   is D.3.3's, by its events: "The host writes step t's inputs into copy t mod k (k being their copies) once the queue
#   has signalled that step t − k is done": `inputs` waits, bounded by MESH_PLATFORM_NS ("The host's wait on that is
#   bounded by MESH_PLATFORM_NS and ends as an error value", ETIMEDOUT), until execution b − HOST_LAG has ended, its end
#   signalled on torch's stream after its last work ("The queue signals each step's end with `signalEvent:value:` after
#   its last commit": rdma/mesh-torch.mm mesh_torch_end, an MPSEvent's notification raising mesh_torch_ended), on the
#   CPU at its run's return; "then sets its input event to t + 1.  Step t's commands that read them are committed after
#   the queue's `waitForEvent:value:` on that event at t + 1": here the host's write is done before `run` encodes
#   execution b's kernels, which torch commits only after they are encoded, so the input event's order is the host's
#   own program order before that commit (C3.1's "A host producer is ordered by the event", the event the host's
#   return from its write).  A host input at a node where no function reading it has a section is placed nowhere
#   there, and `inputs` gives None for it.
# - The API change for the caller: program(block, *example_inputs) returns `run`; `run.inputs()` gives, for the next
#   execution, a tuple with, for each input of the block, the CPU tensor over that input's placed copy (the caller writes
#   it, a host producer: torch's copy_, out=, or numpy over it), None for a non-tensor input, whose example value is
#   compiled into each exported function (A1: "shapes are compiled"), and None where no function here reads it;
#   `run()`, with no argument, runs that execution on the placed inputs, where `run(*inputs)` read the caller's
#   tensors.  `inputs` is called before each `run` whose inputs the caller writes; an input not written keeps its copy's
#   earlier contents.  `run`'s outputs are as before, views of their placed pages (appendix item 7).
# What must stay: no placement and no weight chosen here; every operand of mesh_plan the same at every rank but this
# node's values; a failure after mesh_plan retires the program; every input and invariant region pages
# 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1, R-4,
# R-7 and R-10; "The revisions of 14b9a757"), reversing the notes above where they differ:
# - R-7, "make()'s waits share one deadline": `make` compiles the program first, then takes one deadline,
#   clock_gettime_nsec_np(CLOCK_MONOTONIC) + MESH_PLATFORM_NS, which the retirement of its previous plan (_retire)
#   and mesh_plan's wait for the release share (mesh-collective.h mesh_plan's `deadline`), and so does the retirement
#   after a failure that follows mesh_plan.
# - R-7, "The first ETIMEDOUT ends the run: no later wait runs": the first ETIMEDOUT of `make`, `inputs` or `run` is
#   kept (`timeout`), and every later call of `inputs` or `run` raises ETIMEDOUT at once, waiting for nothing; `inputs`'
#   wait for execution b - HOST_LAG's end ends at MESH_PLATFORM_NS without a rise of the ended count (_wait).
# - R-1: the plan call's functions carry no share.
# - design/problems-of-14b9a757-change-request.md, "Sources for the second stops", 8 (torch 2.14.0
#   torch/export/exported_program.py, ExportedProgram: "a state_dict containing tensor values of all lifted parameters
#   and buffers"): each node loads its own copy of every invariant.
# - R-10: the program's mapping is mesh_view_values' (_mapped).
# - R-4: each compiled function's steps are made once a copy (_steps), inside the plan's failure handling.
# What must stay: no placement and no weight chosen here; every operand of mesh_plan the same at every rank but this
# node's values; a failure after mesh_plan retires the program; every input and invariant region pages
def program(block, *example_inputs):
    tracer = Tracer()
    gm = torch.fx.GraphModule(tracer.root, tracer.trace(block))
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
                for kind, _, key, target in ir.program:
                    if kind == 'constant' and key in places:
                        places[key][0].copy_(operator.attrgetter(target)(ir.gm))
            runs = [dict(zip(('steps', 'results'), _steps(ir, context, f.plan, places, devices, record.spec)),
                         plan=f.plan, executions=0, state=mesh.State.from_address(context.M + f.plan))
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
