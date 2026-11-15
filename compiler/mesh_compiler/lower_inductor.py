# design/compilation-composition-change-request.md, numerical lowerer boundary, items 2 and 5 to 7, S8 and Complete
# replacement; design/aot-compilation-change-request.md A1, A5, A6 and A8 to A10; design/torch-source-lowering-change-
# request.md items 1 to 3, 5 and 6; /Users/mdot/schedule-emit/schedule_emit/lower_inductor.py at f41c64a (adapt,
# recv_strided, lower): each processor's scheduled body given to its target's installed upstream compiler (torch 2.14.0
# torch/_inductor: compile_fx.py compile_fx_aot, the AOT entry, its GraphLowering, Scheduler and the selected backend;
# coremltools 9.0 for the Neural Engine), its returned pre-terminal result retained as that compiler's own objects.
# Fields: output_data/compiler-workgroup-20261114/compiler/numerical68-result.md; upstream hooks:
# compiler/patches/torch-inductor-aot-lowered-result.patch, torch-inductor-graph-lowering-factory.patch,
# torch-inductor-mps-kernel-operands.patch, torch-inductor-upstream-composition.patch and
# torch-inductor-cpu-kernel-results.patch
import contextlib
import dataclasses
import functools
from pathlib import Path

import coremltools as ct
import torch
from torch import Tensor
from torch._higher_order_ops.effects import _EffectType, _register_effectful_op
from torch._inductor import config
from torch._inductor import ir as upstream
from torch._inductor.codegen.cpp import (CppKernel, CppKernelProxy, CppOverrides, CppScheduling, CppTile2DKernel,
                                         CppTile2DOverrides, CppVecKernel, CppVecOverrides, KernelGroup)
from torch._inductor.codegen.cpp_template_kernel import CppTemplateKernel
from torch._inductor.codegen.cpp_utils import LocalBufferContext
from torch._inductor.codegen.mps import MetalKernel, MetalOverrides, MetalScheduling
from torch._inductor.codegen.wrapper import KernelCallLine
from torch._inductor.compile_fx import compile_fx_aot, compile_fx_inner
from torch._inductor.cpp_builder import CppTorchDeviceOptions
from torch._inductor.cpu_vec_isa import pick_vec_isa
from torch._inductor.graph import GraphLowering, SubgraphLowering
from torch._inductor.lowering import fallback_handler, register_lowering
from torch._inductor.scheduler import Scheduler
from torch._inductor.virtualized import V
from torch.utils._cpp_embed_headers import _embed_headers

from .ir import RECV, SEND, channel_of

WITH_EFFECTS = torch.ops.higher_order.with_effects
# driver.py ENGINES: the Neural Engine's target name in Scheduled.engines
ENGINE = "ane"


# compiler/mesh_compiler/schedule.py assemble, module.meta["writers"], Declarations.wires: the logical copies of a
# scheduled body's declared transfers, before any plan. `wires`: channel -> (declared transfer, logical wire value, its
# logical bytes, declared sent view, declared landed view). Each distinct (source, wire value, sent view, read-after-write
# writers) is one write of its members, the SENDs sharing it, in source order
@dataclasses.dataclass
class Copies:
    group: dict
    index: dict
    member: dict


def copies(module, wires):
    writes, sends = {}, [node for node in module.graph.nodes if node.target is SEND and channel_of(node) in wires]
    for writer, reader in module.meta["writers"]:
        writes.setdefault(reader, set()).add(writer)
    group = {}
    for node in sends:
        _, value, _, (offset, element, pairs), _ = wires[channel_of(node)]
        group[node] = (node.args[0], value, offset, element, tuple(map(tuple, pairs)),
                       frozenset(writes.get(node, ())))
    index = {key: k for k, key in enumerate(dict.fromkeys(group.values()))}
    member = {}
    for node in sends:
        member.setdefault(index[group[node]], []).append(node)
    return Copies(group, index, member)


# schedule-emit lower_inductor.py recv_strided and lower_recv_strided: a received value declared at its node's shape,
# stride, dtype and device (Inductor's fake-result and stride contracts), an ORDERED effectful operator as RECV is
# (AOT autograd's functionalization makes each call of one with_effects on the ORDERED token: effects.py
# handle_effects), its storage in GraphLowering's never_reuse_buffers (Lam, PLDI 1988, section 2.3: the sender still
# holds it). Lowered within lowering.py with_effects (`lowerings[op]`, then its StarDep on the previous ORDERED
# operation); SEND, which has no lowering of its own, by that lowering's FallbackKernel
@torch.library.custom_op("mesh::recv_strided", mutates_args=())
def recv_strided(channel: int, shape: list[int], stride: list[int], dtype: torch.dtype,
                 device: torch.device) -> Tensor:
    return RECV(channel)


@recv_strided.register_fake
def _recv_strided_fake(channel, shape, stride, dtype, device):
    return torch.empty_strided(shape, stride, dtype=dtype, device=device)


_register_effectful_op(recv_strided, _EffectType.ORDERED)
RECV_STRIDED = torch.ops.mesh.recv_strided.default
_received = fallback_handler(RECV_STRIDED, add_to_fallback_set=False)


@register_lowering(RECV_STRIDED, type_promotion_kind=None)
def _lower_received(*arguments):
    made = _received(*arguments)
    V.graph.never_reuse_buffers.add(made.get_name())
    return made


# ---- the returned result (numerical68-result.md): the upstream compiler's own objects and their correspondence ----

# body input `position`: `node` the scheduled body's placeholder, `name` the AOT export's user input of that position
# (GraphSignature.user_inputs: export keeps each flat input its own placeholder), `value` its GraphLowering
# placeholder's own result (MeshLowering.mesh_placeholders[qualify_name(name)]: a TensorBox, a sympy expression, a
# TorchBindObject, a generator's, or None for an admitted None input; Neural Engine: its ExportedProgram placeholder)
@dataclasses.dataclass(frozen=True, eq=False)
class Input:
    position: int
    node: object
    name: str
    value: object


# compiled output `position`: `node` the body node it returns (a SEND's value where `sent`: schedule-emit adapt returns
# every sent value, keep_output_stride), `name` the AOT export's user output (GraphSignature.user_outputs; _unlift_graph
# makes every mutation an in-graph copy_, so the outputs are the user outputs only), `value` GraphLowering
# graph_outputs[position] (Neural Engine: its ExportedProgram output node)
@dataclasses.dataclass(frozen=True, eq=False)
class Output:
    position: int
    node: object
    name: str
    value: object
    sent: bool


# codegen/simd.py codegen_node_schedule_with_kernel's second pass for SchedulerNode `node` (after finalize_indexing and
# its LoopBody's indexing_dtype_strength_reduction): `variables` its split_and_set_ranges result, which node.codegen
# gives its LoopBody (each a range-tree entry symbol of the kernel's range_tree_nodes), `reduction` the kernel's
# inside_reduction there (False inside DisableReduction), `accesses` each memory access its node.codegen made, in order,
# as the kernel received it: (method, buffer name, index, store mode, graph, fx node), MetalKernel load, store and
# store_reduction with the index LoopBody bound and SimplifyIndexing simplified, over those variables and the kernel's
# indirect symbols (MeshMetalKernel.mesh_indirect); `graph` and `fx node` loop_body.py InterpreterShim's (V.interpreter)
# at that call: the LoopBodyBlock's own FX graph (the root block's or a subblock's, node._body.subblocks) and its node
# that made the access
@dataclasses.dataclass(eq=False)
class Ranges:
    node: object
    variables: tuple
    reduction: bool
    accesses: list


# codegen/cpp.py CppKernelProxy.codegen_functions: one of its kernels (codegen_kernel: kernel_group.new_kernel of its
# kernel_cls, vec_kernel_cls or tile2d_kernel_cls, then run) codegening SchedulerNode `node` (SchedulerNode.codegen's
# set_current_node): `variables` the (vars, reduction vars) that kernel's set_ranges returned to run, `suffix` whether
# run gave the node to the kernel's write_to_suffix (its vars and no reduction vars: cpp.py run `fn(vars, ())`, its
# code the kernel's reduction_suffix), `accesses` each memory access the node's codegen made, in order, as that kernel
# received it: (method, buffer name, index, store mode, graph, fx node, mask), CppKernel, CppVecKernel or
# CppTile2DKernel load, store and store_reduction with the index LoopBody bound and SimplifyIndexing simplified, over
# those variables and the kernel's indirect symbols (its mesh_indirect); `graph` and `fx node` loop_body.py
# InterpreterShim's at that call, as Ranges'; `mask` the kernel's _load_mask there (CppKernel.masked: CppVecOverrides
# .masked's mask, its masked loads'; None outside one and in a scalar kernel, whose CppOverrides.masked guards its
# lambda instead). The kernel's own fields are its domain as it emits it: itervars, ranges, reduction_depth,
# active_ranges and inner_itervars (codegen_functions'); a vector kernel's tiling_factor, tiling_idx, tail_size and
# num_elems; a tile kernel's tiling_indices, outer_idx, inner and outer tail sizes and num_elems and
# inner_is_tiling_idx
@dataclasses.dataclass(eq=False)
class CppRanges:
    node: object
    variables: tuple
    suffix: bool
    accesses: list


# an actual result a record does not hold: `result` what it is, `owner` the original producer making it, which keeps it
# nowhere this compilation's result reaches. It is no value of that result: not empty, not iterable
@dataclasses.dataclass(frozen=True)
class Unretained:
    result: str
    owner: str


# a CPU template call's unretained results (MeshCppScheduling.codegen_template): its parameter declarations, which
# CppTemplateKernel.def_kernel's render hook prints into its source (cpp_template_kernel.py 113-114, cpp_argdefs at
# render) and returns as text only; and its ranges: its template's own loops, indices and accesses, which its CppTemplate
# renders as source (ir.CppTemplateBuffer.template, cpp_template.py make_kernel_render), and its epilogues', which
# CppTemplateKernel renders through a CppKernelProxy of an upstream KernelGroup() of its own (cpp_template_kernel.py
# 276-278, 348-350), not the selected backend's
TEMPLATE_DECLARED = Unretained("the template definition's parameter declarations (arg_defs)",
                               "cpp_template_kernel.py CppTemplateKernel.def_kernel render hook")
TEMPLATE_RANGES = Unretained("the template's and its epilogues' loops, indices and accesses",
                             "cpp_template.py CppTemplate render; cpp_template_kernel.py CppTemplateKernel epilogue "
                             "CppKernelProxy(KernelGroup())")


# one selected-backend kernel call, its KernelCallLine `line` in its wrapper's lines (line.wrapper), `graph` the
# GraphLowering whose body it implements (a control body's SubgraphLowering), `scheduler` the Scheduler whose codegen
# made it (an inlined control body's: Inlined.scheduler of that inlining; a lifted one's: Lifted.scheduler, its line in
# Lifted.wrapper's lines), `place` "mps" or "cpu", `owner` the
# backend's kernel (MeshMetalKernel: range trees, body, KernelArgs and workspace_args, headers, mesh_indirect,
# metal_entry, metal_source; MeshKernelGroup: its KernelArgs, loops and scheduled nodes; a CPU template's
# CppTemplateKernel: its KernelArgs, local and removed buffers), `launch` its call record as produced (mps:
# torch-inductor-mps-kernel-operands.patch (name, values, types, grid, group, error: its error buffer's argument
# index, c10/metal/error.h ErrorMessages as its declaration names it); cpu: (kernel name, arg_defs, call_args,
# arg_types), its KernelCallLine's kernel_name, call_args and arg_types as its call wrote them, arg_defs its
# definition's own: a group's codegen_group arg_defs (torch-inductor-cpu-kernel-results.patch), a template's
# TEMPLATE_DECLARED). `produced` its (entry, source) as its backend
# made and defined it before its call (mps: metal_entry and metal_source; cpu, a group's and a template's alike:
# define_kernel's declared entry and final source, torch-inductor-upstream-composition.patch defined_kernels).
# `schedule`: mps the node_schedule codegen_node_schedule_with_kernel received,
# its DisableReduction and EnableReduction scopes in it; cpu the group's scheduled nodes, a template's the nodes its
# define_kernel received (its template node, then its epilogue nodes). `ranges`: mps each second-pass SchedulerNode's
# Ranges in order; cpu each finalize_kernel's (CppKernelProxy, nodes, local buffer context, local buffers) in order
# (MeshKernelGroup),
# each proxy's LoopNest and its kernels (MeshCppKernelProxy), each kernel's CppRanges in order with their accesses
# (MeshCppAccesses); a template's TEMPLATE_RANGES (not retained: its effects are not empty; its scheduled nodes'
# read_writes are its own).
# `scheduled` its SchedulerNodes, `origins` the original body nodes their IR implements
@dataclasses.dataclass(eq=False)
class Kernel:
    place: str
    graph: object
    scheduler: object
    line: object
    owner: object
    launch: tuple
    produced: tuple
    schedule: tuple
    ranges: tuple
    scheduled: tuple
    origins: tuple


# one extern call of a scheduler (scheduler.py codegen_extern_call: every ExternKernelSchedulerNode it codegens, whatever
# its operator), its record made at its return: `node` its ir node as lowered, whole (its kernel and op_overload,
# inputs, constant_args, kwargs, layout and outputs as ExternKernel holds them; ir.RandomSeeds aten.randint.low_out
# among them; ir.WhileLoop cond_subgraph, body_subgraph, carried_inputs, additional_inputs, outputs and stack_output;
# ir.InvokeSubgraph subgraph, inputs and outputs; ir.Switch selector, operands, outputs and branches), `graph` the
# GraphLowering it was called in, `scheduler` that scheduler, `lines` every wrapper line its codegen wrote, the line
# objects themselves in order (upstream's control scaffolding among them as its own str lines), `inlined` each control
# body inlining that codegen made, in order (Inlined; while_loop: its cond before the loop, its body, its cond at the
# loop's end), `lifted` each control body that codegen lowered on its own, in order (Lifted; a Switch's branches under
# fx_wrapper). One call's codegen makes one kind: a Switch lifts (wrapper_fxir.py codegen_switch), while_loop and
# invoke_subgraph inline (wrapper.py, aot_mode)
@dataclasses.dataclass(frozen=True, eq=False)
class ExternCall:
    node: object
    graph: object
    scheduler: object
    lines: tuple
    inlined: tuple
    lifted: tuple


# one inlining of a control body (GraphLowering.codegen_subgraph from wrapper.py codegen_subgraph_by_inlining):
# `graph` its SubgraphLowering (the ir node's subgraph's graph: cond_subgraph.graph, body_subgraph.graph,
# subgraph.graph), `scheduler` this inlining's own Scheduler (_update_scheduler: each inlining schedules the body
# again; Kernel.scheduler and ExternCall.scheduler of what it made), `lines` the wrapper lines it wrote, `externs` its
# scheduler's ExternCall records; its inputs and outputs the body's own (graph.mesh_placeholders, graph.graph_outputs)
@dataclasses.dataclass(frozen=True, eq=False)
class Inlined:
    graph: object
    scheduler: object
    lines: tuple
    externs: tuple


# one control body lowered on its own (wrapper.py codegen_subgraph_common: the body's own GraphLowering.codegen under
# its graph handler, its lowered result wrapper_fxir.py lowered_subgraphs[graph.name]): `graph` its SubgraphLowering
# (the ir node's branch's graph, Switch.branches[b].graph), `scheduler` the Scheduler of that codegen, `wrapper` its own
# wrapper (wrapper_fxir.py SubgraphFxWrapperCodegen, its parent_wrapper its parent's: its lines, launched_kernels and
# lowered_subgraphs), `externs`
# its scheduler's ExternCall records. A body is lowered once per wrapper (codegen_subgraph_common's
# already_codegened_subgraphs): a later Switch of the same branch graph calls the one lowered by name, its ExternCall's
# `lifted` empty, its branch's graph the same object as this record's
@dataclasses.dataclass(frozen=True, eq=False)
class Lifted:
    graph: object
    scheduler: object
    wrapper: object
    externs: tuple


# fx_passes/replace_random.py, which _recursive_joint_graph_passes runs on the body and on every control body
# (compile_fx.py _get_subgraph_names): `graph` the GraphLowering that lowered it (the root's or a control body's),
# ir.py RandomSeeds `buffer` that graph's random state it draws through (an ExternKernelOut, aten.randint.low_out: its
# ExternCall that graph's scheduler's), `uses` each inductor_lookup_seed index of that graph's module and the original
# random nodes it serves; its words the body's Generator (operations.py)
@dataclasses.dataclass(eq=False)
class Seeds:
    graph: object
    buffer: object
    uses: tuple


# a Neural Engine body's upstream result: `program` its torch.export ExportedProgram (Core ATen), `model` coremltools'
# MLModel (its MIL Program each operation's EXIR_DEBUG_HANDLE scope, its control flow and state as converted),
# `selected` the conversion's own operands as given (ct.convert's keywords: convert_to, compute_units,
# compute_precision, minimum_deployment_target, skip_model_load; FLOAT32 coremltools' precision selection, no fp16
# rewrite, not a binary64 result), `received` each (user input position, RECV body node) it binds as an input,
# `handles` {debug_handle: the original body nodes of that program node}
@dataclasses.dataclass(eq=False)
class Model:
    program: object
    model: object
    selected: dict
    received: tuple
    handles: dict


# one processor body's returned upstream result: `module` Scheduled.processor_modules[key][processor] as scheduled,
# `body` the adapted copy the upstream compiler traced (its graph and its control bodies' the provenance roots),
# `sources` {(id(copy graph), copy node name): original node}, `sent` the original sent values in output order, `graph`
# compile_fx_aot's own result (aot_inductor.lowered: output_code.py LoweredAOTI's GraphLowering after its scheduler's
# codegen, before its wrapper's generate: buffers, layouts and aliases, graph_inputs, graph_outputs, mutated_inputs,
# never_reuse_buffers, removed_buffers, effectful_ops, scheduler, wrapper_code and its lines and lowered_subgraphs;
# None for the Neural Engine), `signature` the AOT export's GraphSignature of that graph (graph.module.meta; Neural
# Engine: ExportedProgram.graph_signature), `kernels` {id(KernelCallLine): Kernel} over its wrapper and every subgraph
# wrapper (every selected-backend kernel call: an mps kernel's, a cpu group's and a cpu template's), `externs` its
# scheduler's ExternCall records in emission order, every control body's reached through them (ExternCall.inlined and
# .lifted, each with its own ExternCalls, nested), `seeds` the Seeds of the root graph and of every body graph made
# under it, `model` the Neural Engine's Model
@dataclasses.dataclass(eq=False)
class Produced:
    processor: str
    kind: str
    device: object
    module: object
    body: object
    sources: dict
    sent: tuple
    graph: object
    signature: object
    inputs: tuple
    outputs: tuple
    kernels: dict
    externs: tuple
    seeds: tuple
    model: object

    # each original body node of FX nodes `nodes` (an IR node's get_origins(), a program's nodes), in order
    def origins(self, nodes):
        return _origins(self.sources, nodes)


# ---- the selected upstream backends, retaining their own results ----

# codegen/common.py CSEProxy.indirect_indexing: each indirect symbol its kernel's overrides make (its parent_handler's
# indirect_indexing), with the value it indexes by and its size (and whether its bounds are checked), the indirect
# fact its accesses' indices carry; each backend's own overrides with it
class MeshIndirect:
    def indirect_indexing(self, var, size, check=True, wrap_neg=True):
        made = super().indirect_indexing(var, size, check, wrap_neg)
        V.kernel.mesh_indirect[made] = (var, size, check)
        return made


class MeshMetalOverrides(MeshIndirect, MetalOverrides):
    pass


# codegen/mps.py MetalKernel: its node schedule, each second-pass node's ranges (split_and_set_ranges, then
# set_current_node) and access as it receives it, and its call's line
class MeshMetalKernel(MetalKernel):
    overrides = MeshMetalOverrides

    def __init__(self, tiling, **kwargs):
        super().__init__(tiling, **kwargs)
        self.mesh_indirect, self.mesh_ranges, self.mesh_schedule, self.mesh_split = {}, [], (), None

    def split_and_set_ranges(self, lengths):
        made = super().split_and_set_ranges(lengths)
        self.mesh_split = (tuple(map(tuple, made)), self.inside_reduction)
        return made

    @contextlib.contextmanager
    def set_current_node(self, node):
        self.mesh_ranges.append(Ranges(node, *self.mesh_split, []))
        with super().set_current_node(node):
            yield

    def _accessed(self, method, name, index, mode):
        shim = V.interpreter
        self.mesh_ranges[-1].accesses.append((method, name, index, mode, shim.graph, shim.current_node))

    def load(self, name, index):
        self._accessed("load", name, index, None)
        return super().load(name, index)

    def store(self, name, index, value, mode=None):
        self._accessed("store", name, index, mode)
        return super().store(name, index, value, mode)

    def store_reduction(self, name, index, value):
        self._accessed("store_reduction", name, index, None)
        return super().store_reduction(name, index, value)

    # each call's own record, completed at that call: the operands patch's (kernel, launch), then the KernelCallLine
    # its generate_kernel_call wrote (the wrapper's last line), the graph it was called in, that graph's scheduler
    # making it (an inlined control body's own for that inlining) and its source as made and defined (simd.py
    # codegen_node_schedule: codegen_kernel, then define_kernel, before this call): (metal_entry, metal_source)
    def call_kernel(self, name, node=None, deallocate_ws=True):
        wrapper = V.graph.wrapper_code
        super().call_kernel(name, node, deallocate_ws)
        wrapper.launched_kernels[-1] += (wrapper.lines[-1], V.graph, V.graph.scheduler,
                                         (self.metal_entry, self.metal_source))


class MeshMetalScheduling(MetalScheduling):
    kernel_type = MeshMetalKernel

    def codegen_node_schedule_with_kernel(self, node_schedule, kernel):
        kernel.mesh_schedule = tuple(node_schedule)
        super().codegen_node_schedule_with_kernel(node_schedule, kernel)


# codegen/cpp.py CppOverrides, CppVecOverrides and CppTile2DOverrides, each kernel class's own (common.py
# Kernel.__enter__: CSEProxy(kernel, kernel.overrides()))
class MeshCppOverrides(MeshIndirect, CppOverrides):
    pass


class MeshCppVecOverrides(MeshIndirect, CppVecOverrides):
    pass


class MeshCppTile2DOverrides(MeshIndirect, CppTile2DOverrides):
    pass


# codegen/cpp.py CppKernel, CppVecKernel and CppTile2DKernel as their CppKernelProxy's factory makes each
# (codegen_functions' codegen_kernel), each retaining its set_ranges result, its suffix scope, each node it codegens
# (CppRanges) with each access as it receives it, and its indirect symbols (mesh_indirect). An access is taken at the
# outermost of its class's load, store and store_reduction, before the original's own rename_indexing, a tile's
# transform_indexing or a vector load's scalar load, none of which reaches this class again; CppVecOverrides.load_seed's
# own V.kernel.load among them
class MeshCppAccesses:
    def __init__(self, *args):
        super().__init__(*args)
        self.mesh_indirect, self.mesh_ranges, self.mesh_split, self.mesh_suffix = {}, [], None, False

    def set_ranges(self, lengths, reduction_lengths):
        made = super().set_ranges(lengths, reduction_lengths)
        self.mesh_split = tuple(map(tuple, made))
        return made

    @contextlib.contextmanager
    def write_to_suffix(self):
        self.mesh_suffix = True
        with super().write_to_suffix():
            yield
        self.mesh_suffix = False

    @contextlib.contextmanager
    def set_current_node(self, node):
        self.mesh_ranges.append(CppRanges(node, self.mesh_split, self.mesh_suffix, []))
        with super().set_current_node(node):
            yield

    def _accessed(self, method, name, index, mode):
        shim = V.interpreter
        self.mesh_ranges[-1].accesses.append((method, name, index, mode, shim.graph, shim.current_node,
                                              self._load_mask))

    def load(self, name, index):
        self._accessed("load", name, index, None)
        return super().load(name, index)

    def store(self, name, index, value, mode=None):
        self._accessed("store", name, index, mode)
        return super().store(name, index, value, mode)

    def store_reduction(self, name, index, value):
        self._accessed("store_reduction", name, index, None)
        return super().store_reduction(name, index, value)


class MeshCppKernel(MeshCppAccesses, CppKernel):
    overrides = MeshCppOverrides


class MeshCppVecKernel(MeshCppAccesses, CppVecKernel):
    overrides = MeshCppVecOverrides


# a tile kernel is its proxy's vector kernel, as CppTile2DKernel is a CppVecKernel (aggregate_reduction_buffers:
# isinstance(kernels[0], vec_kernel_cls)). Its MRO: MeshCppTile2DKernel, MeshCppVecKernel, MeshCppAccesses,
# CppTile2DKernel, CppVecKernel, CppKernel; its accesses its own as received, CppTile2DKernel's methods then
# CppVecKernel's as the original's
class MeshCppTile2DKernel(MeshCppVecKernel, CppTile2DKernel):
    overrides = MeshCppTile2DOverrides


# codegen/cpp.py CppKernelProxy making these kernels (its kernel_cls, vec_kernel_cls and tile2d_kernel_cls factory;
# torch-inductor-cpu-kernel-results.patch codegen_conditions: a scalar kernel's tail condition by its class, not
# CppKernel's identity). `kernels` its own selection of them (scalar; vector and tail; tile2d and its tails), each one's
# active_ranges as codegen_functions set them, `loop_nest` its LoopNest (LoopLevels as tiled, then as codegen_loops
# marked them), `call_ranges` its own
class MeshCppKernelProxy(CppKernelProxy):
    kernel_cls, vec_kernel_cls, tile2d_kernel_cls = MeshCppKernel, MeshCppVecKernel, MeshCppTile2DKernel


# codegen/cpp.py KernelGroup: each finalize_kernel's (CppKernelProxy, nodes, local buffer context, local buffers) as it
# received it: the proxy (codegen_node's; codegen_outer_loop_node's merge_outer_fusion_kernels result, its LoopNest's
# kernel an OuterLoopFusedKernel whose `inner` LoopNests' kernels are the fused proxies), the nodes,
# V.local_buffer_context at that finalize (codegen_outer_loop_node's LocalBufferContext, cpp.py 5809, finalized under
# it at 5870: its global_buffers, global_to_local and removed_buffers, which its __exit__ keeps; virtualized.py's
# NullHandler outside one, the fallback's and codegen_node's finalize) and that context's local_buffers mapping as it
# holds it at the finalize, {local name: ir.Buffer}, the same Buffer objects add_local_buffer registered, the
# kernel-local buffers its accesses name and codegen_loops_impl declares (cpp.py 2717: under the same isinstance
# test). Its __exit__ clears local_buffers in place (cpp_utils.py 384), so the mapping is taken here; outside a
# LocalBufferContext the kernel's KernelArgs.input and output are unpatched and no name is local ({}). Each call's own
# record, as
# MeshMetalKernel.call_kernel's (wrapper.launched_kernels): (group, launch, its KernelCallLine, the graph it was called
# in, its scheduler, its (declared entry, final source) as define_kernel defined it: torch-inductor-upstream-composition
# .patch defined_kernels); its launch (kernel name, arg_defs, call_args, arg_types): its definition's own parameter
# declarations (torch-inductor-cpu-kernel-results.patch codegen_group arg_defs, flush's codegen_group before this call)
# and that KernelCallLine's kernel_name, call_args and arg_types as KernelGroup.call_kernel's generate_kernel_call wrote
# them
class MeshKernelGroup(KernelGroup):
    def __init__(self):
        super().__init__()
        self.mesh_proxies = []

    def finalize_kernel(self, new_kernel, nodes):
        context = V.local_buffer_context
        local = dict(context.local_buffers) if isinstance(context, LocalBufferContext) else {}
        self.mesh_proxies.append((new_kernel, tuple(nodes), context, local))
        super().finalize_kernel(new_kernel, nodes)

    def call_kernel(self, wrapper, kernel_name):
        super().call_kernel(wrapper, kernel_name)
        line = wrapper.lines[-1]
        wrapper.__dict__.setdefault("launched_kernels", []).append(
            (self, (line.kernel_name, self.arg_defs, tuple(line.call_args), tuple(line.arg_types)), line, V.graph,
             V.graph.scheduler, wrapper.defined_kernels[kernel_name]))


class MeshCppScheduling(CppScheduling):
    kernel_proxy_cls = MeshCppKernelProxy

    def __init__(self, scheduler):
        self.mesh_defined = []
        super().__init__(scheduler)

    def reset_kernel_group(self):
        self.kernel_group = MeshKernelGroup()

    # each kernel definition, the common definition producer of a group (flush) and a template (codegen_template) as
    # made: (the kernel handler it was defined under, the nodes it received, the kernel name it returned). A template's
    # handler is its CppTemplateKernel (codegen_template defines under V.set_kernel_handler(kernel)); a group's flush
    # sets none of its own
    def define_kernel(self, src_code, nodes, kernel_args=None):
        name = super().define_kernel(src_code, nodes, kernel_args)
        self.mesh_defined.append((V.kernel, tuple(nodes), name))
        return name

    # a CPU template's call, its own record made at the return of the original codegen making it (cpp.py
    # codegen_template: render, then define_kernel, then CppTemplateKernel.call_kernel), as MeshKernelGroup.call_kernel
    # makes a group's (wrapper.launched_kernels): (its CppTemplateKernel, launch, its KernelCallLine, the graph it was
    # called in, its scheduler, its (declared entry, final source) as define_kernel defined it, the nodes that
    # definition received). The kernel and nodes are the one definition this codegen made; the line is the one
    # KernelCallLine it wrote (call_kernel's generate_kernel_call, its one call), the launch that line's own call
    # arguments and types as the call produced them
    def codegen_template(self, template_node, epilogue_nodes, prologue_nodes):
        wrapper = V.graph.wrapper_code
        lines, defined = len(wrapper.lines), len(self.mesh_defined)
        super().codegen_template(template_node, epilogue_nodes, prologue_nodes)
        (kernel, nodes, name), = self.mesh_defined[defined:]
        line, = (made for made in wrapper.lines[lines:] if isinstance(made, KernelCallLine))
        wrapper.__dict__.setdefault("launched_kernels", []).append(
            (kernel, (line.kernel_name, TEMPLATE_DECLARED, tuple(line.call_args), tuple(line.arg_types)), line,
             V.graph, V.graph.scheduler, wrapper.defined_kernels[name], nodes))


SELECTED = {"mps": MeshMetalScheduling, "cpu": MeshCppScheduling}


# scheduler.py Scheduler with the selected backends, its own create_backend's selection
class MeshScheduler(Scheduler):
    def __init__(self, nodes):
        self.mesh_externs = []
        super().__init__(nodes)

    def create_backend(self, device):
        V.graph.add_device_info(device)
        return SELECTED[device.type](self)

    # each extern call's own record (codegen_extern_call: its ir node's codegen into its graph's wrapper), made at its
    # return with the control body inlinings (MeshSubgraph.codegen_subgraph) and liftings (MeshSubgraph.codegen) that
    # codegen made, each in order
    def codegen_extern_call(self, scheduler_node):
        wrapper, inlined, lifted = V.graph.wrapper_code, [], []
        start, self.mesh_inlined, self.mesh_lifted = len(wrapper.lines), inlined, lifted
        super().codegen_extern_call(scheduler_node)
        self.mesh_externs.append(ExternCall(scheduler_node.node, V.graph, self, tuple(wrapper.lines[start:]),
                                            tuple(inlined), tuple(lifted)))


# the compilation's own operand: graph.py GraphLowering as torch-inductor-graph-lowering-factory.patch's graph_lowering
# constructs it, and its control bodies' SubgraphLowering (make_subgraph: torch-inductor-upstream-composition.patch
# subgraph_lowering): each scheduled with the selected backends (upstream's _update_scheduler, with MeshScheduler);
# each placeholder's own result by its qualified name (GraphLowering.placeholder as it returned it, keyed as it keys
# graph_inputs and graph_input_names: qualify_name, a named SubgraphLowering's prefixed; a TensorBox, a sympy
# expression, a TorchBindObject, a generator's, or None, which graph_inputs does not hold); each body GraphLowering it
# made, in the order made (make_subgraph, the one constructor of a body's: ir.py InvokeSubgraph, Switch and WhileLoop
# create and SubgraphBuffer)
class _Lowering:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mesh_placeholders, self.mesh_subgraphs = {}, []

    def placeholder(self, target, args, kwargs):
        made = super().placeholder(target, args, kwargs)
        self.mesh_placeholders[self.qualify_name(target)] = made
        return made

    def make_subgraph(self, gm, example_inputs, subgraph_name):
        made = super().make_subgraph(gm, example_inputs, subgraph_name)
        self.mesh_subgraphs.append(made)
        return made

    def _update_scheduler(self):
        with config.patch("triton.store_cubin", False):
            self.scheduler = MeshScheduler(self.operations)


class MeshLowering(_Lowering, GraphLowering):
    pass


class MeshSubgraph(_Lowering, SubgraphLowering):
    # one inlining of this control body (wrapper.py codegen_subgraph_by_inlining, in aot_mode: while_loop's cond
    # before its loop, its body, its cond at the loop's end; invoke_subgraph), recorded for the extern call making it
    def codegen_subgraph(self, parent_graph):
        wrapper = parent_graph.wrapper_code
        start = len(wrapper.lines)
        super().codegen_subgraph(parent_graph)
        parent_graph.scheduler.mesh_inlined.append(Inlined(self, self.scheduler, tuple(wrapper.lines[start:]),
                                                           tuple(self.scheduler.mesh_externs)))

    # this control body lowered on its own (wrapper.py codegen_subgraph_common, under this graph's handler: a Switch's
    # branch under fx_wrapper), recorded for its parent's extern call making it (`parent` make_subgraph's, the graph
    # whose scheduler codegens that call)
    def codegen(self):
        made = super().codegen()
        self.parent.scheduler.mesh_lifted.append(Lifted(self, self.scheduler, self.wrapper_code,
                                                        tuple(self.scheduler.mesh_externs)))
        return made


MeshLowering.subgraph_lowering = MeshSubgraph.subgraph_lowering = MeshSubgraph


# ---- the producer ----

# a copy of `module` (torch/fx/graph.py graph_copy: each node's meta a shallow copy) and of each control body it holds
# (get_attr of a GraphModule), each copied node's original by (id(its graph), its name) in `sources`, each body's copy
# its own output as its HOP's signature has it and then adapted (_adapt). Returns (graph, its returned values,
# {get_attr target: its body's copy})
def _copied(module, sources):
    graph, copied = torch.fx.Graph(), {}
    returned = graph.graph_copy(module.graph, copied)
    for node, made in copied.items():
        sources[id(graph), made.name] = node
    bodies = {}
    for node in module.graph.find_nodes(op="get_attr"):
        held = functools.reduce(getattr, node.target.split("."), module)
        if isinstance(held, torch.fx.GraphModule):
            body, made, nested = _copied(held, sources)
            body.output(made)
            _adapt(body, sources)
            bodies[node.target] = _module(held, body, nested)
    return graph, returned, bodies


def _module(root, graph, bodies):
    made = torch.fx.GraphModule(root, graph)
    for target, body in bodies.items():
        made.add_submodule(target, body)
    return made


# schedule-emit lower_inductor.py adapt, over one copied graph (the body's and each control body's alike): with_effects
# as its operator with the token passed through (effects.py with_effects_dense: (token, *results), one result itself;
# a control body's token its own placeholder, returned as its signature returns it: torch 2.14.0
# invoke_subgraph.py, the one HOP whose body threads an effect token), which AOT autograd's functionalization makes
# again for each effectful operator on its own tokens (invoke_subgraph's in its body); each RECV as recv_strided at its
# node's value
def _adapt(graph, sources):
    for node in list(graph.nodes):
        if node.target is WITH_EFFECTS:
            token, operator, *arguments = node.args
            value = node.meta["val"]
            with graph.inserting_before(node):
                call = graph.call_function(operator, tuple(arguments), dict(node.kwargs))
            call.meta = {**node.meta, "val": tuple(value[1:]) if len(value) > 2 else value[1] if len(value) > 1 else None}
            sources[id(graph), call.name] = sources[id(graph), node.name]
            for getitem in list(node.users):
                k = getitem.args[1]
                if k and len(value) > 2:
                    getitem.args = (call, k - 1)
                else:
                    getitem.replace_all_uses_with(token if k == 0 else call)
                    graph.erase_node(getitem)
            graph.erase_node(node)
            node = call
        if node.target is RECV:
            value = node.meta["val"]
            node.target = RECV_STRIDED
            node.args = (node.args[0], list(value.shape), list(value.stride()), value.dtype, value.device)


# the processor body adapted: its copy and its control bodies' (_copied, _adapt), every SEND's value also returned
# after its own results (schedule-emit adapt; keep_output_stride). A SEND and a RECV are root nodes only:
# compiler/mesh_compiler/placement.py insert_send_recv, their one constructor, inserts them into the function's graph,
# a HOP call one node of one processor, and schedule.py assemble copies that graph's nodes, each HOP's bodies borrowed
# from its owner as they are; a sent value never crosses a HOP's interface. Returns (copy, its sources, the sent
# values' original nodes)
def adapted(module):
    sources = {}
    graph, returned, bodies = _copied(module, sources)
    output = graph.output(tuple(returned) if isinstance(returned, (tuple, list)) else (returned,))
    _adapt(graph, sources)
    sent = tuple(dict.fromkeys(node.args[0] for node in graph.find_nodes(op="call_function", target=SEND)))
    output.args = (output.args[0] + sent,)
    return _module(module, graph, bodies), sources, tuple(sources[id(graph), value.name] for value in sent)


# torch/fx/traceback.py NodeSource: each node's from_node chain (set_current_meta's CREATE under preserve_node_meta, the
# pattern matcher's REPLACE and torch-inductor-aot-lowered-result.patch _transfer_matched's every matched node), nested,
# to each (graph id, name) of `sources`: every original body node, in order
def _origins(sources, nodes):
    made = {}
    for node in nodes:
        _sourced(sources, node.meta.get("from_node", ()), made)
    return tuple(made)


def _sourced(sources, entries, made):
    for entry in entries:
        if (key := (entry.graph_id, entry.name)) in sources:
            made.setdefault(sources[key])
        _sourced(sources, entry.from_node, made)


def _scheduled_origins(scheduled, sources):
    return _origins(sources, [o for node in scheduled for member in node.get_nodes() for o in member.node.get_origins()])


# one call record (owner, launch, line, graph, scheduler, produced) as its call made it
def _mps(owner, launch, line, graph, scheduler, produced, sources):
    scheduled = tuple(r.node for r in owner.mesh_ranges)
    return Kernel("mps", graph, scheduler, line, owner, launch, produced, owner.mesh_schedule,
                  tuple(owner.mesh_ranges), scheduled, _scheduled_origins(scheduled, sources))


def _cpu(owner, launch, line, graph, scheduler, produced, sources):
    scheduled = tuple(owner.scheduled_nodes)
    return Kernel("cpu", graph, scheduler, line, owner, launch, produced, scheduled, tuple(owner.mesh_proxies),
                  scheduled, _scheduled_origins(scheduled, sources))


# a CPU template's call record (MeshCppScheduling.codegen_template), `nodes` its definition's
def _template(owner, launch, line, graph, scheduler, produced, nodes, sources):
    return Kernel("cpu", graph, scheduler, line, owner, launch, produced, nodes, TEMPLATE_RANGES, nodes,
                  _scheduled_origins(nodes, sources))


PLACED = {MeshMetalKernel: _mps, MeshKernelGroup: _cpu, CppTemplateKernel: _template}


# a graph's wrapper and each cond branch's own (wrapper_fxir.py lowered_subgraphs); an inlined control body
# (while_loop, invoke_subgraph in aot_mode) writes into its parent's
def _wrappers(graph):
    yield graph.wrapper_code
    for lowered in graph.wrapper_code.lowered_subgraphs.values():
        yield from _wrappers(lowered)


# GraphLowering `graph` and every body GraphLowering made under it (_Lowering.mesh_subgraphs), each once, depth first
# in the order made
def _graphs(graph):
    yield graph
    for made in graph.mesh_subgraphs:
        yield from _graphs(made)


# each RandomSeeds buffer of every graph `_graphs` yields, by its origin (that graph's lowered inductor_seeds node),
# each of that graph module's lookups' index and original nodes
def _seeds(graph, sources):
    found = []
    for lowered in _graphs(graph):
        owned = {origin: buffer for buffer in lowered.buffers if isinstance(buffer, upstream.RandomSeeds)
                 for origin in buffer.origins}
        uses = {}
        for node in lowered.module.graph.find_nodes(op="call_function",
                                                    target=torch.ops.prims.inductor_lookup_seed.default):
            uses.setdefault(owned[node.args[0]], []).append((node.args[1], _origins(sources, (node,))))
        found += [Seeds(lowered, buffer, tuple(made)) for buffer, made in uses.items()]
    return tuple(found)


def _returned(module):
    returned = module.graph.output_node().args[0]
    return tuple(returned) if isinstance(returned, (tuple, list)) else (returned,)


def _placeholders(graph):
    return graph.find_nodes(op="placeholder")


# one adapted body by compile_fx.py compile_fx_aot, the original AOT entry (V.aot_compilation: _compile_fx_main's
# run_pre_grad_passes, aot_export_module under unlift_effect_tokens, _unlift_graph, its inference compiler's joint and
# post-grad passes, compile_fx_inner without the empty-graph bailout or cache), at its scheduled placeholders' fake
# values under no_grad, `graph_lowering` MeshLowering (the selected backends). config_patches: fx_wrapper (its wrapper
# lines typed, cpp_wrapper off), aot_inductor.lowered (its lowered result: the GraphLowering after its scheduler's
# codegen, its backends' kernel sources made, before its wrapper's generate and any module or kernel compile or load),
# schedule-emit's assume_unaligned_fallback_output for a receiving body, and trace.provenance_tracking_level 1 (every
# rewrite's REPLACE sources)
def _inductor(module, body, sources, sent, processor, kind, device):
    received = any(n.target is RECV_STRIDED for n in body.graph.nodes)
    with torch.no_grad():
        graph = compile_fx_aot(body, [n.meta["val"] for n in _placeholders(body.graph)],
                               inner_compile=functools.partial(compile_fx_inner, graph_lowering=MeshLowering),
                               config_patches={"fx_wrapper": True, "aot_inductor.lowered": True,
                                               "assume_unaligned_fallback_output": received,
                                               "trace.provenance_tracking_level": 1})
    signature = graph.module.meta["graph_signature"]
    returned = _returned(module)
    names = signature.user_inputs
    return Produced(
        processor, kind, device, module, body, sources, sent, graph, signature,
        tuple(Input(k, node, names[k], graph.mesh_placeholders[graph.qualify_name(names[k])])
              for k, node in enumerate(_placeholders(module.graph))),
        tuple(Output(k, node, signature.user_outputs[k], graph.graph_outputs[k], k >= len(returned))
              for k, node in enumerate(returned + sent)),
        {id(record[2]): PLACED[type(record[0])](*record, sources)
         for wrapper in _wrappers(graph) for record in wrapper.__dict__.get("launched_kernels", ())},
        tuple(graph.scheduler.mesh_externs), _seeds(graph, sources), None)


# the Neural Engine's upstream compiler over one adapted body (its control bodies adapted by _copied): each received
# value an input of its own (the model runs once its receives have landed), each SEND erased (its value is returned;
# both root nodes only, adapted), torch.export's non-strict export at its fake values and Core ATen decomposition
# (run_decompositions), each program node's debug_handle its position from 1, coremltools' torch.export frontend
# (converters/mil/frontend/torch ops.py: the EXIR_DEBUG_HANDLE scope) to an ML program for ComputeUnit.CPU_AND_NE,
# compute_precision FLOAT32 (coremltools' selection: no fp16 rewrite), not loaded
def _engine(module, body, sources, sent, processor, kind, device):
    graph = body.graph
    first = next(n for n in graph.nodes if n.op != "placeholder")
    received = []
    for node in list(graph.nodes):
        if node.target is RECV_STRIDED:
            with graph.inserting_before(first):
                made = graph.placeholder(f"received_{node.args[0]}")
            made.meta = {**node.meta}
            sources[id(graph), made.name] = sources[id(graph), node.name]
            node.replace_all_uses_with(made)
            graph.erase_node(node)
            received.append(sources[id(graph), made.name])
        elif node.target is SEND:
            graph.erase_node(node)
    graph.lint()
    body.recompile()
    program = torch.export.export(body, tuple(n.meta["val"] for n in _placeholders(graph)),
                                  strict=False).run_decompositions()
    handles = {}
    for k, node in enumerate(program.graph.nodes, 1):
        node.meta["debug_handle"] = k
        handles[k] = _origins(sources, (node,))
    selected = {"convert_to": "mlprogram", "compute_units": ct.ComputeUnit.CPU_AND_NE,
                "compute_precision": ct.precision.FLOAT32, "minimum_deployment_target": ct.target.macOS26,
                "skip_model_load": True}
    model = ct.convert(program, **selected)
    signature = program.graph_signature
    named = {node.name: node for node in program.graph.nodes}
    placeholders = _placeholders(module.graph)
    returned = _returned(module)
    entry = Model(program, model, selected, tuple((len(placeholders) + k, node) for k, node in enumerate(received)),
                  handles)
    return Produced(
        processor, kind, device, module, body, sources, sent, None, signature,
        tuple(Input(k, node, signature.user_inputs[k], named[signature.user_inputs[k]])
              for k, node in enumerate(placeholders)),
        tuple(Output(k, node, signature.user_outputs[k], named[signature.user_outputs[k]], k >= len(returned))
              for k, node in enumerate(returned + sent)),
        {}, (), (), entry)


# one body at its processor's target: `kind` "cpu", "mps" or ENGINE, `device` the processor's torch.device
def produced(module, processor, kind, device):
    body, sources, sent = adapted(module)
    return (_engine if kind == ENGINE else _inductor)(module, body, sources, sent, processor, kind, device)


# design/compilation-composition-change-request.md, numerical lowerer boundary (schedule-emit lower_inductor.py lower):
# each function's processor bodies (compiler/mesh_compiler/schedule.py schedule's third result, ir.py
# Scheduled.processor_modules {function key: {processor: GraphModule}}) given to the upstream compiler at its
# processor's target, from the full original processor maps: Scheduled.devices its torch.device, Scheduled.engines its
# engine (ENGINE: the Neural Engine, whose fake values share the host's device; driver.py targets). Returns {function
# key: {processor: Produced}}
def lower(scheduled):
    devices, engines = scheduled.devices, scheduled.engines
    return {key: {processor: produced(module, processor, engines.get(processor, devices[processor].type),
                                      devices[processor])
                  for processor, module in bodies.items()}
            for key, bodies in scheduled.processor_modules.items()}


# ---- terminal emission ----

# metal_ir.h struct metal_ir_upstream_parameter: one parameter of a CPU pass's final entry as its final emitter declared
# it, `type` its C type as printed, `slot` the binding it receives (an argument index or a composition slot), `pointer`
# 1 for the bound pointer itself, 0 for the value it points at
@dataclasses.dataclass(frozen=True)
class Parameter:
    type: str
    slot: int
    pointer: int


# the terminal composition of one pass of one Metal kernel, the patched MetalKernel.codegen_kernel's `composition`
# (torch-inductor-upstream-composition.patch): `source` its definitions before the kernel (the executor's callable
# collection as printed, never parsed); `parameters` its own kernel parameters and `builtins` its builtin ones; `enter`
# its statements before the body, `grid` and `group` uint3 expressions the body's thread position in its grid and
# threadgroup after them, `exit` its statements after the body's stores; `paged` {argument index: its arena slot},
# each other argument at its own slot (argument i at buffer i, metal_lower.mm)
@dataclasses.dataclass(frozen=True)
class Composition:
    source: str
    parameters: tuple
    builtins: tuple
    enter: tuple
    grid: str
    group: str
    exit: tuple
    paged: dict

    def argument(self, index, declaration):
        # a paged argument's declaration names its arena, the body's subscript its local accessor of the same name
        if index in self.paged:
            return f"{declaration}_arena [[buffer({self.paged[index]})]]"
        return f"{declaration} [[buffer({index})]]"


# the terminal composition of one pass of one CPU kernel, the patched KernelGroup.codegen_group's `composition`
# (torch-inductor-upstream-composition.patch): `source` its definitions after the prefix (the executor's host C++
# callable collection as printed, never parsed), `parameters` its own after the group's arguments, `enter` its
# statements before the body; `typed` those parameters as their Parameter records
@dataclasses.dataclass(frozen=True)
class Hosted:
    source: str
    parameters: tuple
    enter: tuple
    typed: tuple


# metal_ir.h metal_ir_value_paged: byte b of a paged root at byte frames[b / page] page + b mod page of its arena (the
# arena bound from its first byte, its frame table after the arguments: metal_lower.mm), element i of an argument at
# byte origin + i sizeof(T) of its root (metal_ir_upstream_page `origin`: the argument binding's first byte in its
# root, placed); `P` the argument's own declared pointer type, every load and store its subscript
PAGED = """
template <typename P> struct mesh_paged {
  P arena;
  const device uint* frames;
  ulong origin, page;
  decltype(*P()) operator[](long i) const thread {
    const ulong at = origin + ulong(i) * sizeof(arena[0]);
    return arena[(ulong(frames[at / page]) * page + at % page) / sizeof(arena[0])];
  }
};
"""
# metal-microbench metal_transport.mm: each protocol function's parameters as Tint's MSL printer types them
# (tint/lang/msl/writer/printer: a storage runtime array `{const }device tint_array<T, 1>*`, host C++ (Options::host_cpp,
# provider/dawn.patch) `{const }tint_array<T, 1>*`; an atomic `atomic_uint` or `atomic_ulong`, u64 `ulong`), `votes` its
# workgroup array of 1024 / kSimd (metal_lower.h kSimd 32) u32
VOTES = 1024 // 32
# TransportViews: each word's (word, kind) its parameter; (1, 3) a later pass's `first` before its `supplied`
WORDS = {(0, 1): "step", (1, 0): "phases", (1, 1): "counts", (1, 2): "status", (2, 0): "flags", (3, 0): "list"}
BUILTINS = ("uint3 mesh_lid [[thread_position_in_threadgroup]]", "uint3 mesh_gid [[threadgroup_position_in_grid]]",
            "uint mesh_lane [[thread_index_in_simdgroup]]", "uint3 mesh_threads [[threads_per_threadgroup]]",
            "uint3 mesh_position [[thread_position_in_grid]]")


# the acquire function's parameters after its builtins, in its declared order (metal_ir.h struct
# metal_ir_upstream_protocol: mesh_acquire_first, mesh_acquire_later)
def _acquired(later):
    return ["step", "phases", "counts", *(["status", "first", "list"] if later else ["flags"]), "classes", "members",
            "records"]


# native composition `made`'s pass (`made.later`): each transport word, readiness table (metal_ir.h struct
# metal_ir_upstream_readiness: `classes` at class_slot or later_class_slot, `members` its classes' field membership at
# member_slot or later_member_slot, CONSTANT u32) and record table by its protocol parameter name, its (slot, element
# type, written); and its sent values' arrival counters in `sent` order
def _views(made):
    later = bool(made.later)
    firsts, named, arrivals = ["first", "supplied"] if later else ["supplied"], {}, []
    for w in made.later_words if later else made.words:
        if w.word >= 4:
            name = f"arrivals{w.word - 4}"
            arrivals.append(name)
        else:
            name = WORDS[w.word, w.kind] if (w.word, w.kind) in WORDS else firsts.pop(0)
        element = "ulong" if w.kind % 2 else "uint"
        named[name] = (w.slot, f"atomic_{element}" if w.kind >= 2 else element, bool(w.write))
    if made.readiness is not None:
        r = made.readiness
        named["classes"] = (r.later_class_slot if later else r.class_slot, "uint", False)
        named["members"] = (r.later_member_slot if later else r.member_slot, "uint", False)
    if made.table is not None:
        named["records"] = (made.table.later_slot if later else made.table.slot, "ulong", False)
    return named, arrivals


# the pass's acquire function the collection makes (metal.Callables: `later` mesh_acquire_later for pass 1, `acquire`
# mesh_acquire_first for pass 0; None where it makes none)
def _acquire(made):
    return made.callables.later if made.later else made.callables.acquire


# a Metal kernel's pass: each view a parameter at its slot as the protocol's parameters type it; the collection's
# mesh_acquire_first (pass 0) or mesh_acquire_later (pass 1) first, its threadgroup returning where it does not run (.w
# 0), the body at the position it gives (pass 0 its own: thread_position_in_grid; pass 1 its listed threadgroup's .xyz,
# whole threadgroups, its further lanes outside the body's bound reaching mesh_release); mesh_release after the body's
# stores, its arrival counters in `sent` order; each paged argument's accessor over its arena
def _threaded(kernel, made):
    callables, later, args, values = made.callables, bool(made.later), kernel.owner.args, kernel.launch[1]
    named, arrivals = _views(made)
    parameters = [f"{'device' if write else 'const device'} tint_array<{element}, 1>* mesh_{name} [[buffer({slot})]]"
                  for name, (slot, element, write) in named.items()]
    parameters += [f"const device uint* mesh_frames{slot} [[buffer({slot})]]"
                   for slot in dict.fromkeys(p.frames for p in made.pages)]
    enter, exit_ = [], []
    for p in made.pages:
        outer = values[p.argument]
        inner = args.output_buffers[outer] if outer in args.output_buffers else args.input_buffers[outer]
        enter.append(f"const mesh_paged<decltype({inner}_arena)> {inner}{{{inner}_arena, mesh_frames{p.frames}, "
                     f"{p.origin}ul, {p.page_bytes}ul}};")
    grid, acquire = "mesh_position", _acquire(made)
    if acquire is not None:
        enter += [f"threadgroup tint_array<uint, {VOTES}> mesh_votes;",
                  f"const uint4 mesh_acquired = {acquire}("
                  + ", ".join(["mesh_lid", "mesh_gid", "mesh_lane", "mesh_threads",
                               *(f"mesh_{name}" for name in _acquired(later)), "&mesh_votes"]) + ");",
                  "if (mesh_acquired.w == 0) return;"]
        if later:
            grid = "(mesh_acquired.xyz * mesh_threads + mesh_lid)"
    if callables.release is not None:
        exit_.append(f"{callables.release}(" + ", ".join(["mesh_lid", "mesh_threads", "mesh_step", "mesh_status",
                                                          "mesh_supplied", "mesh_records",
                                                          *(f"mesh_{name}" for name in arrivals)]) + ");")
    return Composition(callables.source + (PAGED if made.pages else ""), tuple(parameters), BUILTINS, tuple(enter),
                       grid, "mesh_lid", tuple(exit_), {p.argument: p.slot for p in made.pages})


# a CPU kernel's pass (a native task: its one invocation; its sends published by its Steps, no release): the
# collection's bool mesh_acquire_first or mesh_acquire_later first, its body run where it returns true, each of its
# parameters the kernel's own at its slot as the host C++ collection types it; a paged argument's binding its flat view
def _hosted(kernel, made):
    named, _ = _views(made)
    acquire = _acquire(made)
    used = _acquired(bool(made.later)) if acquire is not None else []
    typed = tuple(Parameter(f"{'' if named[name][2] else 'const '}tint_array<{named[name][1]}, 1>*", named[name][0], 1)
                  for name in used)
    enter = (f"if (!{acquire}(" + ", ".join(f"mesh_{name}" for name in used) + ")) return;",) if used else ()
    return Hosted(made.callables.source, tuple(f"{p.type} mesh_{name}" for name, p in zip(used, typed)), enter, typed)


# kernel `kernel`'s composition for one pass from its native composition `made` (compiler/mesh_compiler/metal.py
# Composed: metal_ir_upstream_composition's words, later words, pages (each with its argument binding's placed origin),
# readiness and table, the kernel's one callable collection `callables` (metal.Callables: metal_ir_callables_emit's
# source, its binary64 license notice its leading comment, and metal_ir_upstream_protocol's acquire, later and release),
# `later` its pass): a Metal kernel's Composition, a CPU kernel's Hosted
def composed(kernel, made):
    return (_hosted if kernel.place == "cpu" else _threaded)(kernel, made)


# metal_ir.h struct metal_ir_upstream_emission
@dataclasses.dataclass(frozen=True)
class Emission:
    source: str
    entry: str
    parameters: tuple
    compile: tuple
    link: tuple


# metal_ir.h metal_ir_upstream_emitted: the kernel's terminal emission by its own upstream emitter for one pass
# (`composition` its composed() result; None: as produced), from the same retained owner for each pass. Metal: the
# patched MetalKernel.codegen_kernel of the retained kernel, its headers expanded (torch/mps/__init__.py
# compile_shader's _embed_headers). CPU: cpp.py define_kernel's final source and declared entry
# (torch-inductor-upstream-composition.patch defined_kernels of its line's wrapper; a group's and a template's alike),
# or the retained group's patched codegen_group with its composition through define_kernel's defined_source (a
# template's composed pass needs its own upstream hook, cpp_template_kernel.py CppTemplateKernel.def_kernel and its
# body's opening, which no patch has yet: numerical68-result.md); its parameters its call's arguments in
# cpp_argdefs order (each at its own slot; its buffers, inplace, input and output, the bound pointer, its sizevars after
# them the value) and the composition's; codecache.py CppCodeCache's build options (CppTorchDeviceOptions for the cpu,
# CppPythonBindingsCodeCache: no libtorch, shared) in cpp_builder.py CppBuilder's argument form
def emitted(kernel, composition=None):
    owner = kernel.owner
    if kernel.place == "cpu":
        name, _, _, types = kernel.launch
        entry, source = kernel.produced
        values = len(types) - len(owner.args.sizevars)
        parameters = tuple(Parameter(t, k, int(k < values)) for k, t in enumerate(types))
        if composition is not None:
            with V.set_graph_handler(kernel.graph):
                source = CppScheduling.defined_source(owner.codegen_group(None, composition), entry, name)
            parameters += composition.typed
        options = CppTorchDeviceOptions(vec_isa=pick_vec_isa(), include_pytorch=False, device_type="cpu", shared=True)
        compile_ = (options.get_compiler(), *[f for d in options.get_definitions() for f in ("-D", d)],
                    *[f"-I{d}" for d in options.get_include_dirs()], *[f"-{f}" for f in options.get_cflags()],
                    *options.get_passthrough_args())
        link = (*[f"-{f}" for f in options.get_ldflags()], *[f"-L{d}" for d in options.get_libraries_dirs()],
                *[f"-l{lib}" for lib in options.get_libraries()])
        return Emission(source, entry, parameters, compile_, link)
    entry, source = kernel.produced
    if composition is not None:
        with V.set_graph_handler(kernel.graph), V.set_kernel_handler(owner):
            owner.codegen_kernel(entry, composition)
        source = owner.metal_source
    source = _embed_headers([line + "\n" for line in source.split("\n")],
                            [Path(torch.__file__).parent / "include"], set())
    return Emission(source, entry, (), (), ())
