# design/compilation-composition-change-request.md, numerical lowerer boundary, items 2 and 5 to 7, S8 and Complete
# replacement; design/aot-compilation-change-request.md A1, A5, A6 and A8 to A10; design/torch-source-lowering-change-
# request.md items 1 to 3, 5 and 6; /Users/mdot/schedule-emit/schedule_emit/lower_inductor.py at f41c64a (adapt,
# recv_strided, lower): each processor's scheduled body given to its target's installed upstream compiler (torch 2.14.0
# torch/_inductor: compile_fx.py's ahead-of-time inference sequence, AOT autograd's export and effect tokens,
# GraphLowering, its Scheduler and the selected backend; coremltools 9.0 for the Neural Engine), its result retained as
# that compiler's own objects. Fields and pairing: output_data/compiler-workgroup-20261114/compiler/
# numerical65-operands.md; upstream hooks: compiler/patches/torch-inductor-mps-kernel-operands.patch and
# torch-inductor-upstream-composition.patch
import dataclasses
import functools
from pathlib import Path

import coremltools as ct
import torch
from torch import Tensor
from torch._dispatch.python import enable_python_dispatcher
from torch._export.utils import _detect_fake_mode_from_gm
from torch._functorch import config as functorch_config
from torch._functorch.aot_autograd import aot_export_module
from torch._higher_order_ops.effects import _EffectType, _register_effectful_op
from torch._inductor import config
from torch._inductor import ir as upstream
from torch._inductor.codegen.cpp import CppScheduling, KernelGroup
from torch._inductor.codegen.mps import MetalKernel, MetalOverrides, MetalScheduling
from torch._inductor.codegen.simd import DisableReduction, EnableReduction
from torch._inductor.compile_fx import (_recursive_post_grad_passes, _recursive_record_original_output_strides,
                                        _unlift_graph, compile_fx_forward, create_compiler_config_extra,
                                        fake_tensor_prop, get_num_model_outputs, maybe_disable_comprehensive_padding,
                                        run_pre_grad_passes)
from torch._inductor.cpp_builder import CppTorchDeviceOptions
from torch._inductor.cpu_vec_isa import pick_vec_isa
from torch._inductor.debug import DebugContext
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.fx_passes.post_grad import view_to_reshape
from torch._inductor.graph import GraphLowering, SubgraphLowering
from torch._inductor.lowering import fallback_handler, register_lowering
from torch._inductor.scheduler import Scheduler
from torch._inductor.utils import shape_env_from_inputs
from torch._inductor.virtualized import V
from torch.utils._cpp_embed_headers import _embed_headers

from .ir import RECV, SEND, channel_of

WITH_EFFECTS = torch.ops.higher_order.with_effects
# torch/fx/proxy.py _COPY_META_FIELDS "custom", copied under fx/traceback.py preserve_node_meta (AOT autograd's
# create_functional_call runs a GraphModule through an Interpreter under it; a control body's through
# torch/_higher_order_ops/utils.py _maybe_run_with_interpreter), by _inductor/pattern_matcher.py _transfer_meta and by
# torch.export: each lowered node's original body node, its index in Produced.sources
SOURCE = "mesh_source"
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


# ---- the retained result (numerical65-operands.md): the upstream compiler's own objects and their correspondence ----

# body input `position`: `node` the scheduled body's placeholder, `name` the upstream compiler's input (GraphLowering:
# its graph_inputs_original key, storage offset graph_input_storage_offsets[name]; Neural Engine: its ExportedProgram
# user input)
@dataclasses.dataclass(frozen=True)
class Input:
    position: int
    node: object
    name: str


# compiled output `position`: `node` the body node it returns (a SEND's value where `sent`: schedule-emit adapt returns
# every sent value, keep_output_stride), `value` the upstream compiler's (GraphLowering.graph_outputs[position]; Neural
# Engine: its ExportedProgram user output name)
@dataclasses.dataclass(frozen=True)
class Output:
    position: int
    node: object
    value: object
    sent: bool


# codegen/simd.py codegen_node_schedule_with_kernel's second pass for SchedulerNode `node` (after finalize_indexing and
# its LoopBody's indexing_dtype_strength_reduction and convert_index_expr_to_value_expr): `variables` its
# split_and_set_ranges result (each a range-tree entry symbol of the kernel's range_tree_nodes, whose expr is its tree
# root's FloorDiv or ModularIndexing), `reduction` the kernel's inside_reduction there (False inside DisableReduction),
# `accesses` each memory access its node.codegen made, in order, as the kernel received it (MetalKernel load, store and
# store_reduction: (method, buffer name, index, store mode), the index LoopBody.__call__ bound and SimplifyIndexing
# simplified, over those variables and the kernel's indirect symbols, MeshMetalKernel.mesh_indirect)
@dataclasses.dataclass(eq=False)
class Ranges:
    node: object
    variables: tuple
    reduction: bool
    accesses: tuple


# one upstream-generated kernel of `graph` (its GraphLowering, a control body's own): `place` "mps" or "cpu",
# `scheduled` the Scheduler nodes it implements in order (their read_writes the Scheduler's own accesses), `owner` the
# backend's kernel (MeshMetalKernel: range trees, body, KernelArgs, workspace_args, headers, mesh_indirect,
# metal_entry, metal_source; MeshKernelGroup: its KernelArgs and CppKernelProxy loop nests), `launch` its call record
# as produced (mps: torch-inductor-mps-kernel-operands.patch (name, values, types, grid, group, error: its error
# buffer's argument index, c10/metal/error.h ErrorMessages as its declaration names it); cpu: (kernel name, arg_defs,
# call_args, arg_types) of KernelArgs.cpp_argdefs), `ranges` (mps) its Ranges or (cpu) each finalize_kernel's
# (CppKernelProxy, nodes), `origins` the original body nodes it implements
@dataclasses.dataclass(eq=False)
class Kernel:
    place: str
    graph: object
    scheduled: tuple
    owner: object
    launch: tuple
    ranges: tuple
    origins: tuple


# one upstream extern of `graph` (ExternKernelSchedulerNode `scheduled`, its node the ir.py ExternKernel: FallbackKernel,
# EffectfulKernel, ExternKernelOut, RandomSeeds, MultiOutput, mesh send and receive ...), `origins` the original body
# nodes, `bodies` a control extern's (Conditional, WhileLoop, InvokeSubgraph: ir.py get_subgraphs) each subgraph's name
# and its own entries, lowered and scheduled by its own GraphLowering (subgraph.graph)
@dataclasses.dataclass(eq=False)
class Extern:
    graph: object
    scheduled: object
    origins: tuple
    bodies: dict


# fx_passes/replace_random.py: ir.py RandomSeeds `buffer` the body's random state it draws through, `uses` each
# inductor_lookup_seed index and the original random node it serves; its words the body's Generator (operations.py)
@dataclasses.dataclass(eq=False)
class Seeds:
    buffer: object
    uses: tuple


# a Neural Engine body's upstream result: `program` its torch.export ExportedProgram (Core ATen), `model` coremltools'
# MLModel (an ML program for ComputeUnit.CPU_AND_NE at the body's own precision; its MIL Program each operation's
# EXIR_DEBUG_HANDLE scope its SOURCE), `received` each (user input position, RECV body node) it binds as an input
@dataclasses.dataclass(eq=False)
class Model:
    program: object
    model: object
    received: tuple
    origins: tuple


# one processor body's retained upstream result: `module` Scheduled.processor_modules[key][processor] as scheduled,
# `sources` its nodes and its control bodies' (SOURCE), `lowered` the graph the upstream compiler lowered (after AOT
# autograd's export, its unlifting and compile_fx_forward's and the post-grad passes; Neural Engine: its
# ExportedProgram), `signature` its graph signature (aot_export_module's GraphSignature: user inputs and outputs, input
# mutations, tokens; Neural Engine: ExportedProgram.graph_signature), `graph` its GraphLowering (buffers, layouts and
# aliases, graph_inputs_original, graph_input_storage_offsets, mutated_inputs, removed_buffers, effectful_ops,
# scheduler, wrapper_code; None for the Neural Engine), `entries` Kernel and Extern in its Scheduler's emission order (a
# Neural Engine body's one Model), `wrapper` GraphLowering.codegen's result
@dataclasses.dataclass(eq=False)
class Produced:
    processor: str
    kind: str
    device: object
    module: object
    sources: tuple
    lowered: object
    signature: object
    graph: object
    inputs: tuple
    outputs: tuple
    entries: tuple
    seeds: tuple
    wrapper: object


# ---- the selected upstream backends, retaining their own results ----

# codegen/common.py CSEProxy.indirect_indexing: each indirect symbol its kernel's overrides make, with the value it
# indexes by and its size (and whether its bounds are checked), the indirect fact its accesses' indices carry
class MeshMetalOverrides(MetalOverrides):
    def indirect_indexing(self, var, size, check=True, wrap_neg=True):
        made = super().indirect_indexing(var, size, check, wrap_neg)
        V.kernel.mesh_indirect[made] = (var, size, check)
        return made


# codegen/mps.py MetalKernel: its second-pass ranges (split_and_set_ranges after finalize_indexing) and each access as
# it receives it
class MeshMetalKernel(MetalKernel):
    overrides = MeshMetalOverrides

    def __init__(self, tiling, **kwargs):
        super().__init__(tiling, **kwargs)
        self.mesh_indexed, self.mesh_ranges, self.mesh_schedule = False, [], []
        self.mesh_accesses, self.mesh_indirect = [], {}

    def finalize_indexing(self, indices):
        super().finalize_indexing(indices)
        self.mesh_indexed = True

    def split_and_set_ranges(self, lengths):
        made = super().split_and_set_ranges(lengths)
        if self.mesh_indexed:
            self.mesh_ranges.append((tuple(map(tuple, made)), self.inside_reduction))
        return made

    def load(self, name, index):
        self.mesh_accesses.append((self.current_node, ("load", name, index, None)))
        return super().load(name, index)

    def store(self, name, index, value, mode=None):
        self.mesh_accesses.append((self.current_node, ("store", name, index, mode)))
        return super().store(name, index, value, mode)

    def store_reduction(self, name, index, value):
        self.mesh_accesses.append((self.current_node, ("store_reduction", name, index, None)))
        return super().store_reduction(name, index, value)


class MeshMetalScheduling(MetalScheduling):
    kernel_type = MeshMetalKernel

    def codegen_node_schedule_with_kernel(self, node_schedule, kernel):
        super().codegen_node_schedule_with_kernel(node_schedule, kernel)
        kernel.mesh_indexed = False
        kernel.mesh_schedule = [n for n in node_schedule if n is not DisableReduction and n is not EnableReduction]

    def codegen_node(self, node):
        launched = V.graph.wrapper_code.__dict__.setdefault("launched_kernels", [])
        start = len(launched)
        super().codegen_node(node)
        self.scheduler.mesh_made.append(("simd", node, tuple(launched[start:])))


# codegen/cpp.py KernelGroup: each finalize_kernel's CppKernelProxy (its loop nests and local buffers) and its call
class MeshKernelGroup(KernelGroup):
    def __init__(self):
        super().__init__()
        self.mesh_proxies, self.mesh_launch = [], None

    def finalize_kernel(self, new_kernel, nodes):
        self.mesh_proxies.append((new_kernel, tuple(nodes)))
        super().finalize_kernel(new_kernel, nodes)

    def call_kernel(self, wrapper, kernel_name):
        self.mesh_launch = (kernel_name, *map(tuple, self.args.cpp_argdefs()))
        super().call_kernel(wrapper, kernel_name)


class MeshCppScheduling(CppScheduling):
    def reset_kernel_group(self):
        self.kernel_group = MeshKernelGroup()

    def flush(self):
        group = self.kernel_group
        super().flush()
        if group.mesh_launch is not None:
            self.scheduler.mesh_made.append(("cpp", group))


SELECTED = {"mps": MeshMetalScheduling, "cpu": MeshCppScheduling}


# scheduler.py Scheduler with the selected backends (create_backend) and its emission order (codegen_extern_call)
class MeshScheduler(Scheduler):
    def __init__(self, nodes, selected):
        self.mesh_selected, self.mesh_made = selected, []
        super().__init__(nodes)

    def create_backend(self, device):
        V.graph.add_device_info(device)
        return self.mesh_selected[device.type](self)

    def codegen_extern_call(self, scheduler_node):
        super().codegen_extern_call(scheduler_node)
        self.mesh_made.append(("extern", scheduler_node))


def _update_scheduler(self):
    with config.patch("triton.store_cubin", False):
        self.scheduler = MeshScheduler(self.operations, self.mesh_selected)


# graph.py GraphLowering and its control bodies' SubgraphLowering (make_subgraph: torch-inductor-upstream-
# composition.patch subgraph_lowering), each scheduled with the selected backends
class MeshLowering(GraphLowering):
    def __init__(self, gm, *, selected, **kwargs):
        self.mesh_selected = selected
        super().__init__(gm, **kwargs)

    _update_scheduler = _update_scheduler


class MeshSubgraph(SubgraphLowering):
    def __init__(self, parent, *args, **kwargs):
        self.mesh_selected = parent.mesh_selected
        super().__init__(parent, *args, **kwargs)

    _update_scheduler = _update_scheduler


MeshLowering.subgraph_lowering = MeshSubgraph.subgraph_lowering = MeshSubgraph


# ---- the producer ----

# a copy of `module` (torch/fx/graph.py graph_copy: each node's meta a shallow copy) and of each control body it holds
# (get_attr of a GraphModule), each copied node's SOURCE its original's index in `sources`. Returns (graph, its returned
# values, {get_attr target: its body's copy})
def _copied(module, sources):
    graph, copied = torch.fx.Graph(), {}
    returned = graph.graph_copy(module.graph, copied)
    for node, made in copied.items():
        made.meta["custom"] = {**made.meta.get("custom", {}), SOURCE: len(sources)}
        sources.append(node)
    bodies = {}
    for node in module.graph.nodes:
        held = functools.reduce(getattr, node.target.split("."), module) if node.op == "get_attr" else None
        if isinstance(held, torch.fx.GraphModule):
            body, made, nested = _copied(held, sources)
            body.output(made)
            bodies[node.target] = _module(held, body, nested)
    return graph, returned, bodies


def _module(root, graph, bodies):
    made = torch.fx.GraphModule(root, graph)
    for target, body in bodies.items():
        made.add_submodule(target, body)
    return made


# schedule-emit lower_inductor.py adapt over that copy: with_effects as its operator with the token passed through
# (effects.py with_effects_dense: (token, *results), one result itself), which AOT autograd's functionalization makes
# again for each effectful operator on its own tokens; each RECV as recv_strided at its node's value; every SEND's value
# also returned. Returns (copy, the body's and its control bodies' nodes by SOURCE, the sent values' original nodes)
def adapted(module):
    sources = []
    graph, returned, bodies = _copied(module, sources)
    sent = []
    for node in list(graph.nodes):
        if node.target is WITH_EFFECTS:
            token, operator, *arguments = node.args
            value = node.meta["val"]
            with graph.inserting_before(node):
                call = graph.call_function(operator, tuple(arguments), dict(node.kwargs))
            call.meta = {**node.meta, "val": tuple(value[1:]) if len(value) > 2 else value[1] if len(value) > 1 else None}
            for getitem in list(node.users):
                k = getitem.args[1]
                if k and len(value) > 2:
                    getitem.args = (call, k - 1)
                else:
                    getitem.replace_all_uses_with(token if k == 0 else call)
                    graph.erase_node(getitem)
            graph.erase_node(node)
            node = call
        if node.target is SEND:
            sent.append(node.args[0])
        elif node.target is RECV:
            value = node.meta["val"]
            node.target = RECV_STRIDED
            node.args = (node.args[0], list(value.shape), list(value.stride()), value.dtype, value.device)
    sent = tuple(dict.fromkeys(sent))
    returned = tuple(returned) if isinstance(returned, (tuple, list)) else (returned,)
    graph.output(returned + sent)
    sources = tuple(sources)
    return _module(module, graph, bodies), sources, tuple(_source(value, sources) for value in sent)


def _source(node, sources):
    k = node.meta.get("custom", {}).get(SOURCE)
    return None if k is None else sources[k]


def _origins(nodes, sources):
    return tuple(dict.fromkeys(s for n in nodes if (s := _source(n, sources)) is not None))


def _node_origins(scheduled, sources):
    return _origins([o for node in scheduled for member in node.get_nodes() if member.node is not None
                     for o in member.node.get_origins()], sources)


# a SIMD kernel's launch record as its Kernel, its Ranges in its second pass's node order, each with its accesses
def _simd(graph, scheduled, kernel, launch, sources):
    ranges = tuple(Ranges(node, variables, reduction, tuple(a for n, a in kernel.mesh_accesses if n is node))
                   for node, (variables, reduction) in zip(kernel.mesh_schedule, kernel.mesh_ranges))
    return Kernel("mps", graph, (scheduled,), kernel, launch, ranges, _node_origins((scheduled,), sources))


def _cpp(graph, group, sources):
    scheduled = tuple(group.scheduled_nodes)
    return Kernel("cpu", graph, scheduled, group, group.mesh_launch, tuple(group.mesh_proxies),
                  _node_origins(scheduled, sources))


# `graph`'s Scheduler emission (MeshScheduler.mesh_made) as entries, each control extern's bodies their own
def _entries(graph, sources):
    made = []
    for kind, *record in graph.scheduler.mesh_made:
        if kind == "extern":
            (scheduled,) = record
            made.append(Extern(graph, scheduled, _node_origins((scheduled,), sources),
                               {body.name: _entries(body.graph, sources) for body in scheduled.node.get_subgraphs()}))
        elif kind == "simd":
            scheduled, launched = record
            made += [_simd(graph, scheduled, kernel, launch, sources) for kernel, launch in launched]
        else:
            made.append(_cpp(graph, record[0], sources))
    return tuple(made)


# each RandomSeeds buffer by its origin (the lowered inductor_seeds node), each lookup's index and original random node
def _seeds(graph, lowered, sources):
    owned = {origin: buffer for buffer in graph.buffers if isinstance(buffer, upstream.RandomSeeds)
             for origin in buffer.origins}
    uses = {}
    for node in lowered.graph.nodes:
        if node.target is torch.ops.prims.inductor_lookup_seed.default:
            uses.setdefault(owned[node.args[0]], []).append((node.args[1], _source(node, sources)))
    return tuple(Seeds(buffer, tuple(made)) for buffer, made in uses.items())


def _returned(module):
    returned = module.graph.output_node().args[0]
    return tuple(returned) if isinstance(returned, (tuple, list)) else (returned,)


def _placeholders(graph):
    return [n for n in graph.nodes if n.op == "placeholder"]


# compile_fx.py _compile_fx_main's ahead-of-time inference branch over one adapted body: run_pre_grad_passes; AOT
# autograd's aot_export_module (trace_joint False, select_decomp_table, unlift_effect_tokens: each effectful operator a
# with_effects on its token, the tokens _make_token and _sink_tokens; each input mutation an output) under
# enable_python_dispatcher; each constant tensor attribute's fake value; _unlift_graph (its parameters and buffers
# attributes again, each input mutation its copy_); then compile_fx_forward at is_inference (its original output strides,
# joint graph passes and user visible outputs) with, as its inner compiler, _InProcessFxCompile.codegen_and_compile's
# stages (view_to_reshape, fake_tensor_prop, original output strides, post-grad passes, GraphLowering run and codegen),
# its GraphLowering MeshLowering with the selected backends (SELECTED)
def _inductor(module, body, sources, sent, processor, kind, device):
    inputs = [n.meta["val"] for n in _placeholders(body.graph)]
    received = any(n.target is RECV_STRIDED for n in body.graph.nodes)
    returned = _returned(module)

    def compiled(gm, example_inputs, *, static_input_idxs, cudagraphs, graph_id, is_inference,
                 boxed_forward_device_index):
        shape_env = shape_env_from_inputs(example_inputs) if gm.shape_env is None else gm.shape_env
        view_to_reshape(gm)
        with torch.no_grad():
            fake = fake_tensor_prop(gm, example_inputs)
        _recursive_record_original_output_strides(gm)
        with V.set_fake_mode(fake):
            _recursive_post_grad_passes(gm, is_inference=is_inference)
        with V.set_fake_mode(fake), maybe_disable_comprehensive_padding(example_inputs):
            graph = MeshLowering(gm, selected=SELECTED, example_inputs=example_inputs, shape_env=shape_env,
                                 graph_id=graph_id, is_inference=is_inference)
            graph.freeze_runtime_asserts()
            with V.set_graph_handler(graph), V.set_extern_kernel_nodes([]):
                graph.run(*example_inputs)
                wrapper = graph.codegen()
                names = [n.target for n in _placeholders(gm.graph)]
                return Produced(
                    processor, kind, device, module, sources, gm, signature, graph,
                    tuple(Input(k, node, name) for k, (node, name) in enumerate(zip(_placeholders(module.graph),
                                                                                     names))),
                    tuple(Output(k, node, value, k >= len(returned))
                          for k, (node, value) in enumerate(zip(returned + sent, graph.graph_outputs))),
                    _entries(graph, sources), _seeds(graph, gm, sources), wrapper)

    with DebugContext(), config.patch(assume_unaligned_fallback_output=received), enable_python_dispatcher():
        body = run_pre_grad_passes(body, inputs)
        with functorch_config.patch(unlift_effect_tokens=True, selective_decompose=config.selective_decompose):
            exported, signature = aot_export_module(body, inputs, trace_joint=False,
                                                    decompositions=select_decomp_table())
        fake = _detect_fake_mode_from_gm(exported)
        for node in exported.graph.find_nodes(op="get_attr"):
            held = functools.reduce(getattr, node.target.split("."), exported)
            if isinstance(held, Tensor) and "val" not in node.meta:
                node.meta["val"] = fake.from_tensor(held, static_shapes=True)
        lowered = _unlift_graph(body, exported, signature)
        with V.set_fake_mode(fake):
            return compile_fx_forward(lowered, inputs, num_orig_model_outputs=get_num_model_outputs(body),
                                      num_example_inputs=len(inputs),
                                      compiler_config_extra=create_compiler_config_extra(body),
                                      inner_compile=compiled, is_inference=True)


# the Neural Engine's upstream compiler over one adapted body: each received value an input of its own (the model
# runs once its receives have landed), each SEND erased (its value is returned), torch.export's non-strict export at
# its fake values and Core ATen decomposition (run_decompositions), each node's debug_handle its SOURCE, coremltools'
# torch.export frontend (converters/mil/frontend/torch ops.py: the EXIR_DEBUG_HANDLE scope) to an ML program for
# ComputeUnit.CPU_AND_NE, compute_precision FLOAT32 (no fp16 rewrite: each value at its own dtype), not loaded
def _engine(module, body, sources, sent, processor, kind, device):
    graph = body.graph
    first = next(n for n in graph.nodes if n.op != "placeholder")
    received = []
    for node in list(graph.nodes):
        if node.target is RECV_STRIDED:
            with graph.inserting_before(first):
                made = graph.placeholder(f"received_{node.args[0]}")
            made.meta = {**node.meta}
            node.replace_all_uses_with(made)
            graph.erase_node(node)
            received.append(_source(made, sources))
        elif node.target is SEND:
            graph.erase_node(node)
    graph.lint()
    body.recompile()
    placeholders = _placeholders(graph)
    program = torch.export.export(body, tuple(n.meta["val"] for n in placeholders), strict=False).run_decompositions()
    for node in program.graph.nodes:
        if (k := node.meta.get("custom", {}).get(SOURCE)) is not None:
            node.meta["debug_handle"] = k
    model = ct.convert(program, convert_to="mlprogram", compute_units=ct.ComputeUnit.CPU_AND_NE,
                       compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.macOS26,
                       skip_model_load=True)
    signature = program.graph_signature
    count = len(_placeholders(module.graph))
    returned = _returned(module)
    entry = Model(program, model, tuple((count + k, node) for k, node in enumerate(received)),
                  _origins(program.graph.nodes, sources))
    return Produced(
        processor, kind, device, module, sources, program, signature, None,
        tuple(Input(k, node, name) for k, (node, name) in enumerate(zip(_placeholders(module.graph),
                                                                         signature.user_inputs))),
        tuple(Output(k, node, value, k >= len(returned))
              for k, (node, value) in enumerate(zip(returned + sent, signature.user_outputs))),
        (entry,), (), None)


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
# (torch-inductor-upstream-composition.patch): `source` its definitions before the kernel (the executor's helper as
# printed, never parsed); `parameters` its own kernel parameters and `builtins` its builtin ones; `enter` its
# statements before the body, `grid` and `group` uint3 expressions the body's thread position in its grid and
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
# helper as printed, never parsed), `parameters` its own after the group's arguments, `enter` its statements before the
# body; `typed` those parameters as their Parameter records
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
# metal-microbench metal_transport.mm UpstreamHelper: each function's parameters as Tint's MSL printer types them
# (tint/lang/msl/writer/printer: a storage runtime array `{const }device tint_array<T, 1>*`, host C++ (Options::host_cpp,
# provider/dawn.patch) `{const }tint_array<T, 1>*`; an atomic `atomic_uint` or `atomic_ulong`, u64 `ulong`), `votes` its
# workgroup array of 1024 / kSimd (metal_lower.h kSimd 32) u32
VOTES = 1024 // 32
# TransportViews: each word's (word, kind) its parameter; (1, 3) a later pass's `first` before its `supplied`
WORDS = {(0, 1): "step", (1, 0): "phases", (1, 1): "counts", (1, 2): "status", (2, 0): "flags", (3, 0): "list"}
BUILTINS = ("uint3 mesh_lid [[thread_position_in_threadgroup]]", "uint3 mesh_gid [[threadgroup_position_in_grid]]",
            "uint mesh_lane [[thread_index_in_simdgroup]]", "uint3 mesh_threads [[threads_per_threadgroup]]",
            "uint3 mesh_position [[thread_position_in_grid]]")


# the helper's acquire parameters after its builtins, in its declared order (metal_ir.h metal_ir_upstream_helper)
def _acquired(later):
    return ["step", "phases", "counts", *(["status", "first", "list"] if later else ["flags"]), "classes", "spans",
            "records"]


# native composition `made`'s pass (`made.later`): each transport word, readiness table and record table by its helper
# parameter name, its (slot, element type, written); and its sent values' arrival counters in `sent` order
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
        named["spans"] = (r.later_span_slot if later else r.span_slot, "uint", False)
    if made.table is not None:
        named["records"] = (made.table.later_slot if later else made.table.slot, "ulong", False)
    return named, arrivals


def _acquire(made):
    helper = made.helper
    return None if helper is None else helper.later if made.later else helper.acquire


# a Metal kernel's pass: each view a parameter at its slot as the helper's parameters type it; the helper's
# mesh_acquire_first (pass 0) or mesh_acquire_later (pass 1) first, its threadgroup returning where it does not run (.w
# 0), the body at the position it gives (pass 0 its own: thread_position_in_grid; pass 1 its listed threadgroup's .xyz,
# whole threadgroups, its further lanes outside the body's bound reaching mesh_release); mesh_release after the body's
# stores, its arrival counters in `sent` order; each paged argument's accessor over its arena
def _threaded(kernel, made):
    helper, later, args, values = made.helper, bool(made.later), kernel.owner.args, kernel.launch[1]
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
    if helper is not None and helper.release is not None:
        exit_.append(f"{helper.release}(" + ", ".join(["mesh_lid", "mesh_threads", "mesh_step", "mesh_status",
                                                       "mesh_supplied", "mesh_records",
                                                       *(f"mesh_{name}" for name in arrivals)]) + ");")
    source = (helper.source if helper is not None else "") + (PAGED if made.pages else "")
    return Composition(source, tuple(parameters), BUILTINS, tuple(enter), grid, "mesh_lid", tuple(exit_),
                       {p.argument: p.slot for p in made.pages})


# a CPU kernel's pass (a native task: its one invocation; its sends published by its Steps, no release): the helper's
# bool mesh_acquire_first or mesh_acquire_later first, its body run where it returns true, each of its parameters the
# kernel's own at its slot as the host C++ helper types it; a paged argument's binding its flat view
def _hosted(kernel, made):
    named, _ = _views(made)
    acquire = _acquire(made)
    used = _acquired(bool(made.later)) if acquire is not None else []
    typed = tuple(Parameter(f"{'' if named[name][2] else 'const '}tint_array<{named[name][1]}, 1>*", named[name][0], 1)
                  for name in used)
    enter = (f"if (!{acquire}(" + ", ".join(f"mesh_{name}" for name in used) + ")) return;",) if used else ()
    return Hosted(made.helper.source if made.helper is not None else "",
                  tuple(f"{p.type} mesh_{name}" for name, p in zip(used, typed)), enter, typed)


# kernel `kernel`'s composition for one pass from its native composition `made` (compiler/mesh_compiler/metal.py
# Composed: metal_ir_upstream_composition's words, later words, pages (each with its argument binding's placed origin),
# readiness and table, metal_ir_upstream_helper's Helper (None: no transport), `later` its pass): a Metal kernel's
# Composition, a CPU kernel's Hosted
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
# (`composition` its composed() result; None: as produced). Metal: the patched MetalKernel.codegen_kernel of the
# retained kernel, its headers expanded (torch/mps/__init__.py compile_shader's _embed_headers). CPU: cpp.py
# define_kernel's final source and declared entry (torch-inductor-upstream-composition.patch defined_kernels), or the
# retained group's patched codegen_group with its composition through define_kernel's defined_source; its parameters
# its call's arguments in cpp_argdefs order (each at its own slot; its buffers, inplace, input and output, the bound
# pointer, its sizevars after them the value) and the composition's; codecache.py CppCodeCache's build options
# (CppTorchDeviceOptions for the cpu, CppPythonBindingsCodeCache: no libtorch, shared) in cpp_builder.py CppBuilder's
# argument form
def emitted(kernel, composition=None):
    owner = kernel.owner
    if kernel.place == "cpu":
        name, _, _, types = kernel.launch
        entry, source = kernel.graph.wrapper_code.defined_kernels[name]
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
    with V.set_graph_handler(kernel.graph), V.set_kernel_handler(owner):
        owner.codegen_kernel(owner.metal_entry, composition)
    source = _embed_headers([line + "\n" for line in owner.metal_source.split("\n")],
                            [Path(torch.__file__).parent / "include"], set())
    return Emission(source, owner.metal_entry, (), (), ())
