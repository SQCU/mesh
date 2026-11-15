# design/compilation-composition-change-request.md, numerical lowerer boundary, items 2 and 5 to 7, S8 and Complete
# replacement; design/aot-compilation-change-request.md A1, A5, A6 and A8 to A10; design/torch-source-lowering-change-
# request.md items 1 to 3, 5 and 6; /Users/mdot/schedule-emit/schedule_emit/lower_inductor.py at f41c64a (adapt,
# recv_strided, lower): each processor's scheduled body given to its target's installed upstream compiler (torch 2.14.0
# torch/_inductor: compile_fx.py's inference sequence, GraphLowering, its Scheduler and the selected backend;
# coremltools 9.0 for the Neural Engine), its result retained as that compiler's own objects. Fields and pairing:
# output_data/compiler-workgroup-20261114/compiler/numerical65-operands.md; upstream hooks:
# compiler/patches/torch-inductor-mps-kernel-operands.patch and torch-inductor-upstream-composition.patch
import ctypes
import dataclasses
import functools
from pathlib import Path

import coremltools as ct
import torch
from torch import Tensor
from torch._higher_order_ops.effects import _EffectType, _register_effectful_op
from torch._inductor import config
from torch._inductor import ir as upstream
from torch._inductor.codegen.cpp import CppScheduling, KernelGroup
from torch._inductor.codegen.mps import MetalKernel, MetalScheduling
from torch._inductor.codegen.simd import DisableReduction, EnableReduction
from torch._inductor.compile_fx import (_recursive_joint_graph_passes, _recursive_post_grad_passes,
                                        _recursive_record_original_output_strides, fake_tensor_prop)
from torch._inductor.cpp_builder import CppTorchDeviceOptions
from torch._inductor.cpu_vec_isa import pick_vec_isa
from torch._inductor.debug import DebugContext
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.fx_passes.post_grad import view_to_reshape
from torch._inductor.graph import GraphLowering, SubgraphLowering
from torch._inductor.lowering import fallback_handler, register_lowering
from torch._inductor.scheduler import Scheduler
from torch._inductor.virtualized import V
from torch.fx import traceback as fx_traceback
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._cpp_embed_headers import _embed_headers

from .ir import RECV, SEND, channel_of

WITH_EFFECTS = torch.ops.higher_order.with_effects
# torch/fx/proxy.py _COPY_META_FIELDS "custom", copied by make_fx under fx/traceback.py preserve_node_meta (a control
# body's through torch/_higher_order_ops/utils.py _maybe_run_with_interpreter), by _inductor/pattern_matcher.py
# _transfer_meta and by torch.export: each lowered node's original body node, its index in Produced.sources
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
# stride, dtype and device (Inductor's fake-result and stride contracts), its storage in GraphLowering's
# never_reuse_buffers (Lam, PLDI 1988, section 2.3: the sender still holds it); torch/_inductor/lowering.py
# with_effects: each ORDERED effectful operation after the previous one (V.graph.additional_star_deps, effectful_ops)
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
_sent = fallback_handler(SEND, add_to_fallback_set=False)


def _ordered(made, count):
    previous = V.graph.effectful_ops.get(_EffectType.ORDERED)
    for operation in V.graph.operations[count:]:
        operation.has_side_effects = lambda: True
        if previous is not None:
            V.graph.additional_star_deps[operation.get_name()].add(previous.get_name())
        previous = operation
    V.graph.effectful_ops[_EffectType.ORDERED] = previous
    return made


@register_lowering(RECV_STRIDED, type_promotion_kind=None)
def _lower_received(*arguments):
    count = len(V.graph.operations)
    made = _received(*arguments)
    V.graph.never_reuse_buffers.add(made.get_name())
    return _ordered(made, count)


@register_lowering(SEND, type_promotion_kind=None)
def _lower_sent(*arguments):
    count = len(V.graph.operations)
    return _ordered(_sent(*arguments), count)


# ---- the retained result (numerical65-operands.md): the upstream compiler's own objects and their correspondence ----

# torch include/c10/metal/error.h struct ErrorMessage, ErrorMessages (error_message_count 30): the bytes of a launch's
# error buffer (its launch record's `error` index; OperationUtils.mm setErrorBufferIndex), a TEMPORARY of that size
class _ErrorMessage(ctypes.Structure):
    _fields_ = [("file", ctypes.c_char * 128), ("func", ctypes.c_char * 128), ("message", ctypes.c_char * 250),
                ("line", ctypes.c_uint)]


class _ErrorMessages(ctypes.Structure):
    _fields_ = [("count", ctypes.c_uint), ("msg", _ErrorMessage * 30)]


ERROR_BYTES = ctypes.sizeof(_ErrorMessages)


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


# codegen/simd.py codegen_node_schedule_with_kernel's second pass for SchedulerNode `node`, after finalize_indexing:
# `variables` its split_and_set_ranges result (each a range-tree entry symbol of the kernel's range_tree_nodes, whose
# expr is its tree root's FloorDiv or ModularIndexing), `reduction` the kernel's inside_reduction there (False inside
# DisableReduction), `indexing` its LoopBody's index expressions over them (loop_body.py indexing_from_args, the
# expressions LoopBody.__call__ binds; indirect symbols node._body.indirect_var_ranges')
@dataclasses.dataclass(eq=False)
class Ranges:
    node: object
    variables: tuple
    reduction: bool
    indexing: dict


# one upstream-generated kernel of `graph` (its GraphLowering, a control body's own): `place` "mps" or "cpu",
# `scheduled` the Scheduler nodes it implements in order (their read_writes the Scheduler's own accesses), `owner` the
# backend's kernel (MeshMetalKernel: range trees, body, KernelArgs, workspace_args, metal_entry, metal_source;
# MeshKernelGroup: its KernelArgs and CppKernelProxy loop nests), `launch` its call record as produced (mps:
# torch-inductor-mps-kernel-operands.patch (name, values, types, grid, group, error); cpu: (kernel name, arg_defs,
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
# ExternKernelOut, RandomSeeds, MultiOutput, mesh send and receive ...), `origins` the original body nodes, `bodies` a
# control extern's (Conditional, WhileLoop, InvokeSubgraph: ir.py get_subgraphs) each subgraph's name and its own
# entries, lowered and scheduled by its own GraphLowering (subgraph.graph)
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
# `sources` its nodes and its control bodies' (SOURCE), `lowered` the graph the upstream compiler lowered (the fx graph
# after the canonical decomposition, joint and post-grad passes; Neural Engine: its ExportedProgram), `graph` its
# GraphLowering (buffers, layouts and aliases, graph_inputs_original, graph_input_storage_offsets, mutated_inputs,
# removed_buffers, scheduler, wrapper_code; None for the Neural Engine), `entries` Kernel and Extern in its Scheduler's
# emission order (a Neural Engine body's one Model), `wrapper` GraphLowering.codegen's result
@dataclasses.dataclass(eq=False)
class Produced:
    processor: str
    kind: str
    device: object
    module: object
    sources: tuple
    lowered: object
    graph: object
    inputs: tuple
    outputs: tuple
    entries: tuple
    seeds: tuple
    wrapper: object


# ---- the selected upstream backends, retaining their own results ----

# codegen/mps.py MetalKernel: its second-pass ranges (split_and_set_ranges after finalize_indexing)
class MeshMetalKernel(MetalKernel):
    def __init__(self, tiling, **kwargs):
        super().__init__(tiling, **kwargs)
        self.mesh_indexed, self.mesh_ranges, self.mesh_schedule = False, [], []

    def finalize_indexing(self, indices):
        super().finalize_indexing(indices)
        self.mesh_indexed = True

    def split_and_set_ranges(self, lengths):
        made = super().split_and_set_ranges(lengths)
        if self.mesh_indexed:
            self.mesh_ranges.append((tuple(map(tuple, made)), self.inside_reduction))
        return made


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


# schedule-emit lower_inductor.py adapt over that copy: with_effects as its operator, each RECV as recv_strided at its
# node's value, every SEND's value also returned. Returns (copy, the body's and its control bodies' nodes by SOURCE,
# the sent values' original nodes)
def adapted(module):
    sources = []
    graph, returned, bodies = _copied(module, sources)
    sent = []
    for node in list(graph.nodes):
        if node.target is WITH_EFFECTS:
            token, operator, *arguments = node.args
            with graph.inserting_before(node):
                call = graph.call_function(operator, tuple(arguments), dict(node.kwargs))
            call.meta = {**node.meta}
            for getitem in list(node.users):
                getitem.replace_all_uses_with(token if getitem.args[1] == 0 else call)
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


# torch/_inductor/decomposition.py select_decomp_table under make_fx, each traced node carrying its original's meta
# (fx/interpreter.py run_node under preserve_node_meta)
def traced(module, inputs):
    with fx_traceback.preserve_node_meta():
        return make_fx(torch.fx.Interpreter(module).run, decomposition_table=select_decomp_table(),
                       tracing_mode="fake")(*inputs)


def _source(node, sources):
    k = node.meta.get("custom", {}).get(SOURCE)
    return None if k is None else sources[k]


def _origins(nodes, sources):
    return tuple(dict.fromkeys(s for n in nodes if (s := _source(n, sources)) is not None))


def _node_origins(scheduled, sources):
    return _origins([o for node in scheduled for member in node.get_nodes() if member.node is not None
                     for o in member.node.get_origins()], sources)


# a SIMD kernel's launch record as its Kernel, its Ranges in its second pass's node order
def _simd(graph, scheduled, kernel, launch, sources):
    ranges = tuple(Ranges(node, variables, reduction, node._body.indexing_from_args(variables))
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


# compile_fx.py's inference sequence over one adapted body (fw_compiler_base: original output strides, joint graph
# passes; _InProcessFxCompile.codegen_and_compile: view_to_reshape, fake_tensor_prop, post-grad passes, GraphLowering run
# and codegen), its Scheduler with the selected backends (SELECTED)
def _inductor(module, body, sources, sent, processor, kind, device):
    inputs = [n.meta["val"] for n in _placeholders(body.graph)]
    received = any(n.target is RECV_STRIDED for n in body.graph.nodes)
    with DebugContext(), config.patch(assume_unaligned_fallback_output=received):
        lowered = traced(body, inputs)
        _recursive_record_original_output_strides(lowered)
        lowered = _recursive_joint_graph_passes(lowered, input_device=device)
        view_to_reshape(lowered)
        with torch.no_grad():
            fake = fake_tensor_prop(lowered, inputs)
        _recursive_record_original_output_strides(lowered)
        with V.set_fake_mode(fake):
            _recursive_post_grad_passes(lowered, is_inference=True)
            graph = MeshLowering(lowered, selected=SELECTED, example_inputs=inputs, shape_env=fake.shape_env,
                                 is_inference=True)
            graph.freeze_runtime_asserts()
            with V.set_graph_handler(graph), V.set_extern_kernel_nodes([]):
                graph.run(*inputs)
                wrapper = graph.codegen()
                names = [n.target for n in _placeholders(lowered.graph)]
                returned = _returned(module)
                return Produced(
                    processor, kind, device, module, sources, lowered, graph,
                    tuple(Input(k, node, name) for k, (node, name) in enumerate(zip(_placeholders(module.graph),
                                                                                     names))),
                    tuple(Output(k, node, value, k >= len(returned))
                          for k, (node, value) in enumerate(zip(returned + sent, graph.graph_outputs))),
                    _entries(graph, sources), _seeds(graph, lowered, sources), wrapper)


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
        processor, kind, device, module, sources, program, None,
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
# engine (ENGINE: the Neural Engine, whose fake values share the host's device; driver.py). Returns {function key:
# {processor: Produced}}
def lower(scheduled):
    devices, engines = scheduled.devices, scheduled.engines
    return {key: {processor: produced(module, processor, engines.get(processor, devices[processor].type),
                                      devices[processor])
                  for processor, module in bodies.items()}
            for key, bodies in scheduled.processor_modules.items()}


# ---- terminal emission ----

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


# metal_ir.h metal_ir_value_paged: byte b of a paged root at byte frames[b / page] page + b mod page of its arena (the
# arena bound from its first byte, its frame table after the arguments: metal_lower.mm), element i of an argument at
# byte origin + i sizeof(T) of its root; `P` the argument's own declared pointer type, every load and store its subscript
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
# (tint/lang/msl/writer/printer: a storage runtime array `{const }device tint_array<T, 1>*`, an atomic `atomic_uint` or
# `atomic_ulong`, u64 `ulong`), `votes` its workgroup array of 1024 / kSimd (metal_lower.h kSimd 32) u32
VOTES = 1024 // 32
# TransportViews: each word's (word, kind) its parameter; (1, 3) a later pass's `first` before its `supplied`
WORDS = {(0, 1): "step", (1, 0): "phases", (1, 1): "counts", (1, 2): "status", (2, 0): "flags", (3, 0): "list"}
BUILTINS = ("uint3 mesh_lid [[thread_position_in_threadgroup]]", "uint3 mesh_gid [[threadgroup_position_in_grid]]",
            "uint mesh_lane [[thread_index_in_simdgroup]]", "uint3 mesh_threads [[threads_per_threadgroup]]",
            "uint3 mesh_position [[thread_position_in_grid]]")


def _view(name, slot, element, write):
    return f"{'device' if write else 'const device'} tint_array<{element}, 1>* mesh_{name} [[buffer({slot})]]"


# kernel `kernel`'s Composition for one pass from its native composition `made` (compiler/mesh_compiler/metal.py
# Composed: metal_ir_upstream_composition's words, later words, pages (with each argument binding's origin), readiness
# and table, metal_ir_upstream_helper's Helper (None: no transport), `later` its pass): each word, readiness table and
# record table a parameter at its slot as the helper's parameters type it; the helper's mesh_acquire_first (pass 0) or
# mesh_acquire_later (pass 1) first, its threadgroup returning where it does not run (.w 0), the body at the position it
# gives (pass 0 its own: thread_position_in_grid; pass 1 its listed threadgroup's .xyz, whole threadgroups); mesh_release
# after the body's stores, its arrival counters in `sent` order; each paged argument's accessor over its arena
def composed(kernel, made):
    helper, later, args = made.helper, bool(made.later), kernel.owner.args
    values = kernel.launch[1]
    parameters, enter, exit_, arrivals = [], [], [], []
    firsts = ["first", "supplied"] if later else ["supplied"]
    for w in made.later_words if later else made.words:
        if w.word >= 4:
            name = f"arrivals{w.word - 4}"
            arrivals.append(f"mesh_{name}")
        else:
            name = WORDS[w.word, w.kind] if (w.word, w.kind) in WORDS else firsts.pop(0)
        element = "ulong" if w.kind % 2 else "uint"
        parameters.append(_view(name, w.slot, f"atomic_{element}" if w.kind >= 2 else element, w.write))
    if made.readiness is not None:
        r = made.readiness
        parameters += [_view("classes", r.later_class_slot if later else r.class_slot, "uint", False),
                       _view("spans", r.later_span_slot if later else r.span_slot, "uint", False)]
    if made.table is not None:
        parameters.append(_view("records", made.table.later_slot if later else made.table.slot, "ulong", False))
    for slot in dict.fromkeys(p.frames for p in made.pages):
        parameters.append(f"const device uint* mesh_frames{slot} [[buffer({slot})]]")
    for p in made.pages:
        outer = values[p.argument]
        inner = args.output_buffers[outer] if outer in args.output_buffers else args.input_buffers[outer]
        enter.append(f"const mesh_paged<decltype({inner}_arena)> {inner}{{{inner}_arena, mesh_frames{p.frames}, "
                     f"{p.origin}ul, {p.page_bytes}ul}};")
    grid = "mesh_position"
    acquire = (helper.later if later else helper.acquire) if helper is not None else None
    if acquire is not None:
        common = ["mesh_lid", "mesh_gid", "mesh_lane", "mesh_threads", "mesh_step", "mesh_phases", "mesh_counts"]
        own = ["mesh_status", "mesh_first", "mesh_list"] if later else ["mesh_flags"]
        enter += [f"threadgroup tint_array<uint, {VOTES}> mesh_votes;",
                  f"const uint4 mesh_acquired = {acquire}("
                  + ", ".join(common + own + ["mesh_classes", "mesh_spans", "mesh_records", "&mesh_votes"]) + ");",
                  "if (mesh_acquired.w == 0) return;"]
        if later:
            grid = "(mesh_acquired.xyz * mesh_threads + mesh_lid)"
    if helper is not None and helper.release is not None:
        exit_.append(f"{helper.release}(" + ", ".join(["mesh_lid", "mesh_threads", "mesh_step", "mesh_status",
                                                       "mesh_supplied", "mesh_records", *arrivals]) + ");")
    source = (helper.source if helper is not None else "") + (PAGED if made.pages else "")
    return Composition(source, tuple(parameters), BUILTINS, tuple(enter), grid, "mesh_lid", tuple(exit_),
                       {p.argument: p.slot for p in made.pages})


# metal_ir.h struct metal_ir_upstream_emission
@dataclasses.dataclass(frozen=True)
class Emission:
    source: str
    entry: str
    types: tuple
    compile: tuple
    link: tuple


# metal_ir.h metal_ir_upstream_emitted: the kernel's terminal emission by its own upstream emitter. Metal: the patched
# MetalKernel.codegen_kernel of the retained kernel with `composition` (None: as produced), its headers expanded
# (torch/mps/__init__.py compile_shader's _embed_headers). CPU: cpp.py define_kernel's final source and declared entry
# (torch-inductor-upstream-composition.patch defined_kernels), its call's C types, codecache.py CppCodeCache's build
# options (CppTorchDeviceOptions for the cpu, CppPythonBindingsCodeCache: no libtorch, shared) in cpp_builder.py
# CppBuilder's argument form; its pointers each its value's flat bytes
def emitted(kernel, composition=None):
    owner = kernel.owner
    if kernel.place == "cpu":
        entry, source = kernel.graph.wrapper_code.defined_kernels[kernel.launch[0]]
        options = CppTorchDeviceOptions(vec_isa=pick_vec_isa(), include_pytorch=False, device_type="cpu", shared=True)
        compile_ = (options.get_compiler(), *[f for d in options.get_definitions() for f in ("-D", d)],
                    *[f"-I{d}" for d in options.get_include_dirs()], *[f"-{f}" for f in options.get_cflags()],
                    *options.get_passthrough_args())
        link = (*[f"-{f}" for f in options.get_ldflags()], *[f"-L{d}" for d in options.get_libraries_dirs()],
                *[f"-l{lib}" for lib in options.get_libraries()])
        return Emission(source, entry, kernel.launch[3], compile_, link)
    with V.set_graph_handler(kernel.graph), V.set_kernel_handler(owner):
        owner.codegen_kernel(owner.metal_entry, composition)
    source = _embed_headers([line + "\n" for line in owner.metal_source.split("\n")],
                            [Path(torch.__file__).parent / "include"], set())
    return Emission(source, owner.metal_entry, (), (), ())
