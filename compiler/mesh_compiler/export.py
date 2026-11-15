# design/data-flow.md
import functools
import itertools
import math
import operator

import torch
from torch._export.passes.collect_tracepoints_pass import CollectTracepointsPass
from torch._export.utils import node_inline_
from torch._export.wrappers import _wrap_submodules
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
from torch._guards import detect_fake_mode
from torch._higher_order_ops.map import map_dense
from torch._higher_order_ops.scan import generic_scan
from torch._subclasses.fake_tensor import FakeTensorConverter
from torch.export._trace import (_export_to_torch_ir, _get_param_buffer_mapping, _lstsq_fixed_selected,
                                 _make_module_call_signatures)
from torch.export.graph_signature import OutputKind
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree

from .ir import Call, CapturedProgram, Exported, TrainingResult
from .operations import LSTSQ_FIXED

COPY = torch.ops.aten.copy.default
# pub13 section 5.1 case (i): an eigh outlined on a non-CPU engine (compiler/mesh_compiler/decompose.py outline) draws
# its Haar restarts at its own source site, carried to its QDWH steps as data (operations.QDWH_EIGH_STEP `site`); a
# pinv, svd or lstsq (each overload, and compiler/mesh_compiler/operations.py LSTSQ_FIXED, the fixed-result lstsq) is a
# draw source too, its expanded svd and eigh keeping its declared site
DRAWING = (torch.ops.aten._linalg_eigh, torch.ops.aten.linalg_pinv, torch.ops.aten._linalg_svd,
           torch.ops.aten.linalg_lstsq, LSTSQ_FIXED.overloadpacket)


# torch 2.14.0 torch/_higher_order_ops/scan.py generic_scan: the scan HOP's operands (combine_fn, init, xs,
# additional_inputs) in generic_scan's signature
def scanned(combine_fn, init, xs, additional_inputs):
    return generic_scan(combine_fn, init, xs, additional_inputs=additional_inputs)


# torch 2.14.0 torch/_higher_order_ops/map.py map_dense: each fixed map or scan call's upstream dense owner
STATIC = {torch.ops.higher_order.map_impl: map_dense, torch.ops.higher_order.scan: scanned}


def attribute(graph_module, target):
    return functools.reduce(getattr, target.split("."), graph_module)


# torch 2.14.0 torch/fx/experimental/proxy_tensor.py make_fx and torch/_export/utils.py node_inline_: one fixed map's
# or scan's call expanded once in place, its dense owner traced on the call's own fake tensor operands (each tensor
# Node operand a placeholder, every other operand, a body attribute or a literal, the actual captured one, through
# pytree), inlined over the call's operand Nodes (its getitems the traced results, the traced attributes hoisted), each
# made node keeping the call's source metadata, the call's unused body attribute dropped
def expand(graph_module, node):
    flat, spec = pytree.tree_flatten((node.args, node.kwargs))
    held = [k for k, item in enumerate(flat) if isinstance(item, torch.fx.Node)
            and isinstance(item.meta.get("val"), torch.Tensor)]
    given = [attribute(graph_module, item.target) if isinstance(item, torch.fx.Node) and item.op == "get_attr" else item
             for item in flat]

    def operation(*operands):
        values = list(given)
        for k, operand in zip(held, operands):
            values[k] = operand
        args, kwargs = pytree.tree_unflatten(values, spec)
        return STATIC[node.target](*args, **kwargs)
    operands = [flat[k].meta["val"] for k in held]
    with detect_fake_mode(operands):
        traced = make_fx(operation, tracing_mode="fake")(*operands)
    name = next(f"expanded_{i}" for i in itertools.count() if not hasattr(graph_module, f"expanded_{i}"))
    graph_module.add_submodule(name, traced)
    graph, before = graph_module.graph, set(graph_module.graph.nodes)
    with graph.inserting_before(node):
        call = graph.call_module(name, tuple(flat[k] for k in held))
    node.replace_all_uses_with(call)
    graph.erase_node(node)
    for item in flat:
        if isinstance(item, torch.fx.Node) and item.op == "get_attr" and not item.users:
            graph.erase_node(item)
    node_inline_(call)
    for made in [n for n in graph.nodes if n not in before]:
        for key, value in node.meta.items():
            made.meta.setdefault(key, value)


# /Users/mdot/schedule-emit/schedule_emit/export.py export and driver.py 14–21;
# design/compilation-composition-change-request.md "Numerical lowerer boundary"; torch 2.14.0 torch/export/_trace.py
# _strict_export (_export_to_torch_ir, restore_fqn False, preserve_module_call_signature; _lstsq_fixed_selected of
# compiler/patches/torch-lstsq-fixed-results.patch; _get_param_buffer_mapping), _non_strict_export 2341,
# _export_to_aten_ir 1211 and _get_module_call_graph 1633–1679; torch/_export/wrappers.py _wrap_submodules;
# torch/_export/passes/collect_tracepoints_pass.py CollectTracepointsPass
# (compiler/patches/torch-export-tracepoint-calls.patch); torch/_functorch/aot_autograd.py
# aot_export_joint_with_descriptors 1320–1455 (compiler/patches/torch-aot-joint-params-buffers.patch)
def export(stack, module, args, loss=None, kwargs=None, strict=True, functions=()):
    preserved = tuple(path for path in functions if isinstance(path, str))
    if strict:
        source = _export_to_torch_ir(module, args, kwargs, preserve_module_call_signature=preserved, restore_fqn=False,
                                     _log_export_usage=False)
        _lstsq_fixed_selected(source)
        specs = source.meta["module_call_specs"]
    else:
        source, specs = module, {}
    with _wrap_submodules(module, () if strict else preserved, specs):
        joint = aot_export_joint_with_descriptors(stack, source, args, kwargs,
                                                  decompositions=torch.export.default_decompositions(),
                                                  _record_nn_module_stack=True)
    collected = CollectTracepointsPass(_make_module_call_signatures(specs, strict))
    if preserved:
        collected(joint.graph_module)
        joint.graph_module.recompile()
    return Exported(joint, source, _get_param_buffer_mapping(module, source) if strict else None, module, args, kwargs,
                    loss, functions, specs, collected.specs, collected.replaced)


# compiler/patches/torch-export-alias-metadata.patch (torch/_functorch/_aot_autograd/input_output_analysis.py
# create_graph_signature `aot_aliases`, carried by torch/export/experimental/__init__.py
# _copy_graph_module_and_signature)
def aot_aliases(program):
    return program.graph_module.meta["aot_aliases"]


# torch 2.14.0 torch/export/graph_signature.py ExportGraphSignature.backward_signature, gradients_to_parameters and
# gradients_to_user_inputs; compiler/mesh_compiler/ir.py TrainingResult: a program's result, its forward outputs, or
# with a loss its TrainingResult, each gradient map keyed by its output's target in output order
GRADIENTS = OutputKind.GRADIENT_TO_PARAMETER, OutputKind.GRADIENT_TO_USER_INPUT


def returned(forward, outputs, backward):
    if not backward:
        return forward
    parameters, inputs = ({target: 0 for kind, target in outputs if kind == role} for role in GRADIENTS)
    return pytree.tree_structure(TrainingResult(pytree.tree_unflatten([0] * forward.num_leaves, forward), parameters,
                                                inputs))


# design/aot-compilation-change-request.md A10; compiler/mesh_compiler/export.py export and capture
def calls(stack, program, args, functions, loss=None, kwargs=None, strict=True):
    return capture(export(stack, program, args, loss, kwargs, strict, functions))


# design/aot-compilation-change-request.md A1; compiler/mesh_compiler/export.py export and capture
def called(stack, module, args, loss=None, kwargs=None, strict=True):
    return capture(export(stack, module, args, loss, kwargs, strict))


# design/aot-compilation-change-request.md A1 and A10; compiler/mesh_compiler/ir.py CapturedProgram
def capture(exported):
    # torch 2.14.0 torchgen/model.py Tag nondeterministic_seeded (OpOverload.tags); J. K. Salmon et al., SC 2011, §3.1
    sites = itertools.count(1)
    for node in exported.joint.graph_module.graph.nodes:
        if node.op == "call_function" and (torch.Tag.nondeterministic_seeded in getattr(node.target, "tags", ())
                                           or getattr(node.target, "overloadpacket", None) in DRAWING):
            node.meta["site"] = next(sites)
    return CapturedProgram(exported)


# design/aot-compilation-change-request.md A10; compiler/mesh_compiler/decompose.py outline: every call of function
# `key` replaced, in place in the program's call graph, by the calls of `parts` in order, each (key, Exported, inputs,
# results): an input ("call", i) the original call's input i or ("part", p, o) part p's output o; `results` {output o:
# the original call's output k}, whose outer node (its readers, a mutation's role) becomes part output o, every other
# output a new outer node; the program's other calls, functions and outer graph kept. torch 2.14.0
# torch/_subclasses/fake_tensor.py FakeTensorConverter tensor_memo, set_tensor_memo and from_real_tensor, and
# torch/_subclasses/meta_utils.py MetaConverter set_storage_memo and MetaTensorDescriber get_tensor_id,
# get_storage_id and describe_storage (MetaConverter.describer): each part call's one converter the one tensor and
# storage memo, each of its placeholder prototypes the call's actual argument and its storage the actual storage, at
# a byte origin (actual offset bytes less the prototype's, one origin a storage); every output's value its prototype
# realized: a memoized one's memo, one on a bound storage its as_strided view of the actual storage at its own byte
# offset plus that origin (aligned to its element; the anchor's conjugate and negative bits cleared by their view
# toggles, a different dtype by .view over the storage's bytes aligned to both elements, then the prototype's bits and
# requires_grad on that new view) published to the memo, any other the call's own conversion (its fresh storages and
# their aliases its own)
def substitute(captured, key, parts):
    graph = captured.exported.graph_module.graph
    functions = {k: f for k, f in captured.functions.items() if k != key}
    functions.update((k, exported) for k, exported, _, _ in parts)
    calls = {}
    for node, call in captured.calls.items():
        if call.function != key:
            calls[node] = call
            continue
        leaves = [leaf.meta["val"] for leaf in call.inputs + call.outputs
                  if isinstance(leaf, torch.fx.Node) and isinstance(leaf.meta.get("val"), torch.Tensor)]
        fake, made, at = detect_fake_mode(leaves), [], node.prev
        for k, exported, inputs, results in parts:
            arguments = [call.inputs[source[1]] if source[0] == "call" else made[source[1]][source[2]]
                         for source in inputs]
            with graph.inserting_after(at):
                there = graph.node_copy(node, lambda n: n)
            there.args, there.kwargs = tuple(arguments), {}
            there.meta.update(function=k)
            at, outputs = there, []
            converter = FakeTensorConverter()
            describer, bound = converter.meta_converter.describer, {}
            prototypes = [n.meta.get("val") for n in exported.graph_module.graph.nodes if n.op == "placeholder"]
            for prototype, argument in zip(prototypes, arguments, strict=True):
                actual = argument.meta["val"] if isinstance(argument, torch.fx.Node) else argument
                if not (isinstance(prototype, torch.Tensor) and isinstance(actual, torch.Tensor)):
                    continue
                converter.set_tensor_memo(prototype, actual)
                converter.meta_converter.set_storage_memo(describer.describe_storage(prototype.untyped_storage()),
                                                          actual.untyped_storage())
                origin = actual.storage_offset() * actual.element_size() - \
                    prototype.storage_offset() * prototype.element_size()
                held = describer.get_storage_id(actual.untyped_storage()), origin, actual
                storage = describer.get_storage_id(prototype.untyped_storage())
                # rejection filters are forbidden: a correspondence this owner cannot relocate raises, never overwritten
                if bound.setdefault(storage, held)[:2] != held[:2]:
                    raise ValueError(f"{node.name} part {k}: one prototype input storage declared on two actual "
                                     f"storages or origins {bound[storage][:2]} and {held[:2]}")

            def realized(prototype):
                found = converter.tensor_memo.get(describer.get_tensor_id(prototype))
                if found is not None:
                    return found
                storage = describer.get_storage_id(prototype.untyped_storage())
                if storage not in bound:
                    return converter.from_real_tensor(fake, prototype)
                _, origin, anchor = bound[storage]
                byte, element = prototype.storage_offset() * prototype.element_size() + origin, prototype.element_size()
                if byte % element:
                    raise ValueError(f"{node.name} part {k}: output at byte {byte} of its input storage is not aligned "
                                     f"to its {element}-byte elements")
                with fake:
                    plain = anchor.conj() if anchor.is_conj() else anchor
                    plain = plain._neg_view() if plain.is_neg() else plain
                    if plain.dtype != prototype.dtype:
                        span = plain.untyped_storage().nbytes()
                        span -= span % math.lcm(plain.element_size(), element)
                        plain = plain.as_strided((span // plain.element_size(),), (1,), 0).view(prototype.dtype)
                    view = plain.as_strided(prototype.shape, prototype.stride(), byte // element)
                    view = view.conj() if prototype.is_conj() else view
                    view = view._neg_view() if prototype.is_neg() else view
                    if view.requires_grad and not prototype.requires_grad:
                        view = view.detach()
                    elif prototype.requires_grad and not view.requires_grad:
                        view.requires_grad_(True)
                converter.set_tensor_memo(prototype, view)
                return view
            values = exported.graph_module.graph.output_node().args[0]
            for o, value in enumerate(values):
                if o in results:
                    leaf = call.outputs[results[o]]
                    leaf.args = (there, o)
                    at.append(leaf)
                else:
                    with graph.inserting_after(at):
                        leaf = graph.call_function(operator.getitem, (there, o))
                prototype = value.meta["val"] if isinstance(value, torch.fx.Node) else value
                leaf.meta["val"] = realized(prototype) if isinstance(prototype, torch.Tensor) else prototype
                at = leaf
                outputs.append(leaf)
            calls[there] = Call(k, arguments, outputs)
            made.append(outputs)
        graph.erase_node(node)
    captured.exported.graph_module.recompile()
    return CapturedProgram(captured.exported, functions, calls,
                           (*captured.outlined, (captured.functions[key], tuple(part for _, part, _, _ in parts))))
