# design/data-flow.md
import itertools
import operator

import torch
from torch._export.passes.collect_tracepoints_pass import CollectTracepointsPass
from torch._export.wrappers import _wrap_submodules
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
from torch.export._trace import (_export_to_torch_ir, _get_param_buffer_mapping, _lstsq_fixed_selected,
                                 _normalize_nn_module_stack)

from .ir import CapturedProgram, Exported
from .operations import LSTSQ_FIXED

# pub13 section 5.1 case (i): an eigh outlined on a non-CPU engine (compiler/mesh_compiler/decompose.py outline) draws
# its Haar restarts at its own source site, carried to its QDWH steps as data (operations.QDWH_EIGH_STEP `site`); a
# pinv, svd or lstsq (each overload, and compiler/mesh_compiler/operations.py LSTSQ_FIXED, the fixed-result lstsq) is a
# draw source too, its expanded svd and eigh keeping its declared site
DRAWING = (torch.ops.aten._linalg_eigh, torch.ops.aten.linalg_pinv, torch.ops.aten._linalg_svd,
           torch.ops.aten.linalg_lstsq, LSTSQ_FIXED.overloadpacket)


# /Users/mdot/schedule-emit/schedule_emit/export.py export and driver.py 14–21;
# design/compilation-composition-change-request.md "Numerical lowerer boundary"; torch 2.14.0 torch/export/_trace.py
# _strict_export (_export_to_torch_ir, restore_fqn False, preserve_module_call_signature; _lstsq_fixed_selected of
# compiler/patches/torch-lstsq-fixed-results.patch; _normalize_nn_module_stack 1873; _get_param_buffer_mapping),
# _non_strict_export 2341, _export_to_aten_ir 1211 and _get_module_call_graph 1633–1679; torch/_export/wrappers.py
# _wrap_submodules, _export_function_call and _export_tracepoint; torch/_dynamo/symbolic_convert.py
# InliningInstructionTranslator.inline_call; torch/_export/passes/collect_tracepoints_pass.py CollectTracepointsPass
# (compiler/patches/torch-export-tracepoint-calls.patch); torch/_functorch/aot_autograd.py
# aot_export_joint_with_descriptors 1320–1455 (compiler/patches/torch-aot-joint-params-buffers.patch)
def export(stack, module, args, loss=None, kwargs=None, strict=True, functions=()):
    preserved = tuple(functions)
    if strict:
        source = _export_to_torch_ir(module, args, kwargs, preserve_module_call_signature=preserved, restore_fqn=False,
                                     _log_export_usage=False)
        _lstsq_fixed_selected(source)
        _normalize_nn_module_stack(source, type(module))
        specs = source.meta["module_call_specs"]
    else:
        source, specs = module, {}
    with _wrap_submodules(module, () if strict else preserved, specs):
        joint = aot_export_joint_with_descriptors(stack, source, args, kwargs,
                                                  decompositions=torch.export.default_decompositions(),
                                                  _record_nn_module_stack=True)
    collected = CollectTracepointsPass()
    if preserved:
        collected(joint.graph_module)
        for graph_module in joint.graph_module.modules():
            if isinstance(graph_module, torch.fx.GraphModule):
                graph_module.recompile()
    return Exported(joint, source, _get_param_buffer_mapping(module, source) if strict else None, module, args, kwargs,
                    loss, functions, specs, collected)


# compiler/patches/torch-export-alias-metadata.patch (torch/_functorch/_aot_autograd/input_output_analysis.py
# create_graph_signature `aot_aliases`, carried by torch/export/experimental/__init__.py
# _copy_graph_module_and_signature)
def aot_aliases(program):
    return program.graph_module.meta["aot_aliases"]


# design/aot-compilation-change-request.md A1 and A10; compiler/mesh_compiler/ir.py CapturedProgram
def capture(exported):
    # torch 2.14.0 torchgen/model.py Tag nondeterministic_seeded (OpOverload.tags); J. K. Salmon et al., SC 2011, §3.1
    sites = itertools.count(1)
    for node in exported.joint.graph_module.graph.nodes:
        if node.op == "call_function" and (torch.Tag.nondeterministic_seeded in getattr(node.target, "tags", ())
                                           or getattr(node.target, "overloadpacket", None) in DRAWING):
            node.meta["site"] = next(sites)
    return CapturedProgram(exported)


# design/aot-compilation-change-request.md A10; compiler/mesh_compiler/decompose.py outline and _outlined; torch 2.14.0
# torch/fx/graph.py Graph.node_copy and Graph.output; torch/fx/graph_module.py GraphModule (dict root) and _get_attr
def substitute(captured, key, parts):
    program = captured.outlined[-1][1] if captured.outlined else captured.exported.joint.graph_module
    calls = [n for n in program.graph.nodes if n.op == "call_module" and n.meta.get("function") == key]
    graph, env, names, owned = torch.fx.Graph(), {}, {}, {}

    def called(inputs, replaced):
        made, produced = [], {}
        for function, part, sources, results in parts:
            if id(part) not in names:
                names[id(part)] = f"outlined_{len(captured.outlined)}_{len(names)}"
                owned[names[id(part)]] = part
            call = graph.call_module(names[id(part)], tuple(
                made[source[1]][source[2]] if source[0] == "part" else inputs[source[1]] for source in sources))
            call.meta["function"] = function
            outputs = []
            for o, value in enumerate(part.graph.output_node().args[0]):
                leaf = graph.call_function(operator.getitem, (call, o))
                leaf.meta["val"] = (replaced[results[o]] if o in results
                                    else value.meta["val"] if isinstance(value, torch.fx.Node) else value)
                outputs.append(leaf)
            call.meta["val"] = tuple(leaf.meta["val"] for leaf in outputs)
            produced.update((k, outputs[o]) for o, k in results.items())
            made.append(outputs)
        return produced
    if calls:
        for node in program.graph.nodes:
            if node in calls:
                users = {user.args[1]: user for user in node.users}
                produced = called([env[n] if isinstance(n, torch.fx.Node) else n for n in node.args],
                                  {k: user.meta["val"] for k, user in users.items()})
                env.update((users[k], leaf) for k, leaf in produced.items())
            elif node not in env:
                env[node] = graph.node_copy(node, lambda n: env[n])
    else:
        placeholders, output = [n for n in program.graph.nodes if n.op == "placeholder"], program.graph.output_node()
        for node in placeholders:
            env[node] = graph.node_copy(node)
        values = list(output.args[0])
        produced = called([env[node] for node in placeholders],
                          {k: v.meta["val"] if isinstance(v, torch.fx.Node) else v for k, v in enumerate(values)})
        graph.output(tuple(produced[k] for k in range(len(values)))).meta.update(output.meta)
    module = torch.fx.GraphModule({**{n.target: torch.fx.graph_module._get_attr(program, n.target) for n in graph.nodes
                                      if n.op in ("get_attr", "call_module") and n.target not in owned}, **owned}, graph)
    return CapturedProgram(captured.exported, (*captured.outlined, (key, module, tuple(parts))))
