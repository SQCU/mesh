import operator

import torch

from .ir import CapturedProgram


# compiler/patches/torch-export-alias-metadata.patch (torch/_functorch/_aot_autograd/input_output_analysis.py
# create_graph_signature `aot_aliases`, carried by torch/export/experimental/__init__.py
# _copy_graph_module_and_signature)
def aot_aliases(program):
    return program.graph_module.meta["aot_aliases"]


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
