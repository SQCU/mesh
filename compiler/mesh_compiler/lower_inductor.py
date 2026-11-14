# design/compiler-archive-assignment-2026-11-13.md; design/compilation-composition-change-request.md items 2 to 7;
# torch 2.14 torch/_inductor/__init__.py aot_compile, torch/_inductor/package/package.py package_aoti;
# output_data/compiler-workgroup-20261113/executor/native.status.md
import dataclasses
import importlib
import pathlib
import shutil
import tempfile

import torch
from torch._guards import detect_fake_mode
from torch._inductor import aot_compile
from torch._inductor.package import package_aoti

from .ir import RECV, SEND, channel_of, mesh_graph

THREAD, MODEL = "host", "model"
OPTIONS = {"aot_inductor.package": True, "layout_optimization": False, "reorder_for_peak_memory": False}


# compiler/mesh_compiler/bridge.py Published.roots, views and distinct: a declared (start, element, pairs) view of a
# value's byte root
def prepared(root, dtype, sizes, record):
    start, element, pairs = record
    return root.view(dtype).as_strided(sizes, [stride // element for _, stride in pairs], start // element)


# The scheduled body with the caller's immutable constants as buffers of the compiled model, every node keeping its
# original metadata, and each surviving SEND/RECV as fixed native effect sites (rdma/mesh_aoti.py START, WAIT,
# SENT, PUBLISH) on a prepared view of the declared transfer; torch/distributed/fsdp/_fully_shard/_fsdp_param.py
# copy_ stores the sent value into its prepared view. Each emitted site is one task of the record: RECV maps to its
# WAIT and SEND to its PUBLISH for the scheduled dependences contracted through the numerical nodes
# (rdma/mesh_graph.py contract); SENT precedes PUBLISH, START precedes every site, and PUBLISH of execution e - copies
# precedes SENT of execution e (M. Lam, PLDI 1988, section 2.3). `start` is the plan's retained START object on this
# machine, or None: the invocation input and START exist exactly when it does
def body(module, constants, tensors, effects, start):
    import mesh_aoti as native
    importlib.import_module("torch.distributed.fsdp._fully_shard._fsdp_param")
    root, graph, held, examples, bound, roots = torch.nn.Module(), torch.fx.Graph(), {}, [], [], {}
    tasks, operations, dependences, emitted, views = [], [], [], {}, {}
    nodes = list(module.graph.nodes)
    sites = [node for node in nodes if channel_of(node) is not None]
    carried = lambda node: (node.args[0] if node.target is SEND else node).meta["val"]

    def site(made, primitive, key, ordinal, *sources):
        tasks.append((THREAD, (made.name, *(source.name for source in sources))))
        operations.append((THREAD, primitive, len(tasks) - 1, key, 1, ordinal))
        return len(tasks) - 1

    for node in (n for n in nodes if n.op == "placeholder"):
        if node.target in constants:
            name = f"constant{len(held)}"
            root.register_buffer(name, constants[node.target])
            held[node] = graph.get_attr(name)
            held[node].meta.update(node.meta)
        else:
            held[node] = graph.node_copy(node)
            examples.append(node.meta["val"])
    mode = detect_fake_mode([node.meta.get("val") for node in nodes])
    if start is not None:
        invocation = graph.placeholder("invocation")
        examples.append(mode.from_tensor(native.INVOCATION))
        invocation.meta["val"] = examples[-1]
    for node in sites:
        transfer, value, _, extent, sent, landed = effects[channel_of(node)]
        if value not in roots:
            with mode:
                roots[value] = torch.empty(extent, dtype=torch.uint8, device="cpu")
        sizes = [size for size, _ in sent[2]] if node.target is SEND else tuple(carried(node).shape)
        views[node] = graph.placeholder(f"prepared{len(views)}")
        examples.append(prepared(roots[value], carried(node).dtype, sizes, sent if node.target is SEND else landed))
        views[node].meta["val"] = examples[-1]
        bound.append((transfer, "distinct" if node.target is SEND else "views"))
    if start is not None:
        started = site(graph.call_function(native.START, (invocation,)), "bridge_start", start, 1)
    for node in (n for n in nodes if n.op != "placeholder"):
        if node.target is RECV:
            transfer = effects[channel_of(node)][0]
            emitted[node] = site(graph.call_function(native.WAIT, (invocation, transfer, 0, views[node])),
                                 "host_wait", ("arrival", transfer), 1, node)
            held[node] = views[node]
        elif node.target is SEND:
            transfer, _, copies, _, sent, _ = effects[channel_of(node)]
            view, value = views[node], node.args[0]
            released = site(graph.call_function(native.SENT, (invocation, transfer, copies, view)),
                            "host_wait", ("completion", transfer), 1 - copies, node)
            source = graph.call_function(torch.ops.aten.as_strided.default, (
                held[value], [size for size, _ in sent[2]], list(value.meta["val"].stride())))
            graph.call_function(torch.ops.fsdp.copy_.default, (view, source))
            emitted[node] = site(graph.call_function(native.PUBLISH, (invocation, transfer, view)),
                                 "host_publish", ("publication", transfer), 1, node)
            dependences += [(released, emitted[node], 0), (emitted[node], released, copies)]
        elif node.op == "output":
            graph.output(tuple(held[value] for value in tensors)).meta.update(node.meta)
        else:
            held[node] = graph.node_copy(node, held.__getitem__)
    vertices = list(dict.fromkeys([end for pair in module.meta["dependences"] for end in pair] + sites))
    at = {node: k for k, node in enumerate(vertices)}
    dependences += [(emitted[vertices[a]], emitted[vertices[b]], 0) for a, b in mesh_graph.contract(
        [node in emitted for node in vertices], [(at[a], at[b]) for a, b in module.meta["dependences"]])]
    dependences += [(started, k, 0) for k in range(1, len(tasks))]
    entry = [(THREAD, tuple(node.name for node in nodes if node.op == "call_function"))]
    record = {"tasks": tasks or entry, "dependences": sorted(set(dependences)), "operations": operations}
    return torch.fx.GraphModule(root, graph), examples, tuple(bound), record


# compiler/mesh_compiler/schedule.py assemble: one composed body per (machine, target, device) group, compiled once;
# effects: channel -> (declared transfer, its value, the value's solved copies and byte extent, declared sent view,
# declared landed view)
def lower(module, *, constants, effects, start):
    import mesh_aoti as native
    placeholders = [node for node in module.graph.nodes if node.op == "placeholder"]
    given = [node for node in placeholders if node.target not in constants]
    leaves = tuple(next(node for node in module.graph.nodes if node.op == "output").args[0] or ())
    tensors = list(dict.fromkeys(leaf for leaf in leaves if isinstance(leaf, torch.fx.Node)))
    results = tuple(("output", tensors.index(leaf)) if isinstance(leaf, torch.fx.Node) else ("constant", leaf)
                    for leaf in leaves)
    made, examples, bound, record = body(module, constants, tensors, effects, start)
    directory = tempfile.TemporaryDirectory(prefix="mesh-aoti-")
    package = package_aoti(str(pathlib.Path(directory.name) / "package.pt2"), {MODEL: aot_compile(
        made, tuple(examples), options={**OPTIONS, "aot_inductor.custom_ops_to_c_shims": native.SHIMS})})
    sources = {node.name: tuple(task) for node, task in module.meta["tasks"].items()}
    return Target(package, tuple(placeholders.index(node) for node in given), results, bound,
                  None if start is None else len(given), tuple(module.meta["members"]), sources, record, directory)


# design/compiler-native-assignment-2026-11-13.md: one native Model of the whole body; its call takes the arguments
# in scheduled.inputs order and returns the body's output tuple; `prepared` names, per prepared view input after the
# invocation input, its declared transfer and the bridge.Published views it binds
@dataclasses.dataclass
class Target:
    package: str
    arguments: tuple
    results: tuple
    prepared: tuple
    invocation: int
    members: tuple
    sources: dict
    record: dict
    directory: object = None

    def __post_init__(self):
        from mesh_aoti import Model
        inputs = len(self.arguments) + (self.invocation is not None) + len(self.prepared)
        outputs = len({value for kind, value in self.results if kind == "output"})
        self.model = Model(self.package, MODEL, inputs, outputs)

    def __call__(self, *arguments):
        return self.call(self.model, arguments)

    def call(self, entry, arguments):
        made = entry(*(arguments[position] for position in self.arguments))
        return tuple(made[value] if kind == "output" else value for kind, value in self.results)

    # compiler/mesh_compiler/bridge.py prepare, Published; rdma/mesh_aoti.py Model.prepare
    def bind(self, published):
        return self if self.invocation is None else Bound(self, self.model.prepare(
            list(range(len(self.arguments))), [(self.invocation + 1 + k, getattr(published, kind)[transfer])
                                               for k, (transfer, kind) in enumerate(self.prepared)],
            self.invocation, published.context, published.plan, published.execution), published)

    def saved(self, directory, group):
        name = f"{group}.aoti.pt2"
        shutil.copyfile(self.package, pathlib.Path(directory) / name)
        return {"package": name, "arguments": self.arguments, "results": self.results, "prepared": self.prepared,
                "invocation": self.invocation, "members": self.members, "sources": self.sources, "record": self.record}


class Bound:
    def __init__(self, target, prepared, published):
        self.target, self.prepared, self.published = target, prepared, published

    def __call__(self, *arguments):
        return self.target.call(self.prepared, arguments)


def restore(saved, directory):
    record = {**saved["record"], "operations": [[*operation[:3], tuple(operation[3]) if isinstance(operation[3], list)
                                                 else operation[3], *operation[4:]]
                                                for operation in saved["record"]["operations"]]}
    return Target(str(pathlib.Path(directory) / saved["package"]), tuple(saved["arguments"]),
                  tuple(map(tuple, saved["results"])), tuple(map(tuple, saved["prepared"])), saved["invocation"],
                  tuple(saved["members"]),
                  {name: tuple(task) for name, task in saved["sources"].items()}, record)
