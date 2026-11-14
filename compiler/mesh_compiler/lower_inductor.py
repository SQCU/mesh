# design/compiler-archive-assignment-2026-11-13.md; design/compilation-composition-change-request.md items 2 to 7;
# torch 2.14 torch/_inductor/__init__.py aot_compile, torch/_inductor/package/package.py package_aoti;
# output_data/compiler-workgroup-20261113/executor/aoti-binding-contract.md
import dataclasses
import pathlib
import shutil
import tempfile

import torch
from torch._inductor import aot_compile
from torch._inductor.package import package_aoti

THREAD, MODEL = "host", "model"
OPTIONS = {"aot_inductor.package": True, "layout_optimization": False, "reorder_for_peak_memory": False}


# The scheduled body with the caller's immutable constants as buffers of the compiled model; every node keeps its
# original metadata, and the model returns the body's distinct tensor results
def body(module, constants, tensors):
    root, graph, held = torch.nn.Module(), torch.fx.Graph(), {}
    for node in module.graph.nodes:
        if node.op == "placeholder" and node.target in constants:
            name = f"constant{len(held)}"
            root.register_buffer(name, constants[node.target])
            held[node] = graph.get_attr(name)
            held[node].meta.update(node.meta)
        elif node.op == "output":
            graph.output(tuple(held[value] for value in tensors)).meta.update(node.meta)
        else:
            held[node] = graph.node_copy(node, held.__getitem__)
    return torch.fx.GraphModule(root, graph)


# compiler/mesh_compiler/schedule.py assemble: one composed body per (machine, target, device) group, compiled once
def lower(module, *, constants):
    placeholders = [node for node in module.graph.nodes if node.op == "placeholder"]
    given = [node for node in placeholders if node.target not in constants]
    leaves = tuple(next(node for node in module.graph.nodes if node.op == "output").args[0] or ())
    tensors = list(dict.fromkeys(leaf for leaf in leaves if isinstance(leaf, torch.fx.Node)))
    results = tuple(("output", tensors.index(leaf)) if isinstance(leaf, torch.fx.Node) else ("constant", leaf)
                    for leaf in leaves)
    calls = [node for node in module.graph.nodes if node.op == "call_function"]
    directory = tempfile.TemporaryDirectory(prefix="mesh-aoti-")
    package = package_aoti(str(pathlib.Path(directory.name) / "package.pt2"), {
        MODEL: aot_compile(body(module, constants, tensors), tuple(node.meta["val"] for node in given),
                           options=OPTIONS)})
    record = {"tasks": [(THREAD, tuple(node.name for node in calls))], "dependences": [], "operations": []}
    sources = {node.name: tuple(task) for node, task in module.meta["tasks"].items()}
    return Target(package, tuple(placeholders.index(node) for node in given), results,
                  tuple(module.meta["members"]), sources, record, directory)


# design/compiler-native-assignment-2026-11-13.md: one native Model of the whole body; its call takes the arguments
# in scheduled.inputs order and returns the body's output tuple
@dataclasses.dataclass
class Target:
    package: str
    arguments: tuple
    results: tuple
    members: tuple
    sources: dict
    record: dict
    directory: object = None

    def __post_init__(self):
        from mesh_aoti import Model
        self.model = Model(self.package, MODEL, len(self.arguments),
                           len({value for kind, value in self.results if kind == "output"}))

    def __call__(self, *arguments):
        made = self.model(*(arguments[position] for position in self.arguments))
        return tuple(made[value] if kind == "output" else value for kind, value in self.results)

    def saved(self, directory, group):
        name = f"{group}.aoti.pt2"
        shutil.copyfile(self.package, pathlib.Path(directory) / name)
        return {"package": name, "arguments": self.arguments, "results": self.results, "members": self.members,
                "sources": self.sources, "record": self.record}


def restore(saved, directory):
    return Target(str(pathlib.Path(directory) / saved["package"]), tuple(saved["arguments"]),
                  tuple(map(tuple, saved["results"])), tuple(saved["members"]),
                  {name: tuple(task) for name, task in saved["sources"].items()}, saved["record"])
