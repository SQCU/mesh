# design/compiler-archive-assignment-2026-11-13.md; design/compilation-composition-change-request.md items 1 and 7;
# torch 2.14 torch/_inductor/__init__.py aot_compile, torch/_inductor/package/package.py package_aoti;
# output_data/compiler-workgroup-20261113/executor/aoti-binding-contract.md
import dataclasses
import pathlib
import shutil
import tempfile

import torch
from torch._inductor import aot_compile
from torch._inductor.package import package_aoti

from .ir import RECV, SEND, channel_of

THREAD = "host"
OPTIONS = {"aot_inductor.package": True, "layout_optimization": False, "reorder_for_peak_memory": False}
RECIPE = ("regions", "program", "released", "arguments", "results", "layouts")
FIELDS = (*RECIPE, "values", "record")


# design/torch-source-lowering-change-request.md item 3: a received value has the strides of the value sent
def received(node, strides):
    value = node.meta["val"]
    return value.as_strided(value.shape, strides[channel_of(node)], value.storage_offset()) if (
        node.target is RECV) else value


def layout(value):
    return str(value.dtype).removeprefix("torch."), tuple(value.shape), tuple(value.stride()), value.storage_offset()


# One region of the scheduled body: its nodes in scheduled order; values of other entries are its inputs and the
# caller's constants are buffers of the compiled model; each keeps its original node's metadata
def body(nodes, given, folded, written, constants, operands):
    root, graph, held = torch.nn.Module(), torch.fx.Graph(), {}
    for k, value in enumerate(given):
        held[value] = graph.placeholder(f"value{k}")
        held[value].meta.update(value.meta, val=operands[value])
    for k, value in enumerate(folded):
        root.register_buffer(f"constant{k}", constants[value.target])
        held[value] = graph.get_attr(f"constant{k}")
        held[value].meta.update(value.meta)
    for node in nodes:
        held[node] = graph.node_copy(node, held.__getitem__)
    graph.output(tuple(held[value] for value in written))
    return torch.fx.GraphModule(root, graph)


# S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, section 1: the processor's
# compiled order; its numerical regions between channel effects each compile to one model of one package
def lower(module, *, constants, received_strides):
    graph = module.graph
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    output = next(node for node in graph.nodes if node.op == "output")
    results = tuple(output.args[0] or ())
    entries, run = [], []
    for node in (n for n in graph.nodes if n.op == "call_function"):
        if channel_of(node) is None:
            run.append(node)
            continue
        entries += [tuple(run)] * bool(run) + [node]
        run = []
    entries += [tuple(run)] * bool(run)
    place = {node: k for k, entry in enumerate(entries)
             for node in (entry if isinstance(entry, tuple) else (entry,))}
    folds = lambda value, k: value.op == "placeholder" and value.target in constants and isinstance(entries[k], tuple)
    used = {}
    for node, k in place.items():
        for value in node.all_input_nodes:
            if place.get(value) != k and not folds(value, k):
                used.setdefault(value, []).append(k)
    kept = [node for node in [*placeholders, *place] if node in used or node in results or node.target is RECV]
    slot = {value: s for s, value in enumerate(kept)}
    last = {value: max(used.get(value, [place.get(value)])) for value in kept if value not in results}
    operands = {value: received(value, received_strides) for value in kept}
    program, regions, files = [], [], {}
    for k, entry in enumerate(entries):
        if isinstance(entry, tuple):
            inside = set(entry)
            read = list(dict.fromkeys(u for node in entry for u in node.all_input_nodes if u not in inside))
            given = [value for value in read if not folds(value, k)]
            written = [node for node in entry if node in slot]
            name = f"region{len(regions)}"
            made = body(entry, given, [value for value in read if folds(value, k)], written, constants, operands)
            files[name] = aot_compile(made, tuple(operands[value] for value in given), options=OPTIONS)
            program.append(("run", len(regions)))
            regions.append((name, tuple(slot[value] for value in given), tuple(slot[value] for value in written)))
        else:
            sent = entry.target is SEND
            program.append(("send" if sent else "recv", channel_of(entry), slot[entry.args[0] if sent else entry]))
    directory = tempfile.TemporaryDirectory(prefix="mesh-aoti-")
    package = package_aoti(str(pathlib.Path(directory.name) / "package.pt2"), files)
    # compiler/mesh_compiler/schedule.py schedule: its solved edges between this body's nodes, by entry
    record = {
        "tasks": [(THREAD, tuple(n.name for n in entry) if isinstance(entry, tuple) else (entry.name,))
                  for entry in entries],
        "operations": [(THREAD, "send" if entry.target is SEND else "recv", k, channel_of(entry), 0, 0)
                       for k, entry in enumerate(entries) if not isinstance(entry, tuple)],
        "dependences": sorted({(place[a], place[b]) for a, b in module.meta["dependences"] if place[a] != place[b]})}
    return Target(package, tuple(regions), tuple(program),
                  tuple(tuple(slot[v] for v in kept if last.get(v) == k) for k in range(len(entries))),
                  tuple(slot.get(node) for node in placeholders),
                  tuple(("slot", slot[r]) if isinstance(r, torch.fx.Node) else ("constant", r) for r in results),
                  tuple(layout(operands[value]) for value in kept), {value.name: slot[value] for value in kept},
                  record, directory)


# output_data/compiler-workgroup-20261113/executor/aoti-binding-contract.md: one bound native model per region
@dataclasses.dataclass
class Target:
    package: str
    regions: tuple
    program: tuple
    released: tuple
    arguments: tuple
    results: tuple
    layouts: tuple
    values: dict
    record: dict
    directory: object = None

    def __post_init__(self):
        from mesh_aoti import Model
        self.models = [Model(self.package, name, len(inputs), len(outputs)) for name, inputs, outputs in self.regions]

    def saved(self, directory, processor):
        name = f"{processor}.aoti.pt2"
        shutil.copyfile(self.package, pathlib.Path(directory) / name)
        return {"package": name, **{field: getattr(self, field) for field in FIELDS}}


def restore(saved, directory):
    sequence = lambda value: tuple(map(sequence, value)) if isinstance(value, list) else value
    return Target(str(pathlib.Path(directory) / saved["package"]), *(sequence(saved[field]) for field in RECIPE),
                  saved["values"], saved["record"])
