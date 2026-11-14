# design/compiler-archive-assignment-2026-11-13.md; design/compilation-composition-change-request.md items 2 to 7;
# torch 2.14 torch/fx/experimental/proxy_tensor.py make_fx, torch/_inductor/decomposition.py select_decomp_table,
# torch/_inductor/compile_fx.py compile_fx_forward and compile_fx_inner, torch/_inductor/package/package.py
# package_aoti; output_data/compiler-workgroup-20261113/executor/native.status.md
import dataclasses
import functools
import itertools
import operator
import pathlib
import shutil
import tempfile

import torch
from torch._guards import TracingContext, detect_fake_mode, tracing
from torch._inductor import config
from torch._inductor.codecache import code_hash
from torch._inductor.compile_fx import compile_fx_forward, compile_fx_inner, create_compiler_config_extra
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.package import package_aoti
from torch._inductor.virtualized import V
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from .ir import RECV, SEND, channel_of, mesh_graph

THREAD, MODEL = "host", "model"
OPTIONS = {"aot_inductor.package": True, "layout_optimization": False, "reorder_for_peak_memory": False,
           "cpp_wrapper": True}
ATEN = torch.ops.aten


# The scheduled body with the caller's immutable constants as buffers of the compiled model, every node keeping its
# original metadata, and each surviving SEND/RECV as fixed native effect sites (rdma/mesh_aoti.py START, WAIT, SENT,
# PUBLISH) on its typed view of the one arena input: copy e % copies of its value, at the offset the plan solved
# (Plan.offsets, a constant table of the model) plus its declared start. Each emitted site is one task of the record:
# RECV maps to its WAIT and SEND to its PUBLISH for the scheduled dependences contracted through the numerical nodes
# (rdma/mesh_graph.py contract); SENT precedes PUBLISH, START precedes every site, and PUBLISH of execution e - copies
# precedes SENT of execution e (M. Lam, PLDI 1988, section 2.3). `start` is the plan's retained START object on this
# machine, or None: the invocation input and START exist exactly when it does
def body(module, constants, tensors, effects, start, extent):
    import mesh_aoti as native
    root, graph, held, examples = torch.nn.Module(), torch.fx.Graph(), {}, []
    tasks, operations, dependences, emitted, views = [], [], [], {}, {}
    nodes = list(module.graph.nodes)
    sites = [node for node in nodes if channel_of(node) is not None]
    carried = lambda node: (node.args[0] if node.target is SEND else node).meta["val"]
    solved = {value: offsets for _, value, offsets, *_ in effects.values()}
    values = list(dict.fromkeys(effects[channel_of(node)][1] for node in sites))
    first = dict(zip(values, itertools.accumulate((len(solved[value]) for value in values), initial=0)))

    def site(made, primitive, key, ordinal, *sources):
        tasks.append((THREAD, (made.name, *(source.name for source in sources))))
        operations.append((THREAD, primitive, len(tasks) - 1, key, 1, ordinal))
        return len(tasks) - 1

    def typed(node):
        _, value, offsets, size, sent, landed = effects[channel_of(node)]
        offset, element, pairs = sent if node.target is SEND else landed
        sizes = [n for n, _ in pairs] if node.target is SEND else list(carried(node).shape)
        residue = graph.call_function(ATEN.remainder.Scalar, (execution, len(offsets)))
        at = graph.call_function(ATEN.add.Tensor, (graph.call_function(
            ATEN._to_copy.default, (residue,), {"dtype": torch.int64}), first[value]))
        byte = graph.call_function(ATEN._local_scalar_dense.default, (
            graph.call_function(ATEN.index.Tensor, (table, [at])),))
        graph.call_function(ATEN.sym_constrain_range_for_size.default, (byte,), {"min": 0, "max": extent - size})
        return graph.call_function(ATEN.as_strided.default, (
            graph.call_function(ATEN.view.dtype, (arena, carried(node).dtype)), sizes,
            [stride // element for _, stride in pairs],
            graph.call_function(operator.floordiv, (graph.call_function(operator.add, (byte, offset)), element))))

    for node in (n for n in nodes if n.op == "placeholder"):
        if node.target in constants:
            name = f"constant{len(held)}"
            root.register_buffer(name, constants[node.target])
            held[node] = graph.get_attr(name)
            held[node].meta.update(node.meta)
        else:
            held[node] = graph.node_copy(node)
            examples.append(node.meta["val"])
    # torch/_export/non_strict_utils.py make_fake_inputs: the export fake mode, when the body carries none of its own
    mode = detect_fake_mode([node.meta.get("val") for node in nodes]) or FakeTensorMode(
        shape_env=ShapeEnv(tracked_fakes=[], trace_asserts=True), allow_non_fake_inputs=True, export=True)
    if start is not None:
        invocation = graph.placeholder("invocation")
        examples.append(mode.from_tensor(native.INVOCATION))
        invocation.meta["val"] = examples[-1]
    if sites:
        arena = graph.placeholder("arena")
        with mode:
            examples.append(torch.empty(extent, dtype=torch.uint8, device="cpu"))
        arena.meta["val"] = examples[-1]
        root.register_buffer("offsets", torch.tensor([o for value in values for o in solved[value]], dtype=torch.int64))
        table = graph.get_attr("offsets")
        execution = graph.call_function(ATEN.select.int, (invocation, 0, 2))
    if start is not None:
        started = site(graph.call_function(native.START, (invocation,)), "bridge_start", start, 1)
    for node in (n for n in nodes if n.op != "placeholder"):
        if node.target is RECV:
            transfer, *_ = effects[channel_of(node)]
            views[node] = typed(node)
            emitted[node] = site(graph.call_function(native.WAIT, (invocation, transfer, 0, views[node])),
                                 "host_wait", ("arrival", transfer), 1, node)
            held[node] = views[node]
        elif node.target is SEND:
            transfer, _, offsets, _, sent, _ = effects[channel_of(node)]
            view, value = typed(node), node.args[0]
            released = site(graph.call_function(native.SENT, (invocation, transfer, len(offsets), view)),
                            "host_wait", ("completion", transfer), 1 - len(offsets), node)
            graph.call_function(ATEN.copy_.default, (view, graph.call_function(ATEN.as_strided.default, (
                held[value], [n for n, _ in sent[2]], list(value.meta["val"].stride())))))
            emitted[node] = site(graph.call_function(native.PUBLISH, (invocation, transfer, view)),
                                 "host_publish", ("publication", transfer), 1, node)
            dependences += [(released, emitted[node], 0), (emitted[node], released, len(offsets))]
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
    return torch.fx.GraphModule(root, graph), examples, bool(sites), record, mode


# compiler/mesh_compiler/schedule.py assemble: one composed body per (machine, target, device) group, compiled once
# through Inductor's own forward compilation (no second AOT autograd pass), so the body's copies and effects stay
# mutations of their inputs; effects: channel -> (declared transfer, its value, the value's solved copy offsets in the
# arena, its byte extent, declared sent view, declared landed view); extent: the arena's bytes (Plan.arena_bytes)
def lower(module, *, constants, effects, start, extent):
    import mesh_aoti as native
    placeholders = [node for node in module.graph.nodes if node.op == "placeholder"]
    given = [node for node in placeholders if node.target not in constants]
    leaves = tuple(next(node for node in module.graph.nodes if node.op == "output").args[0] or ())
    tensors = list(dict.fromkeys(leaf for leaf in leaves if isinstance(leaf, torch.fx.Node)))
    results = tuple(("output", tensors.index(leaf)) if isinstance(leaf, torch.fx.Node) else ("constant", leaf)
                    for leaf in leaves)
    made, examples, wired, record, mode = body(module, constants, tensors, effects, start, extent)
    traced = make_fx(made, decomposition_table=select_decomp_table(), tracing_mode="fake")(*examples)
    options = {**OPTIONS, "aot_inductor.custom_ops_to_c_shims": native.SHIMS,
               "aot_inductor.output_path": code_hash(traced.code)}
    with (torch.no_grad(), config.patch(options), tracing(TracingContext(mode)), V.set_fake_mode(mode),
          V.set_aot_compilation(True)):
        compiled = compile_fx_forward(traced, examples, num_orig_model_outputs=len(tensors),
                                      num_example_inputs=len(examples),
                                      compiler_config_extra=create_compiler_config_extra(traced),
                                      inner_compile=functools.partial(compile_fx_inner, cpp_wrapper=True),
                                      is_inference=True)
    directory = tempfile.TemporaryDirectory(prefix="mesh-aoti-")
    package = package_aoti(str(pathlib.Path(directory.name) / "package.pt2"), {MODEL: compiled.filename})
    sources = {node.name: tuple(task) for node, task in module.meta["tasks"].items()}
    invocation = None if start is None else len(given)
    return Target(package, tuple(placeholders.index(node) for node in given), results, invocation,
                  len(given) + (start is not None) if wired else None, tuple(module.meta["members"]), sources,
                  record, directory)


# design/compiler-native-assignment-2026-11-13.md: one native Model of the whole body; its call takes the arguments
# in scheduled.inputs order and returns the body's output tuple; `invocation` and `arena` are the positions of the
# invocation input and of the one arena input, or None
@dataclasses.dataclass
class Target:
    package: str
    arguments: tuple
    results: tuple
    invocation: int
    arena: int
    members: tuple
    sources: dict
    record: dict
    directory: object = None

    def __post_init__(self):
        from mesh_aoti import Model
        inputs = len(self.arguments) + (self.invocation is not None) + (self.arena is not None)
        outputs = len({value for kind, value in self.results if kind == "output"})
        self.model = Model(self.package, MODEL, inputs, outputs)

    def __call__(self, *arguments):
        return self.call(self.model, arguments)

    def call(self, entry, arguments):
        made = entry(*(arguments[position] for position in self.arguments))
        return tuple(made[value] if kind == "output" else value for kind, value in self.results)

    # compiler/mesh_compiler/bridge.py prepare, Published.arena; rdma/mesh_aoti.py Model.prepare
    def bind(self, published):
        return self if self.invocation is None else Bound(self, self.model.prepare(
            list(range(len(self.arguments))), [(self.arena, [published.arena])] * (self.arena is not None),
            self.invocation, published.context, published.plan, published.execution), published)

    def saved(self, directory, group):
        name = f"{group}.aoti.pt2"
        shutil.copyfile(self.package, pathlib.Path(directory) / name)
        return {"package": name, "arguments": self.arguments, "results": self.results, "invocation": self.invocation,
                "arena": self.arena, "members": self.members, "sources": self.sources, "record": self.record}


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
                  tuple(map(tuple, saved["results"])), saved["invocation"], saved["arena"], tuple(saved["members"]),
                  {name: tuple(task) for name, task in saved["sources"].items()}, record)
