# design/data-flow.md
import copy
import dataclasses
import json
import operator
import pathlib

import torch
from torch.export import ExportedProgram
from torch.export.exported_program import ModuleCallEntry, ModuleCallSignature
from torch.export.graph_signature import (ConstantArgument, ExportGraphSignature, InputKind,
                                          InputSpec, OutputKind, OutputSpec, TensorArgument)
from torch.utils import _pytree as pytree

from torch._export.verifier import TrainingIRVerifier

from .ir import Exported, Plan, Scheduled, mesh_graph

# design/unequal-shards-change-request.md item 12; design/problems-of-14b9a757-change-request.md items 1 to 3, 6 and 8;
# design/aot-compilation-change-request.md A4
META = ("dimension", "block", "counts", "displs", "sizes", "subsizes", "starts", "view", "invariant", "to", "out",
        "collective")


def tuples(value):
    return tuple(map(tuples, value)) if isinstance(value, list) else value


def wrap(module, constants):
    module = copy.deepcopy(module)
    nodes = list(module.graph.nodes)
    placeholders = [n for n in nodes if n.op == "placeholder"]
    inputs = [InputSpec(InputKind.CONSTANT_TENSOR, TensorArgument(name=n.name),
                        constants[n.target][0]) if n.target in constants else
              InputSpec(InputKind.USER_INPUT, TensorArgument(name=n.name), None)
              for n in placeholders]
    if not nodes[-1].args[0]:
        nodes[-1].args = ((None,),)
    results = nodes[-1].args[0]
    outputs = [OutputSpec(OutputKind.USER_OUTPUT,
                          TensorArgument(name=r.name) if isinstance(r, torch.fx.Node)
                          else ConstantArgument(name="", value=r), None) for r in results]
    free = sum(n.target not in constants for n in placeholders)
    in_spec = pytree.tree_flatten((tuple(range(free)), {}))[1]
    out_spec = pytree.tree_flatten(tuple(range(len(results))))[1]
    calls = [ModuleCallEntry("", ModuleCallSignature([], [], in_spec, out_spec))]
    return ExportedProgram(module, module.graph, ExportGraphSignature(inputs, outputs), {}, {},
                           calls, constants={key: tensor for key, tensor in constants.values()},
                           verifiers=[TrainingIRVerifier])


def save(directory, scheduled):
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    exported = scheduled.exported
    about = json.dumps({
        "output_processor": scheduled.output_processor,
        "processors": {processor: str(device) for processor, device in scheduled.devices.items()},
        "machines": scheduled.machines,
        "engines": scheduled.engines,
        "inputs": scheduled.inputs,
        "channels": scheduled.channels,
        "feedforward": scheduled.feedforward,
        "stored": scheduled.stored,
        "invariant": scheduled.invariant,
        "carried": [[channel, sender, receiver, str(dtype).removeprefix("torch."), shape, stride]
                    for channel, sender, receiver, dtype, shape, stride in scheduled.carried],
        "plan": scheduled.plan and {**dataclasses.asdict(scheduled.plan), "record": scheduled.plan.record.hex()},
        "signature": scheduled.signature and bytes(mesh_graph.from_record(scheduled.signature).digest).hex(),
        "meta": {processor: {node.name: {key: node.meta[key] for key in META if key in node.meta}
                             for node in module.graph.nodes if node.meta.keys() & set(META)}
                 for processor, module in scheduled.modules.items()},
        "tokens": exported.tokens,
        "user_inputs": exported.user_inputs,
        "outputs": [[kind.name, key] for kind, key in exported.outputs],
        "out_spec": pytree.treespec_dumps(exported.out_spec),
    })
    for processor, module in scheduled.modules.items():
        constants = {name: (state_key, tensor.detach().cpu())
                     for name, (state_key, tensor) in exported.state.items()
                     if name in scheduled.inputs[processor]}
        torch.export.save(wrap(module, constants), directory / f"{processor}.pt2",
                          extra_files={"scheduled.json": about})
    if scheduled.signature:
        (directory / "signature.record").write_bytes(scheduled.signature)


def restore(program, processor, targets):
    module = program.graph_module
    targets = iter(targets)
    for node in list(module.graph.nodes):
        if node.op == "placeholder":
            node.target = next(targets)
        elif node.target is operator.getitem and not node.users:
            module.graph.erase_node(node)
        else:
            node.meta["processor"] = processor
        if node.op == "output" and node.args[0] == (None,):
            node.args = ((),)
    module.recompile()
    return module


def load(directory):
    modules, state, about = {}, {}, None
    for path in sorted(pathlib.Path(directory).glob("*.pt2")):
        extra = {"scheduled.json": ""}
        program = torch.export.load(path, extra_files=extra)
        about = json.loads(extra["scheduled.json"])
        targets = about["inputs"][path.stem]
        modules[path.stem] = restore(program, path.stem, targets)
        for node in modules[path.stem].graph.nodes:
            node.meta.update((key, tuples(value)) for key, value in about["meta"][path.stem].get(node.name, {}).items())
        for spec, target in zip(program.graph_signature.input_specs, targets, strict=True):
            if spec.kind == InputKind.CONSTANT_TENSOR:
                state[target] = (spec.target, program.constants[spec.target])
    outputs = [(OutputKind[kind], key) for kind, key in about["outputs"]]
    exported = Exported(None, state, about["tokens"], about["user_inputs"], outputs,
                        pytree.treespec_loads(about["out_spec"]))
    devices = {processor: torch.device(name) for processor, name in about["processors"].items()}
    modules = {processor: modules[processor] for processor in devices if processor in modules}
    carried = [(channel, sender, receiver, getattr(torch, dtype), shape, stride)
               for channel, sender, receiver, dtype, shape, stride in about["carried"]]
    plan = about["plan"] and Plan(**{**about["plan"], "record": bytes.fromhex(about["plan"]["record"])})
    record = pathlib.Path(directory) / "signature.record"
    return Scheduled(exported, modules, devices, about["machines"], about["output_processor"], about["channels"],
                     about["inputs"], about["feedforward"], carried, about["engines"], plan,
                     record.read_bytes() if record.exists() else None, about["stored"],
                     {processor: tuples(records) for processor, records in about["invariant"].items()})

