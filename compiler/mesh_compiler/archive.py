# design/data-flow.md
import copy
import json
import operator
import pathlib

import torch
from torch.export import ExportedProgram
from torch.export.exported_program import ModuleCallEntry, ModuleCallSignature
from torch.export.graph_signature import (ConstantArgument, ExportGraphSignature, InputKind,
                                          InputSpec, OutputKind, OutputSpec, TensorArgument)
from torch.utils import _pytree as pytree

from .ir import Exported, Scheduled


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
                           calls, constants={key: tensor for key, tensor in constants.values()})


def save(directory, scheduled):
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    exported = scheduled.exported
    about = json.dumps({
        "output_processor": scheduled.output_processor,
        "processors": {processor: str(device) for processor, device in scheduled.devices.items()},
        "inputs": scheduled.inputs,
        "channels": scheduled.channels,
        "feedforward": scheduled.feedforward,
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
        for spec, target in zip(program.graph_signature.input_specs, targets, strict=True):
            if spec.kind == InputKind.CONSTANT_TENSOR:
                state[target] = (spec.target, program.constants[spec.target])
    outputs = [(OutputKind[kind], key) for kind, key in about["outputs"]]
    exported = Exported(None, state, about["tokens"], about["user_inputs"], outputs,
                        pytree.treespec_loads(about["out_spec"]))
    devices = {processor: torch.device(name) for processor, name in about["processors"].items()}
    modules = {processor: modules[processor] for processor in devices if processor in modules}
    return Scheduled(exported, modules, devices, about["output_processor"], about["channels"],
                     about["inputs"], about["feedforward"])
