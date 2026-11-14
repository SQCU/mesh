# design/data-flow.md; design/aot-compilation-change-request.md A8 to A10 and appendix item 6: the archive of one
# completed machine program (its native target, joint result and recipe), with the captured program's per-function
# records as explicit typed records indexed by function ordinal (the completed target's body order, CallAnalysis.keys);
# a captured function's key, which may be a callable, is compilation-only and is not saved. Scheduled.plan is the one
# target's joint owner's own result and Scheduled.signature that target's complete signature archive
# (compiler/mesh_compiler/native.py Target.plan and Target.signature, saved and restored with the target;
# compiler/mesh_compiler/driver.py compiled), kept in no record of their own here
import inspect
import json
import pathlib

import torch
from torch.export.graph_signature import InputKind, OutputKind
from torch.utils import _pytree as pytree

from . import native
from .ir import Exported, Scheduled


def tuples(value):
    return tuple(map(tuples, value)) if isinstance(value, list) else value


def save(directory, scheduled):
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    exported, (target,) = scheduled.exported, scheduled.functions.values()
    ordinal = {body.key: k for k, body in enumerate(target.bodies)}
    signature, spec = exported.interface
    torch.save({"state": {name: (key, tensor.detach()) for name, (key, tensor) in exported.state.items()},
                "defaults": {name: parameter.default for name, parameter in signature.parameters.items()
                             if parameter.default is not inspect.Parameter.empty},
                "specialized": exported.specialized}, directory / "state.pt")
    (directory / "scheduled.json").write_text(json.dumps({
        "output_processor": scheduled.output_processor,
        "processors": {processor: str(device) for processor, device in scheduled.devices.items()},
        "machines": scheduled.machines,
        "engines": scheduled.engines,
        "inputs": scheduled.inputs,
        "channels": [{"function": ordinal[key], "channels": count} for key, count in scheduled.channels.items()],
        "feedforward": scheduled.feedforward,
        "stored": scheduled.stored,
        "invariant": [{"function": ordinal[key], "records": records} for key, records in scheduled.invariant.items()],
        "carried": [{"function": ordinal[key],
                     "transfers": [[channel, sender, receiver, str(dtype).removeprefix("torch."), shape, stride]
                                   for channel, sender, receiver, dtype, shape, stride in transfers]}
                    for key, transfers in scheduled.carried.items()],
        "targets": {processor: function.saved(directory, processor)
                    for processor, function in scheduled.functions.items()},
        "state_kinds": {name: kind.name for name, kind in exported.state_kinds.items()},
        "tokens": exported.tokens,
        "user_inputs": exported.user_inputs,
        "outputs": [[kind.name, key] for kind, key in exported.outputs],
        "out_spec": pytree.treespec_dumps(exported.out_spec),
        "interface": {"parameters": [[name, parameter.kind.name] for name, parameter in signature.parameters.items()],
                      "spec": pytree.treespec_dumps(spec)},
    }))


def load(directory):
    directory = pathlib.Path(directory)
    about = json.loads((directory / "scheduled.json").read_text())
    payload = torch.load(directory / "state.pt", map_location="cpu", weights_only=True)
    state, defaults, specialized = payload["state"], payload["defaults"], payload["specialized"]
    outputs = [(OutputKind[kind], key) for kind, key in about["outputs"]]
    signature = inspect.Signature([inspect.Parameter(name, getattr(inspect.Parameter, kind),
                                                     default=defaults.get(name, inspect.Parameter.empty))
                                   for name, kind in about["interface"]["parameters"]])
    exported = Exported(None, state, about["tokens"], about["user_inputs"], outputs,
                        pytree.treespec_loads(about["out_spec"]),
                        {name: InputKind[kind] for name, kind in about["state_kinds"].items()},
                        interface=(signature, pytree.treespec_loads(about["interface"]["spec"])),
                        specialized=specialized)
    devices = {processor: torch.device(name) for processor, name in about["processors"].items()}
    functions = {processor: native.restore(saved, directory) for processor, saved in about["targets"].items()}
    (target,) = functions.values()
    carried = {record["function"]: [(channel, sender, receiver, getattr(torch, dtype), shape, stride)
                                            for channel, sender, receiver, dtype, shape, stride in record["transfers"]]
               for record in about["carried"]}
    channels = {record["function"]: record["channels"] for record in about["channels"]}
    return Scheduled(exported, {}, devices, about["machines"], about["output_processor"], channels,
                     about["inputs"], about["feedforward"], carried, about["engines"], target.plan,
                     signature=target.signature, stored=about["stored"],
                     invariant={record["function"]: tuples(record["records"]) for record in about["invariant"]},
                     functions=functions)
