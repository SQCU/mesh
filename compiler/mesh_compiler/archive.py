# design/data-flow.md; design/compiler-archive-assignment-2026-11-13.md, archive of the completed targets
import dataclasses
import json
import pathlib

import torch
from torch.export.graph_signature import InputKind, OutputKind
from torch.utils import _pytree as pytree

from . import native
from .ir import Exported, Plan, Scheduled
from .native import mesh


def tuples(value):
    return tuple(map(tuples, value)) if isinstance(value, list) else value


def save(directory, scheduled):
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    exported = scheduled.exported
    torch.save({name: (key, tensor.detach()) for name, (key, tensor) in exported.state.items()}, directory / "state.pt")
    (directory / "scheduled.json").write_text(json.dumps({
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
        "plan": scheduled.plan and {
            **{field.name: getattr(scheduled.plan, field.name) for field in dataclasses.fields(scheduled.plan)},
            "record": scheduled.plan.record.hex(), "signature": scheduled.plan.signature.hex(),
            "dependences": [[getattr(d, "from"), d.to, d.lag, d.kinds] for d in scheduled.plan.dependences]},
        "targets": {processor: function.saved(directory, processor)
                    for processor, function in scheduled.functions.items()},
        "state_kinds": {name: kind.name for name, kind in exported.state_kinds.items()},
        "tokens": exported.tokens,
        "user_inputs": exported.user_inputs,
        "outputs": [[kind.name, key] for kind, key in exported.outputs],
        "out_spec": pytree.treespec_dumps(exported.out_spec),
    }))
    if scheduled.signature:
        (directory / "signature.record").write_bytes(scheduled.signature)


def load(directory):
    directory = pathlib.Path(directory)
    about = json.loads((directory / "scheduled.json").read_text())
    state = torch.load(directory / "state.pt", map_location="cpu", weights_only=True)
    outputs = [(OutputKind[kind], key) for kind, key in about["outputs"]]
    exported = Exported(None, state, about["tokens"], about["user_inputs"], outputs,
                        pytree.treespec_loads(about["out_spec"]),
                        {name: InputKind[kind] for name, kind in about["state_kinds"].items()})
    devices = {processor: torch.device(name) for processor, name in about["processors"].items()}
    functions = {processor: native.restore(saved, directory) for processor, saved in about["targets"].items()}
    carried = [(channel, sender, receiver, getattr(torch, dtype), shape, stride)
               for channel, sender, receiver, dtype, shape, stride in about["carried"]]
    plan = about["plan"] and Plan(**{**about["plan"], "record": bytes.fromhex(about["plan"]["record"]),
                                     "signature": bytes.fromhex(about["plan"]["signature"]),
                                     "dependences": (mesh.Dependence * len(about["plan"]["dependences"]))(
                                         *[mesh.Dependence(*record) for record in about["plan"]["dependences"]]),
                                     "nodes": tuple(about["plan"]["nodes"]),
                                     "offsets": tuple(map(tuple, about["plan"]["offsets"])),
                                     "publications": tuple(map(tuple, about["plan"]["publications"])),
                                     "places": [(processor, root, tuple(version))
                                                for processor, root, version in about["plan"]["places"]]})
    record = directory / "signature.record"
    return Scheduled(exported, {}, devices, about["machines"], about["output_processor"], about["channels"],
                     about["inputs"], about["feedforward"], carried, about["engines"], plan,
                     signature=record.read_bytes() if record.exists() else None, stored=about["stored"],
                     invariant={processor: tuples(records) for processor, records in about["invariant"].items()},
                     functions=functions)
