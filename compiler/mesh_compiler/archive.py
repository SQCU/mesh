# design/data-flow.md; design/compiler-archive-assignment-2026-11-13.md, archive of the completed targets
import dataclasses
import json
import pathlib

import torch
from torch.export.graph_signature import OutputKind
from torch.utils import _pytree as pytree

from . import lower_inductor
from .ir import Exported, Plan, Scheduled

# compiler/mesh_compiler/driver.py TARGETS: the same selection, by the processor's engine or device type
RESTORED = {"cpu": lower_inductor.restore}


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
        "plan": scheduled.plan and {**dataclasses.asdict(scheduled.plan), "record": scheduled.plan.record.hex(),
                                    "signature": scheduled.plan.signature.hex()},
        "targets": {processor: function.saved(directory, processor)
                    for processor, function in scheduled.functions.items()},
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
                        pytree.treespec_loads(about["out_spec"]))
    devices = {processor: torch.device(name) for processor, name in about["processors"].items()}
    functions = {processor: RESTORED[about["engines"].get(processor, devices[processor].type)](
                     saved, directory) for processor, saved in about["targets"].items()}
    carried = [(channel, sender, receiver, getattr(torch, dtype), shape, stride)
               for channel, sender, receiver, dtype, shape, stride in about["carried"]]
    plan = about["plan"] and Plan(**{**about["plan"], "record": bytes.fromhex(about["plan"]["record"]),
                                     "signature": bytes.fromhex(about["plan"]["signature"]),
                                     "nodes": tuple(about["plan"]["nodes"]),
                                     "offsets": tuple(map(tuple, about["plan"]["offsets"])),
                                     "publications": tuple(map(tuple, about["plan"]["publications"]))})
    record = directory / "signature.record"
    return Scheduled(exported, {}, devices, about["machines"], about["output_processor"], about["channels"],
                     about["inputs"], about["feedforward"], carried, about["engines"], plan,
                     signature=record.read_bytes() if record.exists() else None, stored=about["stored"],
                     invariant={processor: tuples(records) for processor, records in about["invariant"].items()},
                     functions=functions)
