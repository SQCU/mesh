# design/the-compiler.md; design/compiler-archive-assignment-2026-11-13.md
import dataclasses
import pathlib
import sys

import torch
from torch.export.graph_signature import OutputKind

from . import lower_inductor
from .devices import ENGINES, HOST, on_devices
from .export import export
from .ir import Scheduled
from .placement import place
from .schedule import argument_buffers, carried, schedule, unit_cost
from .signature import signature

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh
import mesh_graph

# design/compilation-composition-change-request.md items 1 and 7: one target lowerer finishes each scheduled body
TARGETS = {"cpu": lambda module, constants, effects, start, extent: lower_inductor.lower(
    module, constants=constants, effects=effects, start=start, extent=extent)}


# rdma/mesh-graph.h struct mesh_graph_signature, mesh_graph_archive_read; rdma/mesh-collective.c plan_signature: the
# BRIDGE_START object the plan's own signature retains on this machine (its index in Plan.nodes), or None
def started(plan):
    made = mesh_graph.from_archive(plan.signature)
    try:
        here = plan.nodes.index(plan.machine)
        return next((operation.object for operation in made.operation[:made.operations]
                     if operation.primitive == mesh_graph.PRIMITIVES.index("bridge_start")
                     and made.processor[operation.processor].index % mesh_graph.NODES == here), None)
    finally:
        mesh_graph.free(made)


# compiler/mesh_compiler/ir.py Planning; compiler/mesh_compiler/bridge.py planning;
# compiler/mesh_compiler/placement.py place, transfers
def compile_module(module, args, processors, placement, cost=unit_cost, stored=(), planning=None, **export_options):
    engines = {name: d for name, (_, d) in processors.items() if d in ENGINES}
    processors = {name: (machine, HOST if name in engines else torch.device(d))
                  for name, (machine, d) in processors.items()}
    exported = export(module, args, **export_options)
    placed, output_processor, channels = place(exported.graph_module, placement)
    devices = {name: device for name, (_, device) in processors.items()}
    machines = {name: machine for name, (machine, _) in processors.items()}
    represented = {}
    groups = {name: represented.setdefault((machines[name], engines.get(name, device.type), device), name)
              for name, device in devices.items()}
    graph = on_devices(placed, processors)
    transfers = graph.meta["transfers"]
    between = carried(transfers, machines)
    if planning is None and any(all(transfer[4]) for transfer in between):
        from .bridge import planning as acquired
        planning = acquired()
    modules, plan = schedule(graph, cost, between, machines,
                             {key for kind, key in exported.outputs if key is not None}, planning, groups=groups)
    held = {name: pages.device for name, (_, pages) in exported.state.items()}
    held.update(zip(exported.user_inputs, map(torch.device, stored), strict=bool(stored)))
    inputs, feedforward, stored = argument_buffers(modules, groups[output_processor], held, transfers, groups=groups)
    ends = [(channel, groups[sender], groups[receiver], *layout) for channel, sender, receiver, *layout in between]
    return Scheduled(exported, modules, {g: devices[g] for g in modules}, {g: machines[g] for g in modules},
                     groups[output_processor], channels, inputs, feedforward, ends,
                     {g: engines[g] for g in modules if g in engines}, plan, stored=stored,
                     invariant=placed.meta["invariant"])


# design/compiler-archive-assignment-2026-11-13.md, "Exact assigned connection"; rdma/mesh-collective.h
# struct mesh_declared, mesh_plan_save: each declared transfer of this machine with its value's solved copies
def lowered(scheduled):
    plan, fields = scheduled.plan, [name for name, _ in mesh.Section._fields_]
    mutated = {key for kind, key in scheduled.exported.outputs if kind == OutputKind.BUFFER_MUTATION}
    constants = {name: tensor.detach() for name, (key, tensor) in scheduled.exported.state.items()
                 if key not in mutated}
    effects = {channel: (transfer, value, plan.offsets[value], plan.sections[value][fields.index("bytes")], sent,
                         landed)
               for transfer, (channel, (*_, value, sent, landed)) in enumerate(zip(plan.channels, plan.declared))
               if value != mesh.ABSENT} if plan else {}
    start = started(plan) if plan else None
    return {group: TARGETS[scheduled.engines.get(group, scheduled.devices[group].type)](
                module, {name: constants[name] for name in scheduled.inputs[group] if name in constants}, effects,
                start, plan.arena_bytes if plan else 0)
            for group, module in scheduled.modules.items() if plan is None or scheduled.machines[group] == plan.machine}


# design/synchronization-signature-change-request.md item 1; compiler/mesh_compiler/ir.py Scheduled.functions
def compiled(scheduled, executions=1):
    scheduled = dataclasses.replace(scheduled, functions=lowered(scheduled))
    return dataclasses.replace(scheduled, signature=signature(scheduled, executions)["record"])
