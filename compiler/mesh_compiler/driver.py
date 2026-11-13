# design/the-compiler.md
import dataclasses

import torch

from .devices import ENGINES, HOST, on_devices
from .export import export
from .ir import Scheduled
from .placement import place
from .schedule import argument_buffers, carried, schedule, unit_cost
from .signature import signature


# compiler/mesh_compiler/ir.py Planning; compiler/mesh_compiler/bridge.py planning
def compile_module(module, args, processors, placement, cost=unit_cost, stored=(), planning=None, **export_options):
    engines = {name: d for name, (_, d) in processors.items() if d in ENGINES}
    processors = {name: (machine, HOST if name in engines else torch.device(d))
                  for name, (machine, d) in processors.items()}
    exported = export(module, args, **export_options)
    placed, output_processor, channels = place(exported.graph_module, placement)
    devices = {name: device for name, (_, device) in processors.items()}
    machines = {name: machine for name, (machine, _) in processors.items()}
    graph = on_devices(placed, processors)
    transfers = graph.meta["transfers"]
    between = carried(transfers, machines)
    if planning is None and any(all(transfer[4]) for transfer in between):
        from .bridge import planning as acquired
        planning = acquired()
    modules, plan = schedule(graph, cost, between, machines,
                             {key for kind, key in exported.outputs if key is not None}, planning)
    held = {name: pages.device for name, (_, pages) in exported.state.items()}
    held.update(zip(exported.user_inputs, map(torch.device, stored), strict=bool(stored)))
    inputs, feedforward, stored = argument_buffers(modules, output_processor, held, transfers)
    return Scheduled(exported, modules, devices, machines, output_processor, channels, inputs, feedforward,
                     between, engines, plan, stored=stored, invariant=placed.meta["invariant"])


def lowered(scheduled, lowerings):
    functions = {}
    for lower in lowerings:
        made = lower(scheduled)
        modules = {processor: function for processor, function in made.items()
                   if isinstance(function, torch.fx.GraphModule)}
        functions.update((processor, function) for processor, function in made.items() if processor not in modules)
        scheduled = dataclasses.replace(scheduled, modules=modules)
    return {**scheduled.modules, **functions}


# design/synchronization-signature-change-request.md item 1; compiler/mesh_compiler/ir.py Scheduled.functions
def compiled(scheduled, lowerings, executions=1):
    scheduled = dataclasses.replace(scheduled, functions=lowered(scheduled, lowerings))
    return dataclasses.replace(scheduled, signature=signature(scheduled, executions)["record"])
