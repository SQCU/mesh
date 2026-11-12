# design/the-compiler.md
import dataclasses

import torch

from .devices import ENGINES, HOST, on_devices
from .export import export
from .ir import Scheduled
from .placement import place
from .schedule import argument_buffers, carried, schedule, unit_cost
from .signature import signature


def compile_module(module, args, processors, placement, cost=unit_cost, stored=(), **export_options):
    engines = {name: d for name, (_, d) in processors.items() if d in ENGINES}
    processors = {name: (machine, HOST if name in engines else torch.device(d))
                  for name, (machine, d) in processors.items()}
    exported = export(module, args, **export_options)
    placed, output_processor, channels = place(exported.graph_module, placement)
    devices = {name: device for name, (_, device) in processors.items()}
    machines = {name: machine for name, (machine, _) in processors.items()}
    graph = on_devices(placed, processors)
    between = carried(graph, machines)
    modules, plan = schedule(graph, cost, between, machines,
                             {key for kind, key in exported.outputs if key is not None})
    held = {name: pages.device for name, (_, pages) in exported.state.items()}
    held.update(zip(exported.user_inputs, map(torch.device, stored), strict=bool(stored)))
    inputs, feedforward, stored = argument_buffers(modules, output_processor, held)
    scheduled = Scheduled(exported, modules, devices, machines, output_processor, channels, inputs, feedforward,
                          between, engines, plan, stored=stored,
                          invariant=placed.meta["invariant"])
    return dataclasses.replace(scheduled, signature=signature(scheduled)["record"])


def lowered(scheduled, lowerings):
    functions = {}
    for lower in lowerings:
        made = lower(scheduled)
        modules = {processor: function for processor, function in made.items()
                   if isinstance(function, torch.fx.GraphModule)}
        functions.update((processor, function) for processor, function in made.items() if processor not in modules)
        scheduled = dataclasses.replace(scheduled, modules=modules)
    return {**scheduled.modules, **functions}


# design/synchronization-signature-change-request.md item 1
def compiled(scheduled, lowerings, executions=1):
    functions = lowered(scheduled, lowerings)
    made = signature(scheduled, functions, executions)
    return dataclasses.replace(scheduled, signature=made["record"]), functions
