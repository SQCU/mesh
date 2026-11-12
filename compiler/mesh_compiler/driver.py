# design/the-compiler.md
import dataclasses

import torch

from .devices import ENGINES, HOST, on_devices
from .export import export
from .ir import Scheduled
from .placement import place
from .schedule import argument_buffers, carried, planned, schedule, stored, unit_cost
from .signature import signature


def compile_module(module, args, processors, placement, cost=unit_cost, **export_options):
    engines = {name: d for name, (_, d) in processors.items() if d in ENGINES}
    processors = {name: (machine, HOST if name in engines else torch.device(d))
                  for name, (machine, d) in processors.items()}
    exported = export(module, args, **export_options)
    placed, output_processor, channels = place(exported.graph_module, placement)
    devices = {name: device for name, (_, device) in processors.items()}
    machines = {name: machine for name, (machine, _) in processors.items()}
    graph = on_devices(placed, processors)
    between = carried(graph, machines)
    plan = planned(graph, between, machines, cost)
    modules = schedule(graph, cost, plan)
    plan = stored(plan, graph, machines, {key for kind, key in exported.outputs if key is not None})
    inputs, feedforward = argument_buffers(modules, output_processor)
    return Scheduled(exported, modules, devices, machines, output_processor, channels, inputs, feedforward,
                     between, engines, plan)


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
def compiled(scheduled, lowerings, executions=1, stored=()):
    functions = lowered(scheduled, lowerings)
    made = signature(scheduled, functions, executions, stored)
    return dataclasses.replace(scheduled, signature=made["record"]), functions
