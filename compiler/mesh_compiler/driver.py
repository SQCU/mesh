# design/the-compiler.md
import torch

from .devices import ENGINES, HOST, on_devices
from .export import export
from .ir import Scheduled
from .placement import place
from .schedule import argument_buffers, carried, schedule, unit_cost


def compile_module(module, args, processors, placement, cost=unit_cost, **export_options):
    engines = {name: d for name, (_, d) in processors.items() if d in ENGINES}
    processors = {name: (machine, HOST if name in engines else torch.device(d))
                  for name, (machine, d) in processors.items()}
    exported = export(module, args, **export_options)
    placed, output_processor, channels = place(exported.graph_module, placement)
    modules = schedule(on_devices(placed, processors), cost)
    inputs, feedforward = argument_buffers(modules, output_processor)
    devices = {name: device for name, (_, device) in processors.items()}
    machines = {name: machine for name, (machine, _) in processors.items()}
    return Scheduled(exported, modules, devices, machines, output_processor, channels, inputs, feedforward,
                     carried(modules, machines), engines)


def lowered(scheduled, lowerings):
    kinds = {processor: scheduled.engines.get(processor, device.type)
             for processor, device in scheduled.devices.items()}
    made = {kind: lower(scheduled) for kind, lower in lowerings.items() if kind in kinds.values()}
    return {processor: made[kinds[processor]][processor] if kinds[processor] in made else module
            for processor, module in scheduled.modules.items()}
