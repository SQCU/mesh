# design/the-compiler.md
import torch

from .devices import on_devices
from .export import export
from .ir import Scheduled
from .placement import place
from .schedule import argument_buffers, schedule, unit_cost


def compile_module(module, args, processors, placement, cost=unit_cost, **export_options):
    processors = {name: (machine, torch.device(d)) for name, (machine, d) in processors.items()}
    exported = export(module, args, **export_options)
    placed, output_processor, channels = place(exported.graph_module, placement)
    modules = schedule(on_devices(placed, processors), cost)
    inputs, feedforward = argument_buffers(modules, output_processor)
    devices = {name: device for name, (_, device) in processors.items()}
    machines = {name: machine for name, (machine, _) in processors.items()}
    return Scheduled(exported, modules, devices, machines, output_processor, channels, inputs, feedforward)
