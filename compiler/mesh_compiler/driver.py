# design/the-compiler.md
import torch

from .devices import on_devices
from .export import export
from .ir import Scheduled
from .placement import place
from .schedule import argument_buffers, schedule, unit_cost


def compile_module(module, args, processors, placement, cost=unit_cost, **export_options):
    devices = {name: torch.device(d) for name, d in processors.items()}
    exported = export(module, args, **export_options)
    placed, output_processor, channels = place(exported.graph_module, placement)
    modules = schedule(on_devices(placed, devices), cost)
    inputs, feedforward = argument_buffers(modules, output_processor)
    return Scheduled(exported, modules, devices, output_processor, channels, inputs, feedforward)
