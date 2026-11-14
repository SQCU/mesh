# design/the-compiler.md; design/compiler-archive-assignment-2026-11-13.md; design/compilation-composition-change-
# request.md items 1, 3, 6 and 7: one construction, one lowering, one plan of its actual results, the native artifact
import dataclasses
import pathlib
import sys

import torch
from torch.export.graph_signature import OutputKind

from . import native
from .devices import ENGINES, HOST, on_devices
from .export import export
from .ir import Scheduled
from .lower_metal import construct
from .placement import place
from .schedule import argument_buffers, carried, schedule, unit_cost

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh_compile


# compiler/mesh_compiler/ir.py Planning; compiler/mesh_compiler/bridge.py planning; compiler/mesh_compiler/placement.py
# place, transfers; compiler/mesh_compiler/schedule.py schedule, Declarations; compiler/mesh_compiler/lower_metal.py
# construct; compiler/mesh_compiler/native.py compile: one group per machine (its first processor its key), its body
# in the retained global order, the source declarations (no plan before the lowering), ONE typed construction with
# every original processor's selected target and device and each output's role (a buffer mutation's by the placeholder
# torch.export's state names it under), and that machine's completed native target, compiled here (this node's machine
# alone where there is a plan's context, whose solved Plan the Scheduled keeps), its machine node the plan's node map
# index (Planning.here) with a context and its source machine without one
def compile_module(module, args, processors, placement, cost=unit_cost, stored=(), planning=None, **export_options):
    engines = {name: d for name, (_, d) in processors.items() if d in ENGINES}
    processors = {name: (machine, HOST if name in engines else torch.device(d))
                  for name, (machine, d) in processors.items()}
    exported = export(module, args, **export_options)
    placed, output_processor, channels = place(exported.graph_module, placement)
    devices = {name: device for name, (_, device) in processors.items()}
    machines = {name: machine for name, (machine, _) in processors.items()}
    represented = {}
    groups = {name: represented.setdefault(machine, name) for name, machine in machines.items()}
    targets = {name: (engines.get(name, device.type), device) for name, device in devices.items()}
    graph = on_devices(placed, processors)
    transfers = graph.meta["transfers"]
    between = carried(transfers, machines)
    if planning is None and any(all(transfer[4]) for transfer in between):
        from .bridge import planning as acquired
        planning = acquired()
    modules, declarations = schedule(graph, cost, between, machines, planning=planning, groups=groups)
    held = {name: pages.device for name, (_, pages) in exported.state.items()}
    held.update(zip(exported.user_inputs, map(torch.device, stored), strict=bool(stored)))
    inputs, feedforward, stored = argument_buffers(modules, groups[output_processor], held, transfers, groups=groups)
    ends = [(channel, groups[sender], groups[receiver], *layout) for channel, sender, receiver, *layout in between]
    sources = {name: tensor.detach() for name, (_, tensor) in exported.state.items()}
    placeholder = {key: name for name, (key, _) in exported.state.items()}
    roles = [(kind, placeholder[key] if kind == OutputKind.BUFFER_MUTATION else key) for kind, key in exported.outputs]

    def compiled_here(group, body):
        generator = mesh_compile.Generator(mesh_compile.seeded(torch.initial_seed()))
        machine = construct(body, targets=targets, kinds=exported.state_kinds, sources=sources,
                            wires=declarations.wires(),
                            output_roles=roles if group == groups[output_processor] else (),
                            generator=generator, machines=mesh_compile._machines())
        return native.compile(machine, declarations, planning.here if planning is not None else machines[group],
                              planning, generator)
    functions = {group: compiled_here(group, body) for group, body in modules.items()
                 if planning is None or machines[group] == declarations.machine}
    plan = None
    if planning is not None:
        (here,) = functions
        plan = functions[here].plan
    return Scheduled(exported, modules, {g: devices[g] for g in modules}, {g: machines[g] for g in modules},
                     groups[output_processor], channels, inputs, feedforward, ends,
                     {g: engines[g] for g in modules if g in engines}, plan, stored=stored,
                     invariant=placed.meta["invariant"], functions=functions)


# design/synchronization-signature-change-request.md item 1; compiler/mesh_compiler/ir.py Scheduled.functions: the
# machine entry's completed native target's own signature record (MetalProgramSignature), no recomposition: the output
# machine's without a plan (every machine compiled here), this node's (the plan's machine, the one compiled) with one
def compiled(scheduled):
    entry = scheduled.output_processor if scheduled.plan is None else next(
        group for group in scheduled.functions if scheduled.machines[group] == scheduled.plan.machine)
    return dataclasses.replace(scheduled, signature=scheduled.functions[entry].signature)
