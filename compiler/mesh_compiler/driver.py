# design/the-compiler.md; design/compiler-archive-assignment-2026-11-13.md; design/compilation-composition-change-
# request.md items 1, 3, 6 and 7; design/aot-compilation-change-request.md A1 and A10: one capture, each captured
# function's construction, one call analysis, one completed native target
import dataclasses

import torch
from torch.export.graph_signature import OutputKind

from . import native
from .decompose import decompose, outline
from .devices import ENGINES, HOST, on_devices
from .export import called, calls
from .ir import Scheduled
from .lower_metal import construct
from .metal import _machines
from .operations import Generator, seeded
from .placement import place
from .schedule import analyse, call_operands, carried, schedule, unit_cost


# design/aot-compilation-change-request.md A1: a module compiled alone, the program of its one call
def compile_module(module, args, processors, placement, cost=unit_cost, planning=None, machine=None,
                   **export_options):
    return compile_captured(called(module, args, **export_options), processors, placement, cost, planning, machine)


# design/aot-compilation-change-request.md A10: a program's calls of its captured functions
def compile_program(program, args, functions, processors, placement, cost=unit_cost, planning=None, machine=None,
                    **export_options):
    return compile_captured(calls(program, args, functions, **export_options), processors, placement, cost, planning,
                            machine)


# compiler/mesh_compiler/ir.py CapturedProgram and Planning; compiler/mesh_compiler/bridge.py planning;
# compiler/mesh_compiler/placement.py place; compiler/mesh_compiler/devices.py on_devices;
# compiler/mesh_compiler/decompose.py decompose; compiler/mesh_compiler/schedule.py analyse, schedule and
# Declarations; compiler/mesh_compiler/lower_metal.py construct; compiler/mesh_compiler/native.py compile_program: one
# group per machine (its first processor its key); each function, in the captured order, placed and put on its
# devices, its placement (its placeholders' stated placements in order, its body's output processor) and the
# program's output processor (the placement of the captured outer output node) given to the calls' operands
# (schedule.py call_operands) before any schedule; a plan's context acquired where a function's transfer crosses
# machines or an operands port (schedule.py CallPort) with a source machine has a consumer machine other than it; the
# compilation machine chosen once (the plan's node map entry Planning.nodes[Planning.here], else `machine`, else the
# processors' one machine) and the seed's words sampled once; a non-CPU eigh's function outlined into its prefix,
# 2n - 1 calls of its QDWH step and its suffix (decompose.py outline) after placement and before the call analysis;
# each function decomposed for its targets, scheduled
# with its source declarations (every machine's body kept, by group), and each machine's body constructed once,
# unlowered (lower_metal.py construct: the program's one construction owner, by (function key, group)), with every
# original processor's selected target and device and each output's role (a buffer mutation's by the placeholder
# torch.export's state names it under), its generator its own copy of the sampled words, the compilation machine's
# body alone with this machine's descriptors (metal.py _machines); the one call analysis (schedule.py analyse) of those
# operands and every construction, which keeps them; ONE completed native target of the compilation machine's
# constructions by function key and the analysis, its node Planning.here with a context and the machine without
def compile_captured(captured, processors, placement, cost=unit_cost, planning=None, machine=None):
    engines = {name: d for name, (_, d) in processors.items() if d in ENGINES}
    processors = {name: (machine, HOST if name in engines else torch.device(d))
                  for name, (machine, d) in processors.items()}
    devices = {name: device for name, (_, device) in processors.items()}
    machines = {name: machine for name, (machine, _) in processors.items()}
    represented = {}
    groups = {name: represented.setdefault(machine, name) for name, machine in machines.items()}
    targets = {name: (engines.get(name, device.type), device) for name, device in devices.items()}
    staged = {}
    for key, exported in captured.functions.items():
        placed, output_processor, channels = place(exported.graph_module, placement)
        graph = on_devices(placed, processors)
        staged[key] = (placed, output_processor, channels, graph, carried(graph.meta["transfers"], machines),
                       [node.meta["stated"] for node in placed.graph.nodes if node.op == "placeholder"])
    captured, staged = outline(captured, staged, targets, processors, machines)
    placements = {key: (stage[5], stage[1]) for key, stage in staged.items()}
    output_processor = placement(captured.exported.graph_module.graph.output_node())
    operands = call_operands(captured, placements, machines, output_processor)
    if planning is None and (any(all(transfer[4]) for stage in staged.values() for transfer in stage[4]) or any(
            port.source is not None and any(machine != port.source for machine in port.machines)
            for port in operands.ports)):
        from .bridge import planning as acquired
        planning = acquired()
    if planning is not None:
        machine = planning.nodes[planning.here]
    elif machine is None:
        (machine,) = dict.fromkeys(machines.values())
    entry, words = represented[machine], seeded(torch.initial_seed())
    constructions, declared, modules = {}, {}, {}
    for key, (placed, produced, channels, graph, between, _) in staged.items():
        exported = captured.functions[key]
        modules[key], declared[key] = schedule(decompose(graph, targets), cost, between, machines, planning=planning,
                                               groups=groups)
        placeholder = {target: name for name, (target, _) in exported.state.items()}
        roles = [(kind, placeholder[target] if kind == OutputKind.BUFFER_MUTATION else target)
                 for kind, target in exported.outputs]
        for group, body in modules[key].items():
            constructions[key, group] = construct(
                body, targets=targets, kinds=exported.state_kinds,
                sources={name: tensor.detach() for name, (_, tensor) in exported.state.items()},
                wires=declared[key].wires(machines[group]), output_roles=roles if group == groups[produced] else (),
                generator=Generator(words.clone()), machines=_machines() if group == entry else None)
    analysis = analyse(operands, constructions, declared)
    constructed = {key: made for (key, group), made in constructions.items() if group == entry}
    target = native.compile_program(captured, constructed, declared, analysis,
                                    planning.here if planning is not None else machine, planning, targets=targets)
    used = list(dict.fromkeys(g for body in modules.values() for g in body))
    return Scheduled(captured.exported, modules, {g: devices[g] for g in used}, {g: machines[g] for g in used},
                     groups[output_processor],
                     {key: stage[2] for key, stage in staged.items()},
                     {entry: [node.name for node in captured.exported.graph_module.graph.nodes
                              if node.op == "placeholder"]}, [],
                     {key: [(channel, groups[sender], groups[receiver], *layout)
                            for channel, sender, receiver, *layout in stage[4]] for key, stage in staged.items()},
                     {g: engines[g] for g in used if g in engines}, target.plan if planning is not None else None,
                     invariant={key: stage[0].meta["invariant"] for key, stage in staged.items()},
                     functions={entry: target})


# design/synchronization-signature-change-request.md item 1; compiler/mesh_compiler/ir.py Scheduled.functions: the
# one completed native target's own signature record (MetalProgramSignature), no recomposition
def compiled(scheduled):
    (target,) = scheduled.functions.values()
    return dataclasses.replace(scheduled, signature=target.signature)
