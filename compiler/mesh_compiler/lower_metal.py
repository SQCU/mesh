# design/compilation-composition-change-request.md items 1, 2, 6 and 7; design/compiler-workgroup-2026-11-13.md;
# ~/schedule-emit/schedule_emit/driver.py and lower_inductor.py lower: a stage takes the scheduled body of one group
# and returns its target's program, built by the one numerical construction (rdma/mesh_compile.py _instruction,
# _Builder); output_data/compiler-workgroup-20261113/frontends/litert/typed-target.status.md, the result contract
import collections
import ctypes as C
import dataclasses
import pathlib
import sys
import weakref

import torch
from torch.export.graph_signature import InputKind
from torch.fx.node import map_arg
from torch.utils import _pytree as pytree

from .ir import RECV, SEND, channel_of
from .lower_inductor import fanout

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh_compile as M

Root = collections.namedtuple("Root", "name kind role root value")
Output = collections.namedtuple("Output", "root value literal metadata")
# torch 2.14.0 torch/export/graph_signature.py InputKind; metal-microbench metal_ir.h METAL_IR_INPUT .. METAL_IR_STATE
KINDS = {InputKind.USER_INPUT: ("user_input", M.INPUT), InputKind.PARAMETER: ("parameter", M.STATE),
         InputKind.BUFFER: ("buffer", M.STATE), InputKind.CONSTANT_TENSOR: ("constant_tensor", M.CONSTANT)}
# a storage several placeholders view: its role the most mutable of theirs (state written across calls, then an
# input varying per call, then constant data)
AUTHORITY = (M.STATE, M.INPUT, M.CONSTANT)
# design/the-compiler.md, "A processor is a device of a machine"; metal-microbench metal_ir.h METAL_IR_GPU .. ANE
PLACES = {"cpu": M.CPU, "mps": M.GPU, "ane": M.ANE}


@dataclasses.dataclass(eq=False)
class Constructed:
    program: int
    compiler: object
    target: str
    place: int
    inputs: tuple
    outputs: tuple
    spec: object
    state: tuple
    received: dict
    sent: dict
    wires: dict
    operations: dict
    advanced: tuple = ()
    fanout: object = None
    writes: dict = dataclasses.field(default_factory=dict)
    identity: int = 0

    def __post_init__(self):
        self.freed = weakref.finalize(self, self.compiler.metal_ir_program_free, self.program)

    def close(self):
        self.freed()


# torch 2.14.0 torch/_tensor_docs.py as_strided, untyped_storage; torch/_export/non_strict_utils.py fakify: one meta root
# a source storage as the export's fake values alias it (every placeholder kind in that one namespace), each
# placeholder its view with its own shape, strides and offset; a CONSTANT root's initial bytes the exported tensor's
def placeholders(nodes, kinds, sources):
    found = {}
    for node in nodes:
        value = node.meta.get("val") if node.op == "placeholder" else None
        if isinstance(value, torch.Tensor):
            found.setdefault(value.untyped_storage()._cdata, []).append(node)
    roots = {}
    for viewed in found.values():
        roles = {KINDS[kinds.get(node.target, InputKind.USER_INPUT)][1] for node in viewed}
        role = next(role for role in AUTHORITY if role in roles)
        first = viewed[0].meta["val"]
        root = torch.empty(first.untyped_storage().nbytes() // first.element_size(), dtype=first.dtype, device="meta")
        data = sources[viewed[0].target] if role == M.CONSTANT else None
        for node in viewed:
            value = node.meta["val"]
            base = root if value.dtype == root.dtype else root.view(value.dtype)
            roots[node] = base.as_strided(value.shape, value.stride(), value.storage_offset()), role, data
    return roots


# compiler/mesh_compiler/driver.py lowered, effects: channel -> (declared transfer, its Plan value, the value's solved
# copy offsets, its byte extent, declared sent view, declared landed view, supplied and completion representatives);
# compiler/mesh_compiler/lower_inductor.py body, typed: a view (offset, element, (size, byte stride) pairs) of one copy
# of the Plan value's bytes; J. K. Salmon et al., SC 2011, §3.1: `generator` the machine's one mesh_compile.Generator,
# its sites numbered across every group it is passed to, its advance in the program built with `advance`
def construct(module, *, target, kinds, sources, effects=None, generator=None, advance=False, page=0, machines=None):
    compiler, place = M._compiler(), PLACES[target]
    builder = M._Builder(compiler, {}, None, 0, 0)
    try:
        if machines:
            compiler.metal_ir_machine(builder.p, C.byref(machines[0]), C.byref(machines[1]))
        nodes, tasks, effects = list(module.graph.nodes), module.meta["tasks"], effects or {}
        drew = generator.sites if generator is not None else 0
        roots, fan = placeholders(nodes, kinds, sources), fanout(module, effects)
        env, records, owners, inputs, landing, sending, held, written = {}, [], [], [], [], [], {}, {}

        def rooted(t, role, data=None):
            return builder.declare(t, role, data), builder.value(M._stored(t))

        def wire(channel, like, landed):
            _, value, _, size, sent, arrived, *_ = effects[channel]
            offset, element, pairs = arrived if landed else sent
            if value not in held:
                held[value] = torch.empty(size // like.element_size(), dtype=like.dtype, device="meta")
            sizes = list(like.shape) if landed else [n for n, _ in pairs]
            return value, held[value].as_strided(sizes, [stride // element for _, stride in pairs], offset // element)

        for node in nodes:
            if node.op == "placeholder":
                kind = KINDS[kinds.get(node.target, InputKind.USER_INPUT)][0]
                if node not in roots:
                    env[node] = node.meta.get("val")
                    inputs.append(Root(node.target, kind, None, None, None))
                    continue
                env[node], role, data = roots[node]
                inputs.append(Root(node.target, kind, role, *rooted(env[node], role, data)))
            elif node.op == "output":
                leaves, spec = pytree.tree_flatten(node.args[0])
        for kind, of in fan.order:
            if kind == "write":
                first = fan.member[of][0]
                value, view = wire(channel_of(first), env[fan.groups[of][0]], False)
                given = env[fan.groups[of][0]]
                made = M._copy(view, given.as_strided(view.shape, given.stride(), given.storage_offset()))
                M._placed(made, place, tasks[first][0])
                records += made
                owners += [(tasks[first], of)] * len(made)
                written[of] = value, view
            elif kind != "node":
                continue
            elif of.target is RECV:
                channel = of.args[0]
                value, env[of] = wire(channel, of.meta["val"], True) if channel in effects else (
                    None, M._held(of.meta["val"]))
                landing.append((channel, value, env[of]))
            elif of.target is SEND:
                env[of] = None
                sending.append((of.args[1], None, env[of.args[0]]))
            else:
                local = M._local_call(of)
                args, kwargs = map_arg((local.args, local.kwargs), env.__getitem__)
                env[of], made = M._instruction(local, args, kwargs, of.meta.get("val"), page, generator)
                M._placed(made, place, tasks[of][0])
                records += made
                owners += [(tasks[of], None)] * len(made)
        sending += [(channel_of(node), *written[fan.index[fan.group[node]]]) for node in fan.sends]
        received, sent, wires = {}, {}, {}
        for channel, value, t in landing:
            received[channel] = Root(channel, "received", M.INPUT, *rooted(t, M.INPUT))
            if value is not None:
                wires[value] = Root(value, "wire", M.INPUT, received[channel].root, None)
        outputs = []
        for leaf in leaves:
            value = env[leaf] if isinstance(leaf, torch.fx.Node) else leaf
            if isinstance(value, torch.Tensor):
                source = leaf.meta["val"] if isinstance(leaf, torch.fx.Node) else value
                outputs.append(Output(*rooted(value, M.OUTPUT), None, M._metadata(source)))
            else:
                outputs.append(Output(None, None, value, None))
        for channel, value, t in sending:
            root, view = (rooted(t, M.TEMPORARY) if value is not None else (builder.root(t), builder.value(M._stored(t))))
            role = builder.roles[M._key(t)]
            sent[channel] = Root(channel, "sent", role, root, view)
            if value is not None:
                wires.setdefault(value, Root(value, "wire", role, root, None))
        state, stepped = (), []
        if generator is not None and (generator.sites > drew or advance and generator.sites):
            state = (Root("generator", "generator", M.STATE, *rooted(generator.state, M.STATE)),)
            if advance:
                stepped = generator.stepped()
                M._placed(stepped, place, len(module.meta["order"]))
        operations, advanced, writes = {}, [], {}
        for owner, g in zip(owners + [None] * len(stepped), records + stepped):
            made = builder.generic(g)
            if owner is None:
                advanced.append(made)
                continue
            task, write = owner
            operations.setdefault(task, []).append(made)
            if write is not None:
                writes.setdefault(write, []).append(made)
    except BaseException:
        compiler.metal_ir_program_free(builder.p)
        raise
    return Constructed(builder.p, compiler, target, place, tuple(inputs), tuple(outputs), spec, state, received, sent,
                       wires, {task: tuple(made) for task, made in operations.items()}, tuple(advanced), fan,
                       {k: tuple(made) for k, made in writes.items()}, builder.base)


# output_data/compiler-workgroup-20261113/frontends/litert/typed-target.inbox.md: compiler/mesh_compiler/native.py
# Target, lower, the paired consumer of the constructed programs and the plan's effects
def lower(module, *, target, kinds, sources, effects, start, extent, generator=None, advance=False, page=0,
          machines=None):
    from . import native
    return native.lower(construct(module, target=target, kinds=kinds, sources=sources, effects=effects,
                                  generator=generator, advance=advance, page=page, machines=machines),
                        effects=effects, start=start, extent=extent)
