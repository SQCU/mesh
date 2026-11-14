# design/data-flow.md; design/compilation-composition-change-request.md item 7; schedule-emit/schedule_emit/runtime.py;
# design/compiler-archive-assignment-2026-11-13.md: the selected group entry's completed Target, Scheduled.functions
import torch
from torch.export.graph_signature import InputKind, OutputKind
from torch.utils import _pytree as pytree

from . import bridge


# torch 2.15 torch/export/graph_signature.py InputKind; torch/utils/_pytree.py tree_flatten; Tensor.to
class Arguments:
    def __init__(self, scheduled, entry=None):
        self.entry = scheduled.output_processor if entry is None else entry
        self.exported, self.device = scheduled.exported, scheduled.devices[self.entry]
        self.live = {name: tensor for name, (_, tensor) in self.exported.state.items()
                     if self.exported.state_kinds[name] in (InputKind.PARAMETER, InputKind.BUFFER)}
        state = {name: tensor.detach() for name, (_, tensor) in self.exported.state.items() if name not in self.live}
        state.update((name, torch.empty(0)) for name in self.exported.tokens)
        names = scheduled.inputs[self.entry]
        self.arguments = [state[name].to(self.device) if name in state else None for name in names]
        self.positions = {}
        for position, name in enumerate(names):
            self.positions.setdefault(name, []).append(position)
        self.key_of = {key: name for name, (key, _) in self.exported.state.items()}

    def bind(self, name, value):
        for position in self.positions.get(name, ()):
            self.arguments[position] = value.to(self.device)

    def prepare(self, args, kwargs):
        for name, tensor in self.live.items():
            self.bind(name, tensor.detach())
        flat, _ = pytree.tree_flatten((args, kwargs))
        self.given = dict(zip(self.exported.user_inputs, flat, strict=True))
        for name, value in self.given.items():
            self.bind(name, value)
        return self.arguments


# torch 2.15 torch/export/graph_signature.py OutputKind USER_OUTPUT, BUFFER_MUTATION, USER_INPUT_MUTATION;
# torch/utils/_pytree.py tree_unflatten; compiler/mesh_compiler/bridge.py prepare
class Program(Arguments):
    def __init__(self, scheduled, entry=None):
        entry, self.function = bridge.prepare(scheduled, entry)
        super().__init__(scheduled, entry)
        self.output_processor = scheduled.output_processor

    def __call__(self, *args, **kwargs):
        results = self.function(*self.prepare(args, kwargs))
        if self.entry != self.output_processor:
            return None
        user = []
        for (kind, key), value in zip(self.exported.outputs, results):
            if kind == OutputKind.USER_OUTPUT:
                user.append(value)
            elif kind == OutputKind.BUFFER_MUTATION:
                self.exported.state[self.key_of[key]][1].detach().copy_(value)
            elif kind == OutputKind.USER_INPUT_MUTATION:
                self.given[key].detach().copy_(value)
        return pytree.tree_unflatten(user, self.exported.out_spec)
