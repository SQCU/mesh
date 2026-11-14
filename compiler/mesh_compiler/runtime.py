# design/data-flow.md; design/compilation-composition-change-request.md item 7; schedule-emit/schedule_emit/runtime.py;
# design/compiler-archive-assignment-2026-11-13.md: the selected group entry's completed Target, Scheduled.functions
import torch
from torch.export.graph_signature import InputKind
from torch.utils import _pytree as pytree

from . import bridge
from .ir import RETURNED


# torch 2.14.0 torch/export/graph_signature.py InputKind; Python inspect Signature.bind and BoundArguments
# apply_defaults; torch/utils/_pytree.py TreeSpec.flatten_up_to; compiler/mesh_compiler/ir.py Exported.interface: each
# placeholder's actual state or user handle, as given (its device the explicit graph copies govern), aliases kept, the
# call's arguments bound by the captured slot Signature (its own errors for a missing, unexpected or wrong-kind
# argument), its defaults applied, and their user leaves the retained TreeSpec's children of the bound values in its
# structure and key order (its own arity and context errors)
class Arguments:
    def __init__(self, scheduled, entry=None):
        self.entry = scheduled.output_processor if entry is None else entry
        self.exported = scheduled.exported
        self.signature, self.spec = self.exported.interface
        self.live = {name: tensor for name, (_, tensor) in self.exported.state.items()
                     if self.exported.state_kinds[name] in (InputKind.PARAMETER, InputKind.BUFFER)}
        state = {name: tensor.detach() for name, (_, tensor) in self.exported.state.items() if name not in self.live}
        state.update((name, torch.empty(0)) for name in self.exported.tokens)
        names = scheduled.inputs[self.entry]
        # rejection filters are forbidden, even in temporary integration validation: an entry input no binding owner
        # provides raises here, never passed as None
        given = {**dict.fromkeys(self.live), **dict.fromkeys(self.exported.user_inputs), **state}
        self.arguments = [given[name] for name in names]
        self.positions = {}
        for position, name in enumerate(names):
            self.positions.setdefault(name, []).append(position)

    def bind(self, name, value):
        for position in self.positions.get(name, ()):
            self.arguments[position] = value

    def prepare(self, args, kwargs):
        for name, tensor in self.live.items():
            self.bind(name, tensor.detach())
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        flat = self.spec.flatten_up_to(tuple(bound.arguments.values()))
        for name, value in zip(self.exported.user_inputs, flat, strict=True):
            self.bind(name, value)
        return self.arguments


# compiler/mesh_compiler/ir.py RETURNED; torch/utils/_pytree.py tree_unflatten; compiler/mesh_compiler/bridge.py
# prepare: the returned outputs; a buffer or user input mutation is the native program's own store into the caller's
# state or input storage (compiler/mesh_compiler/lower_metal.py construct), not copied back
class Program(Arguments):
    def __init__(self, scheduled, entry=None):
        entry, self.function = bridge.prepare(scheduled, entry)
        super().__init__(scheduled, entry)
        self.output_processor = scheduled.output_processor

    def __call__(self, *args, **kwargs):
        results = self.function(*self.prepare(args, kwargs))
        if self.entry != self.output_processor:
            return None
        user = [value for (kind, _), value in zip(self.exported.outputs, results, strict=True)
                if kind in RETURNED]
        return pytree.tree_unflatten(user, self.exported.out_spec)
