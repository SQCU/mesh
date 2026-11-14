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
# structure and key order (its own arity and context errors); prepared once, each live PARAMETER or BUFFER tensor
# with its entry positions and each user input's entry positions (empty where the selected entry has none)
class Arguments:
    def __init__(self, scheduled, entry=None):
        self.entry = scheduled.output_processor if entry is None else entry
        self.exported = scheduled.exported
        self.signature, self.spec = self.exported.interface
        live = {name: tensor for name, (_, tensor) in self.exported.state.items()
                if self.exported.state_kinds[name] in (InputKind.PARAMETER, InputKind.BUFFER)}
        state = {name: tensor.detach() for name, (_, tensor) in self.exported.state.items() if name not in live}
        state.update((name, torch.empty(0)) for name in self.exported.tokens)
        names = scheduled.inputs[self.entry]
        # rejection filters are forbidden, even in temporary integration validation: an entry input no binding owner
        # provides raises here, never passed as None
        given = {**dict.fromkeys(live), **dict.fromkeys(self.exported.user_inputs), **state}
        self.arguments = [given[name] for name in names]
        positions = {}
        for position, name in enumerate(names):
            positions.setdefault(name, []).append(position)
        self.live = tuple((tensor, tuple(positions.get(name, ()))) for name, tensor in live.items())
        self.users = tuple(tuple(positions.get(name, ())) for name in self.exported.user_inputs)

    def bind(self, positions, value):
        for position in positions:
            self.arguments[position] = value

    def prepare(self, args, kwargs):
        for tensor, positions in self.live:
            self.bind(positions, tensor.detach())
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        flat = self.spec.flatten_up_to(tuple(bound.arguments.values()))
        for positions, value in zip(self.users, flat, strict=True):
            self.bind(positions, value)
        return self.arguments


# compiler/mesh_compiler/ir.py RETURNED; torch/utils/_pytree.py tree_structure and tree_unflatten;
# compiler/mesh_compiler/bridge.py prepare; design/compilation-composition-change-request.md item 7: the returned
# outputs, a buffer or user input mutation being the native program's own store into the caller's state or input
# storage (compiler/mesh_compiler/lower_metal.py construct), not copied back; the result recipe prepared once: on the
# output owner the RETURNED output indices under out_spec, elsewhere no index and the one constant leaf None under its
# own TreeSpec; each invocation the native function's fresh results gathered at those indices beside those constants
class Program(Arguments):
    def __init__(self, scheduled, entry=None):
        entry, self.function = bridge.prepare(scheduled, entry)
        super().__init__(scheduled, entry)
        self.output_processor = scheduled.output_processor
        owner = self.entry == self.output_processor
        self.returned = tuple(k for k, (kind, _) in enumerate(self.exported.outputs) if kind in RETURNED) if owner \
            else ()
        self.constants, self.result = ((), self.exported.out_spec) if owner else ((None,), pytree.tree_structure(None))

    def __call__(self, *args, **kwargs):
        results = self.function(*self.prepare(args, kwargs))
        return pytree.tree_unflatten([*self.constants, *(results[k] for k in self.returned)], self.result)
