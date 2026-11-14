# design/data-flow.md; design/compilation-composition-change-request.md item 7; schedule-emit/schedule_emit/runtime.py;
# design/compiler-archive-assignment-2026-11-13.md: the selected group entry's completed Target, Scheduled.functions
import math

import torch
from torch.export.graph_signature import InputKind
from torch.utils import _pytree as pytree

from . import bridge
from .ir import RETURNED


# torch 2.14.0 torch/_export/utils.py _check_input_constraints_for_graph, lines 487–504, its int, float and str literal
# branch as written (a bool by its int class): an argument for a specialized literal input of another type or value, or
# not nan for a nan, does not run the program compiled for that literal
def literal(path, node_val, arg):
    if type(arg) is not type(node_val):
        raise RuntimeError(f"Expected input at {path} to be equal to {node_val}, but got {arg}")
    if isinstance(node_val, float) and math.isnan(node_val):
        if not isinstance(arg, float) or not math.isnan(arg):
            raise RuntimeError(f"Expected input at {path} to be nan, but got {arg}")
    elif arg != node_val:
        raise RuntimeError(f"Expected input at {path} to be equal to {node_val}, but got {arg}")


# torch 2.14.0 torch/export/graph_signature.py InputKind; Python inspect Signature.bind and BoundArguments
# apply_defaults; torch/utils/_pytree.py TreeSpec.flatten_up_to; compiler/mesh_compiler/ir.py Exported.interface: each
# placeholder's actual state or user handle, as given (its device the explicit graph copies govern), aliases kept, the
# call's arguments bound by the captured slot Signature (its own errors for a missing, unexpected or wrong-kind
# argument), its defaults applied, and their user leaves the retained TreeSpec's children of the bound values in its
# structure and key order (its own arity and context errors), each specialized literal user leaf (Exported.specialized,
# a value of the upstream literal branch's classes) checked by `literal`; prepared once, each live PARAMETER or BUFFER
# tensor with its entry positions, each user input's entry positions (empty where the selected entry has none) and
# each specialized literal's (flat user leaf index, compiled literal)
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
        self.specialized = tuple((self.exported.user_inputs.index(name), value)
                                 for name, value in self.exported.specialized if isinstance(value, (int, float, str)))

    def bind(self, positions, value):
        for position in positions:
            self.arguments[position] = value

    def prepare(self, args, kwargs):
        for tensor, positions in self.live:
            self.bind(positions, tensor.detach())
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        flat = self.spec.flatten_up_to(tuple(bound.arguments.values()))
        for k, value in self.specialized:
            literal(self.exported.user_inputs[k], value, flat[k])
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
