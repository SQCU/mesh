# design/data-flow.md; design/compilation-composition-change-request.md item 7; schedule-emit/schedule_emit/runtime.py;
# design/compiler-archive-assignment-2026-11-13.md: the selected group entry's completed Target, Scheduled.functions
import torch
from torch._export.utils import _check_input_constraints_for_graph
from torch.utils import _pytree as pytree

from . import bridge
from .ir import RETURNED


# Python inspect Signature.bind and BoundArguments apply_defaults; torch/utils/_pytree.py TreeSpec.flatten_up_to; compiler/mesh_compiler/ir.py Exported.interface: each
# placeholder's actual state object (Exported.state's own tensor, a name sharing it the same object; native.py
# Bound.invoked returning a direct input output as the object given) or user handle, as given (its device the explicit
# graph copies govern), aliases kept, the
# call's arguments bound by the captured slot Signature (its own errors for a missing, unexpected or wrong-kind
# argument), its defaults applied, and their user leaves the retained TreeSpec's children of the bound values in its
# structure and key order (its own arity and context errors), checked by torch/_export/utils.py
# _check_input_constraints_for_graph itself (as ExportedProgram's call checks its user inputs) over the outer program's
# own user input placeholders in user input order (each one's `val` its compiled prototype or literal), each leaf with
# its key path in that TreeSpec, and the outer program's retained range constraints (Exported.range_constraints, their
# own producer's result at capture); identity and storage overlap among the inputs no part of that check (the AOT
# dispatch wrappers' pre_compile result, not retained by the selected export entry, owns it);
# prepared once, each state and token bound at its entry positions, each user input's entry positions (empty where the
# selected entry has none) bound at each call
class Arguments:
    def __init__(self, scheduled, entry=None):
        self.entry = scheduled.output_processor if entry is None else entry
        self.exported = scheduled.exported
        self.signature, self.spec = self.exported.interface
        state = {name: tensor for name, (_, tensor) in self.exported.state.items()}
        state.update((name, torch.empty(0)) for name in self.exported.tokens)
        names = scheduled.inputs[self.entry]
        # rejection filters are forbidden, even in temporary integration validation: an entry input no binding owner
        # provides raises here, never passed as None
        given = {**dict.fromkeys(self.exported.user_inputs), **state}
        self.arguments = [given[name] for name in names]
        positions = {}
        for position, name in enumerate(names):
            positions.setdefault(name, []).append(position)
        self.users = tuple(tuple(positions.get(name, ())) for name in self.exported.user_inputs)
        placeholders = {node.name: node for node in self.exported.graph_module.graph.nodes if node.op == "placeholder"}
        self.placeholders = [placeholders[name] for name in self.exported.user_inputs]
        self.paths = [path for path, _ in pytree.tree_flatten_with_path(
            pytree.tree_unflatten(list(range(self.spec.num_leaves)), self.spec))[0]]

    def bind(self, positions, value):
        for position in positions:
            self.arguments[position] = value

    def prepare(self, args, kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        flat = self.spec.flatten_up_to(tuple(bound.arguments.values()))
        _check_input_constraints_for_graph(self.placeholders, list(zip(self.paths, flat, strict=True)),
                                           self.exported.range_constraints)
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
