# design/data-flow.md
import operator

import torch
from torch._ops import HigherOrderOperator
from torch.export.experimental import _export_forward_backward
from torch.export.graph_signature import InputKind, OutputKind
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils import _pytree as pytree

from .ir import Exported

WITH_EFFECTS = torch.ops.higher_order.with_effects


# design/unequal-shards-change-request.md item 8; torch 2.15 torch/export/experimental/__init__.py
def export(module, args, loss=None, **export_options):
    program = torch.export.export(module, args, **export_options)
    if loss is not None:
        with sdpa_kernel(SDPBackend.MATH):
            program = _export_forward_backward(program, joint_loss_index=loss)
    return read(program.run_decompositions())


def read(program):
    signature = program.graph_signature
    nodes = list(program.graph_module.graph.nodes)
    control = [n.target.name() for n in nodes
               if isinstance(n.target, HigherOrderOperator) and n.target is not WITH_EFFECTS]
    if control:
        raise NotImplementedError(
            f"{control[0]}: data-dependent control flow is out of scope; only static-capture "
            "programs (fixed operators, shapes and channels) are scheduled")
    attached = [n.target for n in nodes if n.op == "get_attr"]
    if attached:
        raise NotImplementedError(f"get_attr {attached[0]}: a graph that reads an attribute of "
                                  "its module is out of scope; state must be a placeholder")

    for node in nodes:
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor) and not all(isinstance(d, int) for d in value.shape):
            raise NotImplementedError(
                f"{node.name} has shape {tuple(value.shape)}: shapes are compiled; a dimension "
                "known only at run time is out of scope")

    state, tokens, user_inputs = {}, [], []
    for spec in signature.input_specs:
        name = spec.arg.name
        if spec.kind == InputKind.USER_INPUT:
            user_inputs.append(name)
        elif spec.kind == InputKind.TOKEN:
            tokens.append(name)
        elif spec.kind == InputKind.PARAMETER or (spec.kind == InputKind.BUFFER and spec.persistent):
            state[name] = (spec.target, program.state_dict[spec.target])
        elif spec.kind in (InputKind.BUFFER, InputKind.CONSTANT_TENSOR):
            state[name] = (spec.target, program.constants[spec.target])
        else:
            raise NotImplementedError(f"input kind {spec.kind}")

    for node in nodes:
        if node.op == "placeholder" and node.name in state:
            node.meta["state"] = state[node.name][0]

    outputs, gradients, out_spec = [], {}, program.call_spec.out_spec
    for spec in signature.output_specs:
        if spec.kind == OutputKind.GRADIENT_TO_PARAMETER:
            gradients[spec.target] = 0
        elif spec.kind not in (OutputKind.USER_OUTPUT, OutputKind.LOSS_OUTPUT, OutputKind.BUFFER_MUTATION,
                               OutputKind.TOKEN):
            raise NotImplementedError(f"output kind {spec.kind}")
        returned = spec.kind in (OutputKind.LOSS_OUTPUT, OutputKind.GRADIENT_TO_PARAMETER)
        outputs.append((OutputKind.USER_OUTPUT, None) if returned else (spec.kind, spec.target))
    if signature.backward_signature is not None:
        forward = pytree.tree_unflatten([0] * out_spec.num_leaves, out_spec)
        out_spec = pytree.tree_structure((forward, gradients))
    return Exported(program.graph_module, state, tokens, user_inputs, outputs, out_spec)


# design/aot-compilation-change-request.md A1 and A10; P. Barham et al., Pathways, arXiv:2203.12533, section 3;
# J. Reed et al., torch.fx, MLSys 2022
class Calls(torch.fx.Tracer):
    def __init__(self, functions):
        super().__init__(autowrap_functions=tuple(f for f in functions if callable(f)))
        self.functions = functions

    def is_leaf_module(self, module, qualified_name):
        return qualified_name in self.functions


class Function(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class Written(torch.nn.Module):
    def __init__(self, function, written):
        super().__init__()
        self.function, self.written = function, written

    def forward(self, *args, **kwargs):
        result = self.function(*args[:len(args) - len(self.written)], **kwargs)
        return (result, *(self.function.get_buffer(name) for name in self.written))


class Captured(torch.fx.Interpreter):
    def __init__(self, module, functions, options):
        super().__init__(module)
        self.functions, self.options, self.exported = functions, options, {}

    def run_node(self, node):
        self.calling = node
        return super().run_node(node)

    def captured(self, target, module, args, kwargs):
        leaves, structure = pytree.tree_flatten((args, kwargs))
        key = target, str(structure), tuple((str(leaf.dtype), tuple(leaf.shape)) if isinstance(leaf, torch.Tensor)
                                            else repr(leaf) for leaf in leaves)
        if key not in self.exported:
            with torch.enable_grad():
                self.exported[key] = export(module, args, kwargs=kwargs, **self.options)
        self.calling.meta["function"] = key

    def call_module(self, target, args, kwargs):
        self.captured(target, self.fetch_attr(target), args, kwargs)
        return super().call_module(target, args, kwargs)

    def call_function(self, target, args, kwargs):
        if target in self.functions:
            self.captured(target, Function(target), args, kwargs)
        elif target is not operator.getitem or not isinstance(args[0], (tuple, list, dict)):
            raise NotImplementedError(f"{self.calling.name} runs between the program's compiled calls; every "
                                      "function of a program is captured")
        return super().call_function(target, args, kwargs)


def calls(program, args, functions, **export_options):
    graph = Calls(functions).trace(program)
    for node in graph.nodes:
        if node.op not in ("placeholder", "call_module", "call_function", "output"):
            raise NotImplementedError(f"{node.name} runs between the program's compiled calls; every function of a "
                                      "program is captured")
    module = torch.fx.GraphModule(program, graph)
    captured = Captured(module, functions, export_options)
    captured.run(*args)
    held = {}
    for node in [n for n in module.graph.nodes if n.op == "call_module"]:
        written = tuple(target for kind, target in captured.exported[node.meta["function"]].outputs
                        if kind == OutputKind.BUFFER_MUTATION)
        if written:
            name = f"{node.target.replace('.', '_')}_written"
            if not hasattr(module, name):
                module.add_submodule(name, Written(module.get_submodule(node.target), written))
            with module.graph.inserting_before(node):
                buffers = [held.get((node.target, buffer)) or module.graph.get_attr(f"{node.target}.{buffer}")
                           for buffer in written]
                call = module.graph.call_module(name, (*node.args, *buffers), node.kwargs)
                call.meta = {**node.meta, "written": written}
                node.replace_all_uses_with(module.graph.call_function(operator.getitem, (call, 0)))
                held.update(((node.target, buffer), module.graph.call_function(operator.getitem, (call, k + 1)))
                            for k, buffer in enumerate(written))
            module.graph.erase_node(node)
    module.recompile()
    return module, captured.exported
