# design/data-flow.md
import operator

import torch
from torch._ops import HigherOrderOperator
from torch.export.experimental import _export_forward_backward
from torch.export.graph_signature import InputKind, OutputKind
from torch.multiprocessing.reductions import StorageWeakRef
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils import _pytree as pytree

from .ir import Exported

WITH_EFFECTS = torch.ops.higher_order.with_effects


# design/unequal-shards-change-request.md item 8; torch 2.15 torch/export/experimental/__init__.py
def export(module, args, loss=None, **export_options):
    program = torch.export.export(module, args, **export_options)
    graded = {n.name for n in program.graph.find_nodes(op="placeholder")
              if isinstance(n.meta.get("val"), torch.Tensor) and n.meta["val"].requires_grad}
    if loss is not None:
        with sdpa_kernel(SDPBackend.MATH):
            program = _export_forward_backward(program, joint_loss_index=loss)
    return read(program.run_decompositions(), graded)


# torch 2.15 torch/export/exported_program.py _decompose_and_get_gm_with_new_signature_constants, joint outputs;
# torch/_functorch/_aot_autograd/input_output_analysis.py create_graph_signature, gradients_to_user_inputs
def read(program, graded=frozenset()):
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

    graded = iter([name for name in user_inputs if name in graded])
    outputs = [(spec.kind, next(graded) if spec.kind == OutputKind.GRADIENT_TO_USER_INPUT else spec.target)
               for spec in signature.output_specs]
    out_spec = program.call_spec.out_spec
    if signature.backward_signature is not None:
        forward = pytree.tree_unflatten([0] * out_spec.num_leaves, out_spec)
        out_spec = pytree.tree_structure((forward, *({key: 0 for kind, key in outputs if kind == of} for of in (
            OutputKind.GRADIENT_TO_PARAMETER, OutputKind.GRADIENT_TO_USER_INPUT))))
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


class Stated(torch.nn.Module):
    def __init__(self, function, read, written):
        super().__init__()
        self.function, self.read, self.written = function, read, written

    def forward(self, *args, **kwargs):
        result = self.function(*args[:len(args) - len(self.read)], **kwargs)
        return (result, *(self.function.get_buffer(name) for name in self.written)) if self.written else result


class Captured(torch.fx.Interpreter):
    def __init__(self, module, functions, options):
        super().__init__(module)
        self.functions, self.options, self.exported = functions, options, {}

    def run_node(self, node):
        self.calling = node
        return super().run_node(node)

    def captured(self, target, module, args, kwargs):
        leaves, structure = pytree.tree_flatten((args, kwargs))
        key = target, structure, specialized(leaves)
        if key not in self.exported:
            with torch.enable_grad():
                self.exported[key] = export(module, args, kwargs=kwargs, **self.options)
        self.calling.meta["function"] = key

    def call_module(self, target, args, kwargs):
        module = self.fetch_attr(target)
        self.captured(target, module, args, kwargs)
        before = attributes(module)
        result = super().call_module(target, args, kwargs)
        after = attributes(module)
        written = {key for kind, key in self.exported[self.calling.meta["function"]].outputs
                   if kind in (OutputKind.BUFFER_MUTATION, OutputKind.PARAMETER_MUTATION)}
        rebound = [name for name in before if after.get(name) is not before[name] and name not in written]
        if rebound:
            raise NotImplementedError(f"{self.calling.name} rebinds {rebound[0]}, which torch.export captures as no "
                                      "write; a function's writes are its node's results")
        self.calling.meta["held"] = before, after
        return result

    def call_function(self, target, args, kwargs):
        if target in self.functions:
            self.captured(target, Function(target), args, kwargs)
        elif target is not operator.getitem or not isinstance(args[0], (tuple, list, dict)):
            raise NotImplementedError(f"{self.calling.name} runs between the program's compiled calls; every "
                                      "function of a program is captured")
        return super().call_function(target, args, kwargs)


def attributes(module):
    return {**dict(module.named_parameters()), **dict(module.named_buffers())}


# torch 2.15 torch/_subclasses/meta_utils.py MetaConverter.meta_storage, its StorageWeakRef memo;
# torch/_dynamo/guards.py TENSOR_MATCH and EQUALS_MATCH
def specialized(leaves):
    storages = {}

    def operand(leaf):
        if not isinstance(leaf, torch.Tensor):
            return type(leaf), leaf
        held = (storages.setdefault(StorageWeakRef(leaf.untyped_storage()), len(storages)),
                leaf.untyped_storage().nbytes(), leaf.storage_offset(), leaf.stride()) if (
            leaf.layout == torch.strided) else ()
        return (type(leaf), leaf.layout, tuple(leaf.shape), *held, leaf.dtype, leaf.device, leaf.requires_grad,
                leaf.is_conj(), leaf.is_neg())
    return tuple(map(operand, leaves))


def calls(program, args, functions, **export_options):
    graph = Calls(functions).trace(program)
    for node in graph.nodes:
        if node.op not in ("placeholder", "call_module", "call_function", "output"):
            raise NotImplementedError(f"{node.name} runs between the program's compiled calls; every function of a "
                                      "program is captured")
    module = torch.fx.GraphModule(program, graph)
    captured = Captured(module, functions, export_options)
    captured.run(*args)
    held, last = {}, {}
    for node in [n for n in module.graph.nodes if "function" in n.meta]:
        function, called = captured.exported[node.meta["function"]], node.op == "call_module"
        if not function.state:
            continue
        leaf = module.get_submodule(node.target) if called else Function(node.target)
        state = dict(function.state.values())
        before, after = node.meta.pop("held", ({}, {}))
        written = tuple(key for kind, key in function.outputs
                        if kind in (OutputKind.BUFFER_MUTATION, OutputKind.PARAMETER_MUTATION))
        paths = {key: f"{node.target}.{key}" if key in before else f"{node.name}_{key}" for key in state}
        module.add_submodule(f"{node.name}_stated", Stated(leaf, tuple(state), written))
        with module.graph.inserting_before(node):
            for key in state.keys() - before.keys():
                module.register_buffer(paths[key], state[key], persistent=False)
            read = {key: before.get(key, tensor) for key, tensor in state.items()}
            taken = [held[id(tensor)][1] if id(tensor) in held else module.graph.get_attr(paths[key])
                     for key, tensor in read.items()]
            call = module.graph.call_module(f"{node.name}_stated", (*node.args, *taken), node.kwargs)
            call.meta = {**node.meta, "read": tuple(state), **({"written": written} if written else {})}
            node.replace_all_uses_with(module.graph.call_function(operator.getitem, (call, 0)) if written else call)
            for (key, tensor), value in zip(read.items(), taken):
                held.setdefault(id(tensor), (tensor, value))
                if key in written:
                    value = module.graph.call_function(operator.getitem, (call, 1 + written.index(key)))
                    last.pop(id(tensor), None)
                    held[id(after[key])] = after[key], value
                    last[id(after[key])] = paths[key], value
        module.graph.erase_node(node)
    # torch 2.15 torch/export/graph_signature.py OutputKind.BUFFER_MUTATION
    if last:
        output = module.graph.output_node()
        output.args = ((output.args[0], *(value for _, value in last.values())),)
        module.meta["written"] = tuple(path for path, _ in last.values())
    module.recompile()
    return module, captured.exported
