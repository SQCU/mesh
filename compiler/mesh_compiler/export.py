# design/data-flow.md
import inspect
import itertools
import operator

import torch
from torch._guards import detect_fake_mode
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorConverter, FakeTensorMode
from torch.export.experimental import _export_forward_backward
from torch.export.graph_signature import InputKind, OutputKind
from torch.fx._symbolic_trace import PH
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.multiprocessing.reductions import StorageWeakRef
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils import _pytree as pytree

from .ir import RETURNED, Call, CapturedProgram, Exported, TrainingResult, written

WITH_EFFECTS = torch.ops.higher_order.with_effects
COPY = torch.ops.aten.copy.default


# design/unequal-shards-change-request.md item 8; torch 2.14.0 torch/export/experimental/__init__.py
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

    state, state_kinds, tokens, user_inputs = {}, {}, [], []
    for spec in signature.input_specs:
        name = spec.arg.name
        if spec.kind == InputKind.USER_INPUT:
            user_inputs.append(name)
        elif spec.kind == InputKind.TOKEN:
            tokens.append(name)
        elif spec.kind == InputKind.PARAMETER or (spec.kind == InputKind.BUFFER and spec.persistent):
            state[name], state_kinds[name] = (spec.target, program.state_dict[spec.target]), spec.kind
        elif spec.kind in (InputKind.BUFFER, InputKind.CONSTANT_TENSOR):
            state[name], state_kinds[name] = (spec.target, program.constants[spec.target]), spec.kind
        else:
            raise NotImplementedError(f"input kind {spec.kind}")

    for node in nodes:
        if node.op == "placeholder" and node.name in state:
            node.meta["state"] = state[node.name][0]

    outputs = []
    for spec in signature.output_specs:
        if spec.kind not in (OutputKind.USER_OUTPUT, OutputKind.LOSS_OUTPUT, OutputKind.BUFFER_MUTATION,
                             OutputKind.USER_INPUT_MUTATION, OutputKind.TOKEN, *GRADIENTS):
            raise NotImplementedError(f"output kind {spec.kind}")
        outputs.append((spec.kind, spec.target))

    # design/compilation-composition-change-request.md items 6 and 7; torch 2.14.0 torch/export/graph_signature.py
    # OutputKind.USER_INPUT_MUTATION (its target the input placeholder) and BUFFER_MUTATION (its target the buffer,
    # its placeholder the state's): the output node's meta "destination", each mutation output's position to its target
    # placeholder's name, the value at that position computing into it (ir.views), a mutation whose value is another
    # placeholder first copying it there (aten.copy)
    graph, stated = program.graph_module.graph, {target: name for name, (target, _) in state.items()}
    output = graph.output_node()
    named, values, destinations = {node.name: node for node in nodes}, list(output.args[0]), {}
    for k, (kind, target) in enumerate(outputs):
        if kind not in (OutputKind.BUFFER_MUTATION, OutputKind.USER_INPUT_MUTATION):
            continue
        destination = named[stated[target] if kind == OutputKind.BUFFER_MUTATION else target]
        value = values[k]
        if value is destination or not isinstance(value, torch.fx.Node) or written(value) is not None:
            continue
        if value.op == "placeholder":
            with graph.inserting_before(output):
                values[k] = graph.call_function(COPY, (destination, value))
            with detect_fake_mode([destination.meta["val"]]):
                values[k].meta["val"] = COPY(destination.meta["val"], value.meta["val"])
        destinations[k] = destination.name
    output.args, output.meta["destination"] = (tuple(values),), destinations
    program.graph_module.recompile()
    return Exported(program.graph_module, state, tokens, user_inputs, outputs,
                    returned(program.call_spec.out_spec, outputs, signature.backward_signature is not None),
                    state_kinds)


# torch 2.14.0 torch/export/graph_signature.py ExportGraphSignature.backward_signature, gradients_to_parameters and
# gradients_to_user_inputs; compiler/mesh_compiler/ir.py TrainingResult: a program's result, its forward outputs, or
# with a loss its TrainingResult, each gradient map keyed by its output's target in output order
GRADIENTS = OutputKind.GRADIENT_TO_PARAMETER, OutputKind.GRADIENT_TO_USER_INPUT


def returned(forward, outputs, backward):
    if not backward:
        return forward
    parameters, inputs = ({target: 0 for kind, target in outputs if kind == role} for role in GRADIENTS)
    return pytree.tree_structure(TrainingResult(pytree.tree_unflatten([0] * forward.num_leaves, forward), parameters,
                                                inputs))


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


def placeholders(function):
    return [node.name for node in function.graph_module.graph.nodes if node.op == "placeholder"]


# torch 2.14.0 torch/_subclasses/fake_tensor.py FakeTensorMode, the capture's one owner of values: the program's
# arguments and each function's state as its fakes, storages shared where the actual ones are; torch.fx Interpreter:
# the program interpreted once at compile time on them, each call's function exported once per key (its target, its
# arguments' structure, each leaf's dtype, device, requires_grad, shape, stride, offset and storage among the call's
# leaves); torch/_functorch/functional_call.py functional_call: the call's source forward on its fake arguments and
# its module's fake parameters and buffers, its user and loss outputs that forward's result leaves (a returned input
# or view of one that input's fake), a mutation output the written fake itself, a token output its exported
# output's own value and each gradient output its exported output's own value in the capture mode by this call's one
# FakeTensorConverter.from_real_tensor (one converter a call, so a call of a reused export has its own gradients and
# the call's gradients keep their views and storages among themselves); the call's value its function's own result (compiler/mesh_compiler/ir.py RETURNED
# outputs under its out_spec, a TrainingResult with a loss), read between calls by item or by a named field; no
# actual state or argument is read for its contents or written
class Captured(torch.fx.Interpreter):
    def __init__(self, module, functions, options, fake):
        super().__init__(module)
        self.functions, self.options, self.fake, self.exported, self.bound = functions, options, fake, {}, {}
        with fake:
            self.token = torch.empty(0)

    def run_node(self, node):
        self.calling = node
        return super().run_node(node)

    def captured(self, target, module, args, kwargs):
        leaves, structure = pytree.tree_flatten((args, kwargs))
        storages = {}
        key = target, str(structure), tuple(
            (str(leaf.dtype), str(leaf.device), leaf.requires_grad, tuple(leaf.shape), tuple(leaf.stride()),
             leaf.storage_offset(), storages.setdefault(StorageWeakRef(leaf.untyped_storage()), len(storages)))
            if isinstance(leaf, torch.Tensor) else repr(leaf) for leaf in leaves)
        if key not in self.exported:
            with torch.enable_grad():
                self.exported[key] = export(module, args, kwargs=kwargs, **self.options)
        function, names = self.exported[key], placeholders(self.exported[key])
        values = [self.fake.from_tensor(function.state[name][1]) if name in function.state
                  else self.token if name in function.tokens else leaves[function.user_inputs.index(name)]
                  for name in names]
        held = {name: self.fake.from_tensor(tensor) for name, tensor in [
            *module.named_parameters(remove_duplicate=False), *module.named_buffers(remove_duplicate=False)]}
        with self.fake, torch.enable_grad():
            result = torch.func.functional_call(module, held, tuple(args), kwargs)
        forward, exported = iter(pytree.tree_leaves(result)), function.graph_module.graph.output_node().args[0]
        stated, converter = {target: name for name, (target, _) in function.state.items()}, FakeTensorConverter()
        produced = tuple(
            next(forward) if kind in (OutputKind.USER_OUTPUT, OutputKind.LOSS_OUTPUT)
            else values[names.index(stated[target] if kind == OutputKind.BUFFER_MUTATION else target)]
            if kind in (OutputKind.BUFFER_MUTATION, OutputKind.USER_INPUT_MUTATION)
            else converter.from_real_tensor(self.fake, exported[k].meta["val"]) if kind in GRADIENTS
            else exported[k].meta["val"] if isinstance(exported[k], torch.fx.Node) else exported[k]
            for k, (kind, target) in enumerate(function.outputs))
        self.calling.meta["function"] = key
        self.bound[self.calling] = key, values, produced
        return pytree.tree_unflatten([value for (kind, _), value in zip(function.outputs, produced, strict=True)
                                      if kind in RETURNED], function.out_spec)

    def call_module(self, target, args, kwargs):
        return self.captured(target, self.fetch_attr(target), args, kwargs)

    def call_function(self, target, args, kwargs):
        if target in self.functions:
            return self.captured(target, Function(target), args, kwargs)
        if target not in (operator.getitem, getattr) or not isinstance(args[0], (tuple, list, dict)):
            raise NotImplementedError(f"{self.calling.name} runs between the program's compiled calls; every "
                                      "function of a program is captured")
        return super().call_function(target, args, kwargs)


# design/aot-compilation-change-request.md A1 and A10; torch 2.14.0 torch/fx/_symbolic_trace.py PH: the program's call
# graph, its placeholders its flat argument leaves
def calls(program, args, functions, **export_options):
    parameters = inspect.signature(program.forward).parameters
    graph = Calls(functions).trace(program, concrete_args={
        name: pytree.tree_map(lambda _: PH, value) for name, value in zip(parameters, args)})
    for node in graph.nodes:
        if node.op not in ("placeholder", "call_module", "call_function", "output"):
            raise NotImplementedError(f"{node.name} runs between the program's compiled calls; every function of a "
                                      "program is captured")
    return capture(torch.fx.GraphModule(program, graph), args, functions, export_options, program)


# design/aot-compilation-change-request.md A1: a module compiled alone, its program one call of it
def called(module, args, **export_options):
    root, graph = torch.nn.Module(), torch.fx.Graph()
    root.function = module
    leaves, spec = pytree.tree_flatten(args)
    graph.output(graph.call_module("function", tuple(pytree.tree_unflatten(
        [graph.placeholder(f"arg_{k}") for k in range(len(leaves))], spec))))
    return capture(torch.fx.GraphModule(root, graph), args, {"function"}, export_options, module)


# design/aot-compilation-change-request.md A1 and A10; torch 2.14.0 torch/export/graph_signature.py InputKind and
# OutputKind: each call node's arguments its function's placeholders in order (the state it reads, its effect tokens,
# its flat arguments), each the outer node holding that value at the call, and each output spec of its function one
# node after it with its role in its function's outputs; a value held by its latest node (an output, or a buffer or
# user input mutation of it, the mutation's value the written fake, Captured.captured), a state placeholder's target
# its tensor's name in `root` (else its call's name and key) and its kind the function's; the program's outputs its
# token chain, each written state or argument with its role and target, and each leaf of its actual result (the calls'
# own results, a training call's TrainingResult among them) a USER_OUTPUT, under that result's structure
def capture(module, args, functions, options, root):
    fake = FakeTensorMode(shape_env=ShapeEnv(tracked_fakes=[]), static_shapes=True, allow_non_fake_inputs=True)
    leaves = [fake.from_tensor(leaf) if isinstance(leaf, torch.Tensor) else leaf for leaf in pytree.tree_leaves(args)]
    graph, interpreter = module.graph, Captured(module, functions, options, fake)
    outcome = graph.process_outputs(interpreter.run(*leaves, enable_io_processing=False))
    named = {id(tensor): name for name, tensor in reversed([*root.named_parameters(remove_duplicate=False),
                                                            *root.named_buffers(remove_duplicate=False)])}
    first, inputs = next(iter(graph.nodes)), [node for node in graph.nodes if node.op == "placeholder"]
    held, roots, chain, written, state, kinds, tokens, bound = {}, {}, [], {}, {}, {}, [], {}

    def hold(node, value):
        node.meta["val"] = value
        if isinstance(value, torch.Tensor):
            held[id(value)] = value, node
        return node

    def holder(value):
        return held[id(value)][1] if isinstance(value, torch.Tensor) else value

    for node, value in zip(inputs, leaves, strict=True):
        roots[id(value)] = hold(node, value)
    for call, (key, values, produced) in interpreter.bound.items():
        function, names, arguments = interpreter.exported[key], placeholders(interpreter.exported[key]), []
        for name, value in zip(names, values):
            if name in function.tokens:
                while len(chain) <= function.tokens.index(name):
                    with graph.inserting_before(first):
                        chain.append(graph.placeholder("token"))
                    chain[-1].meta["val"] = value
                    tokens.append(chain[-1].name)
                arguments.append(chain[function.tokens.index(name)])
                continue
            if name in function.state and id(value) not in held:
                target, tensor = function.state[name]
                path = named.get(id(tensor), f"{call.name}_{target}")
                with graph.inserting_before(first):
                    roots[id(value)] = hold(graph.placeholder(path.replace(".", "_")), value)
                state[roots[id(value)].name], kinds[roots[id(value)].name] = (path, tensor), function.state_kinds[name]
            arguments.append(holder(value))
        call.args, call.kwargs, outputs, token = tuple(arguments), {}, [], 0
        with graph.inserting_before(call.next):
            for k, ((kind, target), value) in enumerate(zip(function.outputs, produced, strict=True)):
                node = graph.call_function(operator.getitem, (call, k))
                if kind == OutputKind.TOKEN:
                    node.meta["val"], chain[token], token = value, node, token + 1
                else:
                    hold(node, value)
                    if kind in (OutputKind.BUFFER_MUTATION, OutputKind.USER_INPUT_MUTATION) and id(value) in roots:
                        written[roots[id(value)]] = node
                outputs.append(node)
        bound[call] = Call(key, arguments, outputs)
    results = [(OutputKind.TOKEN, None, node) for node in chain]
    results += [(OutputKind.BUFFER_MUTATION, state[base.name][0], node) if base.name in state
                else (OutputKind.USER_INPUT_MUTATION, base.name, node) for base, node in written.items()]
    results += [(OutputKind.USER_OUTPUT, None, holder(value)) for value in pytree.tree_leaves(outcome)]
    graph.output_node().args = (tuple(node for *_, node in results),)
    made = {node for call in bound.values() for node in call.outputs}
    for node in reversed(list(graph.nodes)):
        if node.op == "call_function" and node not in bound and node not in made and not node.users:
            graph.erase_node(node)
    graph.set_codegen(torch.fx.graph.CodeGen())
    module.recompile()
    # torch 2.14.0 torchgen/model.py Tag nondeterministic_seeded (OpOverload.tags); J. K. Salmon et al., SC 2011, §3.1:
    # each seeded source operation's draw site, one source namespace numbered from 1 over the captured functions in
    # their order and each one's graph order, once before placement
    sites = itertools.count(1)
    for function in interpreter.exported.values():
        for node in function.graph_module.graph.nodes:
            if node.op == "call_function" and torch.Tag.nondeterministic_seeded in getattr(node.target, "tags", ()):
                node.meta["site"] = next(sites)
    exported = Exported(module, state, tokens, [node.name for node in inputs],
                        [(kind, target) for kind, target, _ in results], pytree.tree_structure(outcome), kinds)
    return CapturedProgram(exported, interpreter.exported, bound)
