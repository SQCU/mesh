# design/data-flow.md
import dataclasses
import functools
import inspect
import itertools
import math
import operator

import torch
from torch._export.non_strict_utils import make_constraints
from torch._export.passes.lift_constants_pass import lift_constants_pass
from torch._export.utils import _collect_and_set_constant_attrs, node_inline_
from torch._guards import detect_fake_mode
from torch._higher_order_ops.map import map_dense
from torch._higher_order_ops.scan import generic_scan
from torch._subclasses.fake_tensor import FakeTensorConverter, FakeTensorMode, extract_tensor_metadata
from torch.export._trace import _get_inline_constraints
from torch.export.experimental import _export_forward_backward
from torch.export.graph_signature import ConstantArgument, InputKind, OutputKind
from torch.fx._symbolic_trace import PHWithMeta
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.multiprocessing.reductions import StorageWeakRef
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils import _pytree as pytree

from .ir import RETURNED, Call, CapturedProgram, Exported, TrainingResult, written
from .operations import LSTSQ_FIXED

COPY = torch.ops.aten.copy.default
# torchgen/gen_functionalization_type.py gen_composite_view_copy_kernel: a view_copy operator's registered kernel key
VIEW_COPY = torch._C.DispatchKey.CompositeExplicitAutogradNonFunctional
# pub13 section 5.1 case (i): an eigh outlined on a non-CPU engine (compiler/mesh_compiler/decompose.py outline) draws
# its Haar restarts at its own source site, carried to its QDWH steps as data (operations.QDWH_EIGH_STEP `site`); a
# pinv, svd or lstsq (each overload, and compiler/mesh_compiler/operations.py LSTSQ_FIXED, the fixed-result lstsq) is a
# draw source too, its expanded svd and eigh keeping its declared site
DRAWING = (torch.ops.aten._linalg_eigh, torch.ops.aten.linalg_pinv, torch.ops.aten._linalg_svd,
           torch.ops.aten.linalg_lstsq, LSTSQ_FIXED.overloadpacket)


# torch 2.14.0 torch/_higher_order_ops/scan.py generic_scan: the scan HOP's operands (combine_fn, init, xs,
# additional_inputs) in generic_scan's signature
def scanned(combine_fn, init, xs, additional_inputs):
    return generic_scan(combine_fn, init, xs, additional_inputs=additional_inputs)


# torch 2.14.0 torch/_higher_order_ops/map.py map_dense: each fixed map or scan call's upstream dense owner
STATIC = {torch.ops.higher_order.map_impl: map_dense, torch.ops.higher_order.scan: scanned}


def attribute(graph_module, target):
    return functools.reduce(getattr, target.split("."), graph_module)


# torch 2.14.0 torch/fx/experimental/proxy_tensor.py make_fx and torch/_export/utils.py node_inline_: one fixed map's
# or scan's call expanded once in place, its dense owner traced on the call's own fake tensor operands (each tensor
# Node operand a placeholder, every other operand, a body attribute or a literal, the actual captured one, through
# pytree), inlined over the call's operand Nodes (its getitems the traced results, the traced attributes hoisted), each
# made node keeping the call's source metadata, the call's unused body attribute dropped
def expand(graph_module, node):
    flat, spec = pytree.tree_flatten((node.args, node.kwargs))
    held = [k for k, item in enumerate(flat) if isinstance(item, torch.fx.Node)
            and isinstance(item.meta.get("val"), torch.Tensor)]
    given = [attribute(graph_module, item.target) if isinstance(item, torch.fx.Node) and item.op == "get_attr" else item
             for item in flat]

    def operation(*operands):
        values = list(given)
        for k, operand in zip(held, operands):
            values[k] = operand
        args, kwargs = pytree.tree_unflatten(values, spec)
        return STATIC[node.target](*args, **kwargs)
    operands = [flat[k].meta["val"] for k in held]
    with detect_fake_mode(operands):
        traced = make_fx(operation, tracing_mode="fake")(*operands)
    name = next(f"expanded_{i}" for i in itertools.count() if not hasattr(graph_module, f"expanded_{i}"))
    graph_module.add_submodule(name, traced)
    graph, before = graph_module.graph, set(graph_module.graph.nodes)
    with graph.inserting_before(node):
        call = graph.call_module(name, tuple(flat[k] for k in held))
    node.replace_all_uses_with(call)
    graph.erase_node(node)
    for item in flat:
        if isinstance(item, torch.fx.Node) and item.op == "get_attr" and not item.users:
            graph.erase_node(item)
    node_inline_(call)
    for made in [n for n in graph.nodes if n not in before]:
        for key, value in node.meta.items():
            made.meta.setdefault(key, value)


# torch 2.14.0 torch/_export/passes/lift_constants_pass.py lift_constants_pass and torch/_export/utils.py
# _collect_and_set_constant_attrs: a tensor attribute of the program's graph lifted into its signature and constants
def lifted(program, module):
    graph_module = program.graph_module
    if any(n.op == "get_attr" and isinstance(attribute(graph_module, n.target), torch.Tensor)
           for n in graph_module.graph.nodes):
        attributes = _collect_and_set_constant_attrs(program.graph_signature, program.constants, module)
        program.constants.update(lift_constants_pass(graph_module, program.graph_signature, attributes))
        graph_module.recompile()
    return program


# design/unequal-shards-change-request.md item 8; design/torch-source-lowering-change-request.md item 2; torch 2.14.0
# torch/export/experimental/__init__.py _export_forward_backward and torch/export/exported_program.py
# run_decompositions and default_decompositions; torch/_ops.py OpOverload has_kernel_for_dispatch_key and _op_dk;
# torchgen/gen_functionalization_type.py gen_composite_view_copy_kernel: the program's one capture, torch.export's
# strict one unless the caller states `strict` (compiler/patches/torch-lstsq-fixed-results.patch), and its one
# normalization, before it is read: each fixed map or scan call expanded (`expand`, each call once, a call the expansion exposes after it), its
# exposed tensor attributes lifted, then its forward and backward (a loss) and run_decompositions once over the default
# table with each captured view_copy operator (torch.Tag.view_copy) that has a CompositeExplicitAutogradNonFunctional
# kernel decomposed by that registered kernel (its view, then a clone), and any tensor attribute left lifted; the alias
# record (`aot_aliases`) the first AOT result's, with a loss the forward and backward's, popped before
# run_decompositions, else run_decompositions'; run_decompositions' own record, the final program's, its `decomposed`
# (with a loss a distinct stage result, else the same record); the export's own stage-2 callable (`aot_compiled`,
# compiler/patches/torch-export-alias-metadata.patch) popped from the first program before any later pass, its
# `compiled`
def export(module, args, loss=None, kwargs=None, strict=True):
    kwargs = {} if kwargs is None else kwargs
    # torch 2.14.0 torch/export/_trace.py _export_to_torch_ir (strict Dynamo capture, restore_fqn False as
    # _strict_export passes it) and _lstsq_fixed_selected (compiler/patches/torch-lstsq-fixed-results.patch,
    # _strict_export's own selection), its source names by _get_param_buffer_mapping; else the module itself
    source = _export_to_torch_ir(module, args, kwargs, restore_fqn=False, _log_export_usage=False) if strict else module
    if strict:
        _lstsq_fixed_selected(source)
    names = _get_param_buffer_mapping(module, source) if strict else {}
    # torch/_functorch/aot_autograd.py aot_export_joint_with_descriptors and its companion
    # aot_compile_joint_with_descriptors, both inside the one ExitStack the joint export requires live through its
    # compilation, over torch.export.default_decompositions; each compiler the companion's own default
    # (boxed_nop_preserve_node_meta, an abstract compilation) recording the graph and arguments it is given and, for
    # each call of what it returns, the arguments that call receives (the forward's: its graph's own placeholder values)
    stages = {"forward": [], "backward": []}

    def recording(stage):
        def compiler(graph_module, example_inputs):
            made = boxed_nop_preserve_node_meta(graph_module, example_inputs)

            def called(arguments):
                stages[stage][-1][2].append(list(arguments))
                return made(arguments)
            called._boxed_call = True
            stages[stage].append((graph_module, example_inputs, []))
            return called
        return compiler
    with contextlib.ExitStack() as stack, sdpa_kernel(SDPBackend.MATH):
        joint = aot_export_joint_with_descriptors(stack, source, args, kwargs,
                                                  decompositions=torch.export.default_decompositions())
        graph_module = joint.graph_module
        while pending := [n for n in graph_module.graph.nodes if n.op == "call_function" and n.target in STATIC]:
            expand(graph_module, pending[0])
        graph_module.recompile()
        compiled = aot_compile_joint_with_descriptors(joint, fw_compiler=recording("forward"),
                                                      bw_compiler=recording("backward"))
    return read(module, joint, names, compiled, stages, loss)


# compiler/patches/torch-export-alias-metadata.patch (torch/_functorch/_aot_autograd/input_output_analysis.py
# create_graph_signature `aot_aliases`, carried by torch/export/experimental/__init__.py
# _copy_graph_module_and_signature); torch/export/graph_signature.py InputSpec: an AOT result's retained forward
# record, popped from that program once: its ViewAndMutationMeta, each AOT input's role by that program's own input spec
# of the input's Node (a parameter, buffer or constant its InputKind and target, a user input USER_INPUT and its
# position among the user inputs), and each AOT input's and forward output's prototype value (no Node kept)
def aot_aliases(program):
    metadata, inputs, outputs = program.graph_module.meta.pop("aot_aliases")
    specs = {spec.arg.name: spec for spec in program.graph_signature.input_specs}
    users = [spec.arg.name for spec in program.graph_signature.input_specs if spec.kind == InputKind.USER_INPUT]
    roles = tuple((specs[node.name].kind, users.index(node.name) if specs[node.name].kind == InputKind.USER_INPUT
                   else specs[node.name].target) for node in inputs)
    return metadata, roles, *(tuple(leaf.meta.get("val") if isinstance(leaf, torch.fx.Node) else leaf for leaf in leaves)
                               for leaves in (inputs, outputs))


def read(program, aliases=None, decomposed=None, compiled=None):
    graph_module, signature = program.graph_module, program.graph_signature
    nodes = list(graph_module.graph.nodes)

    # rejection filters are forbidden, even in temporary integration validation: a dimension, input kind or output
    # kind this owner cannot compile raises here
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
    # torch 2.14.0 torch/export/graph_signature.py ConstantArgument: each literal user input the export specialized,
    # its placeholder's name and value as the input spec gives them
    specialized = tuple((spec.arg.name, spec.arg.value) for spec in signature.input_specs
                        if isinstance(spec.arg, ConstantArgument))

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
        if value is destination or not isinstance(value, torch.fx.Node) or written(value):
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
                    state_kinds, aliases, specialized=specialized, decomposed=decomposed, signature=signature,
                    range_constraints=program.range_constraints, compiled=compiled)


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
# torch 2.14.0 torch/fx/_symbolic_trace.py Tracer.create_args_for_root, which names a VAR_POSITIONAL parameter's
# placeholder "*" + name and a VAR_KEYWORD's "**" + name, and Tracer._proxy_placeholder, which looks the original name
# up in concrete_args and the Signature: each placeholder's display name mapped once to its parameter's own name
VARIADIC = {inspect.Parameter.VAR_POSITIONAL: "*", inspect.Parameter.VAR_KEYWORD: "**"}


class Calls(torch.fx.Tracer):
    def __init__(self, functions, parameters):
        super().__init__(autowrap_functions=tuple(f for f in functions if callable(f)))
        self.functions = functions
        self.names = {VARIADIC.get(parameter.kind, "") + name: name for name, parameter in parameters.items()}

    def is_leaf_module(self, module, qualified_name):
        return qualified_name in self.functions

    def _proxy_placeholder(self, name, concrete_args, sig, fn_for_analysis):
        return super()._proxy_placeholder(self.names[name], concrete_args, sig, fn_for_analysis)


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
# arguments' structure, each leaf's dtype, device, requires_grad, shape, stride, offset, storage among the call's
# leaves and tensor object among them: torch/_functorch/_aot_autograd/runtime_wrappers.py AOTDedupeWrapper, a trace of
# f(x, x) valid only for a call passing one tensor so, distinct aliasing tensors another case, by tensor identity);
# compiler/mesh_compiler/ir.py Exported.aliases and torch/_functorch/_aot_autograd/runtime_wrappers.py
# _RuntimeForwardEpilogue (capture_orig_inputs, finalize: input mutations, make_output_handler alias replay through
# gen_alias_from_base and the retained ViewMetaSequence): the call's actual fakes in the AOT input order (each state
# target's fake, each user input's leaf), the AOT outputs (mutated inputs, forward outputs, intermediate bases) the
# prototypes realized by this call's one FakeTensorConverter (seeded with each input prototype's actual fake), the
# epilogue applying the mutations to the call's fakes and returning the forward outputs (the intermediate bases only its
# alias operands); its user and loss outputs those values, a mutation output the written fake itself, a token output its exported
# output's own value and each gradient output its exported output's own value through the same converter (so a call
# of a reused export has its own results, keeping their views and storages among themselves); the call's value its
# function's own result (compiler/mesh_compiler/ir.py RETURNED outputs under its out_spec, a TrainingResult with a
# loss), read between calls by item or by a named field; an attribute the program reads its actual value, a tensor's
# the capture mode's fake of it (its target and real tensor recorded); no source forward is run and no actual state or
# argument is read for its contents or written
def described(tensor):
    metadata = extract_tensor_metadata(tensor)
    return tuple(getattr(metadata, field.name) for field in dataclasses.fields(metadata))


class Captured(torch.fx.Interpreter):
    def __init__(self, module, functions, options, fake):
        super().__init__(module)
        self.functions, self.options, self.fake, self.exported, self.bound = functions, options, fake, {}, {}
        self.attributes, self.specialized = {}, {}
        with fake:
            self.token = torch.empty(0)

    def run_node(self, node):
        self.calling = node
        return super().run_node(node)

    def get_attr(self, target, args, kwargs):
        value = super().get_attr(target, args, kwargs)
        if not isinstance(value, torch.Tensor):
            return value
        self.attributes[self.calling] = target, value
        return self.fake.from_tensor(value)

    # one call's function key: its target, the call's own TreeSpec (torch/utils/_pytree.py TreeSpec, its own equality
    # and hash), each Tensor leaf's every torch/_subclasses/fake_tensor.py extract_tensor_metadata field (the FakeTensor
    # dispatch cache's own record) with its storage's and its object's first-seen positions among the call's leaves
    # (torch/_functorch/_aot_autograd/runtime_wrappers.py AOTDedupeWrapper's object identity, merge_view_inputs' shared
    # storage), each other leaf its type and value (torch/_export/utils.py _check_input_constraints_for_graph's own
    # literal contract: the type itself, then equality)
    def captured(self, target, module, args, kwargs):
        leaves, structure = pytree.tree_flatten((args, kwargs))
        storages, objects = {}, {}
        key = target, structure, tuple(
            (described(leaf), storages.setdefault(StorageWeakRef(leaf.untyped_storage()), len(storages)),
             objects.setdefault(id(leaf), len(objects)))
            if isinstance(leaf, torch.Tensor) else (type(leaf), leaf) for leaf in leaves)
        if key not in self.exported:
            with torch.enable_grad():
                self.exported[key] = export(module, args, kwargs=kwargs, **self.options)
        function, names = self.exported[key], placeholders(self.exported[key])
        # torch/utils/_pytree.py TreeSpec.flatten_up_to: each of the call's leaves under the original FX argument it
        # came from (the calling node's own args and kwargs, before capture replaces them); each ConstantArgument user
        # input of the function (Exported.specialized) recorded on that original argument with its compiled literal
        sources, layout = pytree.tree_flatten((self.calling.args, self.calling.kwargs))
        origins = [source for source, part in zip(sources, layout.flatten_up_to((args, kwargs)), strict=True)
                   for _ in pytree.tree_leaves(part)]
        for name, value in function.specialized:
            self.specialized[origins[function.user_inputs.index(name)]] = value
        values = [self.fake.from_tensor(function.state[name][1]) if name in function.state
                  else self.token if name in function.tokens else leaves[function.user_inputs.index(name)]
                  for name in names]
        # the call's forward results: the export's own stage-2 callable (Exported.compiled: its original wrappers and
        # calling convention, RuntimeWrapper's epilogue among them) over the function's parameters, then its buffers,
        # each in its input spec order (JointWithDescriptors params_spec and buffers_spec's order), then the call's
        # leaves; each AOT input's prototype that input (FakeTensorConverter memo: a gradient output's value)
        _, roles, prototypes, _ = function.aliases
        tensors, converter = dict(function.state.values()), FakeTensorConverter()
        actual = [leaves[k] if kind == InputKind.USER_INPUT else self.fake.from_tensor(tensors[k]) for kind, k in roles]
        for prototype, value in zip(prototypes, actual, strict=True):
            if isinstance(prototype, torch.Tensor) and isinstance(value, torch.Tensor):
                converter.set_tensor_memo(prototype, value)
        state = [self.fake.from_tensor(tensor) for kind in (InputKind.PARAMETER, InputKind.BUFFER)
                 for name, (_, tensor) in function.state.items() if function.state_kinds[name] == kind]
        with self.fake, torch.enable_grad():
            forward = iter(function.compiled(*state, *leaves))
        exported = function.graph_module.graph.output_node().args[0]
        stated = {target: name for name, (target, _) in function.state.items()}
        produced = tuple(
            next(forward) if kind in (OutputKind.USER_OUTPUT, OutputKind.LOSS_OUTPUT)
            else values[names.index(stated[target] if kind == OutputKind.BUFFER_MUTATION else target)]
            if kind in (OutputKind.BUFFER_MUTATION, OutputKind.USER_INPUT_MUTATION)
            else converter.from_real_tensor(self.fake, exported[k].meta["val"]) if kind in GRADIENTS
            else exported[k].meta["val"] if isinstance(exported[k], torch.fx.Node) else exported[k]
            for k, (kind, target) in enumerate(function.outputs))
        self.calling.meta["function"] = key
        # each function placeholder's original FX argument where it is a user input (its leaf's origin), the call's
        # own source provenance of that value (capture: the outer placeholder it read, not merely its value's holder)
        self.bound[self.calling] = key, values, produced, {
            name: origins[function.user_inputs.index(name)] for name in names if name in function.user_inputs}
        return pytree.tree_unflatten([value for (kind, _), value in zip(function.outputs, produced, strict=True)
                                      if kind in RETURNED], function.out_spec)

    def call_module(self, target, args, kwargs):
        return self.captured(target, self.fetch_attr(target), args, kwargs)

    def call_function(self, target, args, kwargs):
        if target in self.functions:
            return self.captured(target, Function(target), args, kwargs)
        # rejection filters are forbidden: an operation between calls that no captured function owns raises here
        if target not in (operator.getitem, getattr) or not isinstance(args[0], (tuple, list, dict)):
            raise NotImplementedError(f"{self.calling.name} runs between the program's compiled calls; every "
                                      "function of a program is captured")
        return super().call_function(target, args, kwargs)


# design/aot-compilation-change-request.md A1 and A10; torch 2.14.0 torch/fx/_symbolic_trace.py PHWithMeta,
# _transfer_attrs and Tracer.create_args_for_root (a placeholder for every forward parameter, variadics last); Python
# inspect Signature.bind and BoundArguments.apply_defaults: the program's call graph, its placeholders the flat leaves
# of every forward parameter's bound value (an omitted default's its default), each placeholder Node's `ordinal` its
# leaf's position in those bound values' flat order
def calls(program, args, functions, **export_options):
    bound = inspect.signature(program.forward).bind(*args)
    bound.apply_defaults()
    ordinals = itertools.count()

    def leaf(_):
        marked = PHWithMeta()
        marked.ordinal = next(ordinals)
        return marked
    graph = Calls(functions, bound.signature.parameters).trace(program, concrete_args={
        name: pytree.tree_map(leaf, value) for name, value in bound.arguments.items()})
    # rejection filters are forbidden: a traced node of no capture owner raises here, never dropped from the program.
    # torch 2.14.0 torch/fx/_symbolic_trace.py replace_ph (a flat placeholder's args its parameter's default) and
    # torch/fx/graph.py CodeGen free_vars: every flat placeholder here is an already-bound leaf (bind and apply_defaults
    # above; the defaults the original Signature's, Exported.interface), so none keeps a default argument
    for node in graph.nodes:
        if node.op not in ("placeholder", "get_attr", "call_module", "call_function", "output"):
            raise NotImplementedError(f"{node.name} runs between the program's compiled calls; every function of a "
                                      "program is captured")
        if node.op == "placeholder":
            node.args = ()
    return capture(torch.fx.GraphModule(program, graph), bound, functions, export_options, program)


# design/aot-compilation-change-request.md A1; Python inspect Signature.bind, BoundArguments.apply_defaults, args and
# kwargs: a module compiled alone, its program one call of it; its placeholders the flat leaves of every forward
# parameter's bound value (an omitted default's its default) in parameter order, each placeholder Node's `ordinal` its
# leaf's position there, passed by the BoundArguments of those placeholders' parameter tree as its positional and
# keyword operands; capture's binding the original real bound values
def called(module, args, **export_options):
    root, graph = torch.nn.Module(), torch.fx.Graph()
    root.function = module
    bound = inspect.signature(module.forward).bind(*args)
    bound.apply_defaults()
    leaves, spec = pytree.tree_flatten(tuple(bound.arguments.values()))

    def placeholder(k):
        node = graph.placeholder(f"arg_{k}")
        node.ordinal = k
        return node
    given = inspect.BoundArguments(bound.signature, dict(zip(bound.arguments, pytree.tree_unflatten(
        [placeholder(k) for k in range(len(leaves))], spec))))
    graph.output(graph.call_module("function", given.args, given.kwargs))
    return capture(torch.fx.GraphModule(root, graph), bound, {"function"}, export_options, module)


# design/aot-compilation-change-request.md A1 and A10; torch 2.14.0 torch/export/graph_signature.py InputKind and
# OutputKind: each call node's arguments its function's placeholders in order (the state it reads, its effect tokens,
# its flat arguments), each the outer node holding that value at the call, and each output spec of its function one
# node after it with its role in its function's outputs; a value held by its latest node (an output, or a buffer or
# user input mutation of it, the mutation's value the written fake, Captured.captured), a state placeholder's target
# its tensor's name in `root` (else its call's name and key) and its kind the function's; each tensor attribute the
# program reads (Captured.get_attr, torch.fx Interpreter get_attr) its fake's holder: a state placeholder of its name
# in `root` (else its attribute target) and its role there (PARAMETER, BUFFER, else CONSTANT_TENSOR, its real tensor
# kept), unless that fake is already held, so a written attribute is a BUFFER_MUTATION; the program's outputs its
# token chain, each written state or argument with its role and target, and each leaf of its actual result (the calls'
# own results, a training call's TrainingResult among them) a USER_OUTPUT, under that result's structure; its user
# leaves those of `bound` (Python inspect BoundArguments of `root.forward`), and its call interface (Exported.interface)
# the Signature of exactly the bound parameters (their own Parameter objects, kinds and defaults) and the TreeSpec of
# the bound values in parameter order
def capture(module, bound, functions, options, root):
    fake = FakeTensorMode(shape_env=ShapeEnv(tracked_fakes=[]), static_shapes=True, allow_non_fake_inputs=True)
    values, combined = tuple(bound.arguments.values()), dict(bound.arguments)
    interface = (inspect.Signature([bound.signature.parameters[name] for name in bound.arguments]),
                 pytree.tree_structure(values))
    leaves = [fake.from_tensor(leaf) if isinstance(leaf, torch.Tensor) else leaf for leaf in pytree.tree_leaves(values)]
    graph, interpreter = module.graph, Captured(module, functions, options, fake)
    # `first` stays the insertion anchor for every new state and token placeholder: an attribute's get_attr node, which
    # may be it, keeps its place until the final cleanup erases it unused; each placeholder Node's `ordinal` (calls'
    # PHWithMeta, called's own) is its leaf's position in `leaves`, so the FX placeholders take their leaves by it and
    # the user inputs are those placeholders in leaf order
    first, inputs = next(iter(graph.nodes)), [node for node in graph.nodes if node.op == "placeholder"]
    users = sorted(inputs, key=lambda node: node.ordinal)
    outcome = graph.process_outputs(interpreter.run(*(leaves[node.ordinal] for node in inputs),
                                                    enable_io_processing=False))
    named = {id(tensor): name for name, tensor in reversed([*root.named_parameters(remove_duplicate=False),
                                                            *root.named_buffers(remove_duplicate=False)])}
    held, roots, chain, written, state, kinds, tokens, bound = {}, {}, [], {}, {}, {}, [], {}

    def hold(node, value):
        node.meta["val"] = value
        if isinstance(value, torch.Tensor):
            held[id(value)] = value, node
        return node

    def holder(value):
        return held[id(value)][1] if isinstance(value, torch.Tensor) else value

    for node, value in zip(users, leaves, strict=True):
        roots[id(value)] = hold(node, value)
    parameters = {id(tensor) for _, tensor in root.named_parameters(remove_duplicate=False)}
    buffers = {id(tensor) for _, tensor in root.named_buffers(remove_duplicate=False)}
    for node, (target, tensor) in interpreter.attributes.items():
        value = fake.from_tensor(tensor)
        if id(value) not in held:
            path = named.get(id(tensor), target)
            with graph.inserting_before(first):
                roots[id(value)] = hold(graph.placeholder(path.replace(".", "_")), value)
            state[roots[id(value)].name] = path, tensor
            kinds[roots[id(value)].name] = (InputKind.PARAMETER if id(tensor) in parameters else InputKind.BUFFER
                                            if id(tensor) in buffers else InputKind.CONSTANT_TENSOR)
        node.replace_all_uses_with(holder(value))
    # a user input value's node for one call: the outer placeholder the call's own FX argument names where that
    # placeholder still holds the value (no mutation has re-held it), so a tensor the caller passes in several positions
    # keeps each call's own position (torch.export's AOTDedupeWrapper then treating a call's duplicate leaves as its
    # own); else the value's latest holder
    def provenance(value, origin):
        if (isinstance(origin, torch.fx.Node) and origin.op == "placeholder" and origin.meta.get("val") is value
                and isinstance(value, torch.Tensor) and held[id(value)][1].op == "placeholder"):
            return origin
        return holder(value)
    for call, (key, values, produced, sources) in interpreter.bound.items():
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
            arguments.append(provenance(value, sources[name]) if name in sources else holder(value))
        call.args, call.kwargs, outputs, token = tuple(arguments), {}, [], 0
        with graph.inserting_before(call.next):
            for k, ((kind, target), value) in enumerate(zip(function.outputs, produced, strict=True)):
                node = graph.call_function(operator.getitem, (call, k))
                if kind == OutputKind.TOKEN:
                    node.meta["val"], chain[token], token = value, node, token + 1
                else:
                    hold(node, value)
                    if kind in (OutputKind.BUFFER_MUTATION, OutputKind.USER_INPUT_MUTATION) and id(value) in roots:
                        # a user input mutation's written base the outer node this call passed for that input (its
                        # own provenance), a buffer's its state placeholder
                        given = arguments[names.index(target)] if kind == OutputKind.USER_INPUT_MUTATION else None
                        written[given if isinstance(given, torch.fx.Node) and given.op == "placeholder"
                                else roots[id(value)]] = node
                outputs.append(node)
        bound[call] = Call(key, arguments, outputs)
    results = [(OutputKind.TOKEN, None, node) for node in chain]
    results += [(OutputKind.BUFFER_MUTATION, state[base.name][0], node) if base.name in state
                else (OutputKind.USER_INPUT_MUTATION, base.name, node) for base, node in written.items()]
    results += [(OutputKind.USER_OUTPUT, None, holder(value)) for value in pytree.tree_leaves(outcome)]
    graph.output_node().args = (tuple(node for *_, node in results),)
    made = {node for call in bound.values() for node in call.outputs}
    for node in reversed(list(graph.nodes)):
        if node.op in ("call_function", "get_attr") and node not in bound and node not in made and not node.users:
            graph.erase_node(node)
    graph.set_codegen(torch.fx.graph.CodeGen())
    module.recompile()
    # torch 2.14.0 torchgen/model.py Tag nondeterministic_seeded (OpOverload.tags); J. K. Salmon et al., SC 2011, §3.1:
    # each seeded source operation's draw site, and each source eigh's (DRAWING), one source namespace numbered from 1
    # over the captured functions in their order and each one's graph order, once before placement
    sites = itertools.count(1)
    for function in interpreter.exported.values():
        for node in function.graph_module.graph.nodes:
            if node.op == "call_function" and (torch.Tag.nondeterministic_seeded in getattr(node.target, "tags", ())
                                               or getattr(node.target, "overloadpacket", None) in DRAWING):
                node.meta["site"] = next(sites)
    # Exported.specialized: each outer user placeholder some call's ConstantArgument input came from (Captured.captured),
    # once, in user input order; a call's literal from a source constant or a prior call's result is no user input.
    # Exported.range_constraints: torch/export/_trace.py's own sequence (_get_inline_constraints into the GraphModule's
    # inline_constraints, then _get_range_constraints' make_constraints, no dynamic shapes stated) over this capture's
    # FakeTensorMode, the one every call's export traced in, its non-user placeholders the state and tokens first
    module.meta["inline_constraints"] = _get_inline_constraints(fake)
    exported = Exported(module, state, tokens, [node.name for node in users],
                        [(kind, target) for kind, target, _ in results], pytree.tree_structure(outcome), kinds,
                        interface=interface, specialized=tuple((node.name, interpreter.specialized[node])
                                                               for node in users if node in interpreter.specialized),
                        range_constraints=make_constraints(fake, module, combined, None, len(state) + len(tokens)))
    return CapturedProgram(exported, interpreter.exported, bound)


# design/aot-compilation-change-request.md A10; compiler/mesh_compiler/decompose.py outline: every call of function
# `key` replaced, in place in the program's call graph, by the calls of `parts` in order, each (key, Exported, inputs,
# results): an input ("call", i) the original call's input i or ("part", p, o) part p's output o; `results` {output o:
# the original call's output k}, whose outer node (its readers, a mutation's role) becomes part output o, every other
# output a new outer node; the program's other calls, functions and outer graph kept. torch 2.14.0
# torch/_subclasses/fake_tensor.py FakeTensorConverter tensor_memo, set_tensor_memo and from_real_tensor, and
# torch/_subclasses/meta_utils.py MetaConverter set_storage_memo and MetaTensorDescriber get_tensor_id,
# get_storage_id and describe_storage (MetaConverter.describer): each part call's one converter the one tensor and
# storage memo, each of its placeholder prototypes the call's actual argument and its storage the actual storage, at
# a byte origin (actual offset bytes less the prototype's, one origin a storage); every output's value its prototype
# realized: a memoized one's memo, one on a bound storage its as_strided view of the actual storage at its own byte
# offset plus that origin (aligned to its element; the anchor's conjugate and negative bits cleared by their view
# toggles, a different dtype by .view over the storage's bytes aligned to both elements, then the prototype's bits and
# requires_grad on that new view) published to the memo, any other the call's own conversion (its fresh storages and
# their aliases its own)
def substitute(captured, key, parts):
    graph = captured.exported.graph_module.graph
    functions = {k: f for k, f in captured.functions.items() if k != key}
    functions.update((k, exported) for k, exported, _, _ in parts)
    calls = {}
    for node, call in captured.calls.items():
        if call.function != key:
            calls[node] = call
            continue
        leaves = [leaf.meta["val"] for leaf in call.inputs + call.outputs
                  if isinstance(leaf, torch.fx.Node) and isinstance(leaf.meta.get("val"), torch.Tensor)]
        fake, made, at = detect_fake_mode(leaves), [], node.prev
        for k, exported, inputs, results in parts:
            arguments = [call.inputs[source[1]] if source[0] == "call" else made[source[1]][source[2]]
                         for source in inputs]
            with graph.inserting_after(at):
                there = graph.node_copy(node, lambda n: n)
            there.args, there.kwargs = tuple(arguments), {}
            there.meta.update(function=k)
            at, outputs = there, []
            converter = FakeTensorConverter()
            describer, bound = converter.meta_converter.describer, {}
            prototypes = [n.meta.get("val") for n in exported.graph_module.graph.nodes if n.op == "placeholder"]
            for prototype, argument in zip(prototypes, arguments, strict=True):
                actual = argument.meta["val"] if isinstance(argument, torch.fx.Node) else argument
                if not (isinstance(prototype, torch.Tensor) and isinstance(actual, torch.Tensor)):
                    continue
                converter.set_tensor_memo(prototype, actual)
                converter.meta_converter.set_storage_memo(describer.describe_storage(prototype.untyped_storage()),
                                                          actual.untyped_storage())
                origin = actual.storage_offset() * actual.element_size() - \
                    prototype.storage_offset() * prototype.element_size()
                held = describer.get_storage_id(actual.untyped_storage()), origin, actual
                storage = describer.get_storage_id(prototype.untyped_storage())
                # rejection filters are forbidden: a correspondence this owner cannot relocate raises, never overwritten
                if bound.setdefault(storage, held)[:2] != held[:2]:
                    raise ValueError(f"{node.name} part {k}: one prototype input storage declared on two actual "
                                     f"storages or origins {bound[storage][:2]} and {held[:2]}")

            def realized(prototype):
                found = converter.tensor_memo.get(describer.get_tensor_id(prototype))
                if found is not None:
                    return found
                storage = describer.get_storage_id(prototype.untyped_storage())
                if storage not in bound:
                    return converter.from_real_tensor(fake, prototype)
                _, origin, anchor = bound[storage]
                byte, element = prototype.storage_offset() * prototype.element_size() + origin, prototype.element_size()
                if byte % element:
                    raise ValueError(f"{node.name} part {k}: output at byte {byte} of its input storage is not aligned "
                                     f"to its {element}-byte elements")
                with fake:
                    plain = anchor.conj() if anchor.is_conj() else anchor
                    plain = plain._neg_view() if plain.is_neg() else plain
                    if plain.dtype != prototype.dtype:
                        span = plain.untyped_storage().nbytes()
                        span -= span % math.lcm(plain.element_size(), element)
                        plain = plain.as_strided((span // plain.element_size(),), (1,), 0).view(prototype.dtype)
                    view = plain.as_strided(prototype.shape, prototype.stride(), byte // element)
                    view = view.conj() if prototype.is_conj() else view
                    view = view._neg_view() if prototype.is_neg() else view
                    if view.requires_grad and not prototype.requires_grad:
                        view = view.detach()
                    elif prototype.requires_grad and not view.requires_grad:
                        view.requires_grad_(True)
                converter.set_tensor_memo(prototype, view)
                return view
            values = exported.graph_module.graph.output_node().args[0]
            for o, value in enumerate(values):
                if o in results:
                    leaf = call.outputs[results[o]]
                    leaf.args = (there, o)
                    at.append(leaf)
                else:
                    with graph.inserting_after(at):
                        leaf = graph.call_function(operator.getitem, (there, o))
                prototype = value.meta["val"] if isinstance(value, torch.fx.Node) else value
                leaf.meta["val"] = realized(prototype) if isinstance(prototype, torch.Tensor) else prototype
                at = leaf
                outputs.append(leaf)
            calls[there] = Call(k, arguments, outputs)
            made.append(outputs)
        graph.erase_node(node)
    captured.exported.graph_module.recompile()
    return CapturedProgram(captured.exported, functions, calls,
                           (*captured.outlined, (captured.functions[key], tuple(part for _, part, _, _ in parts))))
