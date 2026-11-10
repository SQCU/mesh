# design/data-flow.md
import torch
from torch._ops import HigherOrderOperator
from torch.export.graph_signature import InputKind, OutputKind

from .ir import Exported

WITH_EFFECTS = torch.ops.higher_order.with_effects


def export(module, args, **export_options):
    return read(torch.export.export(module, args, **export_options).run_decompositions())


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

    outputs = []
    for spec in signature.output_specs:
        if spec.kind not in (OutputKind.USER_OUTPUT, OutputKind.BUFFER_MUTATION, OutputKind.TOKEN):
            raise NotImplementedError(f"output kind {spec.kind}")
        outputs.append((spec.kind, spec.target))
    return Exported(program.graph_module, state, tokens, user_inputs, outputs, program.call_spec.out_spec)
