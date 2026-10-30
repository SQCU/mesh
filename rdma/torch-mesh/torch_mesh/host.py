"""Host numerical declarations, erased into the ordinary native schedule."""
import json
import torch


@torch.library.custom_op('torch_mesh::host_kernel', mutates_args=(),
    schema='(Tensor[] inputs, int[] input_slots, int[][] output_slots, int[][] output_strides, '
           'int[][] shapes, str[] dtypes, str[] devices, str operands, str definition) -> Tensor[]')
def operation(inputs: list[torch.Tensor], input_slots: list[int], output_slots: list[list[int]],
              output_strides: list[list[int]], shapes: list[list[int]], dtypes: list[str], devices: list[str],
              operands: str, definition: str) -> list[torch.Tensor]:
    raise RuntimeError('host kernels require whole-function preparation')


@operation.register_fake
def _shape(inputs, input_slots, output_slots, output_strides, shapes, dtypes, devices, operands, definition):
    from .operand import fake_outputs
    return fake_outputs(shapes, output_strides, dtypes, devices)


def declare(inputs, outputs, definition, *, input_layouts, output_layouts, binding_layout):
    """Capture named uses and return their result storage handles, as for GPU kernels."""
    from .operand import signature
    if inputs.keys() != input_layouts.keys() or outputs.keys() != output_layouts.keys():
        raise ValueError('each host graph operand requires a declared layout')
    names = [*inputs, *outputs]
    layouts = [input_layouts[name] for name in inputs] + [output_layouts[name] for name in outputs]
    requirements = {i: (layout, binding_layout) for i, layout in enumerate(layouts)}
    definition = dict(definition, inputs=list(inputs), outputs=list(outputs),
                      input_layouts=input_layouts, output_layouts=output_layouts, binding_layout=binding_layout)
    operands = signature([*inputs.values(), *outputs.values()], range(len(inputs), len(names)), requirements)
    values = operation(**operands, definition=json.dumps(definition, separators=(',', ':')))
    slots = {slot: value for group, value in zip(operands['output_slots'], values, strict=True) for slot in group}
    return {name: slots[i] for i, name in enumerate(outputs, len(inputs))}


def bound(node):
    return getattr(getattr(node.target, '_schema', None), 'name', '') == 'torch_mesh::host_kernel'


def prepare(inputs, input_slots, output_slots, output_strides, shapes, dtypes, devices, operands, definition, *, outputs):
    """Bind once after graph placement; no provider lookup survives execution."""
    from .operand import bind
    from .engines import Cpu, coreml_graph
    from types import SimpleNamespace
    uses, definition = json.loads(operands), json.loads(definition)
    values = {slot: bind(value, uses[str(slot)]) for slot, value in zip(input_slots, inputs, strict=True)}
    values.update((slot, bind(value, uses[str(slot)])) for slots, value in zip(output_slots, outputs, strict=True) for slot in slots)
    source = {name: values[i] for i, name in enumerate(definition['inputs'])}
    target = {name: values[i] for i, name in enumerate(definition['outputs'], len(source))}
    if definition['provider'] == 'cpu':
        weights = tuple(source[name] for name in definition['weights'])
        shard = weights[0] if definition['operation'] == 'linear' else weights
        return Cpu().bind(SimpleNamespace(kind=definition['operation']), shard, source['input'], target['output'],
                          input_layouts=definition['input_layouts'], output_layouts=definition['output_layouts'],
                          binding_layout=definition['binding_layout'])
    if definition['provider'] == 'coreml':
        return coreml_graph(definition['compiled'], source, target, configuration=definition['configuration'],
                            contract=definition['contract'])
    raise ValueError(f"unknown declared host provider {definition['provider']}")
