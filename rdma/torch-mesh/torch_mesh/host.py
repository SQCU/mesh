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


def assemble(pieces):
    """One result whose declared slices are the original source pages.

    Each piece is (source view, result view). Placement resolves the equalities;
    preparation emits no host operation or numerical copy for this declaration.
    """
    sources = {str(i): source for i, (source, _) in enumerate(pieces)}
    targets = {str(i): target for i, (_, target) in enumerate(pieces)}
    indexed = {name: 'indexed' for name in sources}
    result = declare(sources, targets, {'provider': 'pages', 'output_aliases': {name: name for name in targets}},
                     input_layouts=indexed, output_layouts=indexed, binding_layout='indexed')
    return next(iter(result.values()))


def bound(node):
    return getattr(getattr(node.target, '_schema', None), 'name', '') == 'torch_mesh::host_kernel'


def equalities(node):
    """Named result aliases become the shared typed placement equations."""
    from .operand import graph_bindings
    if not bound(node):
        return
    arguments, uses = graph_bindings(node)
    definition = json.loads(arguments['definition'])
    sources = {name: i for i, name in enumerate(definition['inputs'])}
    targets = {name: i for i, name in enumerate(definition['outputs'], len(sources))}
    for name, source in definition.get('output_aliases', {}).items():
        left, left_view, _ = uses[targets[name]]
        right, right_view, _ = uses[sources[source] if source in sources else targets[source]]
        yield left, left_view, right, right_view


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
    if definition['provider'] == 'pages':
        from .placement import same_view
        for name, original in definition['output_aliases'].items():
            if not same_view(target[name], source[original]):
                raise ValueError(f'{name}: page assembly lost its declared source equality')
        return None
    if definition['provider'] == 'cpu':
        weights = tuple(source[name] for name in definition['weights'])
        shard = weights[0] if definition['operation'] == 'linear' else weights
        return Cpu().bind(SimpleNamespace(kind=definition['operation']), shard, source['input'], target['output'],
                          input_layouts=definition['input_layouts'], output_layouts=definition['output_layouts'],
                          binding_layout=definition['binding_layout'])
    if definition['provider'] == 'coreml':
        from .placement import same_view
        for name, original in definition.get('output_aliases', {}).items():
            previous = source[original] if original in source else target[original]
            if not same_view(target[name], previous):
                raise ValueError(f'{name}: named result alias conflicts with its assigned input/result pages')
        contract = definition['contract']
        if not contract['outputs']:
            return None
        source = {name: source[name] for name in contract['inputs']}
        target = {name: target[name] for name in contract['outputs']}
        return coreml_graph(definition['compiled'], source, target, configuration=definition['configuration'],
                            contract=contract)
    raise ValueError(f"unknown declared host provider {definition['provider']}")
