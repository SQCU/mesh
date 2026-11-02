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
    if definition['provider'] != 'pages' and 'implementation' not in definition:
        raise ValueError('numerical host declarations require an implementation operand')
    names = [*inputs, *outputs]
    layouts = [input_layouts[name] for name in inputs] + [output_layouts[name] for name in outputs]
    requirements = {i: (layout, binding_layout) for i, layout in enumerate(layouts)}
    definition = dict(definition, inputs=list(inputs), outputs=list(outputs),
                      input_layouts=input_layouts, output_layouts=output_layouts, binding_layout=binding_layout)
    operands = signature([*inputs.values(), *outputs.values()], range(len(inputs), len(names)), requirements)
    values = operation(**operands, definition=json.dumps(definition, separators=(',', ':')))
    slots = {slot: value for group, value in zip(operands['output_slots'], values, strict=True) for slot in group}
    return {name: slots[i] for i, name in enumerate(outputs, len(inputs))}


def assemble(pieces, *, output):
    """One result whose declared slices are the original source pages.

    The complete result operand is explicit; slices never imply its extent.
    Each piece is (source view, result view). Placement resolves the equalities;
    preparation emits no host operation or numerical copy for this declaration.
    """
    pieces = tuple(pieces)
    sources = {str(i): source for i, (source, _) in enumerate(pieces)}
    targets = {'result': output, **{str(i): target for i, (_, target) in enumerate(pieces)}}
    indexed = {name: 'indexed' for name in sources}
    result = declare(sources, targets, {'provider': 'pages', 'output_aliases': {name: name for name in sources}},
                     input_layouts=indexed, output_layouts={name: 'indexed' for name in targets}, binding_layout='indexed')
    return result['result']


def bound(node):
    return getattr(getattr(node.target, '_schema', None), 'name', '') == 'torch_mesh::host_kernel'


def describe(node):
    """Named source/result uses from the common captured operand schema."""
    from .operand import graph_bindings
    if not bound(node):
        return None
    arguments, uses = graph_bindings(node)
    definition = json.loads(arguments['definition'])
    sources = {name: uses[i] for i, name in enumerate(definition['inputs'])}
    targets = {name: uses[i] for i, name in enumerate(definition['outputs'], len(sources))}
    return dict(definition=definition, inputs=sources, outputs=targets)


def equalities(node):
    """Named aliases share ordered bytes; their declared uses keep scalar types."""
    from .operand import TensorView, byte_operand
    declared = describe(node)
    if declared is None:
        return
    definition, sources, targets = declared['definition'], declared['inputs'], declared['outputs']
    for name, source in definition.get('output_aliases', {}).items():
        left, left_view, _ = targets[name]
        right, right_view, _ = sources[source] if source in sources else targets[source]
        left = byte_operand(TensorView(left, left_view))
        right = byte_operand(TensorView(right, right_view))
        yield left.storage, left.layout.reshape((left.layout.count,)), right.storage, right.layout.reshape((right.layout.count,))


def lower_views(definition, source, target):
    """Lower provider declarations after placement and before dependency solving.

    Alias equations belong to the shared placement solve. A provider's input
    may use a required contiguous copy while a forwarded result keeps the
    original pages; those two API uses need not have the same physical view.
    """
    from .engines import Cpu
    if definition['provider'] == 'pages':
        return None
    if definition['provider'] == 'cpu':
        return Cpu(implementation=definition['implementation']).declare_views(definition, source, target)
    if definition['provider'] == 'coreml':
        contract = definition['contract']
        if not contract['outputs']:
            return None
        source = {name: source[name] for name in contract['inputs']}
        target = {name: target[name] for name in contract['outputs']}
        from .operand import byte_operand
        from .cpu import compilation
        values = [*source.values(), *target.values()]
        operands = [byte_operand(value) for value in values]
        command = dict(kind='ane', provider=definition, source=ane_source((), (), (), ()),
            operands=operands, source_views=source, target_views=target,
            argument_specs=(), constants={}, types=tuple(torch.uint8 for _ in values),
            binding_origins=tuple(value.binding_origin for value in operands),
            reaches=tuple((slot, 1 if slot < len(source) else 2, value) for slot, value in enumerate(operands)),
            compiler=compilation(definition['implementation']))
        return dict(commands=(command,), output=tuple(target.values()), pages=tuple(value.storage for value in operands))
    raise ValueError(f"unknown declared host provider {definition['provider']}")


def ane_source(fields, helpers, before, after):
    """Specialize the common submit/completion recipe before request binding."""
    from mesh import RDMA
    def effect(name, statements):
        if not statements:
            return f'#define {name}(arguments)\n'
        body = ' '.join(statements)
        return (f'#define {name}(arguments) do {{ '
                'const auto &a = *static_cast<const MeshCpuArguments *>(arguments); '
                + body + ' } while (0)\n')
    return (f'#include "{RDMA}/mesh-publication.h"\n' + ''.join(helpers) +
        'struct MeshCpuArguments {' + ''.join(fields) + '};\n' +
        effect('MESH_ANE_ENTER', before) + effect('MESH_ANE_LEAVE', after) +
        f'#include "{RDMA}/mesh-ane-entry.inc"\n')


def prepare(command):
    """Final resource binding; declarations and dependency source are complete."""
    from . import _stream
    from .cpu import argument_record
    from .engines import coreml_graph
    definition = command['provider']
    arguments, _ = argument_record(command, {
        slot: _stream.page_backing(value)[0] for slot, value in enumerate(command['operands'])})
    return coreml_graph(definition['compiled'], command['source_views'], command['target_views'],
        configuration=definition['configuration'], contract=definition['contract'],
        implementation=definition['implementation'],
        entry=dict(source=command['source'], compiler=command['compiler'], arguments=arguments),
        reaches=command['reaches'])
