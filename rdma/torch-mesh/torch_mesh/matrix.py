"""Native matrix command operands shared by FX and the streamed DNN program."""
from pathlib import Path

import torch
import mesh_matrix
from torch._inductor.codegen.mps import DTYPE_TO_METAL
from .compilation import operands, api_layouts
from .layout import bindings


_OPERANDS = ''.join(Path(__file__).with_name(f).read_text() + '\n' for f in mesh_matrix.LIBRARIES[:-1])
_SOURCE = _OPERANDS + Path(__file__).with_suffix('.metal').read_text()


def row_tiles(rows):
    """The product's compiled row groups; shared by writes and their producer."""
    return mesh_matrix.row_tiles(rows, operands())


def arguments(x, w, out, *, x1=None, bias=None, act=False, alpha=1, beta=1):
    """Bind page operands to the shared metadata-only product declaration."""
    kinds = DTYPE_TO_METAL
    values = dict(x=x, w=w, out=out, x1=x1, bias=bias)
    declarations = {name: None if value is None else dict(shape=tuple(value.shape), type=kinds[value.dtype])
                    for name, value in values.items()}
    from .command_declarations import configuration
    recipe = mesh_matrix.product(**declarations, **configuration(), act=act, alpha=alpha, beta=beta)
    return (recipe['definitions'], [values[name] if name is not None else None for name in recipe['bindings']],
            recipe['grid'], recipe['threads'])


def declaration(operands, parameters, configuration):
    """The shared product source and typed role slots before page binding."""
    from .command_declarations import Commands
    kinds = DTYPE_TO_METAL
    commands = Commands(operands, **configuration)
    roles = ('x', 'w', 'out', 'x1', 'bias')
    descriptions = {name: dict(shape=value['shape'], type=kinds[value['dtype']])
                    for name, value in operands.items() if name in roles}
    options = {name: parameters[name] for name in ('act', 'alpha', 'beta') if name in parameters}
    recipe = mesh_matrix.product(**descriptions, geometry=configuration['geometry'], capabilities=configuration['capabilities'], **options)
    if not all(operands['out']['shape']):
        return commands.declaration()
    from mesh_relation import Algebra
    algebra = commands.algebra
    logical_accesses = mesh_matrix.product_dispatch(algebra, recipe,
        {name: value for name, value in descriptions.items() if name in ('x', 'w', 'out', 'x1', 'bias')})
    dispatch = []
    shift = configuration['geometry']['ACCESS_WRITE_SHIFT']
    readers = parameters.get('readers', {})
    views = [None if role is None or role in readers else commands.view(role) for role in recipe['bindings']]
    access = (3 | (32 if 'bias' in operands else 0) | (64 if 'x1' in operands else 0)
              | (4 << shift))
    indices, access_layouts, definitions, sources, declarations = {}, {}, dict(recipe['definitions']), {}, []
    for role, reader in readers.items():
        logical_slot = recipe['bindings'].index(role)
        access &= ~(1 << logical_slot)
        indices[logical_slot] = reader['layout']
        types, values = [], []
        for input_index, name in enumerate(reader['bindings']):
            slot = len(views)
            views.append(commands.view(name));access |= 1 << slot
            access_layouts[slot] = reader['accesses'][input_index]
            scalar = kinds[operands[name]['dtype']]
            declarations.append(f'device const {scalar}* reader_input{slot} [[buffer({slot})]],')
            types.append(f'mesh_page_read<{slot},{scalar}>')
            values.append(f'mesh_page_read<{slot},{scalar}>{{reader_input{slot}}}')
        source_type = reader['name'] + (f"<{','.join(types)}>" if types else '')
        expression = f"{source_type}{{{','.join(values)}}}"
        definitions[f'MESH_MATRIX_{role.upper()}_READ'] = f'mesh_mapped_value_read<{logical_slot},{source_type}>{{{expression}}}'
        if role in ('x', 'w', 'x1'):
            definitions[dict(x='AFFINE_A', w='AFFINE_B', x1='AFFINE_A1')[role]] = 0
        sources[reader['name']] = reader['source']
    for use in logical_accesses:
        role, relation = use['role'], use['group_relation']
        if role in readers:
            reader = readers[role]
            for name, dependency in zip(reader['bindings'], reader['relations'], strict=True):
                # The retained reader relation starts in its product-role
                # coordinates and ends in its original input storage ordinals.
                selected = relation.then(dependency.reshape((reader['layout'].count,)))
                slot = next(i for i, view in enumerate(views) if view is not None and view['name'] == name)
                dispatch.append(dict(side=use['side'], slot=slot, group_relation=selected))
        else:
            slot = recipe['bindings'].index(role)
            dispatch.append(dict(side=use['side'], slot=slot, group_relation=relation))
    definitions['MESH_MATRIX_PARAMETERS'] = ''.join(declarations)
    from .native import shader_source
    source = _OPERANDS + shader_source(''.join(sources.values())) + Path(__file__).with_suffix('.metal').read_text()
    commands.shader(source, recipe['function'], views, definitions=definitions,
        indices=indices, access_layouts=access_layouts, grid=recipe['grid'], threadgroup=recipe['threads'], access=access, dispatch=tuple(dispatch), coherent_accesses=True)
    return commands.declaration()


def product(x, w, out, *, x1=None, bias=None, act=False, alpha=1, beta=1, layouts=None):
    """Resolve declared layouts and emit the shared product command."""
    layouts = api_layouts('matrix') if layouts is None else layouts
    inputs, before = bindings(dict(x=x, w=w, x1=x1, bias=bias),
        dict(x=layouts['input'], w=layouts['weight'], x1=layouts['input'], bias=layouts['bias']))
    outputs, after = bindings(dict(out=out), dict(out=layouts['output']), writing=True)
    from .command_declarations import descriptions, configuration, bind
    values = dict(**inputs, out=outputs['out'])
    declared = declaration(descriptions(values), dict(act=act, alpha=alpha, beta=beta), configuration())
    return [*before, *bind(declared, values), *after]


@torch.library.custom_op('torch_mesh::product_pages', mutates_args=(),
    schema='(Tensor[] inputs, int[] input_slots, int[][] output_slots, int[][] output_strides, '
           'int[][] shapes, str[] dtypes, str[] devices, str operands) -> Tensor[]')
def operation(inputs: list[torch.Tensor], input_slots: list[int], output_slots: list[list[int]],
              output_strides: list[list[int]], shapes: list[list[int]], dtypes: list[str], devices: list[str],
              operands: str) -> list[torch.Tensor]:
    raise RuntimeError('product page declarations require whole-function preparation')


@operation.register_fake
def _shape(inputs, input_slots, output_slots, output_strides, shapes, dtypes, devices, operands):
    from .operand import fake_outputs
    return fake_outputs(shapes, output_strides, dtypes, devices)


def declare(x, w, out, *, implementation=None):
    """Retain the product equation and page views, before backend selection.

    Both product compilers consume this mathematical domain. No shader, host
    task or provider invocation is bound at this source declaration boundary.
    """
    import json
    from .operand import signature, TensorView
    values = signature((x, w, out), (2,), {slot: ('indexed',) for slot in range(3)})
    declared = json.loads(values['operands'])
    declared['domains'] = [dict(kind='product', operands=dict(x=0, w=1, out=2),
        parameters=dict(alpha=1, beta=1, act=False, implementation=implementation))]
    values['operands'] = json.dumps(declared, separators=(',', ':'))
    return operation(**values)[0]


def _row_group_declaration(operands, requests, configuration):
    """Bind the common row-product recipe to the ordinary page schema."""
    from .command_declarations import Commands
    commands = Commands(operands, **configuration)
    typed = {name: dict(shape=value['shape'], type=DTYPE_TO_METAL[value['dtype']])
             for name, value in operands.items()}
    recipe = mesh_matrix.row_product_group(typed, requests, **configuration)
    from mesh_relation import Algebra
    algebra, dispatch = commands.algebra, []
    for prefix, part, start in recipe['products']:
        shapes = {name[len(prefix):]: value for name, value in typed.items() if name.startswith(prefix)}
        for use in mesh_matrix.product_dispatch(algebra, part, shapes, offset=start,
                total=recipe['grid'][0], group_order=(2, 1, 0)):
            dispatch.append(dict(side=use['side'], slot=recipe['bindings'].index(prefix + use['role']),
                                 group_relation=use['group_relation']))
    commands.shader(_OPERANDS + recipe['source'], recipe['function'],
        [commands.view(name) for name in recipe['bindings']], grid=recipe['grid'],
        threadgroup=recipe['threads'], access=recipe['access'], dispatch=tuple(dispatch), coherent_accesses=True)
    return commands.declaration()


def products(requests, *, layouts=None):
    """Prepare product groups using one capability and page-binding contract."""
    from .command_declarations import descriptions, configuration, bind
    layouts = api_layouts('matrix') if layouts is None else layouts
    configuration = configuration()
    geometry = configuration['geometry']
    result, group, values = [], [], {}

    def flush():
        if not group:
            return
        constraints = {name: layouts[dict(x='input', x1='input', w='weight', bias='bias')[name.split('_', 1)[1]]]
                       for name in values if not name.endswith('_out')}
        inputs, before = bindings({name: value for name, value in values.items() if name in constraints}, constraints)
        outputs, after = bindings({name: value for name, value in values.items() if name.endswith('_out')},
                                  {name: layouts['output'] for name in values if name.endswith('_out')}, writing=True)
        bound = dict(inputs, **outputs)
        declared = _row_group_declaration(descriptions(bound), group, configuration)
        result.extend((*before, *bind(declared, bound), *after))
        group.clear();values.clear()

    for request in requests:
        if not all(request['out'].shape):
            continue
        actual = {role: value for role, value in request.items() if role in ('x', 'w', 'out', 'x1', 'bias') and value is not None}
        options = {key: request[key] for key in ('act', 'alpha', 'beta') if key in request}
        if request['x'].shape[-2] > geometry['LINEAR_ROW_CROSSOVER'] or len(request['x'].shape) != 2:
            flush();result.extend(product(**request, layouts=layouts));continue
        if len(values) + len(actual) > geometry['BUFFER_BINDINGS']:
            flush()
        prefix = f'p{len(group)}_'
        values.update((prefix + role, value) for role, value in actual.items())
        group.append((prefix, options))
    flush()
    return result
