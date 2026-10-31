"""Native matrix command operands shared by FX and the streamed DNN program."""
from pathlib import Path

import torch
import mesh_matrix
from .compilation import operands, api_layouts
from .layout import bindings


_OPERANDS = ''.join(Path(__file__).with_name(f).read_text() + '\n' for f in mesh_matrix.LIBRARIES[:-1])
_SOURCE = _OPERANDS + Path(__file__).with_suffix('.metal').read_text()


def row_tiles(rows):
    """The product's compiled row groups; shared by writes and their producer."""
    return mesh_matrix.row_tiles(rows, operands())


def arguments(x, w, out, *, x1=None, bias=None, act=False, alpha=1, beta=1, ready=None, arrivals=None):
    """Bind page operands to the shared metadata-only product declaration."""
    kinds = {torch.float16: 'half', torch.float32: 'float', torch.bfloat16: 'bfloat',
             torch.int64: 'long', torch.int32: 'int'}
    values = dict(x=x, w=w, out=out, x1=x1, bias=bias, ready=ready, arrivals=arrivals)
    declarations = {name: None if value is None else dict(shape=tuple(value.shape), type=kinds[value.dtype])
                    for name, value in values.items()}
    recipe = mesh_matrix.product(**declarations, geometry=operands(), act=act, alpha=alpha, beta=beta)
    return (recipe['definitions'], [values[name] if name is not None else None for name in recipe['bindings']],
            recipe['grid'], recipe['threads'])


def declaration(operands, parameters, configuration):
    """The shared product source and typed role slots before page binding."""
    from .command_declarations import Commands
    kinds = {torch.float16: 'half', torch.float32: 'float', torch.bfloat16: 'bfloat',
             torch.int64: 'long', torch.int32: 'int'}
    commands = Commands(operands, **configuration)
    descriptions = {name: dict(shape=value['shape'], type=kinds[value['dtype']]) for name, value in operands.items()}
    options = {name: parameters[name] for name in ('act', 'alpha', 'beta') if name in parameters}
    recipe = mesh_matrix.product(**descriptions, geometry=configuration['geometry'], **options)
    if not all(operands['out']['shape']):
        return commands.declaration()
    shift = configuration['geometry']['ACCESS_WRITE_SHIFT']
    commands.shader(_SOURCE, recipe['function'],
        [None if role is None else commands.view(role) for role in recipe['bindings']],
        definitions=recipe['definitions'], grid=recipe['grid'], threadgroup=recipe['threads'],
        access=3 | (32 if 'bias' in operands else 0) | (64 if 'x1' in operands else 0)
        | ((140 if 'ready' in operands else 4) << shift))
    return commands.declaration()


def product(x, w, out, *, x1=None, bias=None, act=False, alpha=1, beta=1, ready=None, arrivals=None, layouts=None):
    """Resolve declared layouts and emit the shared product command."""
    layouts = api_layouts('matrix') if layouts is None else layouts
    inputs, before = bindings(dict(x=x, w=w, x1=x1, bias=bias),
        dict(x=layouts['input'], w=layouts['weight'], x1=layouts['input'], bias=layouts['bias']))
    outputs, after = bindings(dict(out=out), dict(out=layouts['output']), writing=True)
    from .command_declarations import descriptions, configuration, bind
    from .native import command_records
    values = dict(**inputs, out=outputs['out'], ready=ready, arrivals=arrivals)
    declared = declaration(descriptions(values), dict(act=act, alpha=alpha, beta=beta), configuration())
    return [*before, *command_records(bind(declared, values)), *after]


def declare(x, w, out):
    """Carry the same product and API constraints through an FX declaration."""
    from .kernel import declare as kernel
    definitions, buffers, grid, threads = arguments(x, w, out)
    layouts = api_layouts('matrix')
    prefix = ''.join(f'#define {key} {value}\n' for key, value in definitions.items())
    prefix += _OPERANDS
    return kernel('matrix.metal', 'product', buffers, (2,), (), 0, grid, threads,
                  indexed=(), requirements={0: layouts['input'], 1: layouts['weight'], 2: layouts['output']}, prefix=prefix,
                  domains=({'kind': 'product', 'operands': {'x': 0, 'w': 1, 'out': 2},
                            'parameters': {'alpha': 1, 'beta': 1, 'act': False}},))[0]
