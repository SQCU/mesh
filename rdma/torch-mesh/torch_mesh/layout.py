"""Resolve numerical layout relations into fixed views and native commands.

This module runs only during preparation. A required materialization is an
ordinary tiled data dependency, using preassigned region storage.
"""
from functools import lru_cache
from dataclasses import replace
import math
from pathlib import Path

from . import _stream
from .compilation import operands
from .operand import TensorView, descriptor, requirements, required_layout, storage_requirement, storage_origin, graph_uses


def strides(shape):
    result, step = [], 1
    for n in reversed(shape):
        result.append(step)
        step *= max(1, n)
    return tuple(reversed(result))


def _layouts(bindings):
    return {index: value if isinstance(value, _stream.TensorLayout) else _stream.operand_layout(value)
            for index, value in sorted(bindings.items())}


def index_source(bindings, indices=None, origins=None):
    """Emit addresses from the same descriptors carried into command accesses."""
    return (_stream.view_prelude() + ''.join(layout.declaration(index, _binding_origin(bindings[index]) if origins is None else origins[index])
                                           for index, layout in _layouts(bindings).items())
            + ''.join(_stream.index_values(int(index), values) for index, values in (indices or {}).items()))


def _binding_origin(value):
    origin = getattr(value, 'binding_origin', None)
    return None if origin is None else origin // value.element_size()


def index_constants(bindings, origins=None):
    """Every binding's layout as function constants (_stream TensorLayout.constants), as one tuple; None where one
    binding does not have the form (a mapped view), which then keeps the source form (index_source)."""
    constants = []
    for index, layout in _layouts(bindings).items():
        own = layout.constants(index, _binding_origin(bindings[index]) if origins is None else origins[index])
        if own is None:
            return None
        constants.extend(tuple(c) for c in own)
    return tuple(constants)


@lru_cache(None)
def _compiled(text, name):
    return _stream.pipeline(text, name)


@lru_cache(None)
def _specialized(base, instance, name, constants):
    return _stream.pipeline_specialized(base, instance, name, [list(c) for c in constants])


def pipeline(head, bindings, tail, name, indices=None, instance='', origins=None):
    """Kernel `name` of `head`, the bindings' (TensorViews or TensorLayouts) index declarations, `tail` and the explicit
    instantiation `instance` (its line, '' for a kernel that is no template).

    Where every binding is affine and the text reads no layout as a C++ constant expression (mesh_stride), the
    declarations are one prelude's function constants: commands that differ only in layout share one library, and each
    pipeline is specialized from its own values (design/compile-once-2026-10-29.md). The instances of one text wait until
    a pipeline of them is read, then compile as one library (_stream batch_compile). Otherwise they are source text
    (index_source): mapped views' index tables, and mesh_stride's constant strides.
    """
    constants = None if indices or 'mesh_stride<' in head + tail else index_constants(bindings, origins)
    if constants is None:
        return _compiled(head + index_source(bindings, indices, origins) + tail + instance, name)
    return _specialized(head + _stream.index_constant_prelude() + tail, instance, name, constants)


def solve_layout(count, demands, values):
    """The one relative-placement solve, in byte positions for every scalar type."""
    result = prefer_layout(count, (), (), values, hard=demands)
    return (None, 0) if result is None else (result['layout'], result['extent'])


def prefer_layout(count, groups, preferences, values, *, hard=()):
    """Realize declared priorities with the shared hard placement algebra.

    Every group is one consumer's atomic requirements. A rejected direct use
    still requires its ordinary conversion; provider requirements never become
    optional. The same selector serves the native and portable binders.
    """
    def records(demands):
        result = []
        for use, width, kind, view, capacity, alignment in demands:
            target = required_layout(view, kind).components(width)
            flexible = kind if isinstance(kind, str) and kind in ('strided', 'positive_strided') and not capacity and alignment == 1 else ''
            result.append(dict(use=use, target=target, width=width, flexible=flexible,
                               footprint=(target.index(0) if target.count else 0, capacity, alignment)))
        return result
    return _stream.prefer_layout(count, records(hard), [records(group) for group in groups],
                                list(preferences), [item[0] for item in values], [item[1] for item in values])


def declare(graph, placements, materializations=()):
    """Retain logical producer families and API needs before page assignment."""
    from torch.utils._pytree import tree_flatten
    from . import host
    described = [v for n in graph.nodes for v in tree_flatten(n.meta.get('val'))[0]]
    demands = []
    for node in graph.nodes:
        for left, left_view, right, right_view in host.equalities(node):
            placements.equate(left, right, left_view, right_view)
        for value, view, required in graph_uses(node):
            kind, capacity, alignment = storage_requirement(required)
            if kind != 'indexed' or capacity or alignment != 1:
                demands.append((value, view, kind, capacity, alignment))
    placements.declare(described, demands, materializations)


def place(graph, placements):
    """Assign pages to the same declared producer families."""
    declare(graph, placements)
    placements.solve()


@lru_cache(None)
def _copy_source():
    return Path(__file__).with_suffix('.metal').read_text()


def copy_declarations(source, destination, *, algebra=None):
    """Retain a necessary copy's shared source and exact group/page relation."""
    import torch
    from mesh_relation import Algebra
    from .operand import byte_operand
    from .placement import same_view
    if same_view(source, destination) or not source.numel():
        return []
    if source.shape != destination.shape or source.dtype != destination.dtype:
        raise ValueError('layout materialization preserves the declared values and type')
    from .operand import TensorValue
    keep = lambda value: value if isinstance(value, TensorValue) else TensorView.of(value)
    source, destination = keep(source), keep(destination)
    def typed(view, dtype, units):
        """a tile read at the copy's unit type: a value stays a value of its space (design/page-maps-2026-11-02.md)"""
        if isinstance(view, TensorValue):
            return view.retyped(dtype, units)
        return TensorView(view.storage_as(dtype), view.layout.components(units), view.binding_origin)
    configuration = operands()
    limit = max(1, configuration['LAYOUT_COPY_TILE_BYTES'] // source.element_size())
    commands = []
    width = math.gcd(source.element_size(), 8)
    units = source.element_size() // width
    dtype, kind = {1: (torch.uint8, 'uchar'), 2: (torch.uint16, 'ushort'),
                   4: (torch.uint32, 'uint'), 8: (torch.uint64, 'ulong')}[width]
    from .compilation import relations
    algebra = relations() if algebra is None else algebra
    for origin, size in _stream.view_tiles(source.shape, limit):
        left, right = source, destination
        for axis, (first, count) in enumerate(zip(origin, size)):
            left, right = left.slice(axis, first, count), right.slice(axis, first, count)
        left, right = typed(left, dtype, units), typed(right, dtype, units)
        threads = min(configuration['GATHER_THREADS'], _stream.device_facts()[1])
        count = left.numel()
        groups = -(-count // threads)
        relation = algebra.map((groups,), (count,), lambda g, x: [f'{g[0]} * {threads} <= {x[0]} < ({g[0]} + 1) * {threads}'])
        prefix = (f'#define MESH_COPY_INPUT_TYPE {kind}\n#define MESH_COPY_OUTPUT_TYPE {kind}\n#define MESH_COPY_COUNT {count}ul\n'
                  '#define MESH_COPY_SOURCE (mesh_index<0>(i))\n#define MESH_COPY_DESTINATION (mesh_index<1>(i))\n')
        commands.append(dict(kind='shader', source=prefix + _copy_source(), function='mesh_layout_copy',
            operands=[left, right], constants={}, grid=(groups, 1, 1), threadgroup=(threads, 1, 1), thread_grid=False,
            access=1 | (2 << configuration['ACCESS_WRITE_SHIFT']), types=(dtype, dtype), arguments=(),
            binding_origins=(left.binding_origin, right.binding_origin),
            dispatch=(dict(side='reads', slot=0, group_relation=relation), dict(side='writes', slot=1, group_relation=relation)),
            reaches=((0, 1, byte_operand(left)), (1, 2, byte_operand(right)))))
    return commands


def allocate(layout, dtype, device, *, extent=0, alignment=1, binding_origin=None):
    """Assign the declared layout and byte footprint to aligned bridge pages."""
    from . import empty
    width = dtype.itemsize
    first, length = layout.storage_span()
    bound = 0 if binding_origin is None else (layout.index(0) * width + binding_origin if layout.count else 0)
    origin = -min(0, first, bound // width)
    alignment = math.lcm(alignment, width)
    count = -(-max((first + length + origin) * width, extent + bound + origin * width) // width)
    reserved = -(-(count * width + alignment - width) // alignment) * (alignment // width)
    storage = empty((reserved,), dtype=dtype, device=device)
    base, _, backing = _stream.page_backing(storage)
    offset = backing.index(0) if backing.count else backing.offset
    padding = (-(base + offset + bound + origin * width)) % alignment
    return TensorView(storage, layout.shift(origin + padding // width),
                      None if binding_origin is None else bound + origin * width + padding)


def resolve(value, declarations, *, writing=False, device=None):
    """Realize an explicit API requirement before recording the function.

    The result is either the supplied storage view or a preallocated destination
    and ordinary copy edges. Direction comes from the declared input/output role.
    """
    layout = requirements(value, declarations)
    contracts = [storage_requirement(declared) for declared in declarations]
    capacity = max((item[1] for item in contracts), default=0)
    alignment = math.lcm(*(item[2] for item in contracts))
    origin = storage_origin(declarations)
    from .operand import TensorValue
    if isinstance(value, TensorValue):
        # Changed by the LSTM demonstration session (2026-11-02, design/page-maps-2026-11-02.md): a value's fit is
        # read in its space, whose element 0 is page aligned, so no base address enters: its first element's byte
        # position is aligned and its capacity lies in the space. The final lowering turns that position into the
        # argument's address. What must stay: an unmet requirement placed afresh, with its copy edges.
        width = value.dtype.itemsize
        first = value.layout.index(0) if value.layout.count else 0
        bound = first * width if origin is None else first * width + origin
        space_bytes = value.space.pages * value.space.page_bytes
        fits = bound >= 0 and bound % alignment == 0 and bound + capacity <= space_bytes
        if value.layout == layout and fits and origin is None:
            return value, []
        from .placement import allocate_value
        allocation = layout.shift(-(layout.index(0) if layout.count else 0))
        extent = (allocation.index(0) * width if allocation.count else 0) + capacity
        placed = allocate_value(allocation, value.dtype, extent=extent, alignment=alignment,
                                page_bytes=value.space.page_bytes)
        return placed, copy_declarations(placed, value) if writing else copy_declarations(value, placed)
    device = value.device.type if device is None else device
    base, available, backing = _stream.page_backing(value)
    at = backing.index(0) if backing.count else backing.offset
    storage = TensorView.of(value).storage
    first = descriptor(value).index(0) if descriptor(value).count else 0
    bound = first * value.element_size() if origin is None else first * value.element_size() + origin
    at += bound - first * value.element_size()
    fits = (bound >= 0 and (base + at) % alignment == 0 and at + capacity <= available
            and bound + capacity <= storage.untyped_storage().nbytes())
    if descriptor(value) == layout and value.device.type == device and fits:
        return replace(TensorView.of(value), layout=layout,
                       binding_origin=TensorView.of(value).binding_origin if origin is None else bound), []
    allocation = layout.shift(-(layout.index(0) if layout.count else 0))
    extent = (allocation.index(0) * value.element_size() if allocation.count else 0) + capacity
    placed = allocate(allocation, value.dtype, device, extent=extent, alignment=alignment, binding_origin=origin)
    if origin is None:
        placed = replace(placed, binding_origin=TensorView.of(value).binding_origin)
    return placed, copy_declarations(placed, value) if writing else copy_declarations(value, placed)


def bindings(values, declarations, *, writing=False, binding='indexed', devices=None):
    """Resolve named API operands through the one layout/copy lowering."""
    if values.keys() != declarations.keys():
        raise ValueError('each binding requires an explicit layout operand')
    placed, commands = {}, []
    for name, value in values.items():
        if value is None:
            placed[name] = None
            continue
        placed[name], edges = resolve(value, (declarations[name], binding), writing=writing,
                                     device=None if devices is None else devices[name])
        commands.extend(edges)
    return placed, commands
