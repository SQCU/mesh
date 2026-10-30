"""Resolve numerical layout relations into fixed views and native commands.

This module runs only during preparation. A required materialization is an
ordinary tiled data dependency, using preassigned region storage.
"""
from functools import lru_cache
import math
from pathlib import Path

from . import _stream
from .compilation import operands
from .operand import TensorView, descriptor, requirements, graph_uses


def strides(shape):
    result, step = [], 1
    for n in reversed(shape):
        result.append(step)
        step *= max(1, n)
    return tuple(reversed(result))


def _layouts(bindings):
    return {index: value if isinstance(value, _stream.TensorLayout) else _stream.operand_layout(value)
            for index, value in sorted(bindings.items())}


def index_source(bindings, indices=None):
    """Emit addresses from the same descriptors carried into command accesses."""
    return (_stream.view_prelude() + ''.join(layout.declaration(index) for index, layout in _layouts(bindings).items())
            + ''.join(_stream.index_values(int(index), values) for index, values in (indices or {}).items()))


def index_constants(bindings):
    """Every binding's layout as function constants (_stream TensorLayout.constants), as one tuple; None where one
    binding does not have the form (a mapped view), which then keeps the source form (index_source)."""
    constants = []
    for index, layout in _layouts(bindings).items():
        own = layout.constants(index)
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


def pipeline(head, bindings, tail, name, indices=None, instance=''):
    """Kernel `name` of `head`, the bindings' (TensorViews or TensorLayouts) index declarations, `tail` and the explicit
    instantiation `instance` (its line, '' for a kernel that is no template).

    Where every binding is affine and the text reads no layout as a C++ constant expression (mesh_stride), the
    declarations are one prelude's function constants: commands that differ only in layout share one library, and each
    pipeline is specialized from its own values (design/compile-once-2026-10-29.md). The instances of one text wait until
    a pipeline of them is read, then compile as one library (_stream batch_compile). Otherwise they are source text
    (index_source): mapped views' index tables, and mesh_stride's constant strides.
    """
    constants = None if indices or 'mesh_stride<' in head + tail else index_constants(bindings)
    if constants is None:
        return _compiled(head + index_source(bindings, indices) + tail + instance, name)
    return _specialized(head + _stream.index_constant_prelude() + tail, instance, name, constants)


def place(graph, placements, fixed):
    """Place producer storage from all declared consumer subviews.

    Caller-owned storage stays fixed. Uses with mutually inconsistent physical
    requirements retain the shared resolver's explicit conversion edges.
    """
    from . import empty
    from torch.utils._pytree import tree_flatten
    described = [v for n in graph.nodes for v in tree_flatten(n.meta.get('val'))[0]]
    groups = []
    for node in graph.nodes:
        for value, view, required in graph_uses(node):
            if required == 'indexed':
                continue
            group = next((g for g in groups if torch_alias(g[0][0], value)), None)
            if group is None:
                group = []; groups.append(group)
            group.append((value, view, required if isinstance(required, str) else tuple(required)))
    for group in groups:
        anchor = group[0][0]
        required = lambda view, kind: view.required(kind if isinstance(kind, str) else 'fixed',
                                                  () if isinstance(kind, str) else kind)
        if all(view == required(view, kind) for _, view, kind in group):
            continue
        if any(torch_alias(anchor, v) for v in fixed) or any(torch_alias(anchor, v) for v, _ in placements.bindings):
            continue
        # The complete declared storage domain covers siblings and differently
        # typed aliases as well as the constrained consumer slices.
        domain = anchor.as_strided((anchor.untyped_storage().nbytes() // anchor.element_size(),), (1,), 0)
        values = list({(value.element_size(), descriptor(value)): value
                       for value in [domain, *(v for v in described if torch_alias(anchor, v))]}.values())
        group = list({(value.element_size(), view, kind): (value, view, kind)
                      for value, view, kind in group}.values())
        unit = math.gcd(*(value.element_size() for value in values))
        fields = [value.element_size() // unit for value in values]
        widths = [value.element_size() // unit for value, _, _ in group]
        uses = [view.components(width) for (_, view, _), width in zip(group, widths)]
        targets = [required(view, kind).components(width) for (_, view, kind), width in zip(group, widths)]
        typed = [descriptor(value).components(width) for value, width in zip(values, fields)]
        count = anchor.untyped_storage().nbytes() // unit
        # Strided APIs accept a family of layouts, not the source's current
        # distances. First solve exact demands, then retain each flexible use
        # which that placement already satisfies.
        flexible = [i for i, (_, _, kind) in enumerate(group) if kind in ('strided', 'positive_strided')]
        exact = [i for i in range(len(group)) if i not in flexible]
        placed = _stream.place_layout(count, [uses[i] for i in exact], [targets[i] for i in exact], [(0, 0)] * len(exact), typed, fields)
        if placed is not None and flexible:
            scalar_domain = _stream.TensorLayout((count,), (1,))
            needs_layout = False
            for i in flexible:
                view = uses[i].rebase(scalar_domain, placed).elements(widths[i])
                target = required(view, group[i][2])
                needs_layout |= view != target
                targets[i] = target.components(widths[i])
            if needs_layout:
                placed = _stream.place_layout(count, uses, targets, [(0, 0)] * len(uses), typed, fields)
        if placed is None:
            continue
        width = domain.element_size() // unit
        placed = placed.reshape((*domain.shape, width)).elements(width)
        first, length = placed.storage_span()
        extent = first + length if length else 0
        alignment = math.lcm(*(value.element_size() for value in values)) // domain.element_size()
        extent = -(-extent // alignment) * alignment
        storage = empty((extent,), dtype=domain.dtype, device=domain.device.type)
        placements.bind(domain, TensorView(storage, placed))


def torch_alias(left, right):
    import torch
    return isinstance(right, torch.Tensor) and torch._C._is_alias_of(left, right)


@lru_cache(None)
def _copy_pipeline(source, destination, width):
    kind = {1: 'uchar', 2: 'ushort', 4: 'uint', 8: 'ulong'}[width]
    prefix = (f'#define MESH_COPY_INPUT_TYPE {kind}\n#define MESH_COPY_OUTPUT_TYPE {kind}\n#define MESH_COPY_COUNT {source.count}ul\n'
              '#define MESH_COPY_SOURCE (mesh_index<0>(i))\n#define MESH_COPY_DESTINATION (mesh_index<1>(i))\n')
    return pipeline('', {0: source, 1: destination}, prefix + Path(__file__).with_suffix('.metal').read_text(), 'mesh_layout_copy')


def copies(source, destination):
    """Compile a necessary copy into independently schedulable logical tiles."""
    from .placement import same_view
    if same_view(source, destination) or not source.numel():
        return []
    if source.shape != destination.shape or source.dtype != destination.dtype:
        raise ValueError('layout materialization preserves the declared values and type')
    source, destination = TensorView.of(source), TensorView.of(destination)
    configuration = operands()
    limit = max(1, configuration['LAYOUT_COPY_TILE_BYTES'] // source.element_size())
    commands = []
    width = math.gcd(source.element_size(), 8)
    units = source.element_size() // width
    for origin, size in _stream.view_tiles(source.shape, limit):
        left, right = source, destination
        for axis, (first, count) in enumerate(zip(origin, size)):
            left, right = left.slice(axis, first, count), right.slice(axis, first, count)
        handle = _copy_pipeline(_stream.operand_layout(left).components(units), _stream.operand_layout(right).components(units), width)
        threads = min(configuration['GATHER_THREADS'], _stream.pipeline_facts(handle)[0])
        count = left.numel() * units
        commands.append((handle, [left, right], {}, (-(-count // threads), 1, 1), (threads, 1, 1), False,
                         1 | (2 << configuration['ACCESS_WRITE_SHIFT'])))
    return commands


def resolve(value, declarations, *, writing=False, device=None):
    """Realize an explicit API requirement before recording the function.

    The result is either the supplied storage view or a preallocated destination
    and ordinary copy edges. Direction comes from the declared input/output role.
    """
    layout = requirements(value, declarations)
    device = value.device.type if device is None else device
    if descriptor(value) == layout and value.device.type == device:
        return TensorView(TensorView.of(value).storage, layout), []
    from . import empty
    allocation = layout if layout.affine else layout.required('contiguous')
    steps = tuple(allocation.strides)
    reach = [(n - 1) * step for n, step in zip(value.shape, steps)]
    origin = -sum(min(0, step) for step in reach) if value.numel() else 0
    extent = 1 + sum(abs(step) for step in reach) if value.numel() else 0
    placed = TensorView(empty((extent,), dtype=value.dtype, device=device),
                        _stream.TensorLayout(tuple(value.shape), steps, origin))
    return placed, copies(placed, value) if writing else copies(value, placed)


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
