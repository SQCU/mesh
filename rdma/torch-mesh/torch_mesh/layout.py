"""Resolve numerical layout relations into fixed views and native commands.

This module runs only during preparation. A required materialization is an
ordinary tiled data dependency, using preassigned region storage.
"""
from functools import lru_cache
import math
import json
from pathlib import Path

from . import _stream
from .compilation import operands
from .operand import TensorView, descriptor, requirements


def strides(shape):
    result, step = [], 1
    for n in reversed(shape):
        result.append(step)
        step *= max(1, n)
    return tuple(reversed(result))


def index_source(bindings):
    """Emit addresses from the same descriptors carried into command accesses."""
    return _stream.view_prelude() + ''.join(_stream.operand_layout(value).declaration(index)
                                           for index, value in sorted(bindings.items()))


def place(graph, placements, fixed):
    """Place producer storage from all declared consumer subviews.

    Caller-owned storage stays fixed. Uses with mutually inconsistent physical
    requirements retain the shared resolver's explicit conversion edges.
    """
    from .operand import restore
    from . import empty
    from torch.utils._pytree import tree_flatten
    described = [v for n in graph.nodes for v in tree_flatten(n.meta.get('val'))[0]]
    groups = []
    for node in graph.nodes:
        schema = getattr(node.target, '_schema', None)
        if getattr(schema, 'name', '') != 'torch_mesh::native_kernel':
            continue
        args = {a.name: node.args[i] if i < len(node.args) else node.kwargs[a.name]
                for i, a in enumerate(schema.arguments)}
        declarations = json.loads(args['operands'])
        bindings = [(source.meta['val'], slot) for source, slot in zip(args['inputs'], args['input_slots'], strict=True)]
        bindings.extend((value, slot) for value, slots in zip(node.meta['val'], args['output_slots'], strict=True) for slot in slots)
        for value, slot in bindings:
            declaration = declarations[str(slot)]
            required = declaration['requirement']
            if required == 'indexed':
                continue
            view = restore(declaration['view']).rebase(restore(declaration['source']), descriptor(value))
            target = view.required(required if isinstance(required, str) else 'fixed',
                                   () if isinstance(required, str) else tuple(required))
            group = next((g for g in groups if torch_alias(g[0][0], value)), None)
            if group is None:
                group = []; groups.append(group)
            group.append((value, view, target))
    for group in groups:
        anchor = group[0][0]
        if all(view == target for _, view, target in group):
            continue
        if any(torch_alias(anchor, v) for v in fixed) or any(torch_alias(anchor, v) for v, _ in placements.bindings):
            continue
        # The complete declared storage domain covers siblings and differently
        # typed aliases as well as the constrained consumer slices.
        domain = anchor.as_strided((anchor.untyped_storage().nbytes() // anchor.element_size(),), (1,), 0)
        values = list({(value.element_size(), descriptor(value)): value
                       for value in [domain, *(v for v in described if torch_alias(anchor, v))]}.values())
        group = list({(value.element_size(), view, target): (value, view, target)
                      for value, view, target in group}.values())
        unit = math.gcd(*(value.element_size() for value in values))
        fields = [value.element_size() // unit for value in values]
        uses = [view.components(value.element_size() // unit) for value, view, _ in group]
        targets = [target.components(value.element_size() // unit) for value, _, target in group]
        typed = [descriptor(value).components(width) for value, width in zip(values, fields)]
        placed = _stream.place_layout(anchor.untyped_storage().nbytes() // unit, uses, targets, typed, fields)
        if placed is None:
            continue
        width = domain.element_size() // unit
        placed = placed.reshape((*domain.shape, width)).elements(width)
        extent = max((offset + count for offset, count in placed.ordered_spans()), default=0)
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
    return _stream.pipeline(_stream.view_prelude() + source.declaration(0) + destination.declaration(1)
                            + prefix + Path(__file__).with_suffix('.metal').read_text(), 'mesh_layout_copy')


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


def resolve(value, declared, *, writing=False, device=None, binding='indexed'):
    """Realize an explicit API requirement before recording the function.

    The result is either the supplied storage view or a preallocated destination
    and ordinary copy edges. Direction comes from the declared input/output role.
    """
    layout = requirements(value, (declared, binding))
    device = value.device.type if device is None else device
    if descriptor(value) == layout and value.device.type == device:
        return value, []
    from .native import empty_strided
    allocation = layout if layout.affine else layout.required('contiguous')
    placed = empty_strided(tuple(value.shape), tuple(allocation.strides), value.dtype, device)
    return placed, copies(placed, value) if writing else copies(value, placed)
