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


def matches(value, required):
    return descriptor(value).matches(tuple(required))


def _matches(shape, steps, required):
    return _stream.TensorLayout(tuple(shape), tuple(steps)).matches(tuple(required))


def index_source(bindings):
    """Emit addresses from the same descriptors carried into command accesses."""
    return _stream.view_prelude() + ''.join(_stream.operand_layout(value).declaration(index)
                                           for index, value in sorted(bindings.items()))


def place(graph, placements, fixed):
    """Propagate compatible API requirements back to whole-value producers.

    Caller storage and supplied destinations are fixed. A proper subdomain of
    a larger value needs producer splitting; incompatible demands need distinct
    layouts. Those cases retain their explicit materialization edges.
    """
    from .placement import indices
    from .native import empty_strided
    from torch.utils._pytree import tree_flatten
    described = [v for n in graph.nodes for v in tree_flatten(n.meta.get('val'))[0]]
    def whole(value):
        step = 1
        for n, stride in sorted(zip(value.shape, value.stride()), key=lambda axis: axis[1]):
            if n > 1:
                if stride != step:
                    return False
                step *= n
        return value.storage_offset() == 0 and value.numel() * value.element_size() == value.untyped_storage().nbytes()
    groups = []
    for node in graph.nodes:
        schema = getattr(node.target, '_schema', None)
        if getattr(schema, 'name', '') != 'torch_mesh::native_kernel':
            continue
        args = {a.name: node.args[i] if i < len(node.args) else node.kwargs[a.name]
                for i, a in enumerate(schema.arguments)}
        declarations = json.loads(args['operands'])
        for source, slot in zip(args['inputs'], args['input_slots'], strict=True):
            declaration = declarations[str(slot)]
            required = declaration['requirement']
            if required == 'indexed' or declaration['view'] != declaration['source']:
                continue
            value = source.meta['val']
            group = next((g for g in groups if torch_alias(g[0][0], value)), None)
            if group is None:
                group = []; groups.append(group)
            group.append((value, tuple(required)))
    for group in groups:
        if all(matches(value, required) for value, required in group):
            continue
        anchor = next(((v, s) for v, s in group if whole(v)), None)
        if anchor is None:
            continue
        value, required = anchor
        if any(torch_alias(value, v) for v in fixed) or any(torch_alias(value, v) for v, _ in placements.bindings):
            continue
        try:
            compatible = all(_matches(other.shape, indices(value, required, 0, other)[1], steps) for other, steps in group)
            if compatible:
                for other in described:
                    if torch_alias(value, other):
                        indices(value, required, 0, other)
        except ValueError:
            compatible = False
        if compatible:
            placed = empty_strided(tuple(value.shape), required, value.dtype, value.device.type)
            placements.bind(value, placed)


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
