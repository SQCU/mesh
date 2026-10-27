"""Resolve numerical layout relations into fixed views and native commands.

This module runs only during preparation. A required materialization is an
ordinary tiled data dependency, using preassigned region storage.
"""
from functools import lru_cache
import math
from pathlib import Path

from . import _stream
from .compilation import operands


def strides(shape):
    result, step = [], 1
    for n in reversed(shape):
        result.append(step)
        step *= max(1, n)
    return tuple(reversed(result))


def matches(value, required):
    return _matches(value.shape, value.stride(), required)


def _matches(shape, steps, required):
    return all(n <= 1 or a == b for n, a, b in zip(shape, steps, required, strict=True))


def index_source(bindings):
    """Specialize numerical indexing; no layout descriptor survives this pass."""
    return _stream.view_indices(bindings)


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
        for source, slot, required in zip(args['inputs'], args['input_slots'], args['input_strides'], strict=True):
            if slot in args['indexed']:
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
def _copy_pipeline(shape, source, destination, width):
    kind = {1: 'uchar', 2: 'ushort', 4: 'uint', 8: 'ulong'}[width]
    prefix = (f'#define MESH_COPY_INPUT_TYPE {kind}\n#define MESH_COPY_OUTPUT_TYPE {kind}\n#define MESH_COPY_COUNT {math.prod(shape)}ul\n'
              f'#define MESH_COPY_SOURCE ({_stream.view_index(shape, source, "i")})\n'
              f'#define MESH_COPY_DESTINATION ({_stream.view_index(shape, destination, "i")})\n')
    return _stream.pipeline(prefix + Path(__file__).with_suffix('.metal').read_text(), 'mesh_layout_copy')


def copies(source, destination):
    """Compile a necessary copy into independently schedulable logical tiles."""
    from .placement import same_view
    if same_view(source, destination) or not source.numel():
        return []
    if source.shape != destination.shape or source.dtype != destination.dtype:
        raise ValueError('layout materialization preserves the declared values and type')
    configuration = operands()
    limit = max(1, configuration['LAYOUT_COPY_TILE_BYTES'] // source.element_size())
    shape = tuple(source.shape)
    commands = []
    width = math.gcd(source.element_size(), 8)
    units = source.element_size() // width
    for origin, size in _stream.view_tiles(shape, limit):
        region = tuple(slice(first, first + count) for first, count in zip(origin, size))
        left, right = source[region], destination[region]
        sizes = (*left.shape, units)
        a, b = (*[s * units for s in left.stride()], 1), (*[s * units for s in right.stride()], 1)
        handle = _copy_pipeline(sizes, a, b, width)
        threads = min(configuration['GATHER_THREADS'], _stream.pipeline_facts(handle)[0])
        count = left.numel() * units
        commands.append((handle, [left, right], {}, (-(-count // threads), 1, 1), (threads, 1, 1), False,
                         1 | (2 << configuration['ACCESS_WRITE_SHIFT'])))
    return commands


def resolve(value, requirement, *, writing=False, device=None):
    """Resolve one declared API binding before recording the function.

    ``strided`` accepts the supplied view, ``contiguous`` requires row-major
    storage, and an element-stride tuple describes an API-specific layout.
    A materialization's direction follows its declared input/output role.
    """
    required = (value.stride() if requirement == 'strided' else
                strides(value.shape) if requirement == 'contiguous' else tuple(requirement))
    device = value.device.type if device is None else device
    if matches(value, required) and value.device.type == device:
        return value, []
    from .native import empty_strided
    placed = empty_strided(tuple(value.shape), required, value.dtype, device)
    return placed, copies(placed, value) if writing else copies(value, placed)
