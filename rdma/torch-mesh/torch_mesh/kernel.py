"""Numerical kernel declarations lowered to the common prepared command records."""
from functools import lru_cache
from pathlib import Path
import struct
import json

import torch


@torch.library.custom_op('torch_mesh::native_kernel', mutates_args=(),
    schema='(Tensor[] inputs, int[] input_slots, int[][] output_slots, int[][] output_strides, int[][] shapes, str[] dtypes, str[] devices, '
           'str operands, str source, str name, str instance, int[] words, int constant_slot, int[] grid, int[] threads) -> Tensor[]')
def operation(inputs: list[torch.Tensor], input_slots: list[int], output_slots: list[list[int]], output_strides: list[list[int]],
              shapes: list[list[int]], dtypes: list[str], devices: list[str], operands: str, source: str, name: str, instance: str,
              words: list[int], constant_slot: int, grid: list[int], threads: list[int]) -> list[torch.Tensor]:
    raise RuntimeError('native kernels require whole-function preparation')


@operation.register_fake
def _shape(inputs, input_slots, output_slots, output_strides, shapes, dtypes, devices, operands, source, name, instance, words, constant_slot, grid, threads):
    from .operand import fake_outputs
    return fake_outputs(shapes, output_strides, dtypes, devices)


def declare(source, name, buffers, outputs, words, constant_slot, grid, threads, instance='', *, indexed, requirements, indices=None, prefix=''):
    """Carry each use's layout and requirement across FX's tensor schema.

    The serialized operand descriptions contain only compiler metadata. Values,
    including computed indices, stay in the tensor operands. Preparation composes
    these descriptions with producer placement and emits the common records.
    """
    from .operand import signature
    uses = {i: ('indexed' if i in indexed else requirements[i],) for i, value in enumerate(buffers) if value is not None}
    operands = signature(buffers, outputs, uses)
    if indices or prefix:
        declarations = json.loads(operands['operands'])
        if indices:
            declarations['indices'] = indices
        if prefix:
            declarations['prefix'] = prefix
        operands['operands'] = json.dumps(declarations, separators=(',', ':'))
    return operation(**operands, source=source, name=name, instance=instance,
                     words=list(words), constant_slot=constant_slot, grid=list(grid), threads=list(threads))


@lru_cache(None)
def _text(source):
    return Path(__file__).with_name(source).read_text()


def prepare(inputs, input_slots, output_slots, output_strides, shapes, dtypes, devices, operands, source, name, instance, words, constant_slot, grid, threads, *, outputs=None):
    from . import _stream, layout
    from .operand import bind
    from .placement import output
    from .native import empty_strided
    declarations = json.loads(operands)
    results = []
    for shape, steps, dtype, device, out in zip(shapes, output_strides, dtypes, devices,
            outputs if outputs is not None else [None] * len(shapes), strict=True):
        if out is None:
            out = empty_strided(tuple(shape), tuple(steps), getattr(torch, dtype), device)
        results.append(output(shape, getattr(torch, dtype), device, out))
    if any(v == 0 for v in grid):
        return results, []
    count = max([constant_slot, *input_slots, *(i for slots in output_slots for i in slots)]) + 1
    buffers = [None] * count
    read = write = 0
    commands, addresses, after = [], {}, []
    def prepare_binding(index, value, writing):
        declaration = declarations[str(index)]
        view = bind(value, declaration)
        requirements = declaration['requirements']
        placed, copies = layout.resolve(view, requirements, writing=writing)
        (after if writing else commands).extend(copies)
        buffers[index] = placed
        addresses[index] = placed
    for index, value in zip(input_slots, inputs, strict=True):
        prepare_binding(index, value, False)
        read |= 1 << index
    for slots, value in zip(output_slots, results, strict=True):
        for index in slots:
            prepare_binding(index, value, True)
            write |= 1 << index
    constants = {constant_slot: struct.pack(f'{len(words)}I', *(v & 0xffffffff for v in words))} if words else {}
    handle = layout.pipeline(_stream.metal_prelude(), addresses, declarations.get('prefix', '') + _text(source) + '\n', name,
                             declarations.get('indices'), instance)
    commands.append((handle, buffers, constants, tuple(grid), tuple(threads), False,
                     read | (write << _stream.compilation['ACCESS_WRITE_SHIFT'])))
    return results, commands + after
