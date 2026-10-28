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
    return [torch.empty_strided(shape, steps, dtype=getattr(torch, dtype), device=device)
            for shape, steps, dtype, device in zip(shapes, output_strides, dtypes, devices, strict=True)]


def declare(source, name, buffers, outputs, words, constant_slot, grid, threads, instance='', *, indexed, requirements):
    """Carry each use's layout and requirement across FX's tensor schema.

    The serialized operand descriptions contain only compiler metadata. Values,
    including computed indices, stay in the tensor operands. Preparation composes
    these descriptions with producer placement and emits the common records.
    """
    from .operand import TensorView
    views = {i: TensorView.of(value) for i, value in enumerate(buffers) if value is not None}
    declarations = {i: view.declaration('indexed' if i in indexed else tuple(requirements[i])) for i, view in views.items()}
    results, slots = [], []
    for index in outputs:
        value = views[index].storage
        found = next((j for j, held in enumerate(results) if held is value), None)
        if found is None:
            results.append(value); slots.append([]); found = len(results) - 1
        slots[found].append(index)
    input_slots = [i for i in views if i not in outputs]
    return operation([views[i].storage for i in input_slots], input_slots, slots,
                     [list(value.stride()) for value in results], [list(value.shape) for value in results],
                     [str(value.dtype).removeprefix('torch.') for value in results], [str(value.device) for value in results],
                     json.dumps(declarations, separators=(',', ':')),
                     source, name, instance, list(words), constant_slot, list(grid), list(threads))


@lru_cache(None)
def pipeline(source, name, instance, indices):
    from . import _stream
    return _stream.pipeline(_stream.metal_prelude() + indices + Path(__file__).with_name(source).read_text() + '\n' + instance, name)


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
        requirement = declaration['requirement']
        placed, copies = layout.resolve(view, requirement, writing=writing)
        (after if writing else commands).extend(copies)
        buffers[index] = placed
        if requirement == 'indexed':
            addresses[index] = placed
    for index, value in zip(input_slots, inputs, strict=True):
        prepare_binding(index, value, False)
        read |= 1 << index
    for slots, value in zip(output_slots, results, strict=True):
        for index in slots:
            prepare_binding(index, value, True)
            write |= 1 << index
    constants = {constant_slot: struct.pack(f'{len(words)}I', *(v & 0xffffffff for v in words))}
    commands.append((pipeline(source, name, instance, layout.index_source(addresses)), buffers, constants, tuple(grid), tuple(threads), False,
                     read | (write << _stream.compilation['ACCESS_WRITE_SHIFT'])))
    return results, commands + after
