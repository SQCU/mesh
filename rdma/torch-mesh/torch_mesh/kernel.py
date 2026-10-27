"""Numerical kernel declarations lowered to the common prepared command records."""
from functools import lru_cache
from pathlib import Path
import struct

import torch


@torch.library.custom_op('torch_mesh::native_kernel', mutates_args=(),
    schema='(Tensor[] inputs, int[] input_slots, int[][] input_strides, int[] indexed, int[][] output_slots, int[][] output_strides, int[][] shapes, str[] dtypes, '
           'str source, str name, str instance, int[] words, int constant_slot, int[] grid, int[] threads) -> Tensor[]')
def operation(inputs: list[torch.Tensor], input_slots: list[int], input_strides: list[list[int]], indexed: list[int], output_slots: list[list[int]],
              output_strides: list[list[int]], shapes: list[list[int]], dtypes: list[str], source: str, name: str, instance: str,
              words: list[int], constant_slot: int, grid: list[int], threads: list[int]) -> list[torch.Tensor]:
    raise RuntimeError('native kernels require whole-function preparation')


@operation.register_fake
def _shape(inputs, input_slots, input_strides, indexed, output_slots, output_strides, shapes, dtypes, source, name, instance, words, constant_slot, grid, threads):
    return [torch.empty_strided(shape, steps, dtype=getattr(torch, dtype), device=inputs[0].device)
            for shape, steps, dtype in zip(shapes, output_strides, dtypes, strict=True)]


def declare(source, name, buffers, outputs, words, constant_slot, grid, threads, instance='', *, indexed, requirements):
    """Declare indexed bindings and fixed-layout API bindings before tracing.

    Indexed bindings use logical linear element indices in the numerical
    source. Other bindings retain their declared strides as a requirement.
    Both are compiler operands, including every output binding.
    """
    results, slots = [], []
    for index in outputs:
        value = buffers[index]
        found = next((j for j, held in enumerate(results) if held is value), None)
        if found is None:
            results.append(value); slots.append([]); found = len(results) - 1
        if slots[found] and tuple(requirements[index]) != tuple(requirements[slots[found][0]]):
            raise ValueError('aliased kernel outputs require the same physical layout')
        slots[found].append(index)
    inputs, input_slots = [], []
    for index, value in enumerate(buffers):
        if value is not None and index not in outputs:
            inputs.append(value); input_slots.append(index)
    return operation(inputs, input_slots, [list(requirements[index]) for index in input_slots], list(indexed), slots,
                     [list(requirements[group[0]]) for group in slots], [list(value.shape) for value in results],
                     [str(value.dtype).removeprefix('torch.') for value in results], source, name, instance,
                     list(words), constant_slot, list(grid), list(threads))


@lru_cache(None)
def pipeline(source, name, instance, indices):
    from . import _stream
    return _stream.pipeline(_stream.metal_prelude() + indices + Path(__file__).with_name(source).read_text() + '\n' + instance, name)


def prepare(inputs, input_slots, input_strides, indexed, output_slots, output_strides, shapes, dtypes, source, name, instance, words, constant_slot, grid, threads, *, outputs=None):
    from . import _stream
    from . import layout
    from .placement import output
    from .native import empty_strided
    results = []
    for slots, shape, steps, dtype, out in zip(output_slots, shapes, output_strides, dtypes,
            outputs if outputs is not None else [None] * len(shapes), strict=True):
        if out is None:
            out = empty_strided(tuple(shape), tuple(steps), getattr(torch, dtype), inputs[0].device.type)
        results.append(output(shape, getattr(torch, dtype), inputs[0].device, out))
    if any(v == 0 for v in grid):
        return results, []
    count = max([constant_slot, *input_slots, *(i for slots in output_slots for i in slots)]) + 1
    buffers = [None] * count
    read = write = 0
    commands, addresses, after = [], [], []
    def address(index, value):
        offset = value.storage_offset() if value.is_mps else value.data_ptr() // value.element_size()
        addresses.append((index, tuple(value.shape), tuple(value.stride()), offset))
    for index, value, required in zip(input_slots, inputs, input_strides, strict=True):
        if index in indexed:
            address(index, value)
        else:
            value, copies = layout.resolve(value, required)
            commands.extend(copies)
        buffers[index] = value; read |= 1 << index
    for slots, value, required in zip(output_slots, results, output_strides, strict=True):
        # A fixed-layout result is materialized only when it cannot occupy
        # the destination's layout. Most coded outputs are directly indexed.
        placed = value
        if any(index not in indexed for index in slots):
            placed, copies = layout.resolve(value, required, writing=True)
            after.extend(copies)
        for index in slots:
            buffers[index] = placed; write |= 1 << index
            if index in indexed:
                address(index, placed)
    constants = {constant_slot: struct.pack(f'{len(words)}I', *(v & 0xffffffff for v in words))}
    commands.append((pipeline(source, name, instance, layout.index_source(addresses)), buffers, constants, tuple(grid), tuple(threads), False,
                     read | (write << _stream.compilation['ACCESS_WRITE_SHIFT'])))
    return results, commands + after
