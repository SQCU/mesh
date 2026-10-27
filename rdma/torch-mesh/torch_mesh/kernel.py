"""Numerical kernel declarations lowered to the common prepared command records."""
from functools import lru_cache
from pathlib import Path
import struct

import torch


@torch.library.custom_op('torch_mesh::native_kernel', mutates_args=(),
    schema='(Tensor[] inputs, int[] input_slots, int[][] output_slots, int[][] shapes, str[] dtypes, '
           'str source, str name, str instance, int[] words, int constant_slot, int[] grid, int[] threads) -> Tensor[]')
def operation(inputs: list[torch.Tensor], input_slots: list[int], output_slots: list[list[int]],
              shapes: list[list[int]], dtypes: list[str], source: str, name: str, instance: str,
              words: list[int], constant_slot: int, grid: list[int], threads: list[int]) -> list[torch.Tensor]:
    raise RuntimeError('native kernels require whole-function preparation')


@operation.register_fake
def _shape(inputs, input_slots, output_slots, shapes, dtypes, source, name, instance, words, constant_slot, grid, threads):
    return [torch.empty(shape, dtype=getattr(torch, dtype), device=inputs[0].device) for shape, dtype in zip(shapes, dtypes)]


def declare(source, name, buffers, outputs, words, constant_slot, grid, threads, instance=''):
    """Output roles are supplied by the numerical builder, before tracing."""
    results, slots = [], []
    for index in outputs:
        value = buffers[index]
        found = next((j for j, held in enumerate(results) if held is value), None)
        if found is None:
            results.append(value); slots.append([]); found = len(results) - 1
        slots[found].append(index)
    inputs, input_slots = [], []
    for index, value in enumerate(buffers):
        if value is not None and index not in outputs:
            inputs.append(value); input_slots.append(index)
    return operation(inputs, input_slots, slots, [list(value.shape) for value in results],
                     [str(value.dtype).removeprefix('torch.') for value in results], source, name, instance,
                     list(words), constant_slot, list(grid), list(threads))


@lru_cache(None)
def pipeline(source, name, instance):
    from . import _stream
    return _stream.pipeline(Path(__file__).with_name(source).read_text() + '\n' + instance, name)


def prepare(inputs, input_slots, output_slots, shapes, dtypes, source, name, instance, words, constant_slot, grid, threads, *, outputs=None):
    from . import _stream, empty
    from .placement import output
    results = [output(shape, getattr(torch, dtype), inputs[0].device, out)
               for shape, dtype, out in zip(shapes, dtypes, outputs if outputs is not None else [None] * len(shapes), strict=True)]
    if any(not value.is_contiguous() for value in results):
        raise ValueError('opaque native kernel layout is fixed by its declaration; supply its output strides in the kernel interface')
    if any(v == 0 for v in grid):
        return results, []
    count = max([constant_slot, *input_slots, *(i for slots in output_slots for i in slots)]) + 1
    buffers = [None] * count
    read = write = 0
    for index, value in zip(input_slots, inputs):
        buffers[index] = value; read |= 1 << index
    for slots, value in zip(output_slots, results):
        for index in slots:
            buffers[index] = value; write |= 1 << index
    constants = {constant_slot: struct.pack(f'{len(words)}I', *(v & 0xffffffff for v in words))}
    return results, [(pipeline(source, name, instance), buffers, constants, tuple(grid), tuple(threads), False,
                      read | (write << _stream.compilation['ACCESS_WRITE_SHIFT']))]
