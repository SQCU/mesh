"""The selection and its records for torch programs: the host's one way into argmax.metal, its kernels encoded on torch's
MPS stream (metal-microbench docs/kernels.md#speculative-decoding, sampling).  The noise is argmax.metal's mesh_gumbel
alone; nothing here computes it.

  draw(z, seed, keys, first=0, min_p=0)        each row's token at temperature 1 (mesh_draw): over its candidates c the
                                               greatest z(c) + g(seed, key, first + c), its id first + c; with min_p > 0
                                               a candidate below the row's greatest z plus log min_p is never taken
  retain(z, seed, positions, head, tail)       each row's distribution as a retained set (mesh_retain): ids
                                               [..., head + tail], their z, and the row's log Z

z is [..., V], rows of logits (on the GPU in float32: another device's or dtype's copied there); a row's key is the
position it predicts from, one for every row or one a row, so a row computed again draws the same token.

The retained set (the operator, 2026-10-14: "you have to choose a power of 2 and then accept the compromises that come
from the power of 2 thus chosen"): a head, the row's 2^k greatest z, and a tail, 2^m of the rest sampled without
replacement in proportion to their probabilities (the greatest z + g keyed as the draw keys the row: the Gumbel-top-k,
Kool, van Hoof & Welling 2019), each in ascending id order, so a record names no greatest entry, and the token
draw(z, seed, positions) takes is in it.  A divergence over a record takes what it excludes as one bucket, its mass
1 - sum exp(z - log Z) over the ids (the other row's at the same ids, over its own log Z): the coarse-grained KL is at
most the rows' (the log-sum inequality), and equal to it where nothing is excluded."""
import math
from pathlib import Path

import torch

from . import _stream
from .streamed import instance

SOURCE = Path(__file__).with_name('argmax.metal').read_text()
THREADS = 256
_PIPELINES = {}


def _kernel(kernel, *arguments):
    name, line = instance(kernel, THREADS, *arguments)
    if name not in _PIPELINES:
        _PIPELINES[name] = _stream.pipeline(SOURCE + '\n' + line + '\n', name)
    return _PIPELINES[name]


def _rows(z, keys):
    z2 = z.to('mps', torch.float32).reshape(-1, z.shape[-1]).contiguous()
    return z2, torch.as_tensor(keys, dtype=torch.int32, device=z2.device).reshape(-1).expand(z2.shape[0]).contiguous()


def draw(z, seed, keys, first=0, min_p=0):
    """Each row's token, int64 [...] (module docstring)."""
    z2, keys = _rows(z, keys)
    rows, v = z2.shape
    blocks = max(1, min(256, v // 2048))
    cutoff = z2.amax(1) + math.log(min_p) if min_p > 0 else torch.full((rows,), -math.inf, device=z2.device)
    ids = torch.empty(rows, dtype=torch.int64, device=z2.device)
    parts = torch.empty(2 * rows * blocks, device=z2.device)
    arrivals = torch.zeros(rows, dtype=torch.int32, device=z2.device)
    _stream.encode(_kernel('mesh_draw_rows'), [z2, keys, cutoff, ids, parts, arrivals], [v, seed, first, blocks], 6, blocks, rows, THREADS)
    return ids.reshape(z.shape[:-1])


def retain(z, seed, positions, head=2 ** 6, tail=2 ** 6):
    """Each row's retained set (module docstring): ids int64 [..., head + tail], their z float32 [..., head + tail], and
    log Z float32 [...]."""
    if not all(n > 0 and n & (n - 1) == 0 for n in (head, tail)) or head + tail >= z.shape[-1]:
        raise ValueError(f'retain: head {head} and tail {tail} are each a power of two, together fewer than a row\'s {z.shape[-1]}')
    z2, positions = _rows(z, positions)
    rows, v = z2.shape
    ids = torch.empty(rows, head + tail, dtype=torch.int64, device=z2.device)
    kept = torch.empty(rows, head + tail, device=z2.device)
    log_z = torch.empty(rows, device=z2.device)
    _stream.encode(_kernel('mesh_retain_rows', head, tail), [z2, positions, ids, kept, log_z], [v, seed], 5, 1, rows, THREADS)
    return ids.reshape(*z.shape[:-1], -1), kept.reshape(*z.shape[:-1], -1), log_z.reshape(z.shape[:-1])
