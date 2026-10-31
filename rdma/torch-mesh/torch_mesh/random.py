"""Counter random source equations, lowered with the surrounding tensor graph.

Philox4x32-10 uses the same round constants and word arrangement as
ATen/core/PhiloxRNGEngine.h. The key and counter are ordinary page values.
There is no generator object, host sampling call, or runtime dispatch choice.
"""
import math

import torch


def _multiply_words(word, multiplier):
    # Products remain below 2**48, including on signed int64 lowerings.
    low = (word & 0xffff) * multiplier
    upper = (word >> 16) * multiplier + (low >> 16)
    return ((upper & 0xffff) << 16) | (low & 0xffff), upper >> 16


def philox_words(counter, key):
    """Four unsigned 32-bit words represented as nonnegative int64 tensors."""
    a, b = counter & 0xffffffff, (counter >> 32) & 0xffffffff
    c, d = torch.zeros_like(a), torch.zeros_like(a)
    k0, k1 = key[1].to(torch.int64), key[0].to(torch.int64)
    for _ in range(10):
        lo0, hi0 = _multiply_words(a, 0xD2511F53)
        lo1, hi1 = _multiply_words(c, 0xCD9E8D57)
        a, b, c, d = hi1 ^ b ^ k0, lo1, hi0 ^ d ^ k1, lo0
        k0, k1 = (k0 + 0x9E3779B9) & 0xffffffff, (k1 + 0xBB67AE85) & 0xffffffff
    return a, b, c, d


def normal(shape, key, *, device=None, dtype=torch.float32):
    """One normal value per logical element, keyed independently of placement.

    An open float32 uniform interval keeps log finite at both endpoints.
    Repartitioning preserves each value's logical counter and sampling key.
    """
    count = math.prod(shape)
    counter = torch.arange(count, dtype=torch.int64, device=key.device if device is None else device)
    first, second, _, _ = philox_words(counter, key)
    u = ((first >> 9).to(torch.float32) + 0.5) * (2.0 ** -23)
    v = ((second >> 9).to(torch.float32) + 0.5) * (2.0 ** -23)
    return (torch.sqrt(-2.0 * torch.log(u)) * torch.cos(2.0 * math.pi * v)).to(dtype).reshape(shape)
