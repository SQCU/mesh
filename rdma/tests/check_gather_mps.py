"""Actual GPU checks of compact generic gather/sum layouts, with null transport bindings.

This validates producer/consumer kernels and adjacent byte ownership. It makes
no claim about NIC progress; fused-physical.py supplies that cluster check.
"""
import json
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch_mesh import fused, partition
from routing import Fabric


def check(dtype, ranks, dim, strided, zero, adds):
    channels = 2 * ranks
    sizes = [3] * ranks if adds else [r + 1 for r in range(ranks)]
    if zero:
        sizes[-1] = 0
    fabric = Fabric(ranks, [{'nodes': (a, b)} for a in range(ranks) for b in range(a + 1, ranks)])
    tensors, originals, forms = [], [], []
    for rank in range(ranks):
        shape = (sizes[rank], 5) if dim == 0 else (2, sizes[rank], 5)
        original = (torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape) % 29 + rank * 32).to(dtype)
        storage_shape = (*shape[:-1], shape[-1] * 2 + 3) if strided else shape
        storage = torch.empty(storage_shape, dtype=dtype, device='mps')
        source = storage[..., 1:1 + 2 * shape[-1]:2] if strided else storage
        source.copy_(original)
        group = SimpleNamespace(rank=lambda r=rank: r, handle=1, page=16384, fabric=fabric)
        with patch.object(dist.distributed_c10d, '_resolve_process_group', return_value=group):
            form = partition._fused_form(source, None, dim, sizes, 'local-kernel-test', adds, channels)
        tensors.append(source)
        originals.append(original)
        forms.append(form)
    counts, starts = forms[0][15], forms[0][14]
    assert counts == [t.numel() for t in tensors]
    poison = 173 if dtype == torch.uint8 else 0 if dtype == torch.bool else -77
    backing = torch.full((sum(counts) + 6,), poison, dtype=dtype, device='mps')
    slot = backing[3:-3]
    for rank, (source, form) in enumerate(zip(tensors, forms)):
        fused.encode(form[4], [source, slot.narrow(0, starts[rank], counts[rank]), form[17]],
                     form[5], 15, form[7], 1, form[8], channels)
    outputs = []
    for form in forms:
        output = torch.empty(form[16], dtype=dtype, device='mps')
        if form[18] >= 0:
            fused.encode(form[18], [], form[10], 15, 1, 1, form[13], channels)
        fused.encode(form[9], [slot, output], form[10], 15, form[12], 1, form[13], channels)
        outputs.append(output)
    torch.mps.synchronize()
    expected = sum(x.float() for x in originals).to(dtype) if adds else torch.cat(originals, dim)
    for rank, output in enumerate(outputs):
        assert torch.equal(output.cpu(), expected), (dtype, ranks, dim, strided, zero, adds, rank)
    assert torch.all(backing[:3].cpu() == poison) and torch.all(backing[-3:].cpu() == poison)
    for rank, original in enumerate(originals):
        assert torch.equal(slot.narrow(0, starts[rank], counts[rank]).cpu(), original.reshape(-1))


def main():
    cases = 0
    for dtype in (torch.float16, torch.float32, torch.uint8, torch.int8, torch.bool, torch.int32, torch.int64):
        for ranks in (2, 3):
            for dim in (0, 1):
                for strided in (False, True):
                    for zero in (False, True):
                        check(dtype, ranks, dim, strided, zero, False)
                        cases += 1
                    if dtype in (torch.float16, torch.float32):
                        check(dtype, ranks, dim, strided, False, True)
                        cases += 1
    print(json.dumps(dict(passed=True, cases=cases, exact_compact_layouts=True,
                          poisoned_neighbors_preserved=True, transport_bindings='null', nic_test=False)))


if __name__ == '__main__':
    main()
