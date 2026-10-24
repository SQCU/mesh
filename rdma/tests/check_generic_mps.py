"""Exercise actual generic kernels locally; no NIC/protocol performance claim.

Run with the MPS torch environment. The transport prelude is compiled from the
checkout's current header; the gather source comes directly from partition.py.
"""
import ast
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch


root = Path(__file__).resolve().parents[1]
package = root / 'torch-mesh' / 'torch_mesh'
tree = ast.parse((package / 'partition.py').read_text())
source = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == '_GATHER_SOURCE' for t in n.targets))
with tempfile.TemporaryDirectory(prefix='mesh-generic-prelude-') as temporary:
    main, executable = Path(temporary) / 'prelude.m', Path(temporary) / 'prelude'
    main.write_text('#include <stdio.h>\n#include "nccl-mesh-kernels.h"\n'
                    'int main(void) { fputs(STEPS.UTF8String, stdout); return 0; }\n')
    subprocess.run(['clang', '-fobjc-arc', '-framework', 'Foundation', '-I', str(root), str(main), '-o', str(executable)], check=True)
    steps = subprocess.check_output([str(executable)], text=True)
prelude = (root / 'mesh.metal').read_text() + steps + (package / 'argmax.metal').read_text()


def instance(name, *args):
    values = ', '.join(map(str, args))
    symbol = '_'.join([name, *map(str, args)])
    return symbol, f'template [[host_name("{symbol}")]] [[kernel]] decltype({name}<{values}>) {name}<{values}>;'


def words(n, channels, count, pre, post, rank, sizes, starts):
    data = [count, pre, post, rank, 0, *sizes, *starts]
    data += [0] * (len(data) % 2)
    return torch.tensor(data + [0] * (24 * channels), dtype=torch.int32, device='mps')


torch.manual_seed(9451)
checks = []
dummy = torch.zeros(1, dtype=torch.int32, device='mps')
for (n, channels), inline_wait in itertools.product(((2, 1), (3, 3), (3, 5), (3, 6), (4, 9)), (0, 1)):
    declarations = [instance('gather_ready', n, channels, 256)]
    declarations += [instance('gather_land', n, channels, inline_wait, 256)]
    declarations += [instance('sum_land', size, n, channels, inline_wait, 256) for size in (2, 4)]
    declarations += [instance('gather_publish', 3, half, channels, contiguous, 256)
                     for half in (1, 2) for contiguous in (0, 1)]
    shader = torch.mps.compile_shader(prelude + source + '\n'.join(line for _, line in declarations))
    ready, gather, sum_half, sum_float = [getattr(shader, name) for name, _ in declarations[:4]]
    for dtype in (torch.float16, torch.float32):
        half = dtype.itemsize // 2
        for pre, post in ((1, 1), (3, 5), (5, 257)):
            sizes = [r + 1 for r in range(n)]
            operands = [torch.randn(pre, size, post, dtype=dtype) for size in sizes]
            counts = [t.numel() for t in operands]
            starts = [0]
            for count in counts:
                starts.append(starts[-1] + (count + 31) // 32 * 32)
            lease = torch.zeros(starts[-1], dtype=dtype)
            for at, operand in zip(starts, operands):
                lease[at:at + operand.numel()] = operand.flatten()
            expected = torch.cat(operands, dim=1).flatten()
            lease = lease.to('mps')
            for rank in range(n):
                constants = words(n, channels, counts[rank] * half, pre, post * half, rank,
                                  sizes, [at * half for at in starts[:-1]])
                out = torch.full((expected.numel() + 4,), 123, dtype=dtype, device='mps')
                if not inline_wait:
                    ready(*([dummy] * 15), constants, threads=32, group_size=32)
                gather(lease, out, *([dummy] * 13), constants, threads=256 if inline_wait else 4096, group_size=256)
                actual = out.cpu()
                assert torch.equal(actual[:expected.numel()], expected), (n, channels, dtype, pre, post, rank)
                assert torch.all(actual[expected.numel():] == 123), 'gather output guard overwritten'
                checks.append(dict(kind='gather', ranks=n, channels=channels, dtype=str(dtype),
                                   pre=pre, post=post, rank=rank, inline_wait=inline_wait))
                source_operand = operands[rank].to('mps')
                backing = torch.zeros((pre, sizes[rank], post * 2 + 3), dtype=dtype, device='mps')
                strided = backing[..., 1:1 + 2 * post:2]
                strided.copy_(source_operand)
                for operand in (source_operand, strided, source_operand[:1].expand_as(source_operand)):
                    pwords = [counts[rank] * half, 0, *operand.shape, *operand.stride()]
                    publication_words = torch.tensor(pwords + [0] * (24 * channels), dtype=torch.int32, device='mps')
                    name, _ = instance('gather_publish', 3, half, channels, int(operand.is_contiguous()), 256)
                    publish = getattr(shader, name)
                    sent = torch.full((counts[rank] + 4,), 123, dtype=dtype, device='mps')
                    arrival = torch.zeros(1, dtype=torch.int32, device='mps')
                    publish(operand, sent, arrival, *([dummy] * 12), publication_words, threads=4096, group_size=256)
                    actual = sent.cpu()
                    assert torch.equal(actual[:counts[rank]], operand.cpu().flatten()), (n, channels, operand.stride())
                    assert torch.all(actual[counts[rank]:] == 123), 'publication output guard overwritten'
                    checks.append(dict(kind='publish', ranks=n, channels=channels, dtype=str(dtype), pre=pre, post=post,
                                       rank=rank, source_strides=operand.stride(), source_offset=operand.storage_offset()))
        for count in (1, 17, 1025):
            padded = (count + 31) // 32 * 32
            data = [torch.randn(count, dtype=dtype) for _ in range(n)]
            lease = torch.zeros(n * padded, dtype=dtype)
            for rank, operand in enumerate(data):
                lease[rank * padded:rank * padded + count] = operand
            expected = torch.stack(data).float().sum(0).to(dtype)
            constants = words(n, channels, count * half, 1, half, n - 1, [count] * n,
                              [rank * padded for rank in range(n)])
            out = torch.full((count + 4,), 123, dtype=dtype, device='mps')
            (sum_half if dtype == torch.float16 else sum_float)(lease.to('mps'), out, *([dummy] * 13), constants,
                                                               threads=256 if inline_wait else 4096, group_size=256)
            actual = out.cpu()
            assert torch.allclose(actual[:count], expected, atol=1e-6, rtol=1e-6)
            assert torch.all(actual[count:] == 123), 'sum output guard overwritten'
            checks.append(dict(kind='sum', ranks=n, channels=channels, dtype=str(dtype), count=count, inline_wait=inline_wait))
    name, declaration = instance('linear_publish', 4, 4, 2, 1, 2, channels)
    torch.mps.compile_shader(prelude + '\n' +
                             (package / 'linear.metal').read_text() + '\n' + declaration)

report = dict(cases=len(checks), checks=checks, torch=torch.__version__,
              scope='Actual GPU arithmetic, layout, bounds and physical binding extents; null-ready bindings only, no remote NIC test')
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key != 'checks'}))
