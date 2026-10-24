"""design/whole-function.md#measurements"""
import datetime
import json
import os

import numpy as np
import torch
import torch.distributed as dist
import torch_mesh
from torch_mesh import _stream, fused
from torch_mesh.backend import ALLREDUCE, WRITE, Call, ProcessGroupMesh

import mesh
from routing import Fabric, QueueSample

rank, size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
store = dist.TCPStore(os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT']), size, rank == 0,
                     timeout=datetime.timedelta(seconds=60))
group = ProcessGroupMesh(rank, size, store)
fabric = Fabric.of(group.comm)
parts = {f'route:{a}:{b}': [1 if len(p) == 1 else 7 for p in paths] for (a, b), paths in fabric.paths.items()}
fabric.install(group.comm, parts)
count = 1048593
producer = _stream.pipeline(fused.steps() + """
kernel void produce(device uchar *out [[buffer(0)]], constant uint2 &p [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < p.x) mesh_step_write<int>(out, i, int(i % 997) + int(p.y));
}
""", 'produce')
consumer = _stream.pipeline(fused.steps() + """
kernel void consume(device uchar *input [[buffer(0)]], device int *out [[buffer(1)]],
                    constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = mesh_step_read<int>(input, i) + 1;
}
""", 'consume')
expected_index = np.arange(count, dtype=np.int32) % 997
consumed = torch_mesh.empty(count, dtype=torch.int32)
source = torch_mesh.empty(count, dtype=torch.int32)
landing = [torch_mesh.empty(count, dtype=torch.int32) for _ in fabric.paths]
host = torch_mesh.empty(count, dtype=torch.int32, device='cpu')
group.plan([([Call(WRITE, count, torch.int32, root=a, destination=b) for a, b in fabric.paths], 1)])
for t in range(4):
    store.set(f'begin/{t}/{rank}', b'1')
    for peer in range(size):
        store.get(f'begin/{t}/{peer}')
    _stream.encode(producer, [source], [count, rank + 100 * t], 1, (count + 255) // 256, 1, 256)
    landing = [torch_mesh.empty(count, dtype=torch.int32) for _ in fabric.paths]
    group.start_coalescing(torch.device('mps'))
    for (a, b), target in zip(fabric.paths, landing):
        group.write(source, target, a, b)
    group.end_coalescing(torch.device('mps'))
    torch.mps.synchronize()
    for (a, b), target in zip(fabric.paths, landing):
        if b != rank:
            continue
        _stream.encode(consumer, [target, consumed], [count], 2, (count + 255) // 256, 1, 256)
        host.copy_(consumed)
        torch.mps.synchronize()
        status = mesh.ffi.new('ncclResult_t *')
        mesh.check(mesh.lib.ncclCommGetAsyncError(group.comm, status))
        if status[0]:
            raise RuntimeError(mesh.ffi.string(mesh.lib.ncclGetLastError(group.comm)).decode())
        bad = np.flatnonzero(host.numpy() != expected_index + a + 100 * t + 1)
        if len(bad):
            raise RuntimeError(f'GPU write {a}->{b}, iteration {t}: {len(bad)} wrong; first {bad[:8].tolist()}, '
                               f'values {host.numpy()[bad[:8]].tolist()}, expected {a + 100 * t + 1}')
print(json.dumps({'rank': rank, 'gpu_all_pairs': True, 'iterations': 4, 'elements': count}), flush=True)
group.plan([[Call(ALLREDUCE, count, torch.int32)]])
for in_place in (False, True):
    for t in range(4):
        store.set(f'reduce/{in_place}/{t}/{rank}', b'1')
        for peer in range(size):
            store.get(f'reduce/{in_place}/{t}/{peer}')
        _stream.encode(producer, [source], [count, rank + 100 * t], 1, (count + 255) // 256, 1, 256)
        result = source if in_place else torch_mesh.empty(count, dtype=torch.int32)
        group._issue([(ALLREDUCE, [source], [result], [], int(mesh.lib.ncclSum), 0)], [result]).wait()
        _stream.encode(consumer, [result, consumed], [count], 2, (count + 255) // 256, 1, 256)
        host.copy_(consumed)
        torch.mps.synchronize()
        expected = size * expected_index + size * 100 * t + size * (size - 1) // 2 + 1
        bad = np.flatnonzero(host.numpy() != expected)
        if len(bad):
            raise RuntimeError(f'GPU allreduce iteration {t}: {len(bad)} wrong; first {bad[:8].tolist()}, '
                               f'values {host.numpy()[bad[:8]].tolist()}, expected {expected[bad[:8]].tolist()}')
    print(json.dumps({'rank': rank, 'gpu_allreduce': True, 'in_place': in_place, 'iterations': 4, 'elements': count}), flush=True)
samples = QueueSample(group.comm, rank).read()
assert all(s['sent'] > 0 and s['landed'] > 0 for s in samples)
print(json.dumps({'rank': rank, 'cables': samples}), flush=True)
store.set(f'finished/{rank}', b'1')
for peer in range(size):
    store.get(f'finished/{peer}')
store.set(f'drained/{rank}', b'1')
if rank == 0:
    for peer in range(size):
        store.get(f'drained/{peer}')
mesh.check(mesh.lib.ncclCommDestroy(group.comm))
