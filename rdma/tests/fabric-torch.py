"""design/whole-function.md#measurements"""
import datetime
import json
import os

import numpy as np
import torch
import torch.distributed as dist
import torch_mesh
from torch_mesh.backend import ALLREDUCE, WRITE, Call, ProcessGroupMesh

import mesh
from routing import Fabric

rank, size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
store = dist.TCPStore(os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT']), size, rank == 0,
                     timeout=datetime.timedelta(seconds=60))
group = ProcessGroupMesh(rank, size, store)
fabric = Fabric.of(group.comm)
parts = {f'route:{a}:{b}': [1 if len(p) == 1 else 7 for p in paths] for (a, b), paths in fabric.paths.items()}
fabric.install(group.comm, parts, chunk_pages=4)
count = 1048593
source = torch_mesh.empty(count, dtype=torch.int32)
landing = [torch_mesh.empty(count, dtype=torch.int32) for _ in fabric.paths]
host = torch_mesh.empty(count, dtype=torch.int32, device='cpu')
group.plan([([Call(WRITE, count, torch.int32, root=a, destination=b) for a, b in fabric.paths], 1)])
for t in range(4):
    store.set(f'begin/{t}/{rank}', b'1')
    for peer in range(size):
        store.get(f'begin/{t}/{peer}')
    source.fill_(rank + 100 * t)
    for target in landing:
        target.fill_(-1)
    group.start_coalescing(torch.device('mps'))
    for (a, b), target in zip(fabric.paths, landing):
        group.write(source, target, a, b)
    group.end_coalescing(torch.device('mps'))
    torch.mps.synchronize()
    for (a, b), target in zip(fabric.paths, landing):
        if b != rank:
            continue
        host.copy_(target)
        torch.mps.synchronize()
        before = np.flatnonzero(host.numpy() != a + 100 * t)
        print(json.dumps({"rank": rank, "source": a, "iteration": t, "before_bad": len(before), "first": before[:2].tolist()}), flush=True)
        target.add_(1)
        host.copy_(target)
        torch.mps.synchronize()
        status = mesh.ffi.new('ncclResult_t *')
        mesh.check(mesh.lib.ncclCommGetAsyncError(group.comm, status))
        if status[0]:
            raise RuntimeError(mesh.ffi.string(mesh.lib.ncclGetLastError(group.comm)).decode())
        bad = np.flatnonzero(host.numpy() != a + 100 * t + 1)
        if len(bad):
            raise RuntimeError(f'GPU write {a}->{b}, iteration {t}: {len(bad)} wrong; first {bad[:8].tolist()}, '
                               f'values {host.numpy()[bad[:8]].tolist()}, expected {a + 100 * t + 1}')
print(json.dumps({'rank': rank, 'gpu_all_pairs': True, 'iterations': 4, 'elements': count}), flush=True)
group.plan([[Call(ALLREDUCE, count, torch.int32)]])
for t in range(4):
    source.fill_(rank + 100 * t)
    group.allreduce([source])
    host.copy_(source)
    torch.mps.synchronize()
    expected = size * 100 * t + size * (size - 1) // 2
    bad = np.flatnonzero(host.numpy() != expected)
    if len(bad):
        raise RuntimeError(f'GPU allreduce iteration {t}: {len(bad)} wrong; first {bad[:8].tolist()}, '
                           f'values {host.numpy()[bad[:8]].tolist()}, expected {expected}')
print(json.dumps({'rank': rank, 'gpu_allreduce': True, 'iterations': 4, 'elements': count}), flush=True)
mesh.check(mesh.lib.ncclCommDestroy(group.comm))
