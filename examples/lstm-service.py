import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch_mesh
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard
from torch_mesh import partition
from torch_mesh.backend import BROADCAST, Call
from torch_mesh.declarations import Preparations
from torch_mesh.configuration import placement, bind, record
from torch_mesh import _stream
import time

import chain
import lstm

CHUNK = 512


# docs/writing-a-service.md#the-lstm
def weights(config):
    """Each layer's w_ih and w_hh side by side, [4, H, K] (the files' own order: each gate's units in turn), and its
    bias [4, H], in region tensors, read memory-mapped (rank 0's, before the communicator exists, so no peer's wait
    spans this host work)."""
    L, KIN, H, folder = config['layers'], config['input_width'], config['width'], Path(config['weights'])
    load = lambda name: np.load(folder / f'{name}.npy', mmap_mode='r')
    made = []
    for l in range(L):
        w_ih, w_hh = load(f'w_ih_{l}'), load(f'w_hh_{l}')
        W = torch_mesh.empty(4, H, w_ih.shape[1] + H, dtype=torch.half)
        for c in range(0, 4 * H, CHUNK):
            W.view(4 * H, -1)[c:c + CHUNK].copy_(torch.from_numpy(np.concatenate([w_ih[c:c + CHUNK], w_hh[c:c + CHUNK]], 1)))
        made.append((W, torch_mesh.empty(4, H, dtype=torch.half).copy_(torch.from_numpy(load(f'b_{l}').reshape(4, H).copy()))))
    return made


# docs/writing-a-service.md#the-lstm
def main():
    config = json.loads(open(sys.argv[1]).read())
    held = weights(config) if os.environ['RANK'] == '0' else None
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    L, KIN, H = config['layers'], config['input_width'], config['width']
    mesh = DeviceMesh.from_group(dist.group.WORLD, 'mps', mesh_dim_names=('tp',))
    partition.attach(mesh, tp=H)
    steps, rows, inputs = chain.operand(config['input'], rank, KIN)
    configuration = placement(mesh, 'tp', {'function': 'lstm_sequence', 'layers': L, 'input_width': KIN,
                                        'width': H, 'steps': steps, 'rows': rows},
                            config.get('observations', config['output'] + '.observations.jsonl'))
    shape = lambda l: (4, H, (KIN if l == 0 else H) + H)
    split = lambda t: partition.distribute_tensor(t, mesh, [Shard(1)], src_data_rank=0)
    held = held or [(torch_mesh.empty(1, 1, shape(l)[2], dtype=torch.half).expand(shape(l)),
                     torch_mesh.empty(1, 1, dtype=torch.half).expand(4, H)) for l in range(L)]
    tick = lstm.Tick(mesh, [split(w) for w, _ in held], [split(b) for _, b in held], rows, steps)
    del held
    X = torch_mesh.empty(steps, rows, KIN, dtype=torch.half)
    O = torch_mesh.empty(steps, rows, H, dtype=torch.half) if rank == 0 else None
    t, k = torch.empty(1, dtype=torch.int64, device='mps').fill_(L - 1), config['steps_per_replay']
    recording = torch_mesh.record(tick.stepper(X, O, t), k)
    recording.issue(min(steps, k + steps % k))
    for s in range(L - 1):
        tick.starting(s, X[s])
    tick.reset()
    t.fill_(L - 1)
    if rank == 0:
        X.copy_(torch.from_numpy(inputs[:]))
    torch_mesh.check_memory()
    torch.mps.synchronize()
    ticks = steps + L - 1
    group = mesh.get_group('tp')
    groups, forms = recording.schedule(group, steps)
    initial = [partition.gathered(tick.mine, 2, tick.sizes, group)] * (L - 1)
    initial_forms = [(tick.mine, None, 2, tick.sizes, group.group_name, False, 1)] * (L - 1)
    bind(configuration, [[Call(BROADCAST, X.numel(), X.dtype, root=0)], *initial, *groups],
                       Preparations([*initial_forms, *forms]))
    started = time.monotonic()
    dist.broadcast(X, src=0)
    for s in range(L - 1):
        tick.starting(s, X[s])
    recording.issue(steps)
    completion = _stream.completion()
    _stream.publish_boundary()
    blocks = partition.split(mesh, 'tp', H)
    summary = dict(blocks=blocks, steps_per_replay=k, layers=L, rows=rows,
                   input_width=KIN, width=H, steps=steps, ticks=ticks, latency_ticks=L)
    record(configuration, completion, started, steps * rows * H * torch.half.itemsize, summary)
    if O is not None:
        torch.from_numpy(np.lib.format.open_memmap(config['output'], 'w+', np.float16, (steps, rows, H))).copy_(O)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
