import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch_mesh
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard
from torch_mesh import partition
from torch_mesh.backend import BROADCAST, Call
from torch_mesh.declarations import Function, Preparations
from torch_mesh.configuration import placement, bind, record
from torch_mesh import _stream
import time

import chain
import lstm

CHUNK = 512


# docs/writing-a-service.md#the-lstm
def weights(config, values):
    L, H = config['layers'], config['width']
    made = []
    for l in range(L):
        w_ih, w_hh = values[f'w_ih_{l}'], values[f'w_hh_{l}']
        W = torch_mesh.empty(4, H, w_ih.shape[1] + H, dtype=torch.half)
        for c in range(0, 4 * H, CHUNK):
            W.view(4 * H, -1)[c:c + CHUNK].copy_(torch.from_numpy(np.concatenate([w_ih[c:c + CHUNK], w_hh[c:c + CHUNK]], 1)))
        made.append((W, torch_mesh.empty(4, H, dtype=torch.half).copy_(torch.from_numpy(values[f'b_{l}'].reshape(4, H)))))
    return made


# docs/writing-a-service.md#the-lstm
def main():
    config = json.load(sys.stdin if sys.argv[1] == '-' else open(sys.argv[1]))
    steps, rows = config['steps'], config['rows']
    inputs, held = None, None
    if os.environ['RANK'] == '0':
        inputs, values = lstm.inputs(config, steps, rows)
        held = weights(config, values)
        del values
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    L, KIN, H = config['layers'], config['input_width'], config['width']
    mesh = DeviceMesh.from_group(dist.group.WORLD, 'mps', mesh_dim_names=('tp',))
    partition.attach(mesh, tp=H)
    configuration = placement(mesh, 'tp', {'function': 'lstm_sequence', 'layers': L, 'input_width': KIN,
                                        'width': H, 'steps': steps, 'rows': rows},
                            config['observations'])
    shape = lambda l: (4, H, (KIN if l == 0 else H) + H)
    split = lambda t: partition.distribute_tensor(t, mesh, [Shard(1)], src_data_rank=0)
    held = held or [(torch_mesh.empty(1, 1, shape(l)[2], dtype=torch.half).expand(shape(l)),
                     torch_mesh.empty(1, 1, dtype=torch.half).expand(4, H)) for l in range(L)]
    tick = lstm.Tick(mesh, [split(w) for w, _ in held], [split(b) for _, b in held], rows, steps)
    del held
    X = torch_mesh.empty(steps, rows, KIN, dtype=torch.half)
    O = torch_mesh.empty(steps, rows, H, dtype=torch.half) if rank == 0 else None
    t, k = torch.empty(1, dtype=torch.int64, device='mps').fill_(L - 1), config['steps_per_replay']
    group = mesh.get_group('tp')
    full, rest = divmod(steps, k)
    chunks = [k] * full + [rest] * bool(rest)
    functions = {}
    for count in dict.fromkeys(chunks):
        function = Function(tick.stepper(X, O, count), t, compiler='mesh')
        function.declare(group)
        functions[count] = function
    initial = [Function(tick.starting, s, X[s], compiler='mesh') for s in range(L - 1)]
    for function in initial:
        function.declare(group)
    ordered = [*initial, *(functions[count] for count in chunks)]
    runs = tuple(function.run for function in ordered)
    groups = [calls for function in ordered for calls in function.groups.get(group.group_name, [])]
    forms = [form for function in ordered for form in function.forms]
    if rank == 0:
        X.copy_(torch.from_numpy(inputs[:]))
    torch_mesh.check_memory()
    ticks = steps + L - 1
    bind(configuration, [[Call(BROADCAST, X.numel(), X.dtype, root=0)], *groups], Preparations(forms))
    started = time.monotonic()
    dist.broadcast(X, src=0)
    for run in runs:
        run()
    completion = _stream.completion()
    _stream.publish_boundary()
    blocks = partition.split(mesh, 'tp', H)
    summary = dict(blocks=blocks, steps_per_replay=k, layers=L, rows=rows,
                   input_width=KIN, width=H, steps=steps, ticks=ticks, latency_ticks=L)
    record(configuration, completion, started, steps * rows * H * torch.half.itemsize, summary)
    dist.destroy_process_group()
    return O


if __name__ == '__main__':
    main()
