import json
import time
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch_mesh
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard
from torch_mesh import partition
from torch_mesh.compiler import Program
from mesh_statistics import now, measure, emit

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
def main(config=None):
    if config is None:
        config = json.load(sys.stdin if sys.argv[1] == '-' else open(sys.argv[1]))
    steps, rows = config['steps'], config['rows']
    L, KIN, H = config['layers'], config['input_width'], config['width']
    k = config['parameters']['steps_per_replay']['value']
    expected = None
    # every rank draws the same inputs and weights from the configuration's seed and keeps its own share: no broadcast
    # of X into X and no scatter of the weights (design/page-maps-2026-11-02.md, Construction 7)
    inputs, values = lstm.inputs(config, steps, rows)
    held = weights(config, values)
    if os.environ['RANK'] == '0':
        # the outputs the run must give, from the same inputs and weights on the CPU (lstm.reference)
        expected = lstm.reference(config, inputs, values, steps)
    del values
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    mesh = DeviceMesh.from_group(dist.group.WORLD, 'mps', mesh_dim_names=('tp',))
    partition.attach(mesh, tp=H)
    split = lambda t: partition.distribute_tensor(t, mesh, [Shard(1)])
    tick = lstm.Tick(mesh, [split(w) for w, _ in held], [split(b) for _, b in held], rows, steps)
    del held
    X = torch_mesh.empty(steps, rows, KIN, dtype=torch.half)
    O = torch_mesh.empty(steps, rows, H, dtype=torch.half) if rank == 0 else None
    X.copy_(torch.from_numpy(inputs[:]))
    del inputs
    t = torch.empty(1, dtype=torch.int64, device='mps').fill_(L - 1)
    # the wavefront's first L - 1 ticks, then chunks of k steps, each length compiled once and instanced
    # (torch_mesh/compiler.py), the step counter t on the GPU selecting its input and output
    starting = [lambda _, s=s: tick.starting(s, X[s]) for s in range(L - 1)]
    whole, rest = divmod(steps, k)
    stepper = {n: tick.stepper(X, O, n) for n in {k, rest} if n}
    program = Program([(fn, (t,)) for fn in starting] + [(stepper[k], (t,))] * whole +
                      ([(stepper[rest], (t,))] if rest else []))
    torch_mesh.check_memory()
    repeats = config.get('repeats', 1)
    program.prepare(repeats)
    blocks = partition.split(mesh, 'tp', H)
    # the prepared run `repeats` times, each from the same state (the step counter and every layer's h and c set again
    # on the device once the run before has completed); each is one record
    for repeat in range(repeats):
        if repeat:
            t.fill_(L - 1)
            tick.reset()
        started = now()
        program.forward()
        while (ended := program.completion()) is None:
            if now() - started > 8.0:
                parts, landings = program.done[-1].waiting()
                print(json.dumps(dict(phase='timeout', parts_pending=parts, landings_waiting=len(landings), landings=[dict(word=hex(w), value=v, threshold=t) for w, v, t in landings[:16]])), flush=True)
                raise TimeoutError('the run did not complete within 8 s (a run is bounded: docs/facts.md)')
            time.sleep(0)
        summary = dict(blocks=blocks, steps_per_replay=k, layers=L, rows=rows, repeat=repeat, input_width=KIN,
                       width=H, steps=steps, ticks=steps + L - 1, latency_ticks=L)
        emit({**measure(started, ended, steps * rows * H * torch.half.itemsize, steps), 'workload': 'lstm_sequence',
              'rank': rank, 'size': size, 'report': summary})
    if expected is not None:
        # rank 0's outputs, every step's now complete, against the reference: the largest error over every step, row
        # and unit, and over the last step
        error = np.abs(O.cpu().float().numpy() - expected)
        print(json.dumps(dict(phase='checked', max_abs_error=float(error.max()), last_step_max_abs_error=float(error[-1].max()),
                              output_max_abs=float(np.abs(expected).max()))), flush=True)
    dist.destroy_process_group()
    return {'output': O, 'configuration': config.get('placement')}


if __name__ == '__main__':
    main()
