"""A data-parallel training step through the compiler, its gradient summed by the all-reduce (torch_mesh/compiler.py's
REDUCE steps): each rank holds its own shard of a regression batch (X, Y drawn from its rank's seed) and the same W;
a step is P = X W, E = P - Y, G = X^T E, G summed over every rank in place, W -= lr G / batch. `steps` steps a run.
Every rank prints its loss and a checksum of W (the same on every rank: the sum makes them equal), and rank 0 checks W
against the same full-batch descent on the CPU from every rank's shard. A run is bounded at 8 s."""
import json
import sys
import time

import torch
import torch.distributed as dist
import torch_mesh
from torch.distributed.device_mesh import DeviceMesh
from torch_mesh import partition
from torch_mesh.compiler import Program
from mesh_statistics import now, measure, emit


def shard(rank, rows, k, n, seed):
    g = torch.Generator().manual_seed(seed * 1000 + rank)
    x = torch.randn(rows, k, generator=g)
    truth = torch.randn(k, n, generator=torch.Generator().manual_seed(seed))
    return x, x @ truth + 0.01 * torch.randn(rows, n, generator=g)


def main(config=None):
    if config is None:
        config = json.load(sys.stdin if sys.argv[1] == '-' else open(sys.argv[1]))
    rows, k, n, steps, lr, seed = (config[key] for key in ('rows', 'inputs', 'outputs', 'steps', 'lr', 'seed'))
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    mesh = DeviceMesh.from_group(dist.group.WORLD, 'mps', mesh_dim_names=('tp',))
    partition.attach(mesh, tp=k)
    group = mesh.get_group('tp').group_name
    x_cpu, y_cpu = shard(rank, rows, k, n, seed)
    X = torch_mesh.empty(rows, k, dtype=torch.float32).copy_(x_cpu.to('mps'))
    Y = torch_mesh.empty(rows, n, dtype=torch.float32).copy_(y_cpu.to('mps'))
    W = torch_mesh.empty(k, n, dtype=torch.float32)
    G = torch_mesh.empty(k, n, dtype=torch.float32)
    scale = lr / (rows * size)
    t = torch.empty(1, dtype=torch.int64, device='mps').fill_(0)

    def step(t):
        E = X @ W - Y
        G.copy_(X.t() @ E)
        torch.ops.torch_mesh.allreduce_(G, group)
        W.copy_(W - G * scale)
        t.add_(1)
    program = Program([(step, (t,))] * steps)
    repeats = config.get('repeats', 3)
    begun = now()
    program.prepare(repeats)
    print(json.dumps(dict(phase='prepared', seconds=round(now() - begun, 3))), flush=True)
    for repeat in range(repeats):
        W.zero_()
        started = now()
        program.forward()
        while (ended := program.completion()) is None:
            if now() - started > 8.0:
                raise TimeoutError('the run did not complete within 8 s (a run is bounded: docs/facts.md)')
            time.sleep(0)
        torch.mps.synchronize()
        w = W.cpu()
        loss = float(((x_cpu @ w - y_cpu) ** 2).mean())
        flops = steps * 2 * 2 * rows * k * n
        emit({**measure(started, ended, flops, steps), 'workload': 'train_linear', 'rank': rank, 'size': size,
              'report': dict(rows=rows, inputs=k, outputs=n, steps=steps, repeat=repeat, loss=loss,
                             w_checksum=float(w.double().sum()), gflops=round(flops / (ended - started) / 1e9, 3))})
    if rank == 0:
        # the same descent on the CPU over every rank's shard: the gradient is their sum
        shards = [shard(r, rows, k, n, seed) for r in range(size)]
        ref = torch.zeros(k, n)
        for _ in range(steps):
            g = sum(x.t() @ (x @ ref - y) for x, y in shards)
            ref = ref - g * scale
        print(json.dumps(dict(phase='checked', max_abs_error=float((w - ref).abs().max()),
                              w_max_abs=float(ref.abs().max()))), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
