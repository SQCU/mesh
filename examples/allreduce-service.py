"""The all-reduce through the compiler (torch_mesh/compiler.py: a reduce-scatter's REDUCE steps landed in roots of their
own and folded into the operand, then an all-gather): each rank's x filled with rank + 1, summed in place over every
rank, so every element is 1 + 2 + ... + size on every rank; each run's time and the bytes a rank moved. The gradient
sum a trainable program needs (docs/facts.md). A run is bounded at 8 s."""
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


def main(config=None):
    if config is None:
        config = json.load(sys.stdin if sys.argv[1] == '-' else open(sys.argv[1]))
    units, width = config['units'], config['width']
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    mesh = DeviceMesh.from_group(dist.group.WORLD, 'mps', mesh_dim_names=('tp',))
    partition.attach(mesh, tp=units)
    group = mesh.get_group('tp').group_name
    x = torch_mesh.empty(units, width, dtype=torch.half)
    t = torch.empty(1, dtype=torch.int64, device='mps').fill_(0)

    def reduce(t):
        torch.ops.torch_mesh.allreduce_(x, group)
        t.add_(1)
    program = Program([(reduce, (t,))])
    repeats = config.get('repeats', 3)
    begun = now()
    program.prepare(repeats)
    print(json.dumps(dict(phase='prepared', seconds=round(now() - begun, 3))), flush=True)
    total = size * (size + 1) // 2
    for repeat in range(repeats):
        x.fill_(rank + 1)
        started = now()
        program.forward()
        while (ended := program.completion()) is None:
            if now() - started > 8.0:
                raise TimeoutError('the run did not complete within 8 s (a run is bounded: docs/facts.md)')
            time.sleep(0)
        torch.mps.synchronize()
        wrong = int((x.float() != total).sum().item())
        moved = 2 * x.numel() * x.element_size() * (size - 1) // size  # a reduce-scatter's and an all-gather's bytes
        emit({**measure(started, ended, moved, 1), 'workload': 'allreduce', 'rank': rank, 'size': size,
              'report': dict(units=units, width=width, repeat=repeat, elements_wrong=wrong, expected=total)})
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
