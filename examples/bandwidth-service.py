"""The collective's aggregate rate (the goal's second measure, after a machine's own): an all-gather of a [units, width]
half tensor along its units, each rank's block of units its segment, `rounds` gathers a run through the compiler
(torch_mesh/compiler.py) as the LSTM's gather runs; each rank's bytes received over the run's time, and their sum, the
collective's aggregate. A run is bounded at 8 s like every run (docs/facts.md)."""
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
    units, width, rounds = config['units'], config['width'], config['rounds']
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    mesh = DeviceMesh.from_group(dist.group.WORLD, 'mps', mesh_dim_names=('tp',))
    partition.attach(mesh, tp=units)
    group, sizes = mesh.get_group('tp').group_name, partition.split(mesh, 'tp', units)
    first = sum(sizes[:rank])
    h = torch_mesh.empty(units, width, dtype=torch.half)
    h.fill_(0)
    mine = h.narrow(0, first, sizes[rank])
    mine.fill_(rank + 1)
    t = torch.empty(1, dtype=torch.int64, device='mps').fill_(0)

    def gather(t):
        torch.ops.torch_mesh.allgatherv_out(mine, h, 0, sizes, group)
        t.add_(1)
    program = Program([(gather, (t,))] * rounds)
    repeats = config.get('repeats', 3)
    begun = now()
    program.prepare(repeats)
    print(json.dumps(dict(phase='prepared', seconds=round(now() - begun, 3))), flush=True)
    received = (units - sizes[rank]) * width * torch.half.itemsize * rounds
    for repeat in range(repeats):
        started = now()
        program.forward()
        while (ended := program.completion()) is None:
            if now() - started > 8.0:
                raise TimeoutError('the run did not complete within 8 s (a run is bounded: docs/facts.md)')
            time.sleep(0)
        seconds = ended - started
        emit({**measure(started, ended, received, rounds), 'workload': 'allgather_bandwidth', 'rank': rank, 'size': size,
              'report': dict(units=units, width=width, rounds=rounds, repeat=repeat, segments=sizes,
                             received_bytes=received, received_GBps=round(received / seconds / 1e9, 3))})
    # every segment as its rank wrote it: the gathered values, checked on this rank
    torch.mps.synchronize()
    got = h.float().cpu()
    wrong = [r for r in range(size) if not torch.all(got[sum(sizes[:r]):sum(sizes[:r + 1])] == r + 1)]
    print(json.dumps(dict(phase='checked', segments_wrong=wrong)), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
