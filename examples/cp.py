#!/usr/bin/env python3
# docs/measurement.md#the-torch-parallel-runs
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch_mesh import partition
from torch_mesh.configuration import placement, bind, record
from mesh_statistics import now

from tp import DIM, HEADS, STEPS, Attention, held

SEQ, POSITIONS = 16384, 512


class Landed(torch.nn.Module):
    # docs/measurement.md#the-torch-parallel-runs
    def __init__(self, linear, mesh):
        super().__init__()
        self.weight, self.width, self.out = linear.weight, linear.out_features, partition.empty(mesh, "cp", (SEQ, linear.out_features), 0)

    def forward(self, x):
        return torch.matmul(x, self.weight.mT, out=self.out.to_local().view(x.shape[:-1] + (self.width,)))


def main():
    dist.init_process_group(backend=os.environ.get("BACKEND", "mesh"))
    mesh = DeviceMesh.from_group(dist.group.WORLD, "mps", mesh_dim_names=("cp",))
    partition.attach(mesh, cp=SEQ // POSITIONS)
    configuration = placement(mesh, 'cp',
        {'function': 'attention_and_unshard', 'steps': STEPS, 'sequence': SEQ, 'width': DIM, 'heads': HEADS},
        os.environ.get('MESH_OBSERVATIONS', '~/.cache/mesh/observations/cp.jsonl'))
    with torch.device("meta"):
        layer = Attention()
    run = held(layer)
    x = partition.empty(mesh, "cp", (1, SEQ, DIM), 1)
    x.to_local().normal_(std=DIM ** -0.5)
    for name in ("wq", "wk", "wv", "wo"):
        setattr(layer, name, Landed(getattr(layer, name), mesh))

    def ring(q, k, v, is_causal):
        return partition.ring_attention(mesh, "cp", q, k, v, is_causal=is_causal, load_balance=False, sequence_length=SEQ)[0]

    from torch_mesh.declarations import Function, execution_sequence
    function = Function(lambda: partition.context_parallel_unshard(
        mesh, "cp", [run(x.to_local(), ring)], [1], load_balance=False, sequence_length=SEQ))
    with torch.no_grad():
        program = execution_sequence([function] * STEPS)
        bind(configuration, lambda: function.declare(mesh.get_group('cp')), program.realize)
        started = now()
        program.forward()
        completion = program.completion()
        record(configuration, completion, started, STEPS * SEQ * DIM * x.element_size(),
               {'program': 'cp', 'steps': STEPS, 'parts': configuration['assignment']['parts']['cp']})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
