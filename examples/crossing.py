#!/usr/bin/env python3
"""A declared dependency chain; statistics observe its ordinary completion."""
import os
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as collective
import torch_mesh
from torch_mesh.declarations import Function
from mesh_statistics import now
import rollout


HIDDEN = int(os.environ.get('HIDDEN', '1536'))
CROSSINGS = int(os.environ.get('CROSSINGS', '141'))
STEPS = int(os.environ.get('STEPS', '3'))
OVERLAP = os.environ.get('OVERLAP', '0') == '1'


def chain(x, y, weight):
    for _ in range(STEPS * CROSSINGS):
        sent = collective.all_reduce(x, 'sum', dist.group.WORLD)
        if OVERLAP:
            x, y = torch.tanh(weight @ y), collective.wait_tensor(sent)
        else:
            x = collective.wait_tensor(sent)
    return x, y


def main():
    dist.init_process_group(backend='mesh')
    rank, size = dist.get_rank(), dist.get_world_size()
    group = dist.distributed_c10d._get_default_group()._get_backend(torch.device('mps'))
    x, y = (torch_mesh.empty(HIDDEN, dtype=torch.float16).zero_() for _ in range(2))
    weight = torch_mesh.empty(HIDDEN, HIDDEN, dtype=torch.float16).normal_(std=HIDDEN ** -0.5)
    function = Function(chain, x, y, weight)
    group.plan(function.declare(group))
    function.prepare()
    observer = rollout.start(dist.distributed_c10d._get_default_store(), 'crossing', rank, size, [])
    group._observers = [observer]
    started = now()
    function.run()
    completion = function.completion()
    observer[0].put((completion, {'episode': '0', 'parts': {}, 'proposal': 0, 'graphs': {},
        'workload': {'hidden': HIDDEN, 'crossings': CROSSINGS, 'steps': STEPS, 'overlap': OVERLAP},
        'implementation': 'loaded-program', 'started': started, 'rank': rank, 'size': size,
        'useful_work_bytes': STEPS * CROSSINGS * HIDDEN * x.element_size(),
        'report': {'program': 'crossing', 'steps': STEPS, 'calls': STEPS * CROSSINGS, 'overlap': OVERLAP}}))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
