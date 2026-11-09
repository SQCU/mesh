#!/usr/bin/env python3
# design/aot-compilation-change-request.md A4 and A10; metal-microbench/docs/measurement.md
import os

import torch

import mesh_compile as mesh
from tp import arguments, execute


HIDDEN = int(os.environ.get('HIDDEN', '1536'))
CROSSINGS = int(os.environ.get('CROSSINGS', '141'))
OVERLAP = os.environ.get('OVERLAP', '0') == '1'


class Crossing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.randn(HIDDEN, HIDDEN) / HIDDEN ** 0.5)

    def forward(self, x, y):
        received = x.sum(0, keepdim=True).expand_as(x)
        return (torch.tanh(y @ self.weight.mT), received) if OVERLAP else (received, y)


class Chain(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.work = function

    def forward(self, x, y):
        for _ in range(CROSSINGS):
            x, y = self.work(x, y)
        return x, y


def main():
    args = arguments('A compiled all-reduce dependency chain with optional interleaved numerical work.',
                     int(os.environ.get('STEPS', '3')))
    torch.manual_seed(0)
    block = None
    if not args.load:
        model = Crossing().to(device=args.device, dtype=torch.float16)
        def placement(gm, examples):
            return mesh.Placement(list(range(len(args.nodes))), args.nodes,
                                  {node.name: (0, -1) for node in gm.graph.nodes if node.op == 'placeholder'})
        block = Chain(mesh.Function(model, placement, mesh.unit, device=args.device))
    return execute(block, (torch.zeros(len(args.nodes), HIDDEN, dtype=torch.float16),
                           torch.zeros(len(args.nodes), HIDDEN, dtype=torch.float16)), args)


if __name__ == '__main__':
    main()
