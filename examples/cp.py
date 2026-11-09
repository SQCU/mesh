#!/usr/bin/env python3
# design/aot-compilation-change-request.md A2 and A4; metal-microbench/docs/measurement.md
import os

import torch

import mesh_compile as mesh
from tp import Attention, Calls, DIM, arguments, execute


SEQ = int(os.environ.get('SEQ', '16384'))


def main():
    args = arguments('Compiled attention with partitioned query rows and a whole returned output.')
    torch.manual_seed(0)
    x = torch.randn(1, SEQ, DIM) / DIM ** 0.5
    block = None
    if not args.load:
        layer = Attention().to(device=args.device, dtype=torch.float32)
        def placement(gm, examples):
            return mesh.Placement(list(range(len(args.nodes))), args.nodes,
                                  {node.name: (-1, 0, -1) for node in gm.graph.nodes if node.op == 'placeholder'})
        block = Calls(mesh.Function(layer, placement, mesh.unit, device=args.device))
    return execute(block, (x,), args)


if __name__ == '__main__':
    main()
