#!/usr/bin/env python3
# design/the-compiler.md, order 3; design/aot-compilation-change-request.md A4 and A10
import argparse
import io
import json
import os

import torch

import tp


HIDDEN = int(os.environ.get('HIDDEN', '1536'))
CROSSINGS = int(os.environ.get('CROSSINGS', '141'))
OVERLAP = os.environ.get('OVERLAP', '0') == '1'
STEPS = int(os.environ.get('STEPS', '64'))


def arguments(description, steps=STEPS):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--nodes', type=int, nargs='+', required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--steps', type=int, default=steps)
    parser.add_argument('--input')
    parser.add_argument('--save')
    parser.add_argument('--load')
    parser.add_argument('--output')
    return parser.parse_args()


def execute(block, inputs, args):
    import mesh_compile as mesh
    import mesh_disk
    if args.input:
        inputs = torch.load(args.input, map_location='cpu', weights_only=True)
    run = mesh.load(args.load) if args.load else mesh.program(block, *inputs)
    if args.save:
        run.save(args.save)
    print(json.dumps({'structure': run.structure()}), flush=True)
    result = None
    for _ in range(args.steps):
        pending = False
        with torch.no_grad():
            for placed, value in zip(run.inputs(), inputs, strict=True):
                if placed is not None:
                    placed.copy_(value)
                    pending = pending or placed.device.type == 'mps'
        if pending:
            torch.mps.synchronize()
        result = run()
    run.synchronize()
    if args.output:
        output = io.BytesIO()
        torch.save(torch.utils._pytree.tree_map(lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, result), output)
        mesh_disk.write(args.output, output.getvalue(), 'Torch client output')
    leaves = torch.utils._pytree.tree_leaves(result)
    print(json.dumps({'steps': args.steps, 'ended': run.ended(), 'signature': run.signature()['digest'],
                      'outputs': [{'shape': list(x.shape), 'dtype': str(x.dtype)}
                                  for x in leaves if isinstance(x, torch.Tensor)]}), flush=True)
    return result


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


def placement(processors):
    def choose(node):
        if node.name in ('x', 'y') or node.target is torch.ops.aten.expand.default:
            return 0, processors, None
        return processors[0]
    return choose


def main():
    args = tp.arguments('A chain of all-reduces of a sharded value, with optional numerical work between them.')
    model = Chain(Crossing()).to(dtype=torch.float16).eval()
    inputs = tuple(torch.zeros(len(args.nodes), HIDDEN, dtype=torch.float16) for _ in range(2))
    return tp.execute(model, inputs, placement, args)


if __name__ == '__main__':
    main()
