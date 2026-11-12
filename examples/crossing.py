#!/usr/bin/env python3
# design/the-compiler.md, order 3; design/aot-compilation-change-request.md A4 and A10
import os

import torch

import tp


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
