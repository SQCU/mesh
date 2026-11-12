#!/usr/bin/env python3
# design/the-compiler.md, order 3; design/unequal-shards-change-request.md items 9 and 15
import os

import torch

from tp import Attention, DIM, arguments, execute


SEQ = int(os.environ.get('SEQ', '16384'))


def placement(processors):
    def choose(node):
        if node.name == 'x' or node.target is torch.ops.aten.bmm.default:
            return 1, processors, None
        return processors[0]
    return choose


def main():
    args = arguments('Attention with its input and its query rows split over processors along the sequence.')
    torch.manual_seed(0)
    x = torch.randn(1, SEQ, DIM) / DIM ** 0.5
    return execute(Attention().eval(), (x,), placement, args)


if __name__ == '__main__':
    main()
