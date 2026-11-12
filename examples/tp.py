#!/usr/bin/env python3
# design/the-compiler.md, order 3
import argparse
import json
import os
import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).parents[1] / 'compiler'))
from mesh_compiler.driver import compile_module
from mesh_compiler.runtime import Program


DIM, HEADS, FFN, LAYERS, BATCH, SEQ = (int(os.environ.get(k, v)) for k, v in
    (('DIM', 2048), ('HEADS', 16), ('FFN', 5632), ('LAYERS', 4), ('BATCH', 2), ('SEQ', 1024)))


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = DIM // HEADS
        self.wq, self.wk, self.wv, self.wo = (torch.nn.Linear(DIM, DIM, bias=False) for _ in range(4))

    def forward(self, x):
        batch, sequence, _ = x.shape
        def heads(weight):
            return weight(x).view(batch, sequence, HEADS, self.head_dim).transpose(1, 2)
        q, k, v = (heads(weight) for weight in (self.wq, self.wk, self.wv))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).reshape(batch, sequence, DIM))


class FeedForward(torch.nn.Module):
    def __init__(self, width=FFN):
        super().__init__()
        self.w1, self.w3 = torch.nn.Linear(DIM, width, bias=False), torch.nn.Linear(DIM, width, bias=False)
        self.w2 = torch.nn.Linear(width, DIM, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm, self.ffn_norm = torch.nn.RMSNorm(DIM), torch.nn.RMSNorm(DIM)
        self.attention, self.feed_forward = Attention(), FeedForward()

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        return h + self.feed_forward(self.ffn_norm(h))


class Training(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(*(Block() for _ in range(LAYERS)))

    def forward(self, x):
        out = self.model(x)
        return out.sum(), out.detach()


# M. Shoeybi et al., Megatron-LM, arXiv:1909.08053, section 3 and Figure 3; design/unequal-shards-change-request.md
# items 1, 5 and 15
def megatron(processors, columns, rows, units=None):
    def placement(node):
        if node.op == 'placeholder':
            for dimension, layers in enumerate((columns, rows)):
                if any(node.target.endswith(f'_{layer}_weight') for layer in layers):
                    return dimension, processors, units
        return processors[0]
    return placement


def placement(processors, units=None):
    return megatron(processors, ('wq', 'wk', 'wv', 'w1', 'w3'), ('wo', 'w2'), units)


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--machines', type=int, nargs='+', required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--units', type=int, nargs='+')
    return parser.parse_args()


def execute(model, inputs, placement, args):
    processors = {f'p{place}': (machine, args.device) for place, machine in enumerate(args.machines)}
    scheduled = compile_module(model, inputs, processors, placement(tuple(processors)))
    result = Program(scheduled)(*inputs)
    report = {'channels': scheduled.channels, 'between machines': len(scheduled.carried)}
    if result is not None:
        with torch.no_grad():
            expected = model(*inputs)
        pairs = list(zip(torch.utils._pytree.tree_leaves(expected), torch.utils._pytree.tree_leaves(result),
                         strict=True))
        report['outputs'] = [list(got.shape) for _, got in pairs]
        report['error'] = max(((got.cpu() - want).abs().max() / want.abs().max()).item() for want, got in pairs)
    print(json.dumps(report), flush=True)
    return result


def main():
    args = arguments('A transformer forward compiled with its weights split over processors by columns and rows.')
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, DIM) / DIM ** 0.5
    return execute(Training().eval(), (x,), lambda processors: placement(processors, args.units and tuple(args.units)),
                   args)


if __name__ == '__main__':
    main()
