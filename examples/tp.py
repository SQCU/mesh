#!/usr/bin/env python3
# design/aot-compilation-change-request.md A1 to A4 and A10; metal-microbench/docs/measurement.md
import argparse
import io
import json
import os

import torch
import torch.nn.functional as F

import mesh_compile as mesh
import mesh_disk


DIM, HEADS, FFN, LAYERS, BATCH, SEQ = (int(os.environ.get(k, v)) for k, v in
    (('DIM', 2048), ('HEADS', 16), ('FFN', 5632), ('LAYERS', 4), ('BATCH', 2), ('SEQ', 1024)))
STEPS = int(os.environ.get('STEPS', '64'))


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


# M. Shoeybi et al., Megatron-LM, arXiv:1909.08053, Figure 3; V. Korthikanti et al., arXiv:2205.05198
def placement(nodes, sequence_parallel):
    def choose(gm, examples):
        splits = {}
        for node in gm.graph.nodes:
            if node.op == 'get_attr' and node.target.endswith('.weight'):
                layer = node.target.rsplit('.', 2)[-2]
                if layer in ('wq', 'wk', 'wv', 'w1', 'w3'):
                    splits[node.target] = (0, -1)
                elif layer in ('wo', 'w2'):
                    splits[node.target] = (-1, 0)
            elif node.op == 'placeholder' and sequence_parallel:
                splits[node.name] = (-1, 0, -1)
        return mesh.Placement(list(range(len(nodes))), nodes, splits)
    return choose


class Calls(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.work = function

    def forward(self, x):
        return self.work(x)


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
    if args.input:
        inputs = torch.load(args.input, map_location='cpu', weights_only=True)
    run = mesh.load(args.load) if args.load else mesh.program(block, *inputs)
    if args.save:
        run.save(args.save)
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
    print(json.dumps({'steps': args.steps, 'ended': run.ended(),
                      'outputs': [{'shape': list(x.shape), 'dtype': str(x.dtype)}
                                  for x in leaves if isinstance(x, torch.Tensor)]}), flush=True)
    return result


def main():
    args = arguments('A compiled transformer forward and backward with tensor and sequence placement.')
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, DIM) / DIM ** 0.5
    block = None
    if not args.load:
        model = Training().to(device=args.device, dtype=torch.float32)
        block = Calls(mesh.Function(model, placement(args.nodes, os.environ.get('SP', '1') == '1'), mesh.unit,
                                    device=args.device, loss=0))
    return execute(block, (x,), args)


if __name__ == '__main__':
    main()
