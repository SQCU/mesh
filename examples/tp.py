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


# M. Shoeybi et al., Megatron-LM, arXiv:1909.08053, Figure 3; design/unequal-shards-change-request.md item 5;
# Megatron-LM section 3: a block's first weights (`columns`) are cut along their columns (a torch.nn.Linear weight's
# dimension 0) and its second (`rows`) along its rows (dimension 1) by one entry, mesh dimension 0 with the tuple of
# item 1, or with none. Items 6, 7 and 14; Zhang et al., HAP, EuroSys 2024, sections 2.4, 3.2, 5 and 5.2: a tuple for
# each part of the model, a part a block named in `measured['parts']`, "specified by the user (such as using the layers
# of the model)". The program is partitioned with every part's shards even, the tuple of part k k + 1 at every rank,
# and each part's times of (3) are taken from that program's stages (mesh.stage_times: `measured['times']`, each
# rank's measured times of each kind of function, and `measured['collectives']`); a part's entry holds each rank's
# upper bound `upper` and, where its shares are given and not solved for, `shares` (mesh.shard)
def megatron(gm, nodes, columns, rows, splits, measured):
    cut = {}
    for node in gm.graph.nodes:
        if node.op == 'get_attr' and node.target.endswith('.weight'):
            block, layer = node.target.rsplit('.', 2)[:2]
            if layer in columns + rows:
                cut.setdefault(block, []).append((node, int(layer in rows)))

    def placed(cuts):
        return mesh.Placement(list(range(len(nodes))), nodes, {**splits, **{
            node.target: (cuts.get(block, 0), -1) if along == 0 else (-1, cuts.get(block, 0))
            for block, held in cut.items() for node, along in held}})
    if measured is None:
        return placed({})
    marks = {block: mesh.Shard(0, (k + 1,) * len(nodes)) for k, block in enumerate(measured['parts'])}
    ir = mesh.partitioned(gm, placed(marks))
    return placed({block: mesh.shard(0, cut[block][0][0].meta['val'].shape[cut[block][0][1]],
                                     *mesh.stage_times(ir, marks[block], measured['times'], measured['collectives']),
                                     **part)
                   for block, part in measured['parts'].items()})


# V. Korthikanti et al., arXiv:2205.05198
def placement(nodes, sequence_parallel, measured=None):
    def choose(gm, examples):
        return megatron(gm, nodes, ('wq', 'wk', 'wv', 'w1', 'w3'), ('wo', 'w2'),
                        {node.name: (-1, 0, -1) for node in gm.graph.nodes
                         if node.op == 'placeholder' and sequence_parallel}, measured)
    return choose


# design/unequal-shards-change-request.md item 7: PARTS, JSON of `parts`, `times` (one entry a rank of mesh dimension
# 0: a kind of function's measured times by its units) and `collectives`
def measured():
    if 'PARTS' not in os.environ:
        return None
    given = json.loads(os.environ['PARTS'])
    return {**given, 'times': [{kind: {int(x): t for x, t in at.items()} for kind, at in rank.items()}
                               for rank in given['times']]}


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


def main():
    args = arguments('A compiled transformer forward and backward with tensor and sequence placement.')
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, DIM) / DIM ** 0.5
    block = None
    if not args.load:
        model = Training().to(device=args.device, dtype=torch.float32)
        block = Calls(mesh.Function(model, placement(args.nodes, os.environ.get('SP', '1') == '1', measured()),
                                    mesh.unit, device=args.device, loss=0))
    return execute(block, (x,), args)


if __name__ == '__main__':
    main()
