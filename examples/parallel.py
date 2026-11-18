#!/usr/bin/env python3
# design/the-compiler.md, order 3
import argparse
import inspect
import json
import math
import os
import pathlib
import re
import sys

import torch
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).parents[1] / 'compiler'))
sys.path.append(str(pathlib.Path(__file__).parents[1] / 'rdma'))
import mesh_graph
from mesh_compiler import archive, shares
from mesh_compiler.driver import compile_module
from mesh_compiler.ir import TrainingResult
from mesh_compiler.placement import counts
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
# items 1, 5, 6 and 15
def megatron(processors, columns, rows, units={}):
    def placement(node):
        if node.op == 'placeholder':
            for dimension, layers in enumerate((columns, rows)):
                if any(node.target.endswith(f'_{layer}_weight') for layer in layers):
                    return dimension, processors, next((u for part, u in units.items() if f'_{part}_' in node.target),
                                                       None)
        return processors[0]
    return placement


def placement(processors, units={}):
    return megatron(processors, ('wq', 'wk', 'wv', 'w1', 'w3'), ('wo', 'w2'), units)


def segment(node):
    for path, _ in node.meta.get('nn_module_stack', {}).values():
        if re.fullmatch(r'\w+\.\d+', path):
            return path.replace('.', '_')


# design/unequal-shards-change-request.md items 1, 6, 7 and 14; metal-microbench/output_data/window-20260928/
# bandwidth.json; compiler/mesh_compiler/native.py Bound.observe through compiler/mesh_compiler/runtime.py Program:
# the program `execute` runs (its `loss`, as compile_module's): `--times`' supplied costs (shares.supplied), else its
# completed form's executed units at the stated shares (shares.measured); the reports (and the stages' cycles and
# inner receive edges) on stderr, a unit or stage without a cost stopping; with --solve their shares (shares.solved, an
# external receive's collective time `--comm` a byte of its whole value) in whole `heads` (placement.counts, each rank
# between 0 and floor(heads * its `--upper`), the solve's own capacity in whole heads), else each
# part of `--units` adjusted to `--upper` (shares.shifted) by its ranks' loads over the same summed rows, every other
# part at its current share (shares.state)
def chosen(model, inputs, placement, args, heads, loss=None):
    if not args.upper:
        return args.units
    processors, options = stated(args), {} if loss is None else {'loss': loss}
    if args.times:
        scheduled = compile_module(model, inputs, processors, placement(tuple(processors)), **options)
        staged = {key: shares.stages(scheduled, key) for key in scheduled.modules}
        times = {p: {function: {int(x): t for x, t in held.items()} for function, held in functions.items()}
                 for path in args.times for p, functions in json.loads(pathlib.Path(path).read_text()).items()}
        costs = shares.supplied(staged, segment, tuple(processors), times)
        reports = [{**found, 'key': key} for key, stage in staged.items() for found in stage.inside]
    else:
        scheduled = compile_module(model, inputs, processors, placement(tuple(processors), args.units), **options)
        program = Program(scheduled)
        _, observations = program.function.observe(*program.prepare(inputs, {}))
        costs, reports, staged = shares.measured(program.function, scheduled, segment, tuple(processors),
                                                 observations, heads)
    if reports:
        print(json.dumps(reports), file=sys.stderr, flush=True)
    if not all(report['costed'] for report in reports):
        raise ValueError('a measured unit or stage has no cost (the reports on stderr)')

    def comm(node):
        value = node.meta['val']
        along = value.shape[node.meta['dimension']] if 'counts' in node.meta else 0
        return (args.comm * value.nbytes * sum(torch.utils._pytree.tree_leaves(node.meta['counts'])) / along
                if along else 0.0)

    if args.solve:
        solved = shares.solved(staged, segment, costs, comm, args.upper)
        most = [math.floor(heads * share) for share in args.upper]
        return {part: counts(heads, ratios, len(processors), [0] * len(processors), most)
                for part, ratios in solved.items()}
    rows, current = shares.summed(costs), shares.state(staged, segment, tuple(processors), args.units)
    return {part: shares.shifted(given, args.upper, shares.load(rows, current, part))
            for part, given in args.units.items()}


def stated(args):
    return {f'p{place}': (node, args.device) for place, node in enumerate(args.nodes)}


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--nodes', type=int, nargs='+', required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--units', type=json.loads, default={})
    parser.add_argument('--solve', action='store_true')
    parser.add_argument('--upper', type=float, nargs='+')
    parser.add_argument('--times', nargs='+', default=[])
    parser.add_argument('--comm', type=float, default=1.0752e-10)
    parser.add_argument('--archive', type=pathlib.Path)
    parser.add_argument('--prepared', type=int, default=1)
    args = parser.parse_args()
    if args.solve and not args.upper:
        parser.error('--solve takes its bounds from --upper')
    if args.prepared < 1:
        parser.error('--prepared must be positive')
    return args


# design/unequal-shards-change-request.md item 8; compiler/mesh_compiler/ir.py TrainingResult: the eager reference of a
# program with a loss, its forward outputs, its parameters' gradients by name and its gradient-requiring user inputs'
# gradients by their forward parameter names
class Eager(torch.nn.Module):
    def __init__(self, model, loss):
        super().__init__()
        self.model, self.loss = model, loss

    def forward(self, *inputs):
        leaves = {name: value for name, value in inspect.signature(self.model.forward).bind(*inputs).arguments.items()
                  if isinstance(value, torch.Tensor) and value.requires_grad}
        named = dict(self.model.named_parameters())
        names, parameters = tuple(named), tuple(named.values())
        with torch.enable_grad():
            outputs = self.model(*inputs)
            gradients = torch.autograd.grad(torch.utils._pytree.tree_leaves(outputs)[self.loss],
                                            (*parameters, *leaves.values()))
        return TrainingResult(torch.utils._pytree.tree_map(torch.Tensor.detach, outputs),
                              dict(zip(names, gradients[:len(names)])), dict(zip(leaves, gradients[len(names):])))


def execute(model, inputs, placement, args, loss=None, functions=None, **told):
    processors = stated(args)
    options = {} if loss is None else {'loss': loss}
    placed = placement(tuple(processors))
    scheduled = compile_module(model, inputs, processors, placed, functions=functions or (), **options)
    if args.archive is not None:
        archive.save(args.archive, scheduled)
        scheduled = archive.load(args.archive)
    programs, expected = [Program(scheduled) for _ in range(args.prepared)], None
    for index, program in enumerate(programs):
        result = program(*inputs)
        report = {'channels': [{'function': k, 'channels': scheduled.channels[body.key]}
                               for k, body in enumerate(program.function.target.bodies)],
                  'between machines': sum(len(transfers) for transfers in scheduled.carried.values()),
                  'signature': bytes(program.function.signature.digest).hex(), **told}
        if args.prepared > 1:
            report['preparation'] = index
        if result is not None:
            if expected is None:
                with torch.no_grad():
                    expected = (model if loss is None else Eager(model, loss))(*inputs)
            pairs = list(zip(torch.utils._pytree.tree_leaves(expected), torch.utils._pytree.tree_leaves(result),
                             strict=True))
            report['outputs'] = [list(got.shape) for _, got in pairs]
            report['error'] = max(((got.cpu().double() - want.double()).abs().max() /
                                   want.double().abs().max().clamp(min=1e-12)).item() for want, got in pairs)
        print(json.dumps(report), flush=True)
    return result


def main():
    args = arguments('A transformer forward and backward compiled with its weights split over processors by columns '
                     'and rows.')
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, DIM) / DIM ** 0.5
    model = Training().eval()
    given = chosen(model, (x,), placement, args, HEADS, loss=0)
    return execute(model, (x,), lambda processors: placement(processors, given), args, loss=0, units=given)


if __name__ == '__main__':
    main()
