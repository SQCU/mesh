#!/usr/bin/env python3
# design/the-compiler.md, order 3
import argparse
import json
import os
import pathlib
import re
import sys
import time

import torch
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).parents[1] / 'compiler'))
sys.path.append(str(pathlib.Path(__file__).parents[1] / 'rdma'))
import mesh_graph
from mesh_compiler import shares
from mesh_compiler.driver import compile_module
from mesh_compiler.placement import counts
from mesh_compiler.runtime import Arguments, Program


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


# design/unequal-shards-change-request.md items 1, 6 and 14
def units(fixed, processors, times, comm, upper, heads):
    solved = shares.shares(shares.stages(fixed, segment), processors, times, comm, upper)
    return {part: counts(heads, ratios, len(processors)) for part, ratios in solved.items()}


# design/unequal-shards-change-request.md item 7
class Timed(torch.fx.Interpreter):
    def __init__(self, module, device, times):
        super().__init__(module)
        self.device, self.times = device, times

    def run_node(self, node):
        started = time.perf_counter()
        result = super().run_node(node)
        if self.device.type == 'mps':
            torch.mps.synchronize()
        if node.op == 'call_function':
            self.times[node.meta['processor']].setdefault(shares.kind(node), {})[shares.units(node)[0]] = (
                time.perf_counter() - started)
        return result


# design/unequal-shards-change-request.md item 7; compiler/mesh_compiler/schedule.py assemble, members;
# compiler/mesh_compiler/runtime.py Arguments
taken(fixed, inputs):
    prepared = Arguments(fixed)
    module = fixed.modules[prepared.entry]
    times = {p: {} for p in module.meta['members']}
    Timed(module, prepared.device, times).run(*prepared.prepare(inputs, {}))
    return times


# design/unequal-shards-change-request.md items 6, 7 and 14; metal-microbench/output_data/window-20260928/bandwidth.json
def chosen(model, inputs, placement, args, heads):
    if not args.upper:
        return args.units
    processors = stated(args)
    fixed = compile_module(model, inputs, processors, placement(tuple(processors)))
    times = {p: {function: {int(x): t for x, t in held.items()} for function, held in functions.items()}
             for path in args.times for p, functions in json.loads(pathlib.Path(path).read_text()).items()}
    if not args.times:
        times = taken(fixed, inputs)
        print(json.dumps(times), flush=True)

    def comm(node):
        value = node.meta['val']
        along = value.shape[node.meta['dimension']] if 'counts' in node.meta else 0
        between = node.args[0] in {channel for channel, *_ in fixed.carried}
        return (between * args.comm * value.nbytes * sum(torch.utils._pytree.tree_leaves(node.meta['counts'])) / along
                if along else 0.0)

    if args.solve:
        return units(fixed, tuple(processors), times, comm, args.upper, heads)
    slopes = shares.computation(shares.stages(fixed, segment), tuple(processors), times)
    return {part: shares.shifted(given, args.upper,
                                 lambda j, held: sum(stage[j][0] + held[j] / sum(held) * stage[j][1]
                                                     for stage in slopes[part]))
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
    args = parser.parse_args()
    if args.solve and not args.upper:
        parser.error('--solve takes its bounds from --upper')
    return args


# design/unequal-shards-change-request.md item 8
class Eager(torch.nn.Module):
    def __init__(self, model, loss):
        super().__init__()
        self.model, self.loss = model, loss

    def forward(self, *inputs):
        with torch.enable_grad():
            outputs = self.model(*inputs)
            names, parameters = zip(*self.model.named_parameters())
            gradients = torch.autograd.grad(torch.utils._pytree.tree_leaves(outputs)[self.loss], parameters)
        return torch.utils._pytree.tree_map(torch.Tensor.detach, outputs), dict(zip(names, gradients))


def execute(model, inputs, placement, args, loss=None, **told):
    processors = stated(args)
    options = {} if loss is None else {'loss': loss}
    scheduled = compile_module(model, inputs, processors, placement(tuple(processors)), **options)
    result = Program(scheduled)(*inputs)
    report = {'channels': scheduled.channels, 'between machines': len(scheduled.carried),
              'signature': bytes(mesh_graph.from_record(scheduled.signature).digest).hex(), **told}
    if result is not None:
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
    given = chosen(model, (x,), placement, args, HEADS)
    return execute(model, (x,), lambda processors: placement(processors, given), args, loss=0, units=given)


if __name__ == '__main__':
    main()
