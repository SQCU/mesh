#!/usr/bin/env python3
# design/the-compiler.md, order 5 and "Core ML"
import argparse
import collections
import copy
import json
import os
import pathlib
import re
import sys
import time

import torch
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).parents[1] / 'compiler'))
from mesh_compiler import lower_coreml
from mesh_compiler.driver import compile_module, lowered
from mesh_compiler.runtime import Program


ROWS, K, N, LAYERS = (int(os.environ.get(k, v)) for k, v in
    (('ROWS', 1024), ('K', 3840), ('N', 3456), ('LAYERS', 4)))
KIND = os.environ.get('OP', 'ffn')


class Projection(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer('weight', weight)

    def forward(self, x):
        return F.rms_norm(F.linear(x, self.weight), (K,), eps=1e-6)


class Shared(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        for name, weight in zip(('gate', 'up', 'down'), weights):
            self.register_buffer(name, weight)

    def forward(self, x):
        if KIND == 'linear':
            return F.linear(x, self.gate)
        return F.linear(F.gelu(F.linear(x, self.gate), approximate='tanh') * F.linear(x, self.up), self.down)


class Combine(torch.nn.Module):
    def forward(self, *parts):
        if KIND == 'linear':
            return torch.cat(parts, -1)
        result = parts[0]
        for part in parts[1:]:
            result = result + part
        return result


class Layer(torch.nn.Module):
    def __init__(self, projection, parts, combine):
        super().__init__()
        self.projection, self.combine = projection, combine
        self.parts = torch.nn.ModuleList(parts)

    def forward(self, x):
        projected = self.projection(x)
        return self.combine(*(part(projected) for part in self.parts))


class Engines(torch.nn.Module):
    def __init__(self, widths):
        super().__init__()
        layers = []
        for _ in range(LAYERS):
            projection = Projection((torch.randn(K, K) / K ** 0.5).half())
            weights = [(torch.randn(K, K) / K ** 0.5).half()] if KIND == 'linear' else [
                (torch.randn(N, K) / K ** 0.5).half(), (torch.randn(N, K) / K ** 0.5).half(),
                (torch.randn(K, N) / N ** 0.5).half()]
            parts, offset = [], 0
            for count in widths:
                parts.append(Shared([weight[:, offset:offset + count] if j == 2 else weight[offset:offset + count]
                                     for j, weight in enumerate(weights)]))
                offset += count
            layers.append(Layer(projection, parts, Combine()))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def placement(first, parts):
    def placed(node):
        where = ' '.join([str(path) for path, _ in node.meta.get('nn_module_stack', {}).values()] + [str(node.target)])
        found = re.search(r'layers[._]\d+[._]parts[._](\d+)', where)
        return parts[int(found[1])] if found else first
    return placed


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--nodes', type=int, nargs='+', required=True)
    parser.add_argument('--device', default='mps')
    return parser.parse_args()


def execute(model, inputs, engines, args):
    processors = {'projection': (args.nodes[0], args.device)}
    parts = [f'part{place}' for place in range(len(engines))]
    processors.update((part, (args.nodes[place % len(args.nodes)], engine))
                      for place, (part, engine) in enumerate(zip(parts, engines)))
    scheduled = compile_module(model, inputs, processors, placement('projection', parts))
    functions = lowered(scheduled, (lower_coreml.lower,))
    program = Program(scheduled, functions)
    started = time.perf_counter()
    result = program(*inputs)
    report = {'channels': scheduled.channels, 'between machines': len(scheduled.carried),
              'processors': {name: [machine, engine] for name, (machine, engine) in processors.items()},
              'seconds': time.perf_counter() - started,
              'core ml': {name: dict(collections.Counter(
                  f'{operation} on {preferred}' for plan in function.plan()
                  for operation, _, preferred, _ in plan if operation != 'const'))
                  for name, function in functions.items() if scheduled.engines.get(name)}}
    if result is not None:
        with torch.no_grad():
            expected = copy.deepcopy(model).to(args.device)(*(value.to(args.device) for value in inputs))
        pairs = list(zip(torch.utils._pytree.tree_leaves(expected), torch.utils._pytree.tree_leaves(result),
                         strict=True))
        report['outputs'] = [list(got.shape) for _, got in pairs]
        report['error'] = max(((got.cpu().float() - want.cpu().float()).abs().max()
                               / want.cpu().float().abs().max()).item() for want, got in pairs)
    print(json.dumps(report), flush=True)
    return result


def main():
    args = arguments('Projections and FFN or linear partitions compiled over CPU, MPS and ANE processors.')
    engines = os.environ.get('ENGINES', 'mps,cpu,ane').split(',')
    width = K if KIND == 'linear' else N
    widths = [int(x) for x in os.environ['PARTS'].split(',')] if 'PARTS' in os.environ else [
        (j + 1) * width // len(engines) - j * width // len(engines) for j in range(len(engines))]
    if len(widths) != len(engines) or any(n <= 0 for n in widths) or sum(widths) != width:
        raise ValueError('PARTS supplies one positive width per engine, summing to %d' % width)
    torch.manual_seed(0)
    x = torch.randn(ROWS, K).half() / 4
    return execute(Engines(widths).eval(), (x,), engines, args)


if __name__ == '__main__':
    main()
