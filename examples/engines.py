# design/the-compiler.md, order 5 and "Core ML"
import os
import re

import torch
import torch.nn.functional as F


K, N, LAYERS = (int(os.environ.get(k, v)) for k, v in
    (('K', 3840), ('N', 3456), ('LAYERS', 4)))
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
