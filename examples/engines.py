#!/usr/bin/env python3
# design/aot-compilation-change-request.md A1 to A3 and A10; docs/soc_compute_backends.md
import os

import torch
import torch.nn.functional as F


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


class Program(torch.nn.Module):
    def __init__(self, nodes, engines, widths, projection_device, coded=None):
        super().__init__()
        import mesh_compile as mesh
        def placement(gm, examples):
            return mesh.Placement(list(range(len(nodes))), nodes, {})
        def compiled(module, device, encoded=False):
            return mesh.Function(module if encoded else module.to('cpu' if device == 'ane' else device),
                                 placement, mesh.unit, device=device, source_device='cpu' if encoded else None)
        layers = []
        for layer in range(LAYERS):
            projection = compiled(Projection((torch.randn(K, K) / K ** 0.5).half()), projection_device)
            weights = None if coded is not None else [(torch.randn(K, K) / K ** 0.5).half()] if KIND == 'linear' else [
                (torch.randn(N, K) / K ** 0.5).half(), (torch.randn(N, K) / K ** 0.5).half(),
                (torch.randn(K, N) / N ** 0.5).half()]
            parts, offset = [], 0
            for device, count in zip(engines, widths, strict=True):
                if coded is not None:
                    import mesh_code
                    part = mesh_code.FFN(coded[layer], offset, count)
                else:
                    matrices = [weight[:, offset:offset + count] if j == 2 else weight[offset:offset + count]
                                for j, weight in enumerate(weights)]
                    part = Shared(matrices)
                parts.append(compiled(part, device, coded is not None))
                offset += count
            layers.append(Layer(projection, parts, compiled(Combine(), projection_device)))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def main():
    global K, N
    from crossing import arguments, execute
    args = arguments('Compiled projections and FFN or linear partitions across configured CPU, MPS and ANE engines.',
                     int(os.environ.get('STEPS', '40')))
    coded = None
    if os.environ.get('CODES'):
        import mesh_code
        exported = mesh_code.load(os.environ['CODES'])
        first = int(os.environ.get('FIRST', '0'))
        coded = [[exported['model.language_model.layers.%d.mlp.%s_proj' % (layer, part)]
                  for part in ('gate', 'up', 'down')] for layer in range(first, first + LAYERS)]
        N, K = coded[0][0].shape
        if KIND != 'ffn':
            raise ValueError('CODES supplies the gate/up/down matrices of the OP=ffn workload')
    engines = os.environ.get('ENGINES', 'mps,cpu,ane').split(',')
    width = K if KIND == 'linear' else N
    widths = [int(x) for x in os.environ['PARTS'].split(',')] if 'PARTS' in os.environ else [
        (j + 1) * width // len(engines) - j * width // len(engines) for j in range(len(engines))]
    if len(widths) != len(engines) or any(n < 0 for n in widths) or sum(widths) != width:
        raise ValueError('PARTS supplies one nonnegative width per engine, summing to %d' % width)
    torch.manual_seed(0)
    x = torch.randn(ROWS, K).half() / 4
    block = None if args.load else Program(args.nodes, engines, widths, args.device, coded)
    return execute(block, (x,), args)


if __name__ == '__main__':
    main()
