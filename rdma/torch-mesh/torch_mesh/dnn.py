"""The fabric DNN is ordinary captured source over shared page operands."""
import numpy as np
import torch
import torch.distributed._functional_collectives as collectives

import torch_mesh
from mesh_relation import Algebra
from .engines import host
from .declarations import Function, execution_sequence


class DNN:
    """docs/measurement.md#physical-fabric-dnn"""

    def __init__(self, args, group, *, source_scope):
        self.args, self.group, self.source_scope = args, group, source_scope
        self.algebra = Algebra()
        rows, width, expansion = args.rows, args.width, args.expansion
        self.input = torch_mesh.empty(rows, width)
        self.output = torch_mesh.empty(rows, width)
        rng = np.random.default_rng(1729)
        rng.standard_normal(out=host(self.input).numpy(), dtype=np.float32)
        self.weights = []
        for _ in range(args.layers):
            a, b = torch_mesh.empty(width, expansion, device='cpu'), torch_mesh.empty(expansion, width, device='cpu')
            rng.standard_normal(out=a.numpy(), dtype=np.float32)
            rng.standard_normal(out=b.numpy(), dtype=np.float32)
            a.mul_(width ** -0.5)
            b.mul_(expansion ** -0.5)
            self.weights.append((a, b))

    def declare(self, parts, configuration):
        """Capture each tile shape once, then assign its occurrence's pages."""
        rank, shares = self.group.rank(), parts['compute']
        first, width = sum(shares[:rank]), shares[rank]
        self.local = []
        for a, b in self.weights:
            ag, bg = torch_mesh.empty(a.shape[0], width), torch_mesh.empty(width, b.shape[1])
            ag.copy_(a[:, first:first + width])
            bg.copy_(b[first:first + width, :])
            self.local.append((ag, bg))

        def forward(value):
            for a, b in self.local:
                hidden = torch.tanh(value @ a)
                partial = hidden @ b
                value = torch.tanh(collectives.all_reduce(partial, 'sum', self.group))
            return value

        configuration = {key: value for key, value in configuration.items() if key != 'tile_rows'}
        self.functions, groups, shapes = [], [], {}
        occurrences = -(-self.args.rows // self.args.tile_rows)
        for first in range(0, self.args.rows, self.args.tile_rows):
            rows = min(self.args.tile_rows, self.args.rows - first)
            occurrence = (first // self.args.tile_rows, occurrences)
            destination = self.output.narrow(0, first, rows)
            function = Function(forward, self.input.narrow(0, first, rows), configuration=configuration,
                algebra=self.algebra, source_scope=dict(scope=self.source_scope, tile_rows=rows))
            if rows not in shapes:
                calls = function.declare(self.group, destinations=[destination], occurrence=occurrence)
                shapes[rows] = function
            else:
                source = shapes[rows]
                function.arguments = (function.arguments[0], *source.source_arguments[1:])
                function.output_spec, function.result_types = source.output_spec, source.result_types
                function.choices = source.choices
                calls = function.declare_graph(source.prepared['expression']['graph'], self.group,
                    destinations=[destination], prepared=source.prepared, occurrence=occurrence)
            groups.extend(calls)
            self.functions.append(function)
        self.execution = execution_sequence(self.functions, self.output, algebra=self.algebra)
        return groups

    def prepare(self):
        self.execution.realize()
        self.run = self.execution.forward
        self.completion = self.execution.completion()
        return dict(layouts=[choice for function in self.functions for choice in function.layout_choices],
                    choices=[choice for function in self.functions for choice in function.choices])
