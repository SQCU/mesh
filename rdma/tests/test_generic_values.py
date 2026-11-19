"""Exercise production generic data operations without opening cluster cables."""
import ast
import itertools
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[1] / 'torch-mesh/torch_mesh'


def load(filename, names, **namespace):
    class Clean(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None

        def visit_FunctionDef(self, node):
            node.decorator_list = []
            return self.generic_visit(node)
    nodes = [Clean().visit(n) for n in ast.parse((SOURCE / filename).read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    env = {'torch': torch, 'itertools': itertools, **namespace}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), env)
    return env


class GenericValuesTests(unittest.TestCase):
    def test_functional_sum_preserves_source_stride_and_scalar_shape(self):
        class Mesh:
            def size(self):
                return 3
        group, seen = Mesh(), []
        class Input:
            device = SimpleNamespace(type='mps')
            dtype = torch.float32
            def __init__(self, shape):
                self.shape, self.ndim = shape, len(shape)
            def view(self, shape):
                return Input((shape,) if isinstance(shape, int) else shape)
        def fused(x, dim, counts, selected, adds, weights, repeats):
            seen.append((x, counts, selected, repeats))
            return x
        env = load('backend.py', ['all_reduce'], ProcessGroupMesh=Mesh, _fused=fused,
                   dist=SimpleNamespace(distributed_c10d=SimpleNamespace(_resolve_process_group=lambda _: group)))
        source = Input((3, 5))  # deliberately provides no contiguous() operation
        output = env['all_reduce'](source, 'sum', 'mesh', 7)
        self.assertIs(seen[0][0], source)
        self.assertEqual(seen[0][1:], ([3, 3, 3], group, 7))
        self.assertEqual(output.shape, source.shape)
        self.assertEqual(env['all_reduce'](Input(()), 'sum', 'mesh').shape, ())

    def test_nonfloat_and_nonsum_use_native_operation_unchanged(self):
        class Mesh:
            pass
        seen = []
        native = lambda *args: seen.append(args) or 'native'
        proxy = SimpleNamespace(float16=torch.float16, float32=torch.float32, Tensor=object,
                                ops=SimpleNamespace(_c10d_functional=SimpleNamespace(
                                    all_reduce=SimpleNamespace(default=native))))
        env = load('backend.py', ['all_reduce'], ProcessGroupMesh=Mesh,
                   dist=SimpleNamespace(distributed_c10d=SimpleNamespace(_resolve_process_group=lambda _: Mesh())))
        env['torch'] = proxy
        for dtype, operation in [(torch.bool, 'sum'), (torch.int64, 'sum'), (torch.float32, 'product')]:
            source = SimpleNamespace(device=SimpleNamespace(type='mps'), dtype=dtype)
            self.assertEqual(env['all_reduce'](source, operation, 'mesh'), 'native')
            self.assertEqual(seen[-1], (source, operation, 'mesh'))

    def test_cp_unshard_exact_destination_strides_and_head_tail_order(self):
        parts, total = [3, 2, 1], 24
        operand = SimpleNamespace(capacity=parts, units=sum(parts), parts=torch.tensor(parts))
        group = SimpleNamespace(group_name='mesh')
        for halves in (1, 2):
            whole = torch.arange(2 * total * 3, dtype=torch.int32).view(2, total, 3)
            lengths = [p * 4 for p in parts]
            def spans(rank):
                start = sum(lengths[:rank]) // halves
                size = lengths[rank] // halves
                return [(start, start + size)] if halves == 1 else [(start, start + size), (total - start - size, total - start)]
            shards = [torch.cat([whole[:, lo:hi] for lo, hi in spans(rank)], dim=1) for rank in range(3)]
            for rank in range(3):
                # Select every other lane so the production gather must retain source strides.
                storage = torch.empty((2, lengths[rank], 6), dtype=torch.int32)
                source = storage[..., ::2]
                source.copy_(shards[rank])
                observed = []
                def gather(x, output, dim, counts, name):
                    observed.append((x, output, counts))
                    output.copy_(torch.cat(shards, dim=1))
                mesh = SimpleNamespace(get_local_rank=lambda _: rank, get_group=lambda _: group)
                env = load('partition.py', ['context_parallel_unshard', '_sequence_extent', '_units'], operand=lambda *_: operand,
                           _halves=lambda *_: halves, _blocks=lambda *_: (lengths, None),
                           _spans=lambda _o, r, _s, _h: spans(r), _JOINED={}, allgatherv_out=gather,
                           _empty=lambda shape, source, _group: torch.empty(shape, dtype=source.dtype))
                out, = env['context_parallel_unshard'](mesh, 'cp', [source], [-2], halves == 2)
                self.assertTrue(torch.equal(out, whole))
                self.assertIs(observed[-1][0], source)
                if halves == 1:
                    self.assertIs(observed[-1][1], out, 'normal unshard must gather into its final destination')
                again, = env['context_parallel_unshard'](mesh, 'cp', [source], [1], halves == 2)
                self.assertIs(again, out)


if __name__ == '__main__':
    unittest.main()
