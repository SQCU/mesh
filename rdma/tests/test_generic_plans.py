"""Check static example declarations and pause/resume around feedback.

Extract the actual Python functions to exercise metadata without loading a
bridge or torch's native extension; transport behavior is tested separately.
"""
import ast
import collections
import itertools
import math
from pathlib import Path
import resource
from types import SimpleNamespace
import unittest


PARTITION = Path(__file__).resolve().parents[1] / 'torch-mesh/torch_mesh/partition.py'
CALL = collections.namedtuple('Call', 'kind count dtype root op counts destination fused contributors',
                              defaults=(0, 'sum', None, 0, False, None))
DTYPE = SimpleNamespace(itemsize=4)


class WithoutImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def functions(*names, **values):
    tree = ast.parse(PARTITION.read_text())
    nodes = [WithoutImports().visit(node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(Call=CALL, ALLGATHER=3, REDUCE_SCATTER=4, WRITE=8, math=math, itertools=itertools, resource=resource,
                     torch=SimpleNamespace(float32=DTYPE), **values)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PARTITION), 'exec'), namespace)
    return namespace


class Group:
    handle, page = 1, 16384

    def __init__(self, rank, size):
        self._rank, self._size, self.active, self.events = rank, size, False, []
        cables = [{'nodes': (a, b)} for a in range(size) for b in range(a + 1, size)]
        paths = {}
        for e, cable in enumerate(cables):
            a, b = cable['nodes']
            paths[a, b], paths[b, a] = [[e]], [[e]]
        self.fabric = SimpleNamespace(cables=cables, paths=paths)

    def rank(self):
        return self._rank

    def size(self):
        return self._size

    def partials_operator(self, width):
        return ('partials', width)

    def plan_active(self, active):
        self.active = active
        self.events.append(('active', active))


class GenericPlansTests(unittest.TestCase):
    def test_route_reconfiguration_retains_invariant_and_regenerates_shape_declarations(self):
        group, seen = Group(0, 3), []
        mesh = type('Mesh', (), {'get_group': lambda self, _: group})()
        invariant = [[CALL(0, 48, DTYPE)]]
        plans = {(mesh, 'tp'): invariant}
        env = functions('_replan', _PLANS=plans, declare=lambda group, calls: seen.append(calls))
        env['_replan'](mesh, 'tp')
        self.assertIs(seen[-1], invariant, 'route changes must not silently drop the declared graph')
        plans[mesh, 'tp'] = lambda: [[CALL(3, 0, DTYPE, counts=[5, 7, 9])]]
        env['_replan'](mesh, 'tp')
        self.assertEqual(seen[-1][0][0].counts, [5, 7, 9])

    def test_form_separates_rank_layout_from_channel_and_source_stride_metadata(self):
        group, proofs = Group(0, 3), []
        tensor = SimpleNamespace(shape=(4, 2, 5), ndim=3, device='meta', element_size=lambda: 4,
                                 stride=lambda: (32, 16, 2), is_contiguous=lambda: False)
        env = functions('_gather_paths', '_fused_form', _stream=SimpleNamespace(polls=100),
                        dist=SimpleNamespace(distributed_c10d=SimpleNamespace(_resolve_process_group=lambda _: group)),
                        prove_recurrent_gather=lambda n, paths: proofs.append((n, paths)), _GATHER_THREADS=256,
                        _gather_kernel=lambda *args: args)
        env['torch'] = SimpleNamespace(int32='int32', zeros=lambda *_args, **_kwargs: object())
        form = env['_fused_form'](tensor, None, -2, [2, 1, 3], 'group', False, 6)
        self.assertEqual(form[4], ('gather_publish', 3, 2, 2, 6, False))
        self.assertEqual(form[5], [80, 0, 4, 2, 5, 32, 16, 2])
        self.assertEqual(form[6], 1)
        self.assertEqual(form[14:17], ([0, 40, 60, 120], [40, 20, 60], [4, 6, 5]))
        self.assertEqual(form[18:], (-1, 6))
        self.assertEqual(proofs, [(3, {pair: [pair] for pair in group.fabric.paths})])
        tensor.shape = (4, 0, 5)
        form = env['_fused_form'](tensor, None, 1, [0, 1, 3], 'group', False, 6)
        self.assertEqual(form[15], [0, 20, 60])
        self.assertEqual(form[16], [4, 4, 5])
        self.assertEqual(len(proofs), 1, 'empty-rank recurrence must not be certified')

    def test_lifetime_proof_keeps_actual_relays_and_parallel_cables(self):
        fabric = SimpleNamespace(cables=[{'nodes': (0, 1)}, {'nodes': (0, 1)},
                                         {'nodes': (1, 2)}, {'nodes': (0, 2)}],
                                 paths={(0, 2): [[0, 2], [1, 2], [3]]})
        paths = functions('_gather_paths')['_gather_paths'](fabric)
        self.assertEqual(paths, {(0, 2): [(0, 1, 2), (0, 1, 2), (0, 2)]})

    def test_ragged_gather_declares_exact_common_write_program(self):
        env = functions('_units', '_blocks', '_row', '_gather_writes', 'gathered', 'summed', _region_backed=lambda *_: True)
        sizes = [3, 2, 1]
        for rank in range(3):
            tensor = SimpleNamespace(shape=(4, sizes[rank], 13), dtype=DTYPE,
                                     element_size=lambda: 4, numel=lambda: 4 * sizes[rank] * 13)
            gathered = env['gathered'](tensor, 1, sizes, Group(rank, 3))
            expected = [(source, destination, 52 * sizes[source]) for source in range(3)
                        for destination in range(3) if source != destination]
            self.assertEqual([(c.root, c.destination, c.count) for c in gathered], expected)
            self.assertTrue(all(c.kind == 8 and c.fused and c.counts is None for c in gathered))
            self.assertEqual(gathered, env['gathered'](tensor, -2, sizes, Group(rank, 3)))
            summed = env['summed'](tensor, Group(rank, 3))
            self.assertEqual([(c.root, c.destination, c.count) for c in summed],
                             [(source, destination, tensor.numel()) for source in range(3)
                              for destination in range(3) if source != destination])

    def test_empty_sources_emit_no_dummy_packet_and_nonmesh_preserves_backend(self):
        env = functions('_row', '_gather_writes', 'gathered', _region_backed=lambda *_: True)
        tensor = SimpleNamespace(shape=(0,), dtype=DTYPE, numel=lambda: 0)
        calls = env['gathered'](tensor, 0, [0, 3, 0], Group(0, 3))
        self.assertEqual([(c.root, c.destination, c.count) for c in calls], [(1, 0, 3), (1, 2, 3)])
        self.assertEqual(env['gathered'](tensor, 0, [0, 0, 0], Group(0, 3)), [])
        env['_region_backed'] = lambda *_: False
        calls = env['gathered'](tensor, 0, [0, 3, 0], Group(0, 3))
        self.assertEqual(calls, [CALL(3, 0, DTYPE, counts=[0, 3, 0])])

    def test_cp_declares_the_exact_query_partial_and_unshard_counts(self):
        shares = [16, 12, 4]
        operand = SimpleNamespace(capacity=shares, units=sum(shares))
        for rank in range(3):
            group = Group(rank, 3)
            mesh = SimpleNamespace(get_group=lambda _: group, get_local_rank=lambda _: rank)
            env = functions('_units', '_blocks', '_sequence_extent', '_gather_writes', 'ring_calls', 'unshard_calls',
                            operand=lambda *_: operand)
            local = shares[rank] * 512
            calls = env['ring_calls'](mesh, 'cp', (1, 16, local, 128), DTYPE, 128)
            self.assertEqual(calls[0][0].counts, [s * 512 * 16 * 128 for s in shares])
            self.assertEqual(calls[1][0].counts, [s * 512 * 16 * 130 for s in shares])
            self.assertEqual(calls[1][0].op, ('partials', 130))
            self.assertEqual(calls[1][0].contributors, (0, 1, 2))
            unshard = env['unshard_calls'](mesh, 'cp', [(1, local, 2048)], [1], DTYPE)
            self.assertEqual([(c.root, c.destination, c.count) for c in unshard[0]],
                             [(source, destination, shares[source] * 512 * 2048) for source in range(3)
                              for destination in range(3) if source != destination])

    def test_cp_zero_share_keeps_declared_extent_and_excludes_empty_partial_source(self):
        operand = SimpleNamespace(capacity=[0, 4, 4], units=8)
        group = Group(0, 3)
        mesh = SimpleNamespace(get_group=lambda _: group, get_local_rank=lambda _: 0)
        env = functions('_units', '_blocks', '_sequence_extent', '_gather_writes', 'ring_calls', 'unshard_calls',
                        operand=lambda *_: operand)
        calls = env['ring_calls'](mesh, 'cp', (1, 2, 0, 4), DTYPE, 4, sequence_length=16)
        self.assertEqual(calls[0][0].counts, [0, 64, 64])
        self.assertEqual(calls[1][0].contributors, (1, 2))
        self.assertEqual(calls[1][0].counts, [0, 96, 96])
        unshard = env['unshard_calls'](mesh, 'cp', [(1, 0, 8)], [1], DTYPE, sequence_length=16)
        self.assertEqual([(c.root, c.destination, c.count) for c in unshard[0]],
                         [(1, 0, 64), (1, 2, 64), (2, 0, 64), (2, 1, 64)])
        with self.assertRaisesRegex(ValueError, 'declared sequence_length'):
            env['ring_calls'](mesh, 'cp', (1, 2, 0, 4), DTYPE, 4)



if __name__ == '__main__':
    unittest.main()
