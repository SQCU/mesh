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
CALL = collections.namedtuple('Call', 'kind count dtype root op counts fused', defaults=(0, 'sum', None, False))
DTYPE = SimpleNamespace(itemsize=4)


class WithoutImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def functions(*names, **values):
    tree = ast.parse(PARTITION.read_text())
    nodes = [WithoutImports().visit(node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(Call=CALL, ALLGATHER=3, REDUCE_SCATTER=4, math=math, itertools=itertools, resource=resource,
                     torch=SimpleNamespace(float32=DTYPE), **values)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PARTITION), 'exec'), namespace)
    return namespace


class Group:
    handle, page = 1, 16384

    def __init__(self, rank, size):
        self._rank, self._size, self.active, self.events = rank, size, False, []

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
    def test_form_separates_rank_layout_from_channel_and_source_stride_metadata(self):
        group, proofs = Group(0, 3), []
        tensor = SimpleNamespace(shape=(4, 2, 5), ndim=3, device='meta', element_size=lambda: 4,
                                 stride=lambda: (32, 16, 2), is_contiguous=lambda: False)
        env = functions('_paged', '_fused_form', _stream=SimpleNamespace(polls=100),
                        dist=SimpleNamespace(distributed_c10d=SimpleNamespace(_resolve_process_group=lambda _: group)),
                        prove_recurrent_gather=proofs.append, _GATHER_THREADS=256,
                        _gather_kernel=lambda *args: args)
        env['torch'] = SimpleNamespace(int32='int32', zeros=lambda *_args, **_kwargs: object())
        form = env['_fused_form'](tensor, None, -2, [2, 1, 3], 'group', False, 6)
        self.assertEqual(form[4], ('gather_publish', 3, 2, 6, False))
        self.assertEqual(form[5], [80, 0, 4, 2, 5, 32, 16, 2])
        self.assertEqual(form[6], 1)
        self.assertEqual(form[16], [4, 6, 5])
        self.assertEqual(form[18:], (-1, 6))
        self.assertEqual(proofs, [3])
        tensor.shape = (4, 0, 5)
        form = env['_fused_form'](tensor, None, 1, [0, 1, 3], 'group', False, 6)
        self.assertEqual(form[15], [0, 4096, 4096])
        self.assertEqual(form[16], [4, 4, 5])
        self.assertEqual(proofs, [3], 'empty-rank recurrence must not be certified')

    def test_ragged_gather_declares_one_fully_routable_operation(self):
        env = functions('_units', '_blocks', '_paged', '_row', 'gathered', 'summed', _region_backed=lambda *_: True)
        sizes = [3, 2, 1]
        for rank in range(3):
            tensor = SimpleNamespace(shape=(4, sizes[rank], 13), dtype=DTYPE,
                                     element_size=lambda: 4, numel=lambda: 4 * sizes[rank] * 13)
            gathered = env['gathered'](tensor, 1, sizes, Group(rank, 3))
            self.assertEqual(len(gathered), 1)
            self.assertEqual(gathered[0].kind, 3)
            self.assertEqual(gathered[0].counts, [4096, 4096, 4096])
            self.assertTrue(gathered[0].fused)
            self.assertEqual(gathered, env['gathered'](tensor, -2, sizes, Group(rank, 3)))
            summed = env['summed'](tensor, Group(rank, 3))
            self.assertEqual(len(summed), 1)
            self.assertEqual(summed[0].counts, [4096] * 3)

    def test_cp_declares_the_exact_query_partial_and_unshard_counts(self):
        shares = [16, 12, 4]
        operand = SimpleNamespace(capacity=shares, units=sum(shares))
        for rank in range(3):
            group = Group(rank, 3)
            mesh = SimpleNamespace(get_group=lambda _: group, get_local_rank=lambda _: rank)
            env = functions('_units', '_blocks', '_paged', 'ring_calls', 'unshard_calls', operand=lambda *_: operand)
            local = shares[rank] * 512
            calls = env['ring_calls'](mesh, 'cp', (1, 16, local, 128), DTYPE, 128)
            self.assertEqual(calls[0][0].counts, [s * 512 * 16 * 128 for s in shares])
            self.assertEqual(calls[1][0].counts, [s * 512 * 16 * 130 for s in shares])
            self.assertEqual(calls[1][0].op, ('partials', 130))
            unshard = env['unshard_calls'](mesh, 'cp', [(1, local, 2048)], [1], DTYPE)
            self.assertEqual(unshard[0][0].counts, [s * 512 * 2048 for s in shares])

    def test_feedback_cannot_consume_the_function_plan(self):
        mesh, name, group = object(), 'ep', Group(0, 3)
        mesh = type('Mesh', (), {'get_group': lambda self, _: group})()
        state, plans = {(mesh, name): {'balancer': object()}}, {(mesh, name): object()}

        def balance(*_):
            self.assertFalse(group.active)
            group.events.append(('feedback', False))

        env = functions('evaluations', _BALANCING=state, _PLANS=plans, _balance=balance, operand=lambda *_: object())
        for i in env['evaluations'](mesh, name, 3):
            self.assertTrue(group.active)
            group.events.append(('function', i))
        self.assertEqual(group.events,
                         [('active', True), ('function', 0), ('active', False), ('feedback', False),
                          ('active', True), ('function', 1), ('active', False), ('feedback', False),
                          ('active', True), ('function', 2), ('active', False)])

    def test_abandoned_evaluation_releases_plan_scope(self):
        group = Group(0, 3)
        mesh = type('Mesh', (), {'get_group': lambda self, _: group})()
        env = functions('evaluations', _BALANCING={(mesh, 'cp'): {'balancer': object()}},
                        _PLANS={(mesh, 'cp'): object()}, _balance=lambda *_: None, operand=lambda *_: object())
        calls = env['evaluations'](mesh, 'cp', 2)
        next(calls)
        self.assertTrue(group.active)
        calls.close()
        self.assertFalse(group.active)


if __name__ == '__main__':
    unittest.main()
