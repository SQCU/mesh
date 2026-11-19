"""Zero-query/key CP rank keeps its physical participation and no numerical partial source."""
from types import SimpleNamespace
import unittest

import torch

from test_generic_plans import functions
from test_generic_values import load


class CPZeroTests(unittest.TestCase):
    def test_empty_query_unshard_writes_full_declared_output(self):
        operand = SimpleNamespace(capacity=[0, 4, 4], units=8)
        group = SimpleNamespace(group_name='relay')
        mesh = SimpleNamespace(get_local_rank=lambda _: 0, get_group=lambda _: group)
        whole = torch.arange(24, dtype=torch.int32).reshape(1, 8, 3)
        calls = []
        def gather(source, out, dim, sizes, group_name):
            calls.append((source.numel(), sizes, tuple(out.shape)))
            out.copy_(whole)
        env = load('partition.py', ['context_parallel_unshard', '_sequence_extent', '_units', '_blocks'],
                   operand=lambda *_: operand, _halves=lambda *_: 1, _JOINED={}, allgatherv_out=gather,
                   _empty=lambda shape, like, group: torch.empty(shape, dtype=like.dtype))
        result, = env['context_parallel_unshard'](mesh, 'cp', [torch.empty(1, 0, 3, dtype=torch.int32)], [1],
                                                   sequence_length=8)
        self.assertTrue(torch.equal(result, whole))
        self.assertEqual(calls, [(0, [0, 4, 4], (1, 8, 3))])

    def test_empty_rank_issues_full_graph_without_constructing_identity_partial_payload(self):
        events = []
        class Group:
            group_name = 'zero-query'

            def merge_partials(self, output, source, rows):
                events.append(('partials', source.numel(), output.numel(), rows))
        group = Group()
        operand = SimpleNamespace(capacity=[0, 4, 4], units=8, parts=torch.tensor([0, 4, 4]),
                                  cached=lambda _, build: build([0, 4, 4]))

        def gather(blocks, original, **_):
            events.append(('gather', [b.numel() for b in blocks], original.numel()))
            return SimpleNamespace(wait=lambda: events.append(('dependency',)))
        env = functions('_ring', '_ring_attention', '_sequence_extent', '_units', '_blocks', '_positions', '_spans',
                        _OPERANDS={'cp': operand}, _RINGS={}, ProcessGroupMesh=Group,
                        dist=SimpleNamespace(get_rank=lambda _: 0, get_world_size=lambda _: 3, all_gather=gather),
                        _empty=lambda shape, like, group, dtype=None: torch.empty(shape, dtype=dtype or like.dtype),
                        _block=lambda *_: self.fail('an empty key partition must not fabricate a partial input'))
        env['torch'] = torch
        q = torch.empty(1, 2, 0, 4)
        result, lse = env['_ring_attention'](group, 'cp', q, q, q, True, None, 1, 8)
        self.assertEqual(result.shape, (1, 2, 0, 4))
        self.assertEqual(lse.shape, (1, 2, 0))
        self.assertEqual(events, [('gather', [0, 32, 32], 0), ('dependency',), ('partials', 0, 0, [0, 8, 8])])


if __name__ == '__main__':
    unittest.main()
