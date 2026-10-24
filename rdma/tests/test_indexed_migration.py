"""Exercise the actual indexed lowering on tensor views with a delayed WRITE executor.

No native transport is loaded. The executor retains every source/destination
view, checks that each rank declares identical ranges, and completes writes
only after all ranks have returned from realization. Cluster tests separately
check GPU publication, physical routing, and asynchronous source retirement.
"""
import math
from types import SimpleNamespace
import unittest

from test_generic_plans import functions
from test_migration import compile_migration, compile_rows

try:
    import torch
except ImportError:
    torch = None


class Group:
    page = 64

    def __init__(self):
        self.writes, self.events, self.allocations = [], [], []

    def start_coalescing(self, device):
        self.events.append('start')

    def write(self, source, target, a, b):
        assert source is target, 'unused endpoint argument must not allocate a dummy payload'
        self.writes.append((source, a, b))

    def end_coalescing(self, device):
        self.events.append('end')


@unittest.skipUnless(torch is not None, 'requires the torch environment')
class IndexedMigrationTests(unittest.TestCase):
    def lowering(self, old, new):
        def empty(shape, like, group):
            tensor = torch.empty(shape, dtype=like.dtype)
            group.allocations.append(tensor)
            return tensor

        env = functions('_follow_indexed', compile_migration=compile_migration, compile_rows=compile_rows,
                        _OPERANDS={'old': SimpleNamespace(capacity=old), 'new': SimpleNamespace(capacity=new)},
                        _region_backed=lambda *_: True, _empty=empty,
                        address=lambda t: 0 if not t.numel() else 64 + t.storage_offset() * t.element_size())
        env['torch'] = torch
        return env['_follow_indexed']

    def execute(self, shape, dim, old, new):
        full = torch.arange(math.prod(shape), dtype=torch.float32).view(shape)
        grain = shape[dim] // sum(old)
        source = [full.narrow(dim, grain * sum(old[:r]), grain * n).contiguous().clone()
                  for r, n in enumerate(old)]
        groups = [Group() for _ in old]
        lower = self.lowering(old, new)
        output = [lower(SimpleNamespace(key='old', dim=dim), tensor, shape, full.stride(), 'new', rank, group,
                        lambda group=group: group.events.append('publish'))
                  for rank, (tensor, group) in enumerate(zip(source, groups))]
        self.assertTrue(all(t is not None for t in output), 'aligned ranges must use indexed writes')
        writes = [w for w in compile_migration(shape, dim, old, new, 4).writes if w.source != w.destination]
        relays = []
        for group in groups:
            self.assertEqual(group.events, ['publish', 'start', 'end'] if writes else [])
            self.assertEqual([(a, b, t.numel() * t.element_size()) for t, a, b in group.writes],
                             [(w.source, w.destination, w.bytes) for w in writes])
        for i, write in enumerate(writes):
            sending = groups[write.source].writes[i][0]
            landing = groups[write.destination].writes[i][0]
            self.assertEqual(sending.data_ptr(), source[write.source].data_ptr() + write.source_offset)
            self.assertEqual(landing.data_ptr(), output[write.destination].data_ptr() + write.destination_offset)
            for rank, group in enumerate(groups):
                if rank != write.source:
                    target = group.writes[i][0]
                    target.copy_(sending)
                    if rank != write.destination:
                        relays.append(target)
                        self.assertNotEqual(target.untyped_storage().data_ptr(), source[rank].untyped_storage().data_ptr())
                        self.assertNotEqual(target.untyped_storage().data_ptr(), output[rank].untyped_storage().data_ptr())
        self.assertEqual(len({t.untyped_storage().data_ptr() for t in relays}), len(relays),
                         'each simultaneous relay range requires a distinct lease')
        for rank, (actual, original) in enumerate(zip(output, source)):
            self.assertTrue(torch.equal(actual, full.narrow(dim, grain * sum(new[:rank]), grain * new[rank])))
            self.assertTrue(torch.equal(original, full.narrow(dim, grain * sum(old[:rank]), grain * old[rank])),
                            'migration must not overwrite its original ranges')
        return source, output, groups

    def test_leading_axis_aliases_retained_intervals_and_writes_final_destination(self):
        source, output, _ = self.execute((8, 16), 0, [4, 3, 1], [4, 2, 2])
        self.assertEqual(source[0].data_ptr(), output[0].data_ptr())
        self.assertEqual(source[1].data_ptr(), output[1].data_ptr())
        self.assertNotEqual(source[2].data_ptr(), output[2].data_ptr())

    def test_interior_axis_has_disjoint_relay_spans_without_pack_or_unpack(self):
        self.execute((3, 8, 16), 1, [4, 3, 1], [4, 2, 2])

    def test_two_rank_move_and_zero_share_sources_and_destinations(self):
        self.execute((4, 16), 0, [3, 1], [2, 2])
        self.execute((3, 8, 16), 1, [0, 8, 0], [4, 0, 4])

    def test_no_movement_aliases_original_storage_without_publication(self):
        source, output, groups = self.execute((8, 16), 0, [4, 3, 1], [4, 3, 1])
        self.assertTrue(all(a.data_ptr() == b.data_ptr() for a, b in zip(source, output)))
        self.assertTrue(all(not g.allocations for g in groups))

    def test_unaligned_ranges_report_missing_capability_before_issue(self):
        lower = self.lowering([4, 3, 1], [4, 2, 2])
        full = torch.empty((3, 8, 3))
        for rank, part in enumerate([4, 3, 1]):
            group = Group()
            source = torch.empty((3, part, 3))
            self.assertIsNone(lower(SimpleNamespace(key='old', dim=1), source, full.shape, full.stride(),
                                    'new', rank, group, lambda: self.fail('unaligned publish')))
            self.assertFalse(group.allocations or group.events or group.writes)

    def test_declared_source_layout_mismatch_raises_without_issuing(self):
        lower = self.lowering([4, 3, 1], [4, 2, 2])
        group = Group()
        with self.assertRaisesRegex(ValueError, 'declared contiguous'):
            lower(SimpleNamespace(key='old', dim=0), torch.empty((16, 3)).T, (8, 16), (16, 1),
                  'new', 1, group, lambda: self.fail('invalid source publish'))
        self.assertFalse(group.allocations or group.events or group.writes)

    def test_masked_schedule_publishes_independent_sources_before_any_incoming_wait(self):
        events, trains = [], []
        group = Group()
        group.comm = object()
        source = torch.empty((3, 3, 65), dtype=torch.uint8)
        fabric = SimpleNamespace(adjacent=[[(1, 0), (1, 1)]] * 3,
                                 paths={(a, b): [[0], [1]] for a in range(3) for b in range(3) if a != b})

        def writes(first, landings, pairs, group, repeats, stride):
            self.assertIs(first, landings[0])
            self.assertEqual(first.numel(), stride)
            self.assertGreaterEqual(first.untyped_storage().nbytes(), repeats * stride)
            trains.append((pairs, repeats, stride))
            events.append(('bind', pairs))
            return object()

        def encode(*call):
            events.append((call[0][0], call))

        env = functions('_follow_masked', '_migration_words', compile_rows=compile_rows,
                        _OPERANDS={'old': SimpleNamespace(capacity=[4, 3, 1]), 'new': SimpleNamespace(capacity=[3, 3, 2])},
                        _region_backed=lambda *_: True,
                        _empty=lambda shape, like, group, dtype=None: torch.empty(shape, dtype=dtype or like.dtype),
                        address=lambda t: 64 + t.storage_offset() * t.element_size(),
                        Fabric=SimpleNamespace(of=lambda _: fabric), _migration_kernel=lambda *args: args,
                        _GATHER_THREADS=256, fused=SimpleNamespace(writes=writes, encode=encode,
                                                                  encode_many=lambda calls: [encode(*call) for call in calls]))
        env['torch'] = torch
        result = env['_follow_masked'](SimpleNamespace(key='old', dim=1), source, (3, 8, 65), (520, 65, 1),
                                       'new', 1, group)
        self.assertEqual(result.shape, (3, 3, 65))
        self.assertEqual(trains, [([(0, 1)], 2, 128), ([(1, 2)], 2, 128)])
        publishes = [i for i, (kind, _) in enumerate(events) if kind == 'migration_publish']
        consumers = [i for i, (kind, _) in enumerate(events) if kind in ('migration_ready', 'migration_land')]
        self.assertLess(max(publishes), min(consumers))
        for kind in ('migration_publish', 'migration_land'):
            constants = [call[2] for name, call in events if name == kind]
            self.assertEqual([words[14] for words in constants], [0, 1], 'each frame must own its constant snapshot')
            self.assertIsNot(constants[0], constants[1])


if __name__ == '__main__':
    unittest.main()
