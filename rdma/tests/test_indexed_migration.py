"""Actual affine lowering with delayed exact-byte DMA and guarded Torch views.

The mock executor uses original source/final destination pointers and executes
only after every rank finishes configuration. GPU publication and physical
registration cuts are exercised separately by native and cluster tests.
"""
import math
from types import SimpleNamespace
import unittest

from test_generic_plans import functions
from test_migration import compile_rows

try:
    import torch
except ImportError:
    torch = None


class Group:
    def __init__(self):
        self.writes, self.events, self.allocations = [], [], []
        self.fabric = SimpleNamespace(adjacent=[[(1, 0), (1, 1)]] * 3)

    @property
    def page(self):
        raise AssertionError('exact affine writes must not pad to the allocation page')


@unittest.skipUnless(torch is not None, 'requires the torch environment')
class IndexedMigrationTests(unittest.TestCase):
    def guarded(self, shape, dtype):
        size = torch.empty(0, dtype=dtype).element_size()
        prefix, count = (17 if size == 1 else 16), math.prod(shape) * size
        storage = torch.full((prefix + count + 19,), 173, dtype=torch.uint8)
        return storage.narrow(0, prefix, count).view(dtype).view(shape), storage, prefix

    def lowering(self, old, new):
        def empty(shape, like, group, dtype=None):
            tensor, storage, prefix = self.guarded(shape, dtype or like.dtype)
            group.allocations.append((tensor, storage, prefix))
            return tensor

        def write_rows(first, landings, pairs, group, row_bytes, rows, stride):
            self.assertIs(first, landings[0])
            self.assertEqual(first.dtype, torch.uint8)
            self.assertEqual(first.numel(), row_bytes)
            self.assertLessEqual(first.storage_offset() + (rows - 1) * stride + row_bytes,
                                 first.untyped_storage().nbytes())
            group.writes.append((first, pairs[0], row_bytes, rows, stride))
            group.events.append(('bind', pairs[0]))
            return (group, pairs[0])

        def encode(kernel, buffers, words, at, gx, gy, threads, channels, land=None, publish=None):
            if kernel[0] == 'migration_local':
                vector = kernel[1]
                values = [words[k] + (words[k + 1] << 32) for k in range(0, 14, 2)]
                source, destination, row, rows, spitch, dpitch, _ = values
                x, y = [t.view(-1).view(torch.uint8) for t in buffers]
                for r in range(rows):
                    y.narrow(0, (destination + r * dpitch) * vector, row * vector).copy_(
                        x.narrow(0, (source + r * spitch) * vector, row * vector))
            elif kernel[0] == 'migration_publish_existing':
                self.assertEqual(buffers, [], 'publication must not read or copy payload')
                self.assertEqual(words[14], 0, 'affine rows are one logical message, not repeated entries')
                publish[0].events.append(('publish', publish[1]))
            elif kernel[0] == 'migration_ready':
                self.assertEqual(buffers, [], 'direct final writes need no unpack pass')
                self.assertEqual(words[14], 0)
                land[0].events.append(('ready', land[1]))
            else:
                self.fail(f'unexpected payload kernel: {kernel[0]}')

        env = functions('_follow_bytes', '_migration_words', compile_rows=compile_rows,
                        _OPERANDS={'old': SimpleNamespace(capacity=old), 'new': SimpleNamespace(capacity=new)},
                        _region_backed=lambda *_: True, _empty=empty, address=lambda t: t.data_ptr(),
                        _migration_kernel=lambda *args: args, _GATHER_THREADS=256,
                        fused=SimpleNamespace(write_rows=write_rows, encode=encode,
                                              encode_many=lambda calls: [encode(*call) for call in calls]))
        env['torch'] = torch
        return env['_follow_bytes']

    def execute(self, shape, dim, old, new, dtype=None):
        dtype = dtype or torch.float32
        size = torch.empty(0, dtype=dtype).element_size()
        bits = torch.arange(math.prod(shape) * size, dtype=torch.int64).remainder(251).to(torch.uint8)
        bits[:min(16, bits.numel())] = 255  # preserve NaN payloads as raw bytes
        full = bits.view(dtype).view(shape)
        grain, sources, backing = shape[dim] // sum(old), [], []
        for rank, part in enumerate(old):
            block = full.narrow(dim, grain * sum(old[:rank]), grain * part).contiguous()
            source, storage, prefix = self.guarded(block.shape, dtype)
            source.view(-1).view(torch.uint8).copy_(block.view(-1).view(torch.uint8))
            sources.append(source)
            backing.append((storage, storage.clone(), prefix))
        groups = [Group() for _ in old]
        lower = self.lowering(old, new)
        output = [lower(SimpleNamespace(key='old', dim=dim), tensor, shape, full.stride(), 'new', rank, group,
                        lambda group=group: group.events.append(('boundary', None)))
                  for rank, (tensor, group) in enumerate(zip(sources, groups))]
        self.assertTrue(all(t is not None for t in output))
        remote = [w for w in compile_rows(shape, dim, old, new, size).runs if w.source != w.destination]
        relays = []
        for group in groups:
            self.assertEqual([(a, b, width, rows) for _, (a, b), width, rows, _ in group.writes],
                             [(w.source, w.destination, w.row_bytes, w.rows) for w in remote])
            publishes = [i for i, (event, _) in enumerate(group.events) if event == 'publish']
            ready = [i for i, (event, _) in enumerate(group.events) if event == 'ready']
            if publishes and ready:
                self.assertLess(max(publishes), min(ready), 'independent sources must precede incoming waits')
        for i, run in enumerate(remote):
            sending, _, width, rows, spitch = groups[run.source].writes[i]
            self.assertEqual(sending.data_ptr(), sources[run.source].data_ptr() + run.source_offset)
            self.assertEqual(spitch, run.source_stride)
            for rank, group in enumerate(groups):
                if rank == run.source:
                    continue
                landing, _, _, _, pitch = group.writes[i]
                if rank == run.destination:
                    self.assertEqual(landing.data_ptr(), output[rank].data_ptr() + run.destination_offset)
                    self.assertEqual(pitch, run.destination_stride)
                else:
                    self.assertEqual(pitch, width, 'relay bytes are compact, without padding')
                    relays.append(landing)
                    self.assertNotEqual(landing.untyped_storage().data_ptr(), sources[rank].untyped_storage().data_ptr())
                    self.assertNotEqual(landing.untyped_storage().data_ptr(), output[rank].untyped_storage().data_ptr())
                landing.as_strided((rows, width), (pitch, 1)).copy_(sending.as_strided((rows, width), (spitch, 1)))
        self.assertEqual(len({t.untyped_storage().data_ptr() for t in relays}), len(relays))
        for rank, actual in enumerate(output):
            expected = full.narrow(dim, grain * sum(new[:rank]), grain * new[rank]).contiguous()
            self.assertTrue(torch.equal(actual.view(-1).view(torch.uint8), expected.view(-1).view(torch.uint8)))
            self.assertTrue(torch.equal(backing[rank][0], backing[rank][1]), 'original input/guards changed')
            for tensor, storage, prefix in groups[rank].allocations:
                self.assertTrue(torch.all(storage[:prefix] == 173))
                self.assertTrue(torch.all(storage[prefix + tensor.numel() * tensor.element_size():] == 173))
        return sources, output, groups

    def test_leading_axis_aliases_retained_intervals_and_writes_final_destination(self):
        source, output, _ = self.execute((8, 16), 0, [4, 3, 1], [4, 2, 2])
        self.assertEqual(source[0].data_ptr(), output[0].data_ptr())
        self.assertEqual(source[1].data_ptr(), output[1].data_ptr())
        self.assertNotEqual(source[2].data_ptr(), output[2].data_ptr())

    def test_interior_axis_uses_rank_local_pitch_and_direct_exact_bytes(self):
        self.execute((3, 8, 16), 1, [4, 3, 1], [3, 3, 2])
        self.execute((3, 8, 65), 1, [4, 3, 1], [3, 3, 2], torch.uint8)

    def test_two_rank_move_and_zero_share_sources_and_destinations(self):
        self.execute((4, 3), 0, [3, 1], [2, 2], torch.uint8)
        self.execute((3, 8, 16), 1, [0, 8, 0], [4, 0, 4])

    def test_more_than_4096_rows_use_one_binding_per_commodity(self):
        _, _, groups = self.execute((5003, 8, 3), 1, [4, 3, 1], [3, 3, 2], torch.uint8)
        self.assertTrue(all(len(g.writes) == 2 for g in groups))
        self.assertTrue(all(w[3] == 5003 for g in groups for w in g.writes))

    def test_no_movement_aliases_original_storage_without_publication(self):
        source, output, groups = self.execute((8, 16), 0, [4, 3, 1], [4, 3, 1])
        self.assertTrue(all(a.data_ptr() == b.data_ptr() for a, b in zip(source, output)))
        self.assertTrue(all(not g.allocations and not g.events for g in groups))

    def test_declared_source_layout_mismatch_raises_without_issuing(self):
        lower = self.lowering([4, 3, 1], [4, 2, 2])
        group = Group()
        with self.assertRaisesRegex(ValueError, 'declared regional'):
            lower(SimpleNamespace(key='old', dim=0), torch.empty((2, 16)), (8, 16), (16, 1),
                  'new', 1, group, lambda: self.fail('invalid source publish'))
        self.assertFalse(group.allocations or group.events or group.writes)

    def test_noncontiguous_declared_layout_reports_unsupported_before_issue(self):
        group = Group()
        self.assertIsNone(self.lowering([4, 3, 1], [4, 2, 2])(
            SimpleNamespace(key='old', dim=1), torch.empty((3, 3, 5)), (3, 8, 5), (5, 15, 1),
            'new', 1, group, lambda: self.fail('unsupported source publish')))
        self.assertFalse(group.allocations or group.events or group.writes)


if __name__ == '__main__':
    unittest.main()
