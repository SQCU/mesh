import itertools
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from migration import compile_migration, compile_rows


def chunks(shape, dim, parts):
    total, offset, chunks = sum(parts), 0, []
    all_indices = tuple(itertools.product(*(range(n) for n in shape)))
    for part in parts:
        extent = shape[dim] * part // total
        chunks.append([index for index in all_indices if offset <= index[dim] < offset + extent])
        offset += extent
    return chunks


class MigrationTests(unittest.TestCase):
    def assert_ranges(self, shape, dim, old, new, element_size):
        plan = compile_migration(shape, dim, old, new, element_size)
        source = chunks(shape, dim, old)
        expected = chunks(shape, dim, new)
        actual = [[None] * len(rank) for rank in expected]
        reads = [[0] * len(rank) for rank in source]
        for write in plan.writes:
            first, target, count = (value // element_size for value in
                                    (write.source_offset, write.destination_offset, write.bytes))
            for i in range(count):
                self.assertIsNone(actual[write.destination][target + i], 'overlapping destination writes')
                actual[write.destination][target + i] = source[write.source][first + i]
                reads[write.source][first + i] += 1
        self.assertEqual(actual, expected)
        self.assertTrue(all(count == 1 for rank in reads for count in rank))
        return plan

    def test_asymmetric_move_has_a_source_lifetime_without_a_return_write(self):
        plan = self.assert_ranges((8, 4096), 0, [4, 3, 1], [4, 2, 2], 4)
        outgoing = [plan.writes[i] for i in plan.source_uses(1)]
        self.assertTrue(any(w.destination == 2 for w in outgoing))
        self.assertFalse(any(w.source == 2 for w in plan.writes if w.destination == 1))
        self.assertFalse(plan.unaligned(16384))

    def test_interior_axis_writes_original_layout_without_pack_or_unpack(self):
        plan = self.assert_ranges((4, 8, 3), 1, [4, 3, 1], [4, 2, 2], 2)
        moved = [w for w in plan.writes if (w.source, w.destination) == (1, 2)]
        self.assertEqual([(w.source_offset, w.destination_offset, w.bytes) for w in moved],
                         [(12, 0, 6), (30, 12, 6), (48, 24, 6), (66, 36, 6)])
        self.assertTrue(plan.unaligned(4096))

    def test_all_axes_ragged_and_empty_ranks_cover_exactly_once(self):
        rng = random.Random(517)
        for _ in range(100):
            old = [0] * 4
            new = [0] * 4
            for _ in range(8):
                old[rng.randrange(4)] += 1
                new[rng.randrange(4)] += 1
            for dim in range(3):
                shape = [2, 3, 5]
                shape[dim] = 8
                self.assert_ranges(tuple(shape), dim, old, new, rng.choice([1, 2, 4, 8]))

    def test_compact_rows_match_exact_ranges_with_masked_padding_and_poisoned_gaps(self):
        rng = random.Random(921)
        for _ in range(200):
            old, new = [0] * 4, [0] * 4
            for _ in range(8):
                old[rng.randrange(4)] += 1
                new[rng.randrange(4)] += 1
            dim = rng.randrange(3)
            shape = [rng.randrange(1, 5), rng.randrange(1, 5), rng.randrange(1, 7)]
            shape[dim] = 8
            size = rng.choice((1, 2, 4, 8))
            exact = compile_migration(shape, dim, old, new, size)
            compact = compile_rows(shape, dim, old, new, size)
            expected = {(w.source, w.destination, w.source_offset + i, w.destination_offset + i)
                        for w in exact.writes for i in range(w.bytes)}
            actual = set()
            for run in compact.runs:
                frame_bytes = rng.randrange(1, 6) * 16
                destination = bytearray([173]) * (run.destination_byte(run.bytes - 1) + 17)
                touched = set()
                for frame in range(run.frames(frame_bytes)):
                    ingress = bytearray([239]) * frame_bytes
                    valid = min(frame_bytes, run.bytes - frame * frame_bytes)
                    for k in range(valid):
                        i = frame * frame_bytes + k
                        ingress[k] = run.source_byte(i) % 251
                    for k in range(frame_bytes):
                        i = frame * frame_bytes + k
                        if i < run.bytes:
                            at = run.destination_byte(i)
                            destination[at] = ingress[k]
                            touched.add(at)
                            actual.add((run.source, run.destination, run.source_byte(i), at))
                    self.assertTrue(all(b == 239 for b in ingress[valid:]), 'poisoned padding must remain outside validity')
                self.assertTrue(all(v == 173 for i, v in enumerate(destination) if i not in touched),
                                'masked writes modified neighboring live bytes')
            self.assertEqual(actual, expected)

    def test_row_descriptor_size_is_independent_of_prefix_count(self):
        plan = compile_rows((10**9, 8, 3), 1, [4, 3, 1], [4, 2, 2], 2)
        self.assertEqual(len(plan.runs), 4)
        moved = next(r for r in plan.runs if r.source != r.destination)
        self.assertEqual((moved.rows, moved.row_bytes, moved.source_stride, moved.destination_stride),
                         (10**9, 6, 18, 12))
        self.assertEqual(moved.source_byte(6), 30)
        self.assertEqual(moved.destination_byte(6), 12)


if __name__ == '__main__':
    unittest.main()
