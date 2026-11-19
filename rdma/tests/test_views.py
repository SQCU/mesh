"""Shared array-layout metadata maps logical bytes without touching values."""
import importlib.util
import itertools
from pathlib import Path
import random
import unittest

spec = importlib.util.spec_from_file_location('actual_views', Path(__file__).resolve().parents[1]/'views.py')
views = importlib.util.module_from_spec(spec)
spec.loader.exec_module(views)


class ByteViews(unittest.TestCase):
    def check_mapping(self, shape, strides, width):
        shift, spans = views.byte_spans(shape, strides, width)
        actual = [shift+offset+j for offset, length in spans for j in range(length)]
        expected = [sum(i*s for i, s in zip(index, strides))+j
                    for index in itertools.product(*(range(n) for n in shape)) for j in range(width)]
        self.assertEqual(actual, expected)
        self.assertTrue(all(offset >= 0 and length > 0 for offset, length in spans))
        return shift, spans

    def test_contiguous_suffixes_and_affine_rows_stay_compact(self):
        self.assertEqual(self.check_mapping((2, 3, 7), (84, 28, 4), 4), (0, [(0, 168)]))
        shift, spans = views.byte_spans((5003, 257), (263, 1), 1)
        self.assertEqual(shift, 0)
        self.assertEqual(len(spans), 5003)
        self.assertEqual(spans[-1], (5002*263, 257))

    def test_transpose_reverse_broadcast_and_structured_elements(self):
        for shape, strides, width in (((3, 2), (4, 12), 4), ((2, 3, 4), (57, -17, 3), 3),
                                       ((2, 3), (0, 4), 4), ((2, 3), (-20, -4), 4),
                                       ((1, 5), (-999, 1), 1), ((), (), 8)):
            self.check_mapping(shape, strides, width)

    def test_empty_and_invalid_metadata(self):
        self.assertEqual(views.byte_spans((5, 0, 3), (9, 3, 1), 1), (0, []))
        self.assertEqual(views.byte_spans((7,), (0,), 0), (0, []))
        for args in (((-1,), (1,), 1), ((3,), (), 1), ((3,), (1,), -1)):
            with self.assertRaises(ValueError):
                views.byte_spans(*args)
        with self.assertRaises(TypeError):
            views.byte_spans((3.5,), (1,), 1)

    def test_random_metadata_matches_independent_logical_address_enumeration(self):
        rng = random.Random(341)
        for _ in range(300):
            dimensions = rng.randrange(5)
            shape = tuple(rng.randrange(1, 5) for _ in range(dimensions))
            strides = tuple(rng.randrange(-80, 81) for _ in shape)
            self.check_mapping(shape, strides, rng.randrange(1, 9))


if __name__ == '__main__':
    unittest.main()
