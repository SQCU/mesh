"""Exercise the actual live-regression inputs and guard oracle without a NIC."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


def load_fixture():
    root = Path(__file__).resolve().parents[1]
    view_spec = importlib.util.spec_from_file_location('views', root / 'views.py')
    views = importlib.util.module_from_spec(view_spec)
    view_spec.loader.exec_module(views)
    dependencies = {
        'mesh': types.SimpleNamespace(ffi=None, lib=None, check=None),
        'mesh_mpi': types.SimpleNamespace(Comm=None, empty=np.empty),
        'routing': types.SimpleNamespace(Fabric=None, QueueSample=None),
        'views': views,
    }
    spec = importlib.util.spec_from_file_location('live_indexed_fixture', root / 'tests/indexed-writes-physical.py')
    fixture = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(fixture)
    return fixture


class LiveIndexedInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = load_fixture()

    def transfer_for_oracle(self, cases):
        # Independent byte-order copy is only a test oracle. It is absent from
        # the live runner, whose movement is exclusively prepared library WRITE.
        source = cases[0]
        logical = np.concatenate([source['backing'][offset:offset+size]
                                  for offset, size in source['spans']])
        target = cases[1]
        at = 0
        for offset, size in target['spans']:
            target['backing'][offset:offset+size] = logical[at:at+size]
            at += size
        self.assertEqual(at, len(logical))
        for rank, case in enumerate(cases):
            self.fixture.verify(case, rank)

    def test_different_counts_repeated_source_and_guard_corruption(self):
        cases = [self.fixture.mapped_case(rank, np.empty) for rank in range(3)]
        self.assertEqual([len(case['spans']) for case in cases], [3, 2, 3])
        self.assertEqual(cases[0]['spans'][0], cases[0]['spans'][2])
        self.transfer_for_oracle(cases)
        cases[1]['backing'][0] ^= 1
        with self.assertRaises(AssertionError):
            self.fixture.verify(cases[1], 1)

    def test_original_negative_strides_and_value_corruption(self):
        cases = [self.fixture.reversed_case(rank, np.empty) for rank in range(3)]
        self.assertTrue(all(stride < 0 for stride in cases[1]['own'].strides))
        self.assertTrue(all(np.shares_memory(case['backing'], case['own']) for case in cases))
        self.transfer_for_oracle(cases)
        cases[1]['own'][3, 7] ^= 1
        with self.assertRaises(AssertionError):
            self.fixture.verify(cases[1], 1)

    def test_registration_case_only_initializes_sparse_payload_and_guards(self):
        untouched = 229
        allocate = lambda shape, dtype: np.full(shape, untouched, dtype)
        cases = [self.fixture.registration_case(rank, allocate, extent=1 << 16) for rank in range(3)]
        source = cases[0]
        origin, size = source['spans'][0]
        changed = np.count_nonzero(source['backing'][origin:origin+size] != untouched)
        self.assertEqual(changed, sum(len(values) for _, values in source['probes']))
        self.assertLess(changed, size // 100)
        self.assertGreater(size, source['registration_extent'])
        self.assertNotEqual(origin % 4096, 0)
        self.transfer_for_oracle(cases)
        logical, _ = cases[1]['probes'][-1]
        cases[1]['backing'][cases[1]['spans'][0][0]+logical] ^= 1
        with self.assertRaises(AssertionError):
            self.fixture.verify(cases[1], 1)

    def test_counter_identity_uses_physical_wire_and_rejects_changed_fabric(self):
        before = [dict(cable=2, wire=93, peer=1, sent=17, landed=23)]
        after = [dict(cable=2, wire=93, peer=1, sent=31, landed=47)]
        self.assertEqual(self.fixture.traffic(before, after),
                         [dict(cable=2, wire=93, peer=1, sent_bytes=14, landed_bytes=24)])
        after[0]['wire'] += 1
        with self.assertRaises(AssertionError):
            self.fixture.traffic(before, after)


if __name__ == '__main__':
    unittest.main()
