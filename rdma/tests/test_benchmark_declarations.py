"""Benchmark metadata uses the production compiler on arbitrary rank counts."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / 'torch-mesh/torch_mesh/benchmarks.py'
SPEC = importlib.util.spec_from_file_location('collective_benchmarks', PATH)
benchmarks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmarks)


class BenchmarkDeclarations(unittest.TestCase):
    def test_directed_matrix_covers_all_ranks_and_exact_bytes(self):
        for size in (1, 2, 3, 7):
            for count in (1, 3, 16385):
                rows = [benchmarks.exchange_counts(rank, size, count) for rank in range(size)]
                self.assertEqual(len({tuple(row[2]) for row in rows}), 1)
                for source, (sends, receives, matrix) in enumerate(rows):
                    self.assertEqual(sum(sends), count)
                    self.assertEqual(sum(receives), count)
                    for destination in range(size):
                        self.assertEqual(sends[destination], rows[destination][1][source])
                        self.assertEqual(matrix[source * size + destination], sends[destination])

    def test_measurement_barrier_is_outside_function_declaration(self):
        events = []
        fake = types.ModuleType('torch')
        fake.mps = types.SimpleNamespace(synchronize=lambda: events.append('gpu-complete'))
        fake.distributed = types.ModuleType('torch.distributed')
        fake.distributed.barrier = lambda: events.append('control-barrier')
        group = types.SimpleNamespace(plan_active=lambda active: events.append(('active', active)))
        with patch.dict(sys.modules, {'torch': fake, 'torch.distributed': fake.distributed}):
            benchmarks._boundary(group)
        self.assertEqual(events, [('active', False), 'gpu-complete', 'control-barrier',
                                  'gpu-complete', ('active', True)])


if __name__ == '__main__':
    unittest.main()
