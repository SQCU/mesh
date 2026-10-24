"""design/whole-function.md#ahead-of-time"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from routing import Fabric


class RoutingTests(unittest.TestCase):
    def fabric(self):
        pairs = [(0, 1)] * 4 + [(0, 2)] + [(1, 2)] * 2
        return Fabric(3, [{'nodes': list(pair), 'wire': e + 1, 'bits_per_second': 80_000_000_000} for e, pair in enumerate(pairs)])

    def test_parallel_and_relay_capacity(self):
        fabric = self.fabric()
        decisions, result = fabric.allocate({(1, 0): 1_000_000_000})
        self.assertEqual(len(fabric.paths[1, 0]), 6)
        self.assertAlmostEqual(result['fluid_lower_bound_seconds'], 0.02, places=7)
        self.assertAlmostEqual(sum(decisions['route:1:0']['parts']), 4096)
        self.assertEqual(len([x for x in result['directed_load_bytes'] if x > 0]), 7)

    def test_simultaneous_demands_share_directed_cables(self):
        fabric = self.fabric()
        demands = {pair: 1_000_000_000 for pair in fabric.paths}
        _, result = fabric.allocate(demands)
        span = result['fluid_lower_bound_seconds']
        self.assertGreaterEqual(span, 2_000_000_000 / 30_000_000_000 - 1e-8)
        self.assertTrue(all(x > 0 for x in result['directed_load_bytes']))
        self.assertTrue(all(x <= 10_000_000_000 * span * 1.000001 for x in result['directed_load_bytes']))
        # Extra Studio relays can fill all fourteen directions without adding
        # any bytes to the Mini cut. A min-hop face wrongly excludes this.
        self.assertTrue(all(abs(x / 10_000_000_000 - span) < 1e-7 for x in result['directed_load_bytes']))
        for direction in (0, 1):
            parallel = result['directed_load_bytes'][direction:8:2]
            self.assertLess(max(parallel) - min(parallel), 1e-5 * demands[0, 1])

    def test_spare_ragged_parallel_capacity_has_equal_service_times(self):
        fabric = self.fabric()
        fabric.cables[1]['bits_per_second'] //= 2
        demands = {pair: 1_000_000_000 for pair in fabric.paths}
        _, result = fabric.allocate(demands)
        for direction in (0, 1):
            times = [result['directed_load_bytes'][2 * e + direction] /
                     (fabric.cables[e]['bits_per_second'] / 8) for e in range(4)]
            self.assertLess(max(times) - min(times), 1e-7)


if __name__ == '__main__':
    unittest.main()
