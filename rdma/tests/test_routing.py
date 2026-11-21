"""design/whole-function.md#ahead-of-time"""
import pathlib
import ast
import math
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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

    def held(self, topology, destroyed, attached):
        path = pathlib.Path(__file__).resolve().parents[1] / 'torch-mesh/torch_mesh/backend.py'
        cls = next(node for node in ast.parse(path.read_text()).body
                   if isinstance(node, ast.ClassDef) and node.name == 'ProcessGroupMesh')
        held = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_held')
        env = {'ffi': SimpleNamespace(cast=lambda _type, comm: comm),
               'Fabric': SimpleNamespace(of=topology),
               'LIB': SimpleNamespace(ncclCommDestroy=destroyed.append),
               '_stream': SimpleNamespace(attach=lambda *args: attached.append(args))}
        exec(compile(ast.Module(body=[held], type_ignores=[]), str(path), 'exec'), env)
        return env['_held']

    def test_recycled_communicator_address_gets_its_own_topology_snapshot(self):
        first, second = self.fabric(), Fabric(2, [{'nodes': [0, 1], 'bits_per_second': 80}])
        snapshots, reads, destroyed, attached = iter([first, second]), [], [], []

        def topology(comm):
            reads.append(comm)
            return next(snapshots)

        held = self.held(topology, destroyed, attached)
        old, new = SimpleNamespace(), SimpleNamespace()
        held(old, 0, 3, 17)
        held(new, 0, 2, 17)  # allocator can reuse the native address after destroy
        self.assertIs(old.fabric, first)
        self.assertIs(new.fabric, second)
        self.assertEqual(reads, [17, 17])
        self.assertEqual(attached, [(old, 17), (new, 17)])
        self.assertFalse(destroyed)

    def test_unusable_topology_fails_realization_and_releases_communicator(self):
        destroyed, attached = [], []

        def topology(_):
            return Fabric(2, [])

        held = self.held(topology, destroyed, attached)
        with self.assertRaisesRegex(ValueError, 'no path from 0 to 1'):
            held(SimpleNamespace(), 0, 2, 17)
        self.assertEqual(destroyed, [17])
        self.assertFalse(attached)


if __name__ == '__main__':
    unittest.main()
