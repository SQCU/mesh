import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controller import WholeFunctionController
import allocate
from routing import Fabric


class ControllerTests(unittest.TestCase):
    def controller(self):
        return WholeFunctionController({'tp': {'parts': [512, 512], 'grain': 16},
                                        'engine': {'parts': [.5, .5]},
                                        'route': {'parts': [.5, .5]}}, workload='whole DNN')

    def test_calibration_then_four_whole_calls_no_gate(self):
        c = self.controller()
        first = c.proposal()
        self.assertEqual(first['phase'], 'centre')
        self.assertEqual(first['parts']['tp'], [512, 512])
        c.completed(1)
        self.assertEqual(len(c.flow.samples), 0)
        for seconds in (10, 20, 30, 40):
            c.completed(seconds)
        self.assertEqual(c.round, 1)
        self.assertEqual(len(c.history), 1)
        self.assertEqual([s['seconds'] for s in c.history[0]['samples']], [10, 20, 30, 40])
        self.assertFalse(any(k in c.state() for k in ('accepted', 'best', 'rejected', 'radius')))

    def test_identity_scope_and_workload_are_not_optional(self):
        c = self.controller()
        p = c.proposal()
        for patch in ({'scope': 'tensor'}, {'complete': False}, {'workload': 'other'}, {'id': 17}):
            with self.assertRaises(ValueError):
                c.observe({'id': p['id'], 'parts': p['parts'], 'workload': c.workload,
                           'seconds': 1, 'scope': 'function', 'complete': True, **patch})
        c.completed(1, proposal=p)
        with self.assertRaises(ValueError):
            c.completed(1, proposal=p)

    def test_failure_preserves_vector_and_does_not_invent_sample(self):
        c = self.controller()
        before = c.proposal()
        c.failed('compiled layout unavailable')
        self.assertEqual(before, c.proposal())
        self.assertFalse(c.flow.samples)
        self.assertEqual(c.failures[-1]['reason'], 'compiled layout unavailable')

    def test_zero_share_quantized_engine_can_enter_the_function(self):
        c = WholeFunctionController({'engines': {'parts': [512, 0], 'grain': 64}}, workload='graph')
        self.assertEqual(c.parts['engines'], [512, 0])
        c.completed(1)
        plus = c.parts['engines']
        c.completed(1)
        minus = c.parts['engines']
        self.assertEqual(sorted([plus[1], minus[1]]), [0, 64])
        self.assertTrue(all(sum(p) == 512 for p in (plus, minus)))

    def test_fraction_quantum_is_a_latent_resolution_not_a_lower_bound(self):
        c = WholeFunctionController({'engine': {'parts': [1., 0.], 'quantum': .125}}, workload='graph')
        self.assertEqual(c.parts['engine'], [1., 0.])
        c.completed(1)
        plus = allocate.project_parts(c.parts['engine'], 512, 64)
        c.completed(1)
        minus = allocate.project_parts(c.parts['engine'], 512, 64)
        self.assertEqual(sorted([plus[1], minus[1]]), [0, 64])

    def test_round_trip_and_coupled_objective(self):
        c = self.controller()
        def latency(parts):
            x, y, z = parts['tp'][0] / 1024, parts['engine'][0], parts['route'][0]
            return 1 + 4 * (x - .7) ** 2 + 2 * (y - x / 2) ** 2 + 3 * (z - (1 - y)) ** 2
        initial = latency(c.parts)
        for i in range(801):
            c.completed(latency(c.parts))
            if i == 100:
                saved = json.loads(json.dumps(c.state()))
                restored = WholeFunctionController.of(saved)
                self.assertEqual(c.proposal(), restored.proposal())
                c = restored
        self.assertLess(latency(c.proposal(explore=False)['parts']), initial * .85)
        self.assertLessEqual(len(c.history), 64)

    def test_every_physical_path_is_in_same_vector(self):
        fabric = Fabric(3, [{'nodes': [0, 1], 'wire': 0, 'bits_per_second': 80e9},
                            {'nodes': [0, 1], 'wire': 1, 'bits_per_second': 80e9},
                            {'nodes': [0, 2], 'wire': 2, 'bits_per_second': 80e9},
                            {'nodes': [2, 1], 'wire': 3, 'bits_per_second': 80e9}])
        c = WholeFunctionController.from_fabric({'tp': {'parts': [512, 256, 64], 'grain': 64}},
                                                fabric, {(0, 1): 1000000}, workload='DNN')
        self.assertEqual(len(c.parts['route:0:1']), 3)
        self.assertTrue(all(p > 0 for p in c.parts['route:0:1']))
        self.assertIn('tp', c.parts)

    def test_two_communicators_do_not_alias_route_coordinates(self):
        graph = {'nodes': 2, 'cables': [{'nodes': [0, 1], 'wire': 0, 'bits_per_second': 80e9},
                                       {'nodes': [0, 1], 'wire': 1, 'bits_per_second': 80e9}],
                 'demands': [[0, 1, 19]], 'weights': {'route:0:1': [1, 2]}}
        c = WholeFunctionController.from_graphs({'compute': {'parts': [16, 16], 'grain': 4}},
                                                {'forward': graph, 'reverse': graph}, workload='whole')
        self.assertEqual(c.route_operands(), {'forward': {'route:0:1': [1, 2]},
                                             'reverse': {'route:0:1': [1, 2]}})
        self.assertEqual(set(c.parts), {'compute', 'flow:forward/route:0:1', 'flow:reverse/route:0:1'})
        with self.assertRaisesRegex(ValueError, 'name its declared communicator'):
            c.install_routes(None)

    def test_competing_online_interfaces_are_removed(self):
        self.assertFalse(hasattr(allocate, 'Balancer'))
        self.assertFalse(hasattr(allocate, 'Coupled'))
        self.assertTrue(callable(allocate.min_max))
        self.assertTrue(callable(allocate.project_parts))

    def test_native_declaration_retains_real_header_count_arrays(self):
        # Use the production header-derived CFFI types, never a Python layout.
        try:
            import mesh
        except ImportError as error:
            self.skipTest(str(error))
        import types
        from controller import declare, function_demands
        ffi, captured, held = mesh.ffi, [], []
        typed = []
        def plan(comm, calls, count, result):
            import gc
            gc.collect()
            captured.extend((calls[i].what, calls[i].count, calls[i].group, calls[i].repeat,
                             list(calls[i].counts[0:3]) if calls[i].counts else None) for i in range(count))
            typed.extend((list(calls[i].matrix[0:9]) if calls[i].matrix else None,
                          list(calls[i].contributors[0:1]) if calls[i].contributors else None,
                          list(calls[i].recipients[0:1]) if calls[i].recipients else None) for i in range(count))
            entries = ffi.new('uint64_t[][2]', [[5, 7], [11, 13]])
            made = ffi.new('ncclMeshPlan *', {'groups': 2, 'issues': 3, 'entries': entries})
            held.extend((entries, made))
            result[0] = made
            return 0
        def demands(comm, out, count):
            self.assertEqual(count, 9)
            out[1], out[5] = 3, 100000019
            return 0
        fake = types.SimpleNamespace(ffi=ffi, check=mesh.check,
               lib=types.SimpleNamespace(ncclMeshCommPlan=plan, ncclMeshCommPlanDemands=demands))
        fabric = Fabric(3, [{'nodes': [0, 1], 'wire': 0, 'bits_per_second': 80e9},
                            {'nodes': [1, 2], 'wire': 1, 'bits_per_second': 80e9}])
        with patch.dict(sys.modules, {'mesh': fake}), patch.object(Fabric, 'of', return_value=fabric):
            actual = declare(ffi.NULL, [{'what': mesh.lib.MESH_ALLGATHER, 'type': mesh.lib.ncclUint8,
                                         'count': 3, 'counts': [3, 7, 19], 'repeat': 2,
                                         'matrix': list(range(9)), 'contributors': [5], 'recipients': [4]},
                                        {'what': mesh.lib.MESH_ALLREDUCE, 'count': 17,
                                         'type': mesh.lib.ncclFloat32, 'op': mesh.lib.ncclSum}])
            self.assertEqual(actual, [(5, 7), (11, 13)])
            self.assertEqual(captured[0][1:], (3, 0, 2, [3, 7, 19]))
            self.assertEqual(captured[1][1:], (17, 1, 1, None))
            self.assertEqual(typed, [(list(range(9)), [5], [4]), (None, None, None)])
            self.assertEqual(function_demands(ffi.NULL)[1], {(0, 1): 3, (1, 2): 100000019})


if __name__ == '__main__':
    unittest.main()
