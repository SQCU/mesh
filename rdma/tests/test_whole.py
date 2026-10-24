"""design/whole-function.md#finite-differences"""
import copy
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from whole import FunctionFlow, project
from allocate import equal_finish


class WholeFunctionTests(unittest.TestCase):
    def search(self):
        return FunctionFlow({'compute': {'parts': [64, 64], 'grain': 0}, 'route': {'parts': [64, 64], 'grain': 0}})

    @staticmethod
    def latency(parts):
        x, y = parts['compute'][0], parts['route'][0]
        return 1 + (x - 94) ** 2 / 1000 + (y - (128 - x) * 0.75) ** 2 / 1000

    def test_coupled_finite_differences_and_drift(self):
        search = self.search()
        initial = self.latency(search.centre)
        for iteration in range(1200):
            request = search.ask()
            observation = {**request, 'scope': 'function', 'complete': True, 'workload': 'full-dnn',
                           'seconds': self.latency(request['parts']) + iteration * 0.0005,
                           'queues': {'0': {'peak_bytes': 1000, 'integral_byte_seconds': request['parts']['route'][0]}}}
            restored = FunctionFlow.of(search.state())
            self.assertEqual(search.observe(observation), restored.observe(observation))
            for values in search.parts.values():
                self.assertAlmostEqual(sum(values), 128)
                self.assertGreaterEqual(min(values), 1)
        self.assertLess(self.latency(search.centre), 1.05)
        self.assertLess(self.latency(search.centre), initial * 0.6)
        self.assertEqual(len(search.history), 300)
        self.assertTrue(all('gradient' in record and 'accepted' not in record for record in search.history))

    def test_every_stencil_updates_without_acceptance_state(self):
        search = self.search()
        before = search.centre
        for seconds in (60, 70, 90, 120):
            request = search.ask()
            self.assertEqual(request['phase'], 'finite_difference')
            self.assertTrue(any(v != int(v) for p in request['parts'].values() for v in p))
            search.observe({**request, 'scope': 'function', 'complete': True, 'workload': 'dnn', 'seconds': seconds})
        self.assertEqual(search.round, 1)
        self.assertNotEqual(search.centre, before)
        self.assertNotIn('standing', search.state())
        self.assertNotIn('radius', search.state())
        self.assertNotIn('accepted', search.history[0])

    def test_centre_measurement_preserves_optimizer_state(self):
        search = self.search()
        before = search.state()
        request = search.ask(explore=False)
        self.assertEqual(request['continuous'], search.centre)
        self.assertEqual(request['offset'], 0)
        self.assertEqual(request['phase'], 'centre')
        self.assertEqual(search.state(), before)

    def test_congestion_continuously_moves_routes_at_equal_latency(self):
        flow = FunctionFlow({'route': {'parts': [64, 64], 'grain': 0}})
        for _ in range(400):
            request = flow.ask()
            congestion = (request['parts']['route'][0] - 100) ** 2 / 1000
            flow.observe({**request, 'scope': 'function', 'complete': True, 'workload': 'dnn',
                          'seconds': 1.0, 'congestion_seconds': congestion})
        self.assertAlmostEqual(flow.centre['route'][0], 100, delta=2)

    def test_partial_or_wrong_configuration_cannot_train(self):
        search = self.search()
        sample = {**search.ask(), 'scope': 'tensor', 'complete': True, 'seconds': 1}
        with self.assertRaises(ValueError):
            search.observe(sample)
        sample['scope'] = 'function'
        sample['parts'] = copy.deepcopy(sample['parts'])
        sample['parts']['route'] = [63, 65]
        with self.assertRaises(ValueError):
            search.observe(sample)
        self.assertEqual(search.serial, 0)

    def test_projection_bounds_and_ragged_grains(self):
        self.assertEqual(project([2, 80, 4], 96, 8, [8, 8, 8], [48, 48, 48]), [24, 48, 24])
        with self.assertRaises(ValueError):
            project([1, 1], 3, 2, [0, 0], [4, 4])

    def test_historical_equal_finish_accounts_for_fixed_cost_and_tiles(self):
        tiles, continuous, finish = equal_finish(30, 2, [2, 2, 2], [30, 30, 30], [1, 3, 5], [1, 1, 3])
        self.assertAlmostEqual(sum(continuous), 30)
        self.assertTrue(all(abs(a + b * s - finish) < 1e-10 for a, b, s in zip([1, 3, 5], [1, 1, 3], continuous)))
        actual = max(a + b * s for a, b, s in zip([1, 3, 5], [1, 1, 3], tiles))
        brute = min(max(1 + x, 3 + y, 5 + 3 * (30 - x - y)) for x in range(2, 29, 2)
                    for y in range(2, 30 - x, 2))
        self.assertEqual(actual, brute)
        self.assertLess(tiles[2], min(tiles[:2]))

    def test_complete_function_compute_model_moves_work_off_slow_node(self):
        flow = FunctionFlow({'compute': {'parts': [512, 512, 512], 'grain': 64, 'low': [64] * 3,
                                         'high': [1536] * 3}}, fit_compute=True)
        for _ in range(100):
            request = flow.ask()
            times = [0.001 + b * x for b, x in zip([1e-5, 1.2e-5, 5e-5], request['parts']['compute'])]
            tails = [0.002, 0.002, 0.004]
            observation = {**request, 'scope': 'function', 'complete': True, 'workload': 'dnn',
                           'seconds': max(t + d for t, d in zip(times, tails)),
                           'compute_seconds': times, 'delivery_tail_seconds': tails}
            restored = FunctionFlow.of(flow.state())
            self.assertEqual(flow.observe(observation), restored.observe(observation))
        self.assertLess(flow.centre['compute'][2], min(flow.centre['compute'][:2]) / 3)
        self.assertEqual(flow.round, 25)
        self.assertAlmostEqual(flow.compute_model['fixed'][2], 0.001, delta=0.0001)
        self.assertAlmostEqual(flow.compute_model['slopes'][2], 5e-5, delta=1e-6)

    def test_finish_residual_steers_routes_at_equal_function_latency(self):
        flow = FunctionFlow({'route': {'parts': [64, 64], 'grain': 0}})
        for _ in range(400):
            request = flow.ask()
            residual = (request['parts']['route'][0] - 90) ** 2 / 1000
            flow.observe({**request, 'scope': 'function', 'complete': True, 'workload': 'dnn',
                          'seconds': 1.0, 'finish_variance': residual})
        self.assertAlmostEqual(flow.centre['route'][0], 90, delta=2)

    def test_compute_allocation_accounts_for_late_input_release(self):
        flow = FunctionFlow({'compute': {'parts': [100, 100], 'grain': 1, 'low': [1, 1], 'high': [199, 199]}},
                            fit_compute=True)
        for _ in range(80):
            request = flow.ask()
            times = [0.01 + x * 0.001 for x in request['parts']['compute']]
            release = [times[0], times[1] + 0.08]
            origin = min(release)
            flow.observe({**request, 'scope': 'function', 'complete': True, 'workload': 'late-input-dnn',
                          'seconds': max(release), 'compute_seconds': times, 'delivery_tail_seconds': [0.1, 0.1],
                          'producer_release_seconds': [t - origin for t in release], 'finishes': {'stages': [{}]}})
        model = flow.compute_model
        self.assertAlmostEqual(model['start_delay'][1], 0.08, places=8)
        self.assertLess(model['target'][1], 65)
        self.assertAlmostEqual(model['target'][0] - model['target'][1], 80, delta=3)

    def test_physical_finish_correction_handles_unequal_capacity_and_release(self):
        from routing import Fabric
        fabric = Fabric(2, [{'nodes': [0, 1], 'bits_per_second': 80}, {'nodes': [0, 1], 'bits_per_second': 160}])
        demands = {(0, 1): 96, (1, 0): 96}
        decisions, _ = fabric.allocate(demands)
        model = fabric.completion_model(demands)
        flow = FunctionFlow(decisions, routing_model=model)

        def measured(parts):
            loads = [0.0] * 4
            for name, spec in model['paths'].items():
                for weight, arcs in zip(parts[name], spec['arcs']):
                    for e in arcs:
                        loads[e] += spec['bytes'] * weight / sum(parts[name])
            finish = [q / c + delay for q, c, delay in zip(loads, model['capacities'], [0, 0, 0.7, 0.7])]
            arcs = {str(e): {'bytes': loads[e], 'start_ns': [0, 0], 'finish_ns': [t * 1e9] * 2} for e, t in enumerate(finish)}
            return finish, arcs

        for _ in range(80):
            request = flow.ask()
            finish, arcs = measured(request['parts'])
            observation = {**request, 'scope': 'function', 'complete': True, 'workload': 'full-program',
                           'seconds': max(finish), 'finishes': {'stages': [{'arcs': arcs}]}}
            restored = FunctionFlow.of(flow.state())
            self.assertEqual(flow.observe(observation), restored.observe(observation))
        finish, _ = measured(flow.centre)
        self.assertLess(max(finish) - min(finish), max(finish) * 0.04)
        self.assertEqual(len(flow.history), 20)

    def test_release_model_is_fixed_during_each_finite_difference_stencil(self):
        flow = FunctionFlow({'compute': {'parts': [128, 128], 'grain': 64, 'low': [64, 64], 'high': [256, 256]}},
                            fit_compute=True)
        requests = []
        for i in range(12):
            request = flow.ask()
            requests.append(request)
            flow.observe({**request, 'scope': 'function', 'complete': True, 'workload': 'full-dnn', 'seconds': 1.0,
                          'compute_seconds': [0.01 * x for x in request['parts']['compute']],
                          'delivery_tail_seconds': [0.1, 0.1], 'producer_release_seconds': [0.001 * i, 0.002 * i],
                          'finishes': {'stages': [{}]}})
        for start in (0, 4, 8):
            self.assertEqual(requests[start]['producer_seconds'], requests[start + 3]['producer_seconds'])
            self.assertEqual(requests[start + 1]['producer_seconds'], requests[start + 2]['producer_seconds'])

    def test_finish_model_learns_cross_flow_queue_sensitivity(self):
        from routing import Fabric
        fabric = Fabric(2, [{'nodes': [0, 1], 'bits_per_second': 80}, {'nodes': [0, 1], 'bits_per_second': 160}])
        demands = {(0, 1): 96, (1, 0): 96}
        decisions, _ = fabric.allocate(demands)
        model = fabric.completion_model(demands)
        flow = FunctionFlow(decisions, routing_model=model)

        def measured(parts):
            loads = [0.0] * 4
            for name, spec in model['paths'].items():
                for weight, arcs in zip(parts[name], spec['arcs']):
                    for e in arcs:
                        loads[e] += spec['bytes'] * weight / sum(parts[name])
            finish = [q / c for q, c in zip(loads, model['capacities'])]
            # Cross-flow upstream delay changes with the other commodity's
            # route. A constant fitted cable offset cannot predict that move.
            finish[0] += loads[3] / 20
            finish[1] += loads[2] / 20
            finish[2] += 0.7
            finish[3] += 0.7
            arcs = {str(e): {'bytes': loads[e], 'start_ns': [0, 0], 'finish_ns': [t * 1e9] * 2} for e, t in enumerate(finish)}
            return finish, arcs

        for _ in range(160):
            request = flow.ask()
            finish, arcs = measured(request['parts'])
            flow.observe({**request, 'scope': 'function', 'complete': True, 'workload': 'cross-flow-dnn',
                          'seconds': max(finish), 'finishes': {'stages': [{'arcs': arcs}]}})
        finish, _ = measured(flow.centre)
        self.assertLess(max(finish) - min(finish), max(finish) * 0.05)
        self.assertLess(flow.routing_fit['residual_fit_rms_seconds'], 0.02)
        self.assertGreater(max(abs(x) for row in flow.routing_fit['residual_jacobian_seconds'] for x in row), 0.1)


if __name__ == '__main__':
    unittest.main()
