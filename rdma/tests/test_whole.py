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


if __name__ == '__main__':
    unittest.main()
