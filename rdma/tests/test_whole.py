"""design/whole-function.md#finite-differences"""
import copy
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from whole import FunctionSearch, project


class WholeFunctionTests(unittest.TestCase):
    def search(self):
        return FunctionSearch({'compute': {'parts': [64, 64], 'grain': 1}, 'route': {'parts': [64, 64], 'grain': 1}})

    @staticmethod
    def latency(parts):
        x, y = parts['compute'][0], parts['route'][0]
        return 1 + (x - 94) ** 2 / 1000 + (y - (128 - x) * 0.75) ** 2 / 1000

    def test_coupled_finite_differences_and_drift(self):
        search = self.search()
        initial = self.latency(search.centre)
        for iteration in range(220):
            request = search.ask()
            observation = {**request, 'scope': 'function', 'complete': True, 'workload': 'full-dnn',
                           'seconds': self.latency(request['parts']) + iteration * 0.0005,
                           'queues': {'0': {'peak_bytes': 1000, 'integral_byte_seconds': request['parts']['route'][0]}}}
            restored = FunctionSearch.of(search.state())
            self.assertEqual(search.observe(observation), restored.observe(observation))
            for values in search.parts.values():
                self.assertEqual(sum(values), 128)
                self.assertGreaterEqual(min(values), 1)
        self.assertLess(self.latency(search.centre), 1.02)
        self.assertLess(self.latency(search.centre), initial * 0.6)
        self.assertTrue(any(record.get('accepted') for record in search.history))
        self.assertTrue(all('derivatives' in record for record in search.history))

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


if __name__ == '__main__':
    unittest.main()
