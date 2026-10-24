"""design/whole-function.md#finite-differences"""
import copy
import json
import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from whole import DelayPosterior, FunctionFlow, delay_observations, performance, pipeline_delay_prior, project
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

    def test_congestion_alone_is_not_a_performance_reward(self):
        flow = FunctionFlow({'route': {'parts': [64, 64], 'grain': 0}})
        before = flow.centre
        for _ in range(400):
            request = flow.ask()
            congestion = (request['parts']['route'][0] - 100) ** 2 / 1000
            flow.observe({**request, 'scope': 'function', 'complete': True, 'workload': 'dnn',
                          'seconds': 1.0, 'congestion_seconds': congestion})
        self.assertEqual(flow.centre, before)

    def test_dominated_performance_cannot_win_by_emptying_queues(self):
        fast = {'seconds': 0.010, 'congestion_seconds': 0.010, 'finish_variance': 1,
                'useful_work_bytes': 4096,
                'roofline': {'useful_bytes': 8192, 'wire_bytes': 32768,
                            'window_seconds_bounds': [0.007, 0.008]}}
        slow = {**fast, 'seconds': 0.011, 'congestion_seconds': 0, 'finish_variance': 0,
                'roofline': {'useful_bytes': 8192, 'wire_bytes': 16384,
                            'window_seconds_bounds': [0.008, 0.009]}}
        for weight in (0, 0.1, 1, 10):
            self.assertLess(performance(fast, weight)['cost'], performance(slow, weight)['cost'])
        # Reproduce the audited former objective's reversed ordering.
        self.assertGreater(math.log(fast['seconds']) + fast['congestion_seconds'] / fast['seconds'],
                           math.log(slow['seconds']) + slow['congestion_seconds'] / slow['seconds'])

    def test_congestion_cannot_steer_toward_slower_same_work_functions(self):
        flow = FunctionFlow({'route': {'parts': [64, 64], 'grain': 0}})
        for _ in range(400):
            request = flow.ask()
            x = request['parts']['route'][0]
            flow.observe({**request, 'scope': 'function', 'complete': True, 'workload': 'dnn',
                          'seconds': 1 + (x - 94) ** 2 / 1000,
                          'congestion_seconds': (x - 20) ** 2 / 10,
                          'finish_variance': (x - 30) ** 2 / 100})
        self.assertAlmostEqual(flow.centre['route'][0], 94, delta=2)

    def test_explicit_transaction_preference_moves_routes_at_equal_graph_latency(self):
        flow = FunctionFlow({'route': {'parts': [64, 64], 'grain': 0}}, transaction_weight=1)
        for _ in range(400):
            request = flow.ask()
            window = 1 + (request['parts']['route'][0] - 100) ** 2 / 1000
            flow.observe({**request, 'scope': 'function', 'complete': True, 'workload': 'dnn',
                          'seconds': 10, 'useful_work_bytes': 64,
                          'roofline': {'useful_bytes': 128, 'window_seconds_bounds': [window, window]}})
        self.assertAlmostEqual(flow.centre['route'][0], 100, delta=2)

    def test_wire_amplification_is_never_useful_work(self):
        observation = {'seconds': 1, 'useful_work_bytes': 1024,
                       'roofline': {'useful_bytes': 2048, 'wire_bytes': 4096,
                                   'window_seconds_bounds': [0.5, 0.6]}}
        more_hops = copy.deepcopy(observation)
        more_hops['roofline']['wire_bytes'] *= 100
        for weight in (0, 1):
            self.assertEqual(performance(observation, weight), performance(more_hops, weight))

    def test_work_accounting_is_fixed_and_survives_serialization(self):
        flow = self.search()
        sample = {**flow.ask(), 'scope': 'function', 'complete': True, 'workload': 'dnn',
                  'seconds': 1, 'useful_work_bytes': 1024}
        flow.observe(sample)
        restored = FunctionFlow.of(json.loads(json.dumps(flow.state())))
        sample = {**flow.ask(), **{k: sample[k] for k in ('scope', 'complete', 'workload', 'seconds', 'useful_work_bytes')}}
        self.assertEqual(flow.observe(sample), restored.observe(sample))
        before = flow.state()
        with self.assertRaises(ValueError):
            flow.observe({**sample, **flow.ask(), 'useful_work_bytes': 2048})
        self.assertEqual(flow.state(), before)

    def test_old_cost_stencils_cannot_silently_mix_with_new_objective(self):
        state = self.search().state()
        del state['objective_version']
        with self.assertRaisesRegex(ValueError, 'obsolete congestion objective'):
            FunctionFlow.of(state)

    def test_window_preference_requires_an_invariant_delivery_contract(self):
        flow = FunctionFlow({'route': {'parts': [64, 64]}}, transaction_weight=1)
        sample = {**flow.ask(), 'scope': 'function', 'complete': True, 'workload': 'dnn', 'seconds': 1}
        before = flow.state()
        with self.assertRaisesRegex(ValueError, 'transaction-window observations'):
            flow.observe(sample)
        self.assertEqual(flow.state(), before)
        sample['roofline'] = {'useful_bytes': 128, 'window_seconds_bounds': [0.5, 0.6]}
        flow.observe(sample)
        before = flow.state()
        with self.assertRaisesRegex(ValueError, 'invariant useful work'):
            flow.observe({**sample, **flow.ask(), 'roofline': {'useful_bytes': 256,
                                                             'window_seconds_bounds': [0.5, 0.6]}})
        self.assertEqual(flow.state(), before)

    @staticmethod
    def delay_batch(value=30e-6, repeat=1, clock_variance=0):
        return {'labels': ['node:0', 'node:1', 'link:0', 'function'],
                'design': [[1, 1, 1, 1]] * repeat, 'seconds': [value] * repeat,
                'variance': [1e-14] * repeat, 'clock_design': [[1]] * repeat,
                'clock_variance': [clock_variance], 'groups': ['link:0'] * repeat}

    def test_delay_posterior_keeps_unidentifiable_node_effects_uncertain(self):
        import numpy as np
        batch = self.delay_batch()
        posterior = DelayPosterior(batch['labels'], process_stddev=0)
        self.assertEqual(posterior.mean, [0] * 4)
        prior = np.asarray(posterior.covariance)
        for _ in range(20):
            posterior.update(batch)
        covariance = np.asarray(posterior.covariance)
        row = np.ones(4)
        contrast = np.array([1, -1, 0, 0])
        self.assertAlmostEqual(sum(posterior.mean), 30e-6, delta=1e-8)
        self.assertLess(float(row @ covariance @ row), 1e-12)
        self.assertAlmostEqual(float(contrast @ covariance @ contrast), float(contrast @ prior @ contrast), delta=1e-12)
        self.assertLess(covariance[0, 1], 0)

    def test_configurable_node_prior_is_common_without_identifying_node_delays(self):
        import numpy as np
        batch = self.delay_batch()
        posterior = DelayPosterior(batch['labels'], prior_mean=80e-6, process_stddev=0)
        self.assertEqual(posterior.mean, [80e-6, 80e-6, 0, 0])
        contrast = np.array([1, -1, 0, 0])
        prior_variance = float(contrast @ np.asarray(posterior.covariance) @ contrast)
        posterior.update(batch)
        self.assertAlmostEqual(posterior.mean[0], posterior.mean[1], delta=1e-12)
        self.assertAlmostEqual(float(contrast @ np.asarray(posterior.covariance) @ contrast),
                               prior_variance, delta=1e-12)
        self.assertAlmostEqual(sum(posterior.mean), 30e-6, delta=1e-8)

    def test_shared_clock_uncertainty_is_not_averaged_away(self):
        import numpy as np
        batch = self.delay_batch(repeat=64, clock_variance=1e-8)
        posterior = DelayPosterior(batch['labels'], process_stddev=0)
        posterior.update(batch)
        variance = float(np.ones(4) @ np.asarray(posterior.covariance) @ np.ones(4))
        self.assertGreater(variance, 0.99e-8)

    def test_delay_state_replays_and_tracks_changed_congestion(self):
        posterior = DelayPosterior(self.delay_batch()['labels'])
        for _ in range(8):
            posterior.update(self.delay_batch())
        before = sum(posterior.mean)
        restored = DelayPosterior.of(json.loads(json.dumps(posterior.state())))
        for _ in range(8):
            batch = self.delay_batch(90e-6)
            self.assertEqual(posterior.update(batch), restored.update(batch))
        self.assertGreater(sum(posterior.mean), before + 50e-6)
        self.assertGreater(posterior.noise_variance['link:0'], 0)

    def test_first_frame_evidence_does_not_sum_entire_transfer_duration(self):
        cables = [{'nodes': [0, 1], 'bits_per_second': 80_000_000_000}]
        sent = {'cable': 0, 'peer': 1, 'first': 0, 'last': 2, 'direction': 0, 'bytes': 32768,
                'start_lo_ns': 999900, 'start_hi_ns': 1000100, 'end_lo_ns': 2000000, 'end_hi_ns': 3000000}
        landed = {**sent, 'peer': 0, 'direction': 1, 'start_lo_ns': 1009900, 'start_hi_ns': 1010100,
                  'end_lo_ns': 10000000, 'end_hi_ns': 20000000}
        ranks = [{'rank': 0, 'windows': [sent]}, {'rank': 1, 'windows': [landed]}]
        batch = delay_observations(cables, ranks, {'0': [0, 0], '1': [-500, 500]})
        self.assertAlmostEqual(batch['seconds'][0], 10e-6 - 16384 / 10_000_000_000)
        self.assertEqual(batch['clock_design'], [[-1, 1]])
        self.assertEqual(batch['design'], [[1, 1, 1, 0, 1]])
        sent['end_hi_ns'] *= 100
        landed['end_hi_ns'] *= 100
        self.assertEqual(batch, delay_observations(cables, ranks, {'0': [0, 0], '1': [-500, 500]}))

    @staticmethod
    def delay_trace(send_ranges=((0, 6),), receive_ranges=((0, 6),)):
        cables = [{'nodes': [0, 1], 'bits_per_second': 80_000_000_000}]
        ranks = []
        for rank, ranges in enumerate((send_ranges, receive_ranges)):
            windows = []
            for first, last in ranges:
                start = 1_000_000 + first * 100_000 + rank * 10_000
                end = 2_000_000 + last * 100_000 + rank * 10_000
                windows.append({'cable': 0, 'peer': 1 - rank, 'direction': rank,
                                'first': first, 'last': last, 'bytes': (last - first) * 16384,
                                'start_lo_ns': start - 100, 'start_hi_ns': start + 100,
                                'end_lo_ns': end - 100, 'end_hi_ns': end + 100})
            ranks.append({'rank': rank, 'windows': windows})
        return cables, ranks, {'0': [0, 0], '1': [-500, 500]}

    def test_refined_sends_use_only_the_unsplit_copys_observed_first_boundary(self):
        baseline = delay_observations(*self.delay_trace())
        batch = delay_observations(*self.delay_trace(send_ranges=((0, 2), (2, 5), (5, 6))))
        self.assertEqual({k: v for k, v in batch.items() if k != 'coverage'},
                         {k: v for k, v in baseline.items() if k != 'coverage'})
        self.assertEqual(batch['coverage'], {'send_entries': 3, 'receive_entries': 1,
                         'wire_bytes': 6 * 16384, 'observed_shared_first_boundaries': 1,
                         'send_entries_without_destination_first_observation': 2,
                         'receive_entries_without_source_first_observation': 0})

    def test_refined_copies_do_not_invent_interior_source_eligibility(self):
        batch = delay_observations(*self.delay_trace(receive_ranges=((0, 2), (2, 5), (5, 6))))
        self.assertEqual(batch['coverage']['observed_shared_first_boundaries'], 1)
        self.assertEqual(batch['coverage']['receive_entries_without_source_first_observation'], 2)
        self.assertAlmostEqual(batch['seconds'][0], 10e-6 - 16384 / 10_000_000_000)

    def test_crossing_ragged_boundaries_ignore_unobserved_first_arrivals(self):
        cables, ranks, offsets = self.delay_trace(send_ranges=((0, 2), (2, 5), (5, 6)),
                                                 receive_ranges=((0, 3), (3, 5), (5, 6)))
        batch = delay_observations(cables, ranks, offsets)
        self.assertEqual(batch['coverage']['observed_shared_first_boundaries'], 2)
        self.assertEqual(batch['coverage']['wire_bytes'], 6 * 16384)
        self.assertEqual(batch['seconds'], [10e-6 - 16384 / 10_000_000_000] * 2)
        for rank in ranks:
            hidden = rank['windows'][1]
            hidden['start_lo_ns'] += 7000
            hidden['start_hi_ns'] += 9000
            for window in rank['windows']:
                window['end_hi_ns'] *= 100
            rank['windows'].reverse()
        self.assertEqual(batch, delay_observations(cables, ranks, offsets))

    def test_delay_inference_checks_all_coverage_even_for_omitted_boundaries(self):
        for defect in ('duplicate_send', 'duplicate_receive', 'missing_send', 'missing_receive',
                       'wrong_page_size', 'unfinished_receive'):
            with self.subTest(defect=defect):
                cables, ranks, offsets = self.delay_trace(send_ranges=((0, 2), (2, 5), (5, 6)))
                if defect == 'duplicate_send':
                    ranks[0]['windows'].append(copy.deepcopy(ranks[0]['windows'][1]))
                elif defect == 'duplicate_receive':
                    ranks[1]['windows'].append(copy.deepcopy(ranks[1]['windows'][0]))
                elif defect == 'missing_send':
                    del ranks[0]['windows'][1]
                elif defect == 'missing_receive':
                    ranks[1]['windows'][0].update(last=5, bytes=5 * 16384)
                elif defect == 'wrong_page_size':
                    ranks[0]['windows'][1]['bytes'] *= 2
                else:
                    ranks[1]['windows'][0]['end_hi_ns'] = 0
                with self.assertRaises(ValueError):
                    delay_observations(cables, ranks, offsets)

    def test_refined_trace_updates_posterior_once_per_complete_function(self):
        batch = delay_observations(*self.delay_trace(send_ranges=((0, 2), (2, 5), (5, 6))))
        flow = FunctionFlow({'route': {'parts': [64, 64]}})
        for iteration in range(2):
            sample = {**flow.ask(), 'scope': 'function', 'complete': False, 'workload': 'refined-dnn',
                      'seconds': 0.001, 'delay_observations': batch}
            before = flow.state()
            with self.assertRaises(ValueError):
                flow.observe(sample)
            self.assertEqual(flow.state(), before)
            flow.observe({**sample, 'complete': True})
            self.assertEqual(flow.delay_posterior['updates'], iteration + 1)
            self.assertEqual(flow.delay_posterior['last']['observations'], 1)
            self.assertEqual(flow.delay_posterior['last']['coverage'], batch['coverage'])

    def test_serial_stages_and_delay_posterior_survive_complete_function_updates(self):
        model = {'capacities': [1, 1], 'paths': {'route': {'bytes': 1, 'arcs': [[0], [1]]}},
                 'serial_stages': True, 'directed_nodes': [[0, 1], [1, 0]]}
        flow = FunctionFlow({'route': {'parts': [64, 64]}}, routing_model=model)
        batch = {'labels': ['node:0', 'node:1', 'link:0', 'link:1', 'function'],
                 'design': [[1, 1, 1, 0, 1], [1, 1, 0, 1, 1]], 'seconds': [20e-6, 40e-6],
                 'variance': [1e-12, 1e-12], 'clock_design': [[-1, 1], [1, -1]],
                 'clock_variance': [0, 1e-12], 'groups': ['link:0', 'link:1']}
        stages = [{'arcs': {str(e): {'start_ns': [0, 0], 'finish_ns': [t * 1e9] * 2}
                            for e, t in enumerate(times)}} for times in ([10, 1], [1, 10])]
        for _ in range(8):
            request = flow.ask()
            sample = {**request, 'scope': 'function', 'complete': True, 'workload': 'two-serial-stages',
                      'seconds': 20, 'finishes': {'stages': stages}, 'delay_observations': batch}
            restored = FunctionFlow.of(json.loads(json.dumps(flow.state())))
            self.assertEqual(flow.observe(sample), restored.observe(sample))
        self.assertEqual(flow.routing_fit['serial_stages'], 2)
        self.assertEqual(flow.routing_fit['measured_finish_seconds'], [10, 1, 1, 10])
        self.assertEqual(flow.delay_posterior['updates'], 8)
        self.assertTrue(all(0 < x < 1 for x in flow.routing_fit['link_model_reliability']))

    def test_posterior_means_guide_pipeline_routes_without_bulk_hop_penalty(self):
        import numpy as np
        model = {'capacities': [10, 10, 10], 'paths': {'route': {'bytes': 100, 'arcs': [[0, 2], [1, 2]]}},
                 'directed_nodes': [[0, 1], [0, 1], [1, 2]], 'page_bytes': 1}
        labels = ['node:0', 'node:1', 'node:2', 'link:0', 'link:1', 'link:2', 'function']
        posterior = DelayPosterior(labels).state()
        posterior['mean'][3:6] = [1, 3, 2]
        x = np.array([0.25, 0.75])
        arrival, derivative = pipeline_delay_prior(model, posterior, x)
        self.assertAlmostEqual(arrival[2], 4.6)
        numerical = (pipeline_delay_prior(model, posterior, x + [1e-5, -1e-5])[0]
                     - pipeline_delay_prior(model, posterior, x - [1e-5, -1e-5])[0]) / 2e-5
        np.testing.assert_allclose(derivative @ [1, -1], numerical, atol=1e-8)
        bigger = copy.deepcopy(model)
        bigger['paths']['route']['bytes'] *= 10000
        np.testing.assert_allclose(pipeline_delay_prior(bigger, posterior, x)[0], arrival)
        # The same observed binding shared link, with unobserved route
        # sensitivities: opposite learned first-frame priors must produce
        # opposite continuous proposals, not merely different reports.
        centres = []
        for first in ([1, 3, 2], [3, 1, 2]):
            flow = FunctionFlow({'route': {'parts': [64, 64]}}, routing_model=model)
            flow.delay_model = copy.deepcopy(posterior)
            flow.delay_model['mean'][3:6] = first
            stage = {'arcs': {str(e): {'start_ns': [0, 0], 'finish_ns': [t * 1e9] * 2}
                              for e, t in enumerate([1, 1, 20])}}
            flow.routing_observations = [{'parts': copy.deepcopy(flow.centre), 'stages': [stage]} for _ in range(4)]
            flow._routing_step()
            centres.append(flow.centre['route'][0])
        self.assertGreater(centres[0], 64)
        self.assertLess(centres[1], 64)

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

    def test_finish_variance_alone_is_not_a_performance_reward(self):
        flow = FunctionFlow({'route': {'parts': [64, 64], 'grain': 0}})
        before = flow.centre
        for _ in range(400):
            request = flow.ask()
            residual = (request['parts']['route'][0] - 90) ** 2 / 1000
            flow.observe({**request, 'scope': 'function', 'complete': True, 'workload': 'dnn',
                          'seconds': 1.0, 'finish_variance': residual})
        self.assertEqual(flow.centre, before)

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
