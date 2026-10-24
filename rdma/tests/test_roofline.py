"""design/whole-function.md#roofline"""
import pathlib
import sys
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from roofline import causal_offsets, roofline, union_ns, finishes


def window(cable, start, end, count):
    return {'cable': cable, 'bytes': count, 'direction': 0, 'first': 0, 'last': count,
            'start_lo_ns': int(start * 1e9), 'start_hi_ns': int(start * 1e9),
            'end_lo_ns': int(end * 1e9), 'end_hi_ns': int(end * 1e9)}


def landings(cables, ranks):
    for rank in ranks:
        for w in list(rank['windows']):
            if w['direction']:
                continue
            w['peer'] = next(r for r in cables[w['cable']]['nodes'] if r != rank['rank'])
            target = next(r for r in ranks if r['rank'] == w['peer'])
            target['windows'].append({**w, 'peer': rank['rank'], 'direction': 1})
    return ranks


class RooflineTests(unittest.TestCase):
    def test_relay_wait_does_not_make_compute_give_work_to_its_slow_producer(self):
        cables = [{'nodes': [0, 1], 'bits_per_second': 80}]
        ranks = [{'rank': r, 'windows': []} for r in range(2)]
        for rank in ranks:
            own = window(0, 1, 3, 20)
            relay = window(0, 2, 10, 20)
            relay.update(first=20, last=40, relay=1)
            rank['windows'] = [own, relay]
        r = finishes(cables, landings(cables, ranks), 1, {'0': [0, 0], '1': [0, 0]})
        self.assertEqual(r['delivery_tail_seconds'], [2, 2])
        self.assertEqual(r['stages'][0]['arcs']['0']['window_seconds_bounds'], [9, 9])

    def test_completion_spread_uses_stage_and_remote_landing(self):
        cables = [{'nodes': [0, 1], 'bits_per_second': 80}]
        ranks = [{'rank': r, 'windows': []} for r in range(2)]
        for r in range(2):
            for stage in range(2):
                w = window(0, 1 + stage * 10, 3 + stage * 10, 20)
                w.update(first=stage * 20, last=(stage + 1) * 20)
                ranks[r]['windows'].append(w)
        landings(cables, ranks)
        for w in ranks[1]['windows']:
            if w['direction']:
                w['end_lo_ns'] += 1_000_000_000
                w['end_hi_ns'] += 1_000_000_000
        r = finishes(cables, ranks, 2, {'0': [0, 0], '1': [0, 0]})
        self.assertEqual([s['spread_seconds_bounds'] for s in r['stages']], [[1, 1], [1, 1]])
        self.assertAlmostEqual(r['finish_variance'], 0.25 / 9)
        self.assertEqual(r['delivery_tail_seconds'], [6, 4])

    def test_idle_links_still_count_and_overlaps_are_not_summed(self):
        cables = [{'nodes': [0, 1], 'bits_per_second': b} for b in (80, 160)]
        ranks = [{'rank': 0, 'windows': [window(0, 1, 3, 20)]}, {'rank': 1, 'windows': []}]
        r = roofline(cables, landings(cables, ranks), 20, {'0': [0, 0], '1': [0, 0]})
        self.assertEqual(r['capacity_bytes_per_second'], 60)
        self.assertEqual(r['window_seconds_bounds'], [2, 2])
        self.assertEqual(r['wire_bandwidth_bytes_per_second_bounds'], [10, 10])
        self.assertAlmostEqual(r['capacity_utilization_bounds'][0], 1 / 6)
        self.assertEqual(r['unused_capacity_bytes_bounds'], [100, 100])
        self.assertEqual(union_ns([(1, 3), (2, 4), (7, 8)]), 4)

    def test_relay_bytes_and_clock_uncertainty(self):
        cables = [{'nodes': [0, 1], 'bits_per_second': 80}, {'nodes': [1, 2], 'bits_per_second': 80}]
        ranks = [{'rank': 0, 'windows': [window(0, 1, 3, 20)]},
                 {'rank': 1, 'windows': [window(1, 2, 4, 20)]}, {'rank': 2, 'windows': []}]
        r = roofline(cables, landings(cables, ranks), 20, {'0': [0, 0], '1': [-100_000_000, 100_000_000], '2': [0, 0]})
        self.assertEqual(r['hop_and_padding_amplification'], 2)
        self.assertAlmostEqual(r['window_seconds_bounds'][0], 3)
        self.assertAlmostEqual(r['window_seconds_bounds'][1], 3)
        self.assertEqual(r['all_links_ideal_seconds'], 1)
        self.assertEqual(r['current_routing_bottleneck_seconds'], 2)

    def test_unfinished_transaction_is_not_a_bandwidth_result(self):
        w = window(0, 1, 3, 20); w['end_hi_ns'] = 0
        with self.assertRaises(ValueError):
            roofline([{'nodes': [0, 1], 'bits_per_second': 80}], [{'rank': 0, 'windows': [w]}], 20, {'0': [0, 0]})

    def test_source_completion_does_not_substitute_for_destination_landing(self):
        cables = [{'nodes': [0, 1], 'bits_per_second': 80}]
        ranks = landings(cables, [{'rank': 0, 'windows': [window(0, 1, 2, 20)]}, {'rank': 1, 'windows': []}])
        ranks[1]['windows'][0]['end_lo_ns'] = 3_000_000_000
        ranks[1]['windows'][0]['end_hi_ns'] = 3_000_000_000
        r = roofline(cables, ranks, 20, {'0': [0, 0], '1': [-100_000_000, 100_000_000]})
        self.assertAlmostEqual(r['window_seconds_bounds'][0], 2.0)
        self.assertAlmostEqual(r['window_seconds_bounds'][1], 2.1)
        self.assertAlmostEqual(r['nodes'][1]['ingress']['wire_bandwidth_bytes_per_second_bounds'][0], 20 / 2.1)

    def test_causal_clock_bounds_preserve_asymmetric_delay(self):
        cables = [{'nodes': [0, 1], 'bits_per_second': 80}]
        ranks = landings(cables, [{'rank': 0, 'windows': [window(0, 1, 2, 20)]},
                                 {'rank': 1, 'windows': [window(0, 3, 4, 20)]}])
        # Root minus rank 1 is 5 seconds. Forward delay is 10 ns, reverse
        # delay 30 ns: no midpoint/symmetric-delay assumption is justified.
        for rank in ranks:
            for w in rank['windows']:
                for field in ('start_lo_ns', 'start_hi_ns', 'end_lo_ns', 'end_hi_ns'):
                    w[field] += (-5_000_000_000 if rank['rank'] else 0)
                    if w['direction']:
                        w[field] += 10 if rank['rank'] else 30
        raw = {'0': [0, 0], '1': [4_000_000_000, 6_000_000_000]}
        self.assertEqual(causal_offsets(ranks, raw)['1'], [4_999_999_990, 5_000_000_030])
        self.assertEqual(raw['1'], [4_000_000_000, 6_000_000_000])

    def test_inconsistent_clock_evidence_is_not_a_speedup(self):
        cables = [{'nodes': [0, 1], 'bits_per_second': 80}]
        ranks = landings(cables, [{'rank': 0, 'windows': [window(0, 1, 2, 20)]},
                                 {'rank': 1, 'windows': []}])
        ranks[1]['windows'][0]['start_hi_ns'] = 500_000_000
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            causal_offsets(ranks, {'0': [0, 0], '1': [0, 0]})
