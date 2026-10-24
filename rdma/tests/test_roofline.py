"""design/whole-function.md#roofline"""
import pathlib
import sys
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from roofline import causal_offsets, roofline, union_ns, finishes, matched_transfers


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
    def refined(self, sends=((0, 10, 1, 3), (10, 20, 4, 8)), receives=((0, 20, 2, 8),)):
        ranks = [{'rank': 0, 'windows': []}, {'rank': 1, 'windows': []}]
        for first, last, start, end in sends:
            w = window(0, start, end, last - first)
            w.update(first=first, last=last, peer=1)
            ranks[0]['windows'].append(w)
        for first, last, start, end in receives:
            w = window(0, start, end, last - first)
            w.update(first=first, last=last, peer=0, direction=1)
            ranks[1]['windows'].append(w)
        return ranks

    def test_fragmented_send_preserves_interior_timestamp_uncertainty(self):
        ranks = self.refined()
        first, last = matched_transfers(ranks)
        self.assertEqual(first['arrival_ns'], [2_000_000_000] * 2)
        self.assertEqual(first['completion_ns'], [2_000_000_000, 8_000_000_000])
        self.assertTrue(first['arrival_exact'])
        self.assertFalse(first['completion_exact'])
        self.assertEqual(last['arrival_ns'], [2_000_000_000, 8_000_000_000])
        self.assertFalse(last['arrival_exact'])
        self.assertEqual(last['completion_ns'], [8_000_000_000] * 2)
        self.assertTrue(last['completion_exact'])
        result = roofline([{'nodes': [0, 1], 'bits_per_second': 80}], ranks, 20, {'0': [0, 0], '1': [0, 0]})
        self.assertEqual(result['wire_bytes'], 20)
        self.assertEqual(result['directed_wire_bytes'], {'0': 20})
        self.assertEqual(result['nodes'][1]['ingress']['wire_bytes'], 20)
        self.assertEqual(result['window_seconds_bounds'], [5, 7])
        self.assertEqual(result['useful_bandwidth_bytes_per_second_bounds'], [20 / 7, 20 / 5])

    def test_unsplit_send_split_receives_keep_observed_outer_boundaries(self):
        ranks = self.refined(sends=((0, 20, 1, 8),), receives=((0, 10, 2, 3), (10, 20, 4, 8)))
        match, = matched_transfers(ranks)
        self.assertEqual(len(match['receives']), 2)
        self.assertTrue(match['arrival_exact'] and match['completion_exact'])
        self.assertEqual(match['arrival_ns'], [2_000_000_000] * 2)
        self.assertEqual(match['completion_ns'], [8_000_000_000] * 2)
        result = roofline([{'nodes': [0, 1], 'bits_per_second': 80}], ranks, 20, {'0': [0, 0], '1': [0, 0]})
        self.assertEqual(result['wire_bytes'], 20)
        self.assertEqual(result['window_seconds_bounds'], [7, 7])

    def test_crossing_entry_boundaries_match_coverage_without_duplication(self):
        ranks = self.refined(sends=((0, 12, 1, 6), (12, 20, 5, 8)),
                             receives=((0, 8, 2, 3), (8, 20, 4, 8)))
        matches = matched_transfers(ranks)
        self.assertEqual([len(m['receives']) for m in matches], [2, 1])
        self.assertEqual(sum(m['send']['bytes'] for m in matches), 20)
        self.assertEqual(matches[0]['completion_ns'], [4_000_000_000, 8_000_000_000])
        self.assertEqual(matches[1]['arrival_ns'], [4_000_000_000, 8_000_000_000])

    def test_fragment_clock_constraints_do_not_reuse_enclosing_first_arrival(self):
        ranks = self.refined()
        offset = 5_000_000_000
        for w in ranks[1]['windows']:
            for field in ('start_lo_ns', 'start_hi_ns', 'end_lo_ns', 'end_hi_ns'):
                w[field] -= offset
        clocks = {'0': [0, 0], '1': [offset, offset]}
        # SEND 2 is eligible at t=4, after the enclosing receive's first
        # arrival at t=2. Reusing that arrival would invent negative delay.
        self.assertEqual(causal_offsets(ranks, clocks), clocks)
        self.assertEqual(roofline([{'nodes': [0, 1], 'bits_per_second': 80}], ranks, 20, clocks)
                         ['window_seconds_bounds'], [5, 7])

    def test_refined_stage_finishes_use_remote_final_boundary(self):
        cables = [{'nodes': [0, 1], 'bits_per_second': 80}]
        ranks = self.refined()
        ranks[1]['windows'].extend({**w, 'peer': 0} for w in ranks[0]['windows'])
        ranks[0]['windows'].append({**ranks[1]['windows'][0], 'peer': 1})
        result = finishes(cables, ranks, 1, {'0': [0, 0], '1': [0, 0]})
        for arc in result['stages'][0]['arcs'].values():
            self.assertEqual(arc['bytes'], 20)
            self.assertEqual(arc['finish_ns'], [8_000_000_000] * 2)
            self.assertEqual(arc['window_seconds_bounds'], [5, 7])

    def test_refined_stage_entry_count_alone_does_not_identify_stages(self):
        cables = [{'nodes': [0, 1], 'bits_per_second': 80}]
        ranks = [{'rank': rank, 'windows': []} for rank in range(2)]
        for rank in ranks:
            first = 0
            for size in (5, 5, 2, 8):
                w = window(0, 1 + first, 2 + first, size)
                w.update(first=first, last=first + size)
                first += size
                rank['windows'].append(w)
        landings(cables, ranks)
        with self.assertRaisesRegex(ValueError, 'layout does not repeat'):
            finishes(cables, ranks, 2, {'0': [0, 0], '1': [0, 0]})

    def test_repeated_refined_sends_and_unsplit_receives_identify_stage_layout(self):
        cables = [{'nodes': [0, 1], 'bits_per_second': 80}]
        ranks = self.refined(sends=((0, 10, 1, 3), (10, 20, 4, 8), (20, 30, 11, 13), (30, 40, 14, 18)),
                             receives=((0, 20, 2, 8), (20, 40, 12, 18)))
        incoming = [dict(w) for w in ranks[1]['windows']]
        ranks[1]['windows'].extend({**w, 'peer': 0} for w in ranks[0]['windows'])
        ranks[0]['windows'].extend({**w, 'peer': 1} for w in incoming)
        report = finishes(cables, ranks, 2, {'0': [0, 0], '1': [0, 0]})
        for layer, stage in enumerate(report['stages']):
            self.assertEqual(stage['arcs']['0']['bytes'], 20)
            self.assertEqual(stage['arcs']['0']['finish_ns'], [int((8 + 10 * layer) * 1e9)] * 2)

    def test_actual_missing_duplicate_or_inconsistent_coverage_fails(self):
        missing = self.refined(sends=((0, 9, 1, 3), (10, 20, 4, 8)))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            matched_transfers(missing)
        missing = self.refined(receives=((0, 19, 2, 8),))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            matched_transfers(missing)
        for rank in (0, 1):
            duplicated = self.refined()
            duplicated[rank]['windows'].append(dict(duplicated[rank]['windows'][0]))
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                matched_transfers(duplicated)
        wrong_bytes = self.refined()
        wrong_bytes[1]['windows'][0]['bytes'] = 40
        with self.assertRaisesRegex(ValueError, 'byte counts'):
            matched_transfers(wrong_bytes)
        wrong_wire = self.refined()
        for w in wrong_wire[0]['windows']:
            w['wire'] = 100
        wrong_wire[1]['windows'][0]['wire'] = 200
        with self.assertRaisesRegex(ValueError, 'wire identities'):
            matched_transfers(wrong_wire)

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
