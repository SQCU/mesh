import copy
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trace_observation import SpanTrace, tick_summary


def observations(ticks=3):
    ranks = [dict(rank=0, windows=[]), dict(rank=1, windows=[])]
    for tick in range(ticks):
        for source in (0, 1):
            for cable in (0, 1):
                first = tick*2
                when = 1000+tick*1000+source*250+cable*10
                # Two SEND entries, one receive entry; same peer on two wires.
                for direction, parts in ((0, ((first, [1]), (first+1, [2]))), (1, ((first, [1, 2]),))):
                    rank = source if direction == 0 else 1-source
                    for start, lengths in parts:
                        ranks[rank]['windows'].append(dict(cable=cable, wire=111+cable, peer=1-rank,
                            direction=direction, relay=0, first=start, last=start+len(lengths), bytes=sum(lengths),
                            unit_kind='byte_extent', unit_bytes=lengths,
                            start_lo_ns=when+direction*20, start_hi_ns=when+direction*20+10,
                            end_lo_ns=when+100+direction*100, end_hi_ns=when+110+direction*100))
    return ranks


class PhysicalTickTests(unittest.TestCase):
    def test_parallel_cables_fragmented_sends_and_exact_bytes(self):
        result = tick_summary(observations(), 0, 3)
        self.assertTrue(result['tick_layout_supported'])
        self.assertEqual(result['ticks'], 3)
        self.assertEqual(result['entries'], 18)
        self.assertEqual(result['sent_payload_bytes'], 18)
        self.assertEqual(result['landed_payload_bytes'], 18)
        self.assertEqual(len(result['physical_landed_after_publish_us']), 2)
        self.assertEqual(result['tick_us']['median'], 1)
        self.assertEqual(result['publish_to_landed_us']['median'], .46)
        self.assertEqual(result['publish_to_landed_us_bounds']['lower']['median'], .45)
        self.assertEqual(result['publish_to_landed_us_bounds']['upper']['median'], .47)

    def test_converse_fragmentation_is_independent_per_direction(self):
        ranks = observations()
        for rank in ranks:
            for w in rank['windows']:
                w['direction'] = 1-w['direction']
        self.assertTrue(tick_summary(ranks, 0, 3)['tick_layout_supported'])

    def test_coalesced_entry_does_not_invent_tick_boundaries(self):
        ranks = observations()
        for rank in ranks:
            selected = [w for w in rank['windows'] if w['cable'] == 1]
            rank['windows'] = [w for w in rank['windows'] if w['cable'] != 1]
            for direction in (0, 1):
                part = sorted((w for w in selected if w['direction'] == direction), key=lambda w: w['first'])
                combined = dict(part[0], last=part[-1]['last'], bytes=sum(w['bytes'] for w in part),
                                unit_bytes=sum((w['unit_bytes'] for w in part), []),
                                end_lo_ns=part[-1]['end_lo_ns'], end_hi_ns=part[-1]['end_hi_ns'])
                rank['windows'].append(combined)
        result = tick_summary(ranks, 0, 3)
        self.assertFalse(result['tick_layout_supported'])
        self.assertNotIn('ticks', result)
        self.assertIn('divide', result['tick_layout_reason'])
        self.assertTrue(result['windows'])

    def test_cross_rank_coverage_and_length_mismatch_rejected(self):
        for mutate in ('missing', 'duplicate', 'length', 'wire'):
            ranks = observations()
            window = ranks[0]['windows'][0]
            if mutate == 'missing':
                ranks[0]['windows'].pop(0)
            elif mutate == 'duplicate':
                ranks[0]['windows'].append(copy.deepcopy(window))
            elif mutate == 'length':
                window['bytes'] = 5; window['unit_bytes'] = [5]
            else:
                window['wire'] += 99
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                tick_summary(ranks, 0, 3)

    def test_entry_count_alone_is_not_a_repetition_proof(self):
        ranks = observations()
        # Preserve equal counts and matched lengths but change one tick's shape.
        for rank in ranks:
            for w in rank['windows']:
                if w['first'] >= 2 and w['last'] <= 4:
                    w['unit_bytes'] = [n+3 for n in w['unit_bytes']]
                    w['bytes'] = sum(w['unit_bytes'])
        result = tick_summary(ranks, 0, 3)
        self.assertFalse(result['tick_layout_supported'])
        self.assertIn('shape changes', result['tick_layout_reason'])

    def test_relay_start_is_not_local_producer_publication(self):
        ranks = observations()
        for w in ranks[0]['windows']:
            if w['direction'] == 0 and w['cable'] == 1:
                w['relay'] = 1
                w['start_lo_ns'] -= 200
                w['start_hi_ns'] -= 200
        self.assertEqual(tick_summary(ranks, 0, 3)['publish_to_landed_us']['median'], .46)

    def test_invalid_tick_count(self):
        with self.assertRaises(ValueError):
            tick_summary(observations(), 0, 0)


class TraceCaptureTests(unittest.TestCase):
    def test_lengths_freeze_before_stop_and_survive_free(self):
        import cffi
        ffi = cffi.FFI()
        ffi.cdef('''typedef void *ncclMeshTrace_t;
            struct ncclMeshWindow { unsigned long long start_lo_ns,start_hi_ns,end_lo_ns,end_hi_ns,bytes,wire,first,last,
                peak_queue_bytes,cable_peak_queue_bytes; double queue_byte_seconds; unsigned cable,peer,direction,relay; };
            struct ncclMeshTraceStats { unsigned long long polls,max_gap_ns; unsigned incomplete; };''')
        calls = []
        def start(comm, room, handle):
            calls.append(('start', room)); handle[0] = ffi.cast('void *', 1); return 0
        def lengths(handle, out, room, count):
            calls.append(('lengths', out != ffi.NULL)); count[0] = 3
            if out != ffi.NULL:
                out[0], out[1], out[2] = 1, 4097, 19
            return 0
        def stop(handle, windows, room, count, stats):
            calls.append(('stop', room)); count[0] = 1
            windows[0].first, windows[0].last, windows[0].bytes = 17, 20, 4117
            stats.polls = 42
            return 0
        mesh = types.SimpleNamespace(ffi=ffi, check=lambda status: self.assertEqual(status, 0),
            lib=types.SimpleNamespace(ncclMeshTraceStart=start, ncclMeshTraceSpanLengths=lengths, ncclMeshTraceStop=stop))
        with patch.dict(sys.modules, mesh=mesh):
            trace = SpanTrace(object()); trace.start(); report = trace.finish()
        self.assertEqual(calls, [('start', 4096), ('lengths', False), ('lengths', True), ('stop', 4096)])
        self.assertEqual(report['windows'][0]['unit_bytes'], [1, 4097, 19])
        self.assertEqual(report['windows'][0]['bytes'], 4117)


if __name__ == '__main__':
    unittest.main()
