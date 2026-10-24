"""Counterexamples to ACK-free reuse without a causal ownership proof."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lifetimes import Event as E, Lease as L, Read, Transfer, compile_lifetimes, prove_recurrent_gather


class LifetimeTests(unittest.TestCase):
    def sink(self, source_done=False):
        events = [E('source', 1, 'publish'), E('sink', 2, 'land'),
                  E('sink_used', 2, 'read_done', ('sink',)),
                  E('unrelated', 1, 'land'), E('later', 1, 'publish', ('unrelated',))]
        if source_done:
            events.append(E('dma_done', 1, 'source_done', channel='out:session:2', prefix=16384))
        return compile_lifetimes(events, [L('source_pages', 1, 1, 'source'), L('sink_pages', 2, 1, 'sink')],
                                 [Read('sink_pages', 'sink_used')],
                                 [Transfer('source_pages', 'source', 'sink', 'dma_done' if source_done else None)])

    def test_asymmetric_unrelated_landing_cannot_release_source(self):
        plan = self.sink()
        tracker = plan.tracker(1)
        for event in ('source', 'unrelated', 'later'):
            self.assertEqual(tracker.observe(event), ())
        self.assertEqual(tracker.retained, ('source_pages',))
        self.assertEqual(plan.rules(1)[0].unproved, ('transfer:sink',))

    def test_local_dma_retirement_is_not_destination_consumer_retirement(self):
        plan = self.sink(source_done=True)
        data, bridge, destination = plan.tracker(1), plan.tracker(1, bridge=True), plan.tracker(2)
        self.assertEqual(data.observe('dma_done'), ())
        self.assertEqual(bridge.observe('dma_done'), ('source_pages',))
        self.assertEqual(bridge.observe('dma_done'), ())
        self.assertEqual(destination.observe('sink'), ())
        self.assertEqual(destination.observe('sink_used'), ('sink_pages',))

    def test_returning_data_proves_remote_consumption(self):
        events = [E('source', 0, 'publish'), E('sink', 1, 'land'),
                  E('read', 1, 'read_done', ('sink',)), E('result', 1, 'publish', ('read',)),
                  E('returned', 0, 'land')]
        leases = [L('source_pages', 0, 1, 'source'), L('sink_pages', 1, 1, 'sink'), L('result_pages', 1, 1, 'result')]
        plan = compile_lifetimes(events, leases, [Read('sink_pages', 'read')],
                                 [Transfer('source_pages', 'source', 'sink'), Transfer('result_pages', 'result', 'returned')])
        self.assertTrue(plan.precedes('read', 'returned'))
        self.assertEqual(plan.tracker(0).observe('returned'), ('source_pages',))
        # Remote evidence is useful only where it naturally arrives.
        self.assertEqual(plan.tracker(1).observe('returned'), ())

    def test_all_physical_fragments_required_even_to_same_peer(self):
        events = [E('p', 0, 'publish'), E('a', 1, 'land'), E('b', 1, 'land'),
                  E('done_a', 0, 'source_done', channel='cable:a', prefix=16384),
                  E('done_b', 0, 'source_done', channel='cable:b', prefix=16384)]
        plan = compile_lifetimes(events, [L('x', 0, 2, 'p')], transfers=[
            Transfer('x', 'p', 'a', 'done_a'), Transfer('x', 'p', 'b', 'done_b')])
        tracker = plan.tracker(0, bridge=True)
        self.assertEqual(tracker.observe('done_a'), ())
        self.assertEqual(tracker.observe('done_b'), ('x',))

    def test_contiguous_prefix_proves_prior_fragments_only_in_same_generation(self):
        events = [E('p', 0, 'publish'), E('a', 1, 'land'), E('b', 1, 'land'),
                  E('early', 0, 'source_done', channel='cable:epoch1', prefix=16384),
                  E('final', 0, 'source_done', channel='cable:epoch1', prefix=32768),
                  E('other', 0, 'source_done', channel='cable:epoch2', prefix=1000000)]
        plan = compile_lifetimes(events, [L('x', 0, 2, 'p')], transfers=[
            Transfer('x', 'p', 'a', 'early'), Transfer('x', 'p', 'b', 'final')])
        tracker = plan.tracker(0, bridge=True)
        self.assertEqual(tracker.observe('early'), ())
        self.assertEqual(tracker.observe('other'), ())
        self.assertEqual(tracker.observe('final'), ('x',))
        self.assertEqual(plan.tracker(0, bridge=True).observe('final'), ('x',))

    def test_local_consumers_remain_after_dma_done(self):
        events = [E('p', 0, 'publish'), E('l', 1, 'land'), E('d', 0, 'source_done'),
                  E('r', 0, 'read_done', ('p',))]
        plan = compile_lifetimes(events, [L('x', 0, 1, 'p')], [Read('x', 'r')], [Transfer('x', 'p', 'l', 'd')])
        tracker = plan.tracker(0, bridge=True)
        self.assertEqual(tracker.observe('d'), ())
        self.assertEqual(tracker.observe('r'), ('x',))

    def test_reuse_requires_old_reads_before_new_first_write(self):
        events = [E('p', 0, 'publish'), E('read', 0, 'read_done', ('p',)),
                  E('write', 0, 'write_begin', ('read',)), E('next', 0, 'publish', ('write',))]
        leases = [L('old', 0, 4, 'p'), L('new', 0, 2, 'next', 'write')]
        plan = compile_lifetimes(events, leases, [Read('old', 'read')])
        self.assertEqual(plan.allocations['old'].page, plan.allocations['new'].page)

    def test_output_publication_after_read_does_not_prove_inplace_safety(self):
        events = [E('p', 0, 'publish'), E('read', 0, 'read_done', ('p',)),
                  E('write', 0, 'write_begin', ('p',)), E('next', 0, 'publish', ('read', 'write'))]
        leases = [L('old', 0, 4, 'p'), L('new', 0, 4, 'next', 'write')]
        plan = compile_lifetimes(events, leases, [Read('old', 'read')])
        self.assertNotEqual(plan.allocations['old'].page, plan.allocations['new'].page)

    def test_unknown_first_write_conservatively_prevents_alias(self):
        events = [E('p', 0, 'publish'), E('next', 0, 'publish', ('p',))]
        plan = compile_lifetimes(events, [L('old', 0, 1, 'p'), L('new', 0, 1, 'next')])
        self.assertNotEqual(plan.allocations['old'].page, plan.allocations['new'].page)

    def test_local_read_on_relay_outlives_its_forwarding(self):
        events = [E('source', 0, 'publish'), E('relay', 1, 'land'),
                  E('forward', 1, 'publish', ('relay',)), E('dest', 2, 'land'),
                  E('dma_done', 1, 'source_done'), E('read', 1, 'read_done', ('relay',))]
        plan = compile_lifetimes(events, [L('s', 0, 1, 'source'), L('r', 1, 1, 'relay')], [Read('r', 'read')],
                                 [Transfer('s', 'source', 'relay'), Transfer('r', 'forward', 'dest', 'dma_done')])
        tracker = plan.tracker(1, bridge=True)
        self.assertEqual(tracker.observe('dma_done'), ())
        self.assertEqual(tracker.observe('read'), ('r',))

    def test_cannot_smuggle_driver_wait_into_data_graph(self):
        with self.assertRaisesRegex(ValueError, 'cannot wait on bridge'):
            compile_lifetimes([E('done', 0, 'source_done'), E('next', 0, 'publish', ('done',))], [])

    def test_remote_dependencies_need_actual_transfer(self):
        with self.assertRaisesRegex(ValueError, 'only a Transfer'):
            compile_lifetimes([E('remote', 1, 'read_done'), E('local', 0, 'publish', ('remote',))], [])

    def test_cycles_rejected_during_compilation(self):
        with self.assertRaisesRegex(ValueError, 'cycle'):
            compile_lifetimes([E('a', 0, 'publish', ('b',)), E('b', 0, 'publish', ('a',))], [])

    def test_retirement_does_not_require_allgather_when_local_read_done(self):
        events = [E('p', 2, 'land'), E('used', 2, 'read_done', ('p',))]
        plan = compile_lifetimes(events, [L('x', 2, 16, 'p')], [Read('x', 'used')])
        self.assertEqual(plan.tracker(2).observe('used'), ('x',))

    def allgather(self, *, relayed=False, recurrent=True):
        events, leases, reads, transfers = [], [], [], []
        # Three ranks and three crossings expose reuse of generation zero's
        # source/destination slots by generation two. Every rank consumes all
        # other ranks' pieces; publications carry actual useful payload.
        for i in range(3):
            for rank in range(3):
                start, pub, consume = f'w{rank}:{i}', f'p{rank}:{i}', f'c{rank}:{i}'
                previous = (f'c{rank}:{i - 1}',) if i and recurrent else ()
                events.extend([E(start, rank, 'write_begin', previous), E(pub, rank, 'publish', (start,))])
                leases.append(L(f'source{rank}:{i}', rank, 1, pub, start))
                reads.append(Read(f'source{rank}:{i}', consume))
                incoming = [f'land{source}:{rank}:{i}' for source in range(3) if source != rank]
                events.append(E(consume, rank, 'read_done', tuple([pub] + incoming)))
            for source in range(3):
                for dest in range(3):
                    if source == dest:
                        continue
                    begin, land = f'begin{source}:{dest}:{i}', f'land{source}:{dest}:{i}'
                    events.extend([E(begin, dest, 'write_begin'), E(land, dest, 'land')])
                    leases.append(L(f'dest{source}:{dest}:{i}', dest, 1, land, begin))
                    reads.append(Read(f'dest{source}:{dest}:{i}', f'c{dest}:{i}'))
                    if relayed:
                        relay = 3 - source - dest
                        rb, rl, rp = f'rbegin{source}:{dest}:{i}', f'rland{source}:{dest}:{i}', f'rpub{source}:{dest}:{i}'
                        events.extend([E(rb, relay, 'write_begin'), E(rl, relay, 'land'), E(rp, relay, 'publish', (rl,))])
                        leases.append(L(f'relay{source}:{dest}:{i}', relay, 1, rl, rb))
                        transfers.extend([Transfer(f'source{source}:{i}', f'p{source}:{i}', rl, begin=rb),
                                          Transfer(f'relay{source}:{dest}:{i}', rp, land, begin=begin)])
                    else:
                        transfers.append(Transfer(f'source{source}:{i}', f'p{source}:{i}', land, begin=begin))
        return compile_lifetimes(events, leases, reads, transfers)

    def test_two_slot_collective_recurrence_proves_source_and_destination_reuse(self):
        for relayed in (False, True):
            plan = self.allgather(relayed=relayed)
            for rank in range(3):
                for _, witnesses in plan.obligations[f'source{rank}:0']:
                    self.assertTrue(any(plan.precedes(w, f'w{rank}:2') for w in witnesses))
                for peer in range(3):
                    if peer != rank:
                        self.assertTrue(plan.precedes(f'c{rank}:0', f'begin{peer}:{rank}:2'))
                        self.assertFalse(plan.precedes(f'c{rank}:0', f'begin{peer}:{rank}:1'))
            if relayed:
                for source in range(3):
                    for dest in range(3):
                        if source != dest:
                            for _, witnesses in plan.obligations[f'relay{source}:{dest}:0']:
                                self.assertTrue(any(plan.precedes(w, f'rbegin{source}:{dest}:2') for w in witnesses))

    def test_two_slot_reuse_is_not_valid_for_independent_feedforward_publications(self):
        plan = self.allgather(recurrent=False)
        self.assertFalse(plan.precedes('c0:0', 'begin1:0:2'))
        self.assertFalse(any(plan.precedes(w, 'w0:2')
                             for label, witnesses in plan.obligations['source0:0'] if label != 'born'
                             for w in witnesses))

    def test_public_gather_certificate_includes_parallel_and_external_relay_paths(self):
        paths = {(a, b): [(a, b), (a, b), (a, 99, b)] for a in range(3) for b in range(3) if a != b}
        certificate = prove_recurrent_gather(3, paths)
        self.assertEqual(certificate.paths, 18)
        self.assertEqual(certificate.slots, 2)
        self.assertTrue(any('/transfer:' in obligation for obligation, _ in certificate.proofs))
        self.assertTrue(any('/read:' in obligation for obligation, _ in certificate.proofs))

    def test_certificate_rejects_incomplete_or_wrong_endpoint_routing(self):
        with self.assertRaisesRegex(ValueError, 'every source/destination'):
            prove_recurrent_gather(3, {(0, 1): [(0, 1)]})
        with self.assertRaisesRegex(ValueError, 'matching endpoints'):
            prove_recurrent_gather(2, {(0, 1): [(0, 2)], (1, 0): [(1, 0)]})

    def masked_frame_train(self, frames=3):
        """A one-way padded train with serial masked stores on its receiver.

        The transport sees full two-page frames. Valid-byte masks affect only
        pack/unpack arithmetic, never the extent whose DMA ownership persists.
        Descriptor pages are ordinary closure inputs, retained through uses.
        """
        events = [E('input', 0, 'publish'), E('pack_descriptor', 0, 'publish'),
                  E('mask_descriptor', 1, 'publish')]
        leases = [L('original', 0, 3, 'input'), L('pack_closure', 0, 1, 'pack_descriptor'),
                  L('mask_closure', 1, 1, 'mask_descriptor')]
        reads, transfers = [], []
        for i in range(frames):
            events.extend([E(f'pack_begin:{i}', 0, 'write_begin', ('input', 'pack_descriptor')),
                           E(f'packed:{i}', 0, 'publish', (f'pack_begin:{i}',),
                             channel='outgoing/session1', prefix=2 * (i + 1)),
                           E(f'landing_begin:{i}', 1, 'write_begin'),
                           E(f'landed:{i}', 1, 'land', channel='incoming/session1', prefix=2 * (i + 1)),
                           E(f'dma_read_done:{i}', 0, 'source_done', channel='outgoing/session1', prefix=2 * (i + 1))])
            leases.extend([L(f'source_frame:{i}', 0, 2, f'packed:{i}', f'pack_begin:{i}'),
                           L(f'landing_frame:{i}', 1, 2, f'landed:{i}', f'landing_begin:{i}')])
            transfers.append(Transfer(f'source_frame:{i}', f'packed:{i}', f'landed:{i}',
                                      f'dma_read_done:{i}', f'landing_begin:{i}'))
            # Serial destination computation orders consumers, but does not
            # order a remote producer or an already preposted next write.
            prior = (f'masked_stores_done:{i - 1}',) if i else ()
            events.extend([E(f'masked_stores_begin:{i}', 1, 'write_begin',
                             (f'landed:{i}', 'mask_descriptor') + prior),
                           E(f'masked_stores_done:{i}', 1, 'publish', (f'masked_stores_begin:{i}',))])
            reads.extend([Read('original', f'packed:{i}'), Read('pack_closure', f'packed:{i}'),
                          Read(f'landing_frame:{i}', f'masked_stores_done:{i}'),
                          Read('mask_closure', f'masked_stores_done:{i}')])
        events.append(E('output', 1, 'publish', tuple(f'masked_stores_done:{i}' for i in range(frames))))
        leases.append(L('output_pages', 1, 3, 'output', 'masked_stores_begin:0'))
        return compile_lifetimes(events, leases, reads, transfers)

    def test_one_way_masked_train_needs_distinct_landing_generations(self):
        plan = self.masked_frame_train()
        frames = [plan.allocations[f'landing_frame:{i}'] for i in range(3)]
        self.assertEqual(len({frame.page for frame in frames}), 3)
        for i, frame in enumerate(frames):
            for other in frames[i + 1:]:
                self.assertTrue(frame.page + frame.pages <= other.page or other.page + other.pages <= frame.page)
        self.assertTrue(plan.precedes('masked_stores_done:0', 'masked_stores_begin:2'))
        self.assertFalse(plan.precedes('masked_stores_done:0', 'landing_begin:2'))
        self.assertFalse(plan.precedes('masked_stores_done:0', 'packed:2'))

    def test_source_dma_completion_does_not_release_masked_consumer_input(self):
        plan = self.masked_frame_train()
        source, destination = plan.tracker(0, bridge=True), plan.tracker(1, bridge=True)
        self.assertIn('source_frame:0', source.observe('dma_read_done:0'))
        self.assertEqual(destination.observe('dma_read_done:0'), ())
        self.assertEqual(destination.observe('landed:0'), ())
        self.assertIn('landing_frame:0', destination.retained)
        self.assertIn('landing_frame:0', destination.observe('masked_stores_done:0'))

    def test_mask_closure_and_final_output_have_all_frame_dependencies(self):
        plan = self.masked_frame_train()
        destination = plan.tracker(1)
        for i in range(2):
            released = destination.observe(f'masked_stores_done:{i}')
            self.assertNotIn('mask_closure', released)
            self.assertIn('output_pages', destination.retained)
        self.assertIn('mask_closure', destination.observe('masked_stores_done:2'))
        self.assertIn('output_pages', destination.retained)
        self.assertIn('output_pages', destination.observe('output'))
        # Padded landing bytes remain a full transport allocation even if only
        # a few valid elements are written to the final output's current tile.
        self.assertEqual(plan.leases['landing_frame:0'].pages, 2)
        self.assertNotEqual(plan.allocations['landing_frame:0'].page, plan.allocations['output_pages'].page)

    def test_pack_releases_original_after_local_reads_not_remote_acks(self):
        plan = self.masked_frame_train()
        source = plan.tracker(0)
        self.assertNotIn('original', source.observe('packed:0'))
        self.assertNotIn('original', source.observe('packed:1'))
        self.assertIn('original', source.observe('packed:2'))
        self.assertTrue(all(f'source_frame:{i}' in source.retained for i in range(3)))

    def test_publication_prefix_cannot_claim_an_unfinished_earlier_frame(self):
        with self.assertRaisesRegex(ValueError, 'cycle'):
            compile_lifetimes([
                E('frame0', 0, 'publish', ('frame1',), channel='frames/session1', prefix=2),
                E('frame1', 0, 'publish', channel='frames/session1', prefix=4),
            ], [])


if __name__ == '__main__':
    unittest.main()
