# A holding's end is its grace period: reclamation without a per-entry handshake, 2026-10-30

The bridge's last check on its data path after 33274e73 is `metadata_pin` + `holding_kept` (rdma/mesh-link.c): a hazard
pointer (Michael 2004) kept per entry, five sequentially consistent stores and three loads each time the walk enters an
entry and five more to clear it at the end of every send or post loop, a relay's idle spin included. Its contents are
never compared (mesh-dataflow.c mesh_interval_pinned tests it for nonzero). It exists because two reclaimers
(mesh_lease_return, mesh_retired_release) act on a snapshot rather than an acknowledgment, which store buffering needs
sequential consistency on both sides for.

The acknowledgment exists already: stream_free's release store of `owner` runs only after the progress thread stopped
reading the holding (stream_turn sees the give, HOLD_END; the pass after, stream_end -> stream_free). A reclaimer that
acquires `owner` no longer the client's happens-after every table and word read of that holding (message passing,
release/acquire; Pulte et al. 2018). So a holding's end is a grace period, quiescent-state reclamation with the pass as
the quiescent state (McKenney & Slingwine 1998; Desnoyers et al. 2012), acknowledged per holding, with no fence per
pass or entry. A lease the library returns under its compiled lifetime (mesh_lease_retire_ready: completed >= end)
needs no check at all (Hart et al. 2007: QSBR cheapest where natural quiescent states exist).

## Defects in the current form (found reading 07d68ea4)

1. The pin's offset, bytes and word are never compared: a busy flag at the price of five stores.
2. Retention is global: mesh_retired_release asks with client 0, so any busy stream on the node keeps every dead
   client's rows (33274e73).
3. A hole after a dirty end (units outstanding when a holding ends): the owner word is GIVEN alone, so a lease
   return filtered by client skips the stream and reclaims pages posted receives still name (communicator destroyed
   with a call aborted, then ncclMemFree; nccl-mesh.c lease_detachable -> lease_give -> mesh_lease_return) (33274e73).
4. mesh_holds reports a child in its exit gone before the kernel released its device objects (mesh-dataflow.h).
5. The per-unit pin rings (`rows`, about 384 KiB a stream) are leased and written by nothing.

## The form (A + H), its steps

1. Layout 154 -> 155: struct mesh_stream's `metadata` (struct mesh_cpu_pin) becomes `_Atomic uint64_t unfinished`
   (the owner word of the holding that ended with units outstanding, 0 none) and padding, offsets kept; the pin ring
   helpers (mesh.h mesh_dma_pin, mesh_unit_pin, mesh_pin_store, mesh_pin_load) and the `rows` ring deleted.
2. The bridge: holding_kept, metadata_pin and unit_told deleted, and their calls in stream_send and stream_post.
3. stream_end's dirty branch, before its stores: each lane's posted and done counts, and `unfinished` = the holding's
   owner, relaxed; stream_free's release store publishes them. posts[].done stored only while the hold is HOLD_GONE.
4. child_reaped stores each of the port's streams' `child` 0 (release) after waitpid: the proof that no posting child
   remains. Reclaimers test `child == 0`, not mesh_holds.
5. mesh_interval_pinned becomes mesh_holding_retained(m, client): a stream with child 0 retains nothing; else retained
   where (owner & ~GIVEN) == client (the bridge still reads), or `unfinished` == client with a lane's done < posted.
   mesh_retired_release asks it once a distinct closed owner; mesh_lease_return asks nothing (its callers return with
   the compiled proof).
6. nccl-mesh.c lease_detachable: the hold-in-use test kept, then EBUSY where mesh_holding_retained holds for the client;
   the table special case deleted.
7. Optional: at a dirty end, outstanding units' spans recorded so only intersecting rows wait.

Rejected, ranked: per-port QSBR pass counters (a fence a pass), epochs (a fence a pass and a shared line), hazard
pointers (today's), flushing the queue pairs to Error (rubric M3; may be added to A to bound a dead stream's W(t) to
about 0.4 s), copying tables at take (still reads client memory), generation tokens (a check a unit, no DMA cover).
