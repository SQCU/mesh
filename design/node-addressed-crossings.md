# Node-addressed transactions (2026-11-18)

## The misgiving

The region exposes the bridge's wires to its clients. `struct mesh_stream` (mesh.h) is one cable's ordered lane pair;
a client *takes* it by compare-and-swap on its `owner` word (mesh-call.c `mesh_stream_held`: EBUSY when another holds
it), learns its `wire` identity, and presents its sends and landings on that cable's lists, synchronised by that
cable's cumulative `ready` / `sent` / `landed` counts. Everything above follows from that one choice:

- the exclusivity that belongs to the process driving a wire (one owner of a device's lanes, so no contradictory
  submissions reach the driver) is re-exported to every client as a lock on a wire;
- clients address cables, so a crossing is "per channel" in the library, in its fused records and in the caller's
  kernels (one release and one await per cable);
- a cable's counts are cumulative over one holder's list, so releasing frame k+1 declares frame k ready: one global
  release order per communicator, and a second communicator toward the same node is refused (EBUSY);
- striping, cable count and lane order leak into every client and every measurement made through them.

The wires are the bridge's alone. A client should name a node and some bytes, and never see a wire.

macOS 26.5 presents 10 queue pairs per device (the Mini); macOS 27.0 presents 2 (the Studios). Queue pairs are
therefore a property of the driver and firmware, not of cables, links, or anything a client of the bridge experiences
or triggers; with node addressing no caller ever sees them.

## Node-addressed transactions

A transaction is: destination node, source pages, a conversation (the communicator's identity, equal at both ends)
and a sequence number in it. The client submits it to its own queue toward that node; it holds nothing.

- **Submission.** One queue per (client, peer node) in the region, written by that client only: entries
  `{conversation, sequence, source page, bytes}`, a `ready` count the client advances (one store per transaction or per
  frame), nothing else. No owner word, no wire identity, no per-cable list.
- **Scheduling.** The bridge, the wires' one owner, takes ready entries from every client's queue toward a node and
  sends them on whichever of that node's cables is free. That is its job and nobody else's.
- **Landing.** Each cable's receives are always posted on bridge-owned free pages of the region, so a unit may arrive
  in any order from any conversation without a matching post. Every unit is self-describing: its conversation,
  sequence and byte count travel with it, and the receiving bridge records it in the conversation's landing index
  (sequence → page) and advances that conversation's `landed` count.
- **Consumption.** The consumer reads landed bytes in place, through the landing index: `await(conversation, seq)` is
  one spin on one word, and the data is read where the NIC put it (no copy, as the fused kernels already do). The
  page goes back to the bridge's free pool when the consumer's count passes it.

## What disappears

Client-held streams and the EBUSY between communicators; per-cable records, loops and counters in the library and
the caller's kernels; the global release order (conversations are independent; order exists only within one); any
client knowledge of cable count, lanes or striping. Microbatches, communicators and processes toward the same node
proceed independently, and the bridge alone decides how bytes cross the wires.

## Work

1. mesh.h: the per-(client, peer) submission queue and the per-conversation landing index and counts; remove the
   client-visible `owner` / `wire` / per-cable lists from the client contract.
2. mesh-link.c (each cable's child): take ready entries from the node's queues, send self-describing units; keep its
   receives posted on the free pool; record landings in the conversation's index.
3. nccl-mesh / meshmlx: a crossing becomes `publish(conversation, seq)` and `await(conversation, seq)`, one word each.
4. Re-measure everything measured through the per-cable contract (host path, GPU path, fused crossings, TP decode).
