# Node-addressed crossings (proposal, 2026-11-18)

## What is measured today

A fused crossing hands the caller one binding record per *cable stream* (`ncclMeshFused` publish/land records), and the
region keeps `ready` / `sent` / `landed` unit counts per `mesh_stream` (mesh.h), one stream per cable per holder. The
caller's kernels loop over cables, and each cable's counts are cumulative over its whole list.

Two consequences measured on the m5ult2–m5ult1 pair (gemma-4-e2b-pareto, meshmlx):
- every crossing pays per-cable bookkeeping in the caller's kernels;
- cumulative per-stream counts force a single global release order across everything a communicator sends: releasing
  frame k+1 declares frame k ready. Independent work (microbatches) cannot release independently.

A third, physical fact found while checking: every v135 Studio port reports a queue-pair budget of 2
("queue pairs for the link: this port's 2"), so `link_plan` gives each cable one stream (one lane pair); the Mini
(macOS 26.5) reports 10. Whether the Studios' 2 is macOS 27's `max_qp` or queue pairs held in the driver is not yet
determined (a reboot distinguishes them). With one ordered lane pair per cable, sends on a cable land in post order,
so independent orders on one cable need separate streams, which the budget does not allow.

## The API

The caller addresses peers, never wires:

    publish(frame j, peer q)   one store: plan p's ready word for q  := j + 1
    await(frame j, peer q)     one load-spin: plan p's landed word for q >= j + 1

- One `ready` and one `landed` word per (plan, peer) in the region, at offsets the fused issue returns.
- The bridge stripes a frame's bytes over the peer's cables. Each cable's child reads the node-level `ready` word in
  place of its stream's own (a `ready_at` region offset in `mesh_stream`), so entry e on cable c is ready once
  `ready ≥ frame(e) + 1`.
- Each cable's child adds the frames it has fully landed to the node-level `landed` word (an atomic add in the
  shared region; the children are separate processes), so `landed` counts frames complete on every cable.
- Per-plan words make plans independent wherever the lanes allow it (more than one stream per cable). On one stream
  per cable they share its order, and the bridge, not the caller, keeps that order.

## What it removes from the caller

The per-cable loops and records, the global release-order constraint at the API, and any knowledge of cable count
or striping. Kernels touch one word per peer per frame.
