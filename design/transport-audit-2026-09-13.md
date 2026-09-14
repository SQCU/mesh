# Explicit transfer ownership and failed-post continuation

The September 13 review of asynchronous index push identified two instances
where known transfer metadata was discarded. The implementation now keeps it
through the complete send lifetime. Prior art is the asynchronous communication
agent and two-sided verbs mechanism documented in
[algorithm sources](algorithm-sources.md#async-index-push-contract), including
Apple TN3205's receive-credit guarantee.

## Fanout retains every reader

Each realized transfer records local row/page, peer row/page, binding/offset,
reader plane, and its configured transfer index. The row-wide single send-plane
array has been removed. Two transfers reading the same source therefore retain
two reader planes; completing one transfer does not consume the other's read.
Constant sources also receive distinct transport reader planes, so a constant
fanout is emitted once per configured destination rather than repeatedly.

The bridge keeps a setup-allocated active byte per transfer. Announcing a
transfer activates that particular descriptor. It becomes inactive at its
physical send completion, which consumes its recorded reader plane. A new
source publication clears read planes through the existing publication lifecycle.
Eligibility is full-block presence and that transfer's unread plane, together
with its own inactive state; another reader sending the same pages does not
prevent this transfer from being posted.

The bridge also keeps per-row and per-page counts of announced or posted sends.
An announcement increments the counts for its exact source ranges. Completion
or completed QP teardown releases precisely those ranges. The last release clears
the corresponding hot bits. This prevents one completed fanout branch from
making another branch's source pages available for allocation. Numerical output
reuse continues to follow the canonical reader mask, including every fanout
branch. No payload staging buffer is involved.

## An announced payload retains its place

Before a descriptor frame is accepted by `ibv_post_send`, selection changes no
transfer ownership. A rejected index post therefore leaves every selected
transfer eligible for a later pass.

After an accepted index post, each exact transfer is placed in that data QP's
fixed announced-payload vector and its source is retained. Posting drains this
vector in order. If a payload post fails, its entry remains at the head, with
its source retained; subsequent polling passes attempt that same entry. Later
payloads cannot occupy its peer receive slot. Other QPs and completion polling
continue. Successful posting moves ownership into the existing posted-WR entry,
which retains row, page, reader plane, and transfer index through completion.

The sum of announced and posted payloads is bounded by the provider's send
capacity. All descriptor vectors, active bytes, and ownership counters are
allocated before numerical execution. Descriptor frames remain registered until
their own send completion; after copying their small metadata into the fixed
announced vector, no payload submission depends on re-reading a reusable frame.

The descriptor is now 32 bytes, permitting 127 tuples plus the eight-byte frame
header in each 4096-byte index message. Region version 23 identifies this layout.

## Evidence and limits

`make -C rdma mesh-flow libmesh.dylib` compiled the implementation. Source review
traces each accepted announcement to one retained payload entry and each source
retention to its completion/teardown release. No harness or synthetic fault test
was added, and these fixes do not establish an operational throughput gain.

A target buffer reused while its prior readers remain active still carries a
real storage-lifetime dependency. Distinct target buffers permit independent
value instances to proceed. Work-completion failures remain visible through the
existing port metadata; preserving a failed post's unsent tuple is not a claim
that an already failed physical transfer succeeded.
