# What holds a SEND, a receive or a read, against what the provider does (2026-11-02)

Read from mesh 212f8e8e. The audited lines are unchanged from 62382113 and 81ad6576. Static reading only: no run, no
probe, no build. The rows are data in [data/send-receive-wrappings-2026-11-02.json](data/send-receive-wrappings-2026-11-02.json).
This extends [fake-guards-2026-10-30.md](fake-guards-2026-10-30.md),
[runtime-waits-2026-10-30.md](runtime-waits-2026-10-30.md) and
[compiler-runtime-inventory-2026-10-31.md](compiler-runtime-inventory-2026-10-31.md). It does not restate them.

The operator, 2026-11-02: "incoherent wrappings or interpretations of what the bridge dma driver *can* do (send in like
4 ns) vs what the bridge dma driver is being used for by the library in this form (lets wrap all of the sends with
waits for no reason because we forgot what the api does even though the api handles the thing we're trying to
reproduce in the wrapping)".

## The provider's contract

From [hardware-and-discipline.md](hardware-and-discipline.md) section 2 and [transport.md](transport.md) (Apple TN3205):
- UC queue pairs, SEND and RECV only, one SGE each.
- A SEND fills the oldest posted receive of the same frame count.
- A credit in the hardware holds a SEND until its receive is posted.
- Every request ends in a completion with a status.
- A queue accepts posts past its frame capacity and corrupts later. The capacity is a device number
  (mesh-verbs.h:168-173).

## In brief

- The SEND side is mostly clean. Receipts, credits and waits on `sent` were deleted before (22228bb3, bdde7903), and so
  were the landing gate (f1f4122e) and the GPU presents (1f96e30b). What still holds a SEND is its data edge, the
  frame gate and the lane order. One wrapping is left on SENDs: the link (T3). It adds no wait today.
- The receive side carries the cost. Four holds answer page reuse with a wait: the link's position (T2), and the
  landing's and the combine's waits on another SEND's `sent` (T11, T12, G6). Page assignment should answer them.
- Each held receive holds every later receive on its lane (T15, TN3205's order). The peer's SENDs then wait in the
  credit. So one hold stalls a cable, and a stalled cable stalls its peer.
- Two per-unit checks in the bridge can never fail (T4a, T5). One per-pass literal (T14) was a measured gain. It goes
  with the per-unit doorbell, not alone.

## The rows

Classes: (a) a true data edge; (b) a named hardware limit with its number; (c) a re-implementation of what the
provider guarantees; (d) a guard against page reuse that page assignment should answer; (e) defensive or bound
machinery.

Scope and violation follow [page-maps-2026-11-02.md](page-maps-2026-11-02.md): a chooser decides on pages; an effector
turns a decided map into addresses and decides nothing. A violation is "effector decides", "effector computes a map"
or "chooser holds pointer". The JSON adds what each row handles.

### The bridge's walks (rdma/mesh-link.c stream_send 245-278, stream_post 282-312)

| ID | file:line | Holds | Waits on (advanced by) | Class | Cost on the hot path | Scope, violation |
|---|---|---|---|---|---|---|
| T1 | mesh-link.c:221-238, 248, 285 | SEND and receive posts of a stream | owner word, served lanes (holder; event loop; port child) | e | a pass a stream: 1 acquire load, 2 compares; none a unit | effector, decides |
| T2 | mesh-link.c:292, 127; nccl-mesh.c:4865-4885; nccl-mesh-metal.m:507, 529, 865-868; _stream.mm:1129 | every receive of the occurrence and after it on the stream | predecessor's `next` (the GPU link dispatch at the occurrence's own position) | d | 1 dispatch a channel an occurrence (1-2 µs); receives post only when the GPU gets there: the LSTM's 77-101 ms ticks | chooser, holds pointer |
| T3 | mesh-link.c:255-256, 127 | SEND post | the same word | c | none beyond T2's dispatch | chooser, holds pointer |
| T4a | mesh-link.c:258, 295; nccl-mesh.c:4196-4201 | SEND and receive posts | spans word, static table (preparation) | e | a unit, both walks: 1 acquire load, always passes | effector, decides |
| T4b | the same; nccl-mesh.c:5964-5977 | masked units | spans word (the mask command) | a | a unit: 1 acquire load | effector, decides |
| T5 | mesh-link.c:262, 299; layout.h:14 | SEND and receive posts | requests in flight < 4096 (the bridge) | e | a unit, both walks: 1 subtract, 1 compare; cannot bind before T9 or T13 | effector, decides |
| T6 | mesh-link.c:263, 142-148; nccl-mesh.c:4403-4437 | SEND post | `landed` or summed at each followed receive's end (bridge; combine) | a | a unit: 1 acquire load a list entry, a per-entry fact | effector, decides |
| T7 | mesh-link.c:263; nccl-mesh.c:6020-6080 | SEND post | `landed` of the collectives its source landed from (bridge) | a | a unit: 1 acquire load a word | effector, decides |
| T8 | mesh-link.c:264; nccl-mesh.c:4703-4727; mesh.metal:114-118, 154-168 | SEND post | `ready`, `landed`, summed, a producer or copy word (producer's last threadgroup; bridge; combine) | a | a unit: 1 acquire load; a producer threadgroup: 1 barrier, 2-3 system fences, 1-3 system atomics a cut, a prefix CAS loop | effector, decides |
| T9 | mesh-link.c:267, 640, 668; mesh-bridge.h:89-99 | SEND post | frames in flight within capacity (SEND completions) | b | a unit: 1 add, 1 compare | effector, decides |
| T10 | mesh-link.c:300; nccl-mesh.c:4461-4472, 4511-4528 | receive post | `sent` past an earlier SEND of the same call and frame (bridge, after the peer's receive) | a | a unit: 1 acquire load a channel | effector, decides |
| T11 | the same | receive post | `sent` past an earlier frame's or call's SEND of the same issue | d | the same; bound: a wait across nodes | effector, decides |
| T12 | mesh-link.c:300; nccl-mesh.c:4542-4621, 4674-4678 | receive post | `sent` past another issue's last SEND of the caller's pages | d | the same; bound: a wait across nodes | effector, decides |
| T13 | mesh-link.c:302, 749-754 | receive post | frames posted within capacity (receive completions) | b | a unit: 1 add, 1 compare | effector, decides |
| T14 | mesh-link.c:251, 288; layout.h:16 | posts past 64 a lane a pass | the next pass (progress thread) | e | a ready 65th unit waits one completion pass | effector, decides |
| T15 | mesh-link.c:251-256, 288-293, 123-138 | every later unit on the lane | the head unit's own condition | b | multiplies T2 and T10-T12; a unit: the entry re-found and a 64-byte entry copied | effector, computes a map |
| T16 | nccl-mesh.c:5413-5429, 5454-5456, 4717-4721; nccl-mesh-kernels.h:111-117 | SEND post | the premultiplied tile's word (the copy) | a | a whole-operand pass before the first SEND | effector, decides |
| T17 | the same; nccl-mesh-kernels.h:79-84 | SEND post | the copied tile's word (the copy) | d | a whole-source copy before the first SEND | effector, decides |

T2 is the known instance (metal-microbench docs/facts.md, 2026-11-02). zinth declares writes to every landing range
(nccl-mesh.c:4873-4877), and then marks the link administrative. command_emit drops an administrative command's
relations (nccl-mesh-metal.m:507) and gives it frame UINT32_MAX (529). The native bind puts such a command at the
occurrence's base (_stream.mm:1129). So the link is ordered by position alone, after the readers of reused pages.
extras/wait_report.py (534f5709) shows it on d-r41: every tick's link follows the readers of the reused h, and the next
waits on its words are 85-92 ms.

T3: a SEND also waits on its payload word (T8), and TN3205's credit holds it for its receive. So the link adds nothing
to a SEND. Where a SEND reads an input made outside the function, the link's part carries that input's event
(nccl-mesh.c:6094-6117). That edge is the part's queue wait (H1), not the link.

T10 and T11 are one compiled list. A requirement from the same call and frame is a value-version edge of an in-place
call: class (a). One from an earlier frame or call is page reuse: class (d).

### The GPU's waits

| ID | file:line | Holds | Waits on (advanced by) | Class | Cost on the hot path | Scope, violation |
|---|---|---|---|---|---|---|
| G1 | metal-page-effects.m:1433, 1244-1246; nccl-mesh-kernels.h:52-54; mesh.metal:28-38 | consumer read | `landed` past the packet's end (bridge, once a pass) | a | a threadgroup and word: ≥1 poll (fence ≤0.13 µs, load 0.23-0.26 µs), 1 barrier, 1 system fence a thread | effector, decides |
| G2 | nccl-mesh-kernels.h:43-51 | nothing on success | the wait's returned status | e | 2 register compares | effector, decides |
| G3 | mesh.metal:36; metal.def:14 | the wait, on failure | the dispatch's claim word (first failing wait) | e | 1 relaxed load a 1024 polls | effector, decides |
| G4 | mesh.metal:18-20, 36; metal.def:13 | the wait, at its bound | the poll count | b | 1 increment, 1 compare a poll | effector, decides |
| G5 | nccl-mesh-metal.m:729-738; nccl-mesh.c:5113-5118 | a combine's store | `sent` past an earlier SEND of the same call and frame (bridge) | a | a threadgroup: a poll on `sent` | effector, decides |
| G6 | nccl-mesh-metal.m:729-738; nccl-mesh.c:5107-5129 | a combine's store | `sent` past another issue's or an earlier frame's SEND (bridge) | d | the same; bound: a wait across nodes | effector, decides |

G4's number: 4.5 M polls, about 1.1 s. The GPU watchdog ends a command buffer after about 5 s, and an unbounded spin
reset the Mini on 2026-10-21 (mesh.metal:18-20).

### The host and the lifecycle

| ID | file:line | Holds | Waits on (advanced by) | Class | Cost on the hot path | Scope, violation |
|---|---|---|---|---|---|---|
| H1 | metal-recording.m:2394-2395, 2223-2249 | a part's start, and every link in it | predecessor and external events | a | 1 queue wait a prerequisite a part | effector, none |
| H2 | metal-recording.m:1895-1910 | a host task | a countdown of its events (listeners) | a | a callback and a decrement a prerequisite | effector, decides |
| H3 | metal-recording.m:1787-1793 | a host-resource use | the use `capacity` earlier | b | an edge at preparation | chooser, none |
| H4 | mesh-link.c:454-476, 513-528 | every post on a port | the guard word or served lanes (the child) | b | off the hot path; re-looks every 3 s | effector, decides |
| H5 | mesh-link.c:505-512 | a settle or a guard raise | the progress thread's told word | e | off the hot path; 5 s deadline | effector, decides |
| H6 | mesh-link.c:1619-1650 | the child's exit | flush completions, 0.5 s quiet | e | off the hot path | effector, decides |
| H7 | mesh-dataflow.c:504-539 | a row's reuse | a dead holding's posts done (completions; the child's exit) | b | off the hot path | chooser, none |
| H8 | mesh-dataflow.c:240-261 | a client's map wait | the map serial (record flooding) | e | off the hot path; deadline | chooser, none |
| H9 | nccl-mesh.c:1204 | init | the root's listener | e | off the hot path | chooser, none |
| H10 | nccl-mesh.c:1527-1541 | teardown | init and configuration transactions | e | off the hot path | chooser, none |

H1's waits are compiled at preparation and emitted as they are. Metal's queue evaluates them, so the effector decides
nothing there. H3's limit is the provider resource's own capacity. H4's is Apple's provider moving the device's doorbell
mapping when a post crosses another queue pair's teardown ([RDMA-KERNEL-RECOVERY.md](RDMA-KERNEL-RECOVERY.md), "The
stripe writes of October 1"). H7's is verbs: a posted request owns its buffer until it completes or its queue pair is
gone.

No row: rdma/mesh-records.c holds no post, receive or read. metal-recording.m:1186 waits during a replay's preparation.

## What removes each row of classes c, d and e

**T2 and T3 (d, c).** Both halves are needed:
1. distinct landing generations where landings are reused (the LSTM's h). torch_mesh/chain.py `operands` already does
   it: two page generations, and each window's recurrent gather one framed occurrence at the window's start.
2. each link recorded at the earliest order its landing pages allow: the function's start where no earlier command
   touches them.

The docs and the history:
- [eager-execution.md](eager-execution.md): "Page reuse is answered by page assignment".
- [whole-function.md](whole-function.md):235: "Distinct landing generations or an existing causal descendant proof are
  required".
- [page-maps-2026-11-02.md](page-maps-2026-11-02.md): with lifetimes bin-packed, "every receive of an invocation can be
  posted at its start".
- Until d4916d1e (10-25) the host presented a call's lists at issue, before the GPU reached the call
  ([fake-guards-2026-10-30.md](fake-guards-2026-10-30.md), the landing gate's history). zinth put the link at the
  call's place on 10-31 (1f96e30b).
- On 2026-09-30, receives posted ahead from the leading cut took the Mini's "network past the wire" from about 200 µs
  to 2 µs a call (metal-microbench docs/measurement.md, part C, mesh 7c2bbaa; f1-sp1, mesh 55f8a1a).
- The fabric DNN posts its receives during encoding into the function's separate landing leases (docs/measurement.md,
  "Physical fabric DNN").

**T11, T12, G6 (d).** The caller's reused pages, and the combine's output pages, get distinct generations. Then the
`after` lists and the combine's `sent` waits are empty. `sent` is read between calls without waiting.
- [eager-execution.md](eager-execution.md): "Receipts, credits, waits on `sent`, spare pools and reuse queries are
  never the answer"; and "Where a guard would come from": `sent` "read between calls without waiting" (rubric C5).
- [pages-and-functions.md](pages-and-functions.md) Freeing 7: "Two NFEs can be in flight because rows for NFE k+1
  point at other pages".
- [zorvek.md](zorvek.md), "A landing's after", names the cases: a relay forwarding from its recvbuff, and an in-place
  operand's SEND. Its first, coarser form deadlocked the chain service on all five. Found at an issue's first call, it
  ran 64 MiB reduce-scatter and all-reduce up to 1.5x slower.
- The docs disagree on the release proof. pages-and-functions.md Freeing 6 now names the bridge's `sent` and rejects a
  downstream landing as proof (76de94ce, 10-31). eager-execution.md counts both proofs. Either way the wait binds only
  where pages are reused inside the program's window, which page assignment removes.

**T17 (d).** The result gets its own pages, and the source is read in place: transport by view ([zorvek.md](zorvek.md)
item 7). T16 is a real transform (a); its cost goes when the scalar is folded into the last combine (the same item;
fake-guards-2026-10-30, the staging finding).

**T5 (e).** Fold the request bound into the gate's and the receive queue's frame capacity once, at connect
(mesh-link.c:668, 751-752). Delete both tests. Every request covers at least a frame, and each lane's frame capacity is
at most MESH_LANE_UNITS - 1 (mesh-verbs.h:170-173): 2048 frames a lane on 0xc links. runtime-waits-2026-10-30 item 3
called it "cannot bind".

**T4a (e).** A static table names no spans word, and the walk loads none. [transport.md](transport.md) section 14:
"Static tables bind an explicitly completed word". The fact is fixed at preparation.

**T14 (e).** Keep its effect. It was measured: 0310d51f took the 64 MiB all-gather from 6,570 to 5,632 µs and the
all-reduce from 9,393 to 8,450 µs, by reaping completions between bursts. The literal goes with the per-unit doorbell:
one chained request list a lane a pass, sized by the gate's room ([compiler-runtime-inventory-2026-10-31.md](compiler-runtime-inventory-2026-10-31.md)
C10; fake-guards-2026-10-30, the bridge table's open row).

**Keep:** T1 (per pass; the holding's grace period, 2548cbce), G2 (no load on success; without it a bound reads as a
landing), G3 (a load a 1024 polls), H5, H6, H8, H9, H10 (off the hot path, each under its deadline).

## What each row handles

Every violation is corrected at once, with the problem it handled, so none is invented again.

- T1: post nothing of a holding whose holder gave it or died; free the stream once nothing is outstanding.
- T2: a receive must not land into pages an earlier GPU reader of the function still reads.
- T3: post no SEND before the program reaches its call; `ready` and the credit already do this.
- T4a: never read a descriptor before its table is complete.
- T4b: read a masked unit's length only after the GPU's mask producer stored it.
- T5: keep requests in flight within the queue's request depth.
- T6: a SEND reads bytes only after the receive or sum that made them finished.
- T7: a SEND of a function input forwards bytes only after they landed.
- T8: a SEND reads its source only after its producer's stores reach the NIC.
- T9, T13: never post past a queue's frame capacity.
- T10: an in-place call's landing must not overwrite its source before its SEND read it.
- T11: a later frame's landing must not overwrite pages an earlier frame's SEND still reads.
- T12: a later call's receive must not land on caller bytes an earlier call's SEND still reads.
- T14: tell `landed`, `sent` and the frame credits promptly.
- T15: keep each lane's n-th SEND matched to the peer's n-th receive.
- T16: the premultiplied value must exist before it is sent.
- T17: an in-place call whose result overlaps its source at a shift keeps the original until its SENDs read it.
- G1: a consumer reads landed bytes only after the NIC wrote them.
- G2: tell a lost link, a revoke or a bound from a landing, and record which.
- G3: once one wait of a dispatch fails, its other waits end soon.
- G4: a GPU spin with no end halts the node.
- G5, G6: a combine must not overwrite bytes a SEND still reads.
- H1, H2: a part or a host task starts only after what it reads.
- H3: no more uses of a host resource in flight than it holds.
- H4: no post on a device's queue pair while another is made or changed.
- H5: change verbs objects only once the progress thread holds; end the child if it does not.
- H6: leave nothing posted at the child's exit.
- H7: no row is reused while a posted request still names it.
- H8, H9, H10: wait for the map, the root or a communicator's transactions, each within a deadline.

Effectors that compute a map, not on a hold: dma_span (mesh-link.c:90-94) looks up each unit's memory region and key
by extent (two dependent loads a unit); link_lane (324-328) scans up to ten queue-pair numbers a completion, where the
request id already names the span.

## Order of repair, by hot-path cost

1. T2 with T3: milliseconds a tick (the LSTM's 77-101 ms). One change removes both.
2. T12, T11, G6: a wait across nodes when bound, a load a unit or a threadgroup otherwise. Distinct generations remove
   all three.
3. T17: a whole-source copy before the first SEND.
4. T5, T4a: a compare and an acquire load a unit, on both walks.
5. T14: one pass's delay past 64 units. Replace it with the chained post; do not delete it alone.
6. Keep the rest listed above.

## The prior inventories at today's source

Fixed:
- The bridge's span, table and frame-count checks (faa21d00) and the per-unit tell and pin (2548cbce): absent from
  mesh-link.c.
- The tracks pool, its workers, claims, per-chunk re-waits and summed publishes (a9bd4e7a, 53c1fe72): deleted. Every
  pool row of fake-guards-2026-10-30 is moot.
- The landing gate (f1f4122e) and the GPU presents with their barriers and ring waits (1f96e30b, 11807bec).
- A forwarded SEND's publication and whole-consumption wait: forwarded SENDs carry neither (nccl-mesh.c:4414, 5314).
- One poll a pass: link_send and link_receive now drain (mesh-link.c:351-364, 396-409).
- metal_barrier and metal_order (compiler-runtime C1): gone. barrierAfterQueueStages: gone.

Open:
- One `ibv_post_send` and IBV_SEND_SIGNALED a unit (mesh-link.c:269-272); the release store of a private counter a unit
  (275); the lane scan (324-328).
- The 4096 request test (T5) and the per-pass literal (T14, C10).
- `after` words read in the posting loop a unit (T6, T7, T10-T12; compiler-runtime C9) and the lane-ordered walk
  (T15, C11).
- The lane's completion share counted in frames (T9; fake-guards P5).
- `struct mesh_stream` on 64-byte records (etc/bridge/layout.h:38): `ready` at 128 and `sent` at 192 share one
  128-byte line, so the holder's stores and the bridge's share it (fake-guards P4).
- The staged source generation for premultiplied calls and shifted aliases (T16, T17; fake-guards P2 staging).
- A sum's SEND not unit-aligned with its landing waits for the whole sum (nccl-mesh.c:4382; T6).
- Queue waits at part granularity (H1, H2; compiler-runtime section 4).
- Landed payloads are read as 32-bit volatile system-coherent words, whatever their type (mesh.metal:48-61;
  fake-guards P5 kernels).

## Not settled by this reading

Which publication raises `ready` past a SEND that reads a sum it cannot forward unit by unit? nccl-mesh.c:4382 leaves
its `dep_h` unset, and page_effects publishes original SENDs only (nccl-mesh.c:5314). This is for the nccl-mesh.c
audit.
