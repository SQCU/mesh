# Disjoint-pages rulings (Appendix D): the coordinator's contract and its routing record

The coordinating agent's contract for design/disjoint-pages-change-request.md, Appendix D: items D.3.2 to D.3.6, D.3.7
as it applies to them, and the D.4 entries that belong to them, from mesh 5c54134c and metal-microbench a062db81. The
requirements are the change request's: D.1 and C1 hold the operator's turns verbatim and are the rulings; D.3 holds
wherever it differs from C3 or anything earlier; D.2 states the data model of a transfer. This file holds the file split,
the names files share, and the decisions this contract takes, each with its source; then the record of each
sub-assignment, its commits, each review finding and where it went, the builds, and what is left open.

D.3.1 (any number of programs at once) is not in this unit: it waits on the operator's ruling recorded in D.2. Nothing
here changes the client word, the attach, the region header's one plan (hdr `client`, `hops`, `plan`, `state`), the
ports' `prepared`, the bridge's one flow per port, or the launchers.
(2026-11-04, after the work began: the change request gained D.5, "What the plan gives the bridge" (mesh 4cdff8b3),
which settles D.2's open question and on which D.3.1 now stands; D.5 holds over everything earlier, D.3.5's u rounds in
flight among it ("hold per chain"). This unit was dispatched before D.5, with D.3.1 excluded; D.5's names, chains,
placement tables and priorities are not in it, and its D.3.5 is per lane in order, the bridge's one flow per port that
D.3.1 replaces. Both are open for the unit that implements D.3.1 and D.5.)

design/disjoint-pages-revision-coordination.md, the previous coordinator's contract, stays in force except where D.3
changes it: its four rules, its rules for every sub-agent, the form of its file split, the names files share, its
compile commands and its commit rules. Its departures D7 (the decode as one Metal 3 command buffer), D11 (the
delegate's weights and unrenamed buffers left outside the allocator) and D12 (the per-step host inputs sized by the
run) are reversed by D.3.4, D.3.2 and D.3.3, and are not repeated here in another form.

## The operator's rulings of 2026-11-04, as they bind this unit (D.1, verbatim there)

1. **No transport facts outside the bridge.** The bridge's private interface (its connections, and any count of them)
   differs from run to run and between macOS 26 and macOS 27. Nothing outside the bridge's own files names or reasons
   from it: no contract text, no comment, no design in collective, data, GPU or program code. Transfers are described in
   D.2's terms: per adjacent node, page lists of views (pages by name), released by words, drained apart. The bridge is
   an opaque driver consuming queues of desired transfers. The bridge's own files are mesh rdma/mesh-flow.c,
   mesh-link.c, mesh-ports.c, mesh-records.c, mesh-stat.c, mesh-events.c, mesh-verbs.h, mesh-bridge.h, mesh-holders.h,
   mesh-disk.h and the lifecycle files (mesh-heartbeat.c, mesh-children.c, mesh-child.c, mesh-lanes.c), and the
   bridge's design records (design/prepared-machine.md).
2. **Both OSes, no single-casing.** Bridge code takes whatever its devices report on each run, by the failport's code
   (tag pre-return-20261103, rdma/mesh-verbs.h `max_pairs`: what each device reports on the run, both ends'
   smaller). No code single-cases an OS or assumes a count.
3. **No Metal 3 command buffers.** Every command buffer this code makes is an MTL4CommandBuffer from an
   MTL4CommandAllocator, committed to an MTL4CommandQueue. The one exception is the delegate's capture interface
   (metal_recording.m MetalCommandRecorder, MetalQueueRecorder, `recordingQueue`), and nothing the run executes passes
   through it.

## Rules for this unit (beside the previous contract's rules for every sub-agent, which apply verbatim)

- Never open docs/facts.md at any revision. Never cite design/transport.md.
- Run nothing: no bridge, no e2b_replay, no program, no test, no measurement, no ssh, on any machine. Compiling is
  allowed. Write no tests or evaluators.
- Never change the lifecycle files (rdma/mesh-heartbeat.c, mesh-children.c, mesh-child.c, mesh-lanes.c), the deployment
  and launch scripts (rdma/mesh_hourly.py, mesh_preflight.py, mesh_graph.py, bin/, etc/, metal-microbench tools/mesh/),
  or xonotic/; never change what D.3.1 leaves standing (above).
- Work in the literal repo directories on main: no worktrees, no branches, no copies of source anywhere (no /tmp,
  scratchpad or backup copies). Never git reset, clean or stash. Write nothing outside the two repos.
- Commit and push as each piece compiles, by explicit pathspec; never `git add -A` or `git add .`; never commit mesh's
  untracked artifacts (rdma/_mesh_c.*, rdma/libnccl-mesh.*, rdma/libsleef.a, rdma/linear-chain, rdma/mesh-heartbeat,
  rdma/torch-mesh/, NODE-STANDUPS.md) or tailscale.key. Every message ends with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- No invented mechanisms: each comes from D.3, C3, the literature they cite, or a decision below that names its source.
  One configuration per runtime; no call site handles two forms; a replacement removes the old code in the same work.
- Every failure is an error value. Every wait ends in bounded time. No GPU kernel spins.
- A departure only where the real API or hardware makes D.3 impossible, recorded with its evidence (code, SDK header,
  documentation), never one taken for ease.
- Comments: each file's existing form (mesh rdma block comments citing the design by section; metal-microbench
  `// ../dox/mesh/...` citations and dated notes). Cite "design/disjoint-pages-change-request.md, D.3.n" and this
  contract by its decision (R1 ...). Where a reversal is made, a dated note (2026-11-04) at the spot: what, why, what
  must stay.

## Where things live (owner = sub-assignment)

| file | owner |
|---|---|
| mesh rdma/mesh-flow.c, rdma/mesh-records.c, rdma/mesh-bridge.h, rdma/mesh-dataflow.c (mesh_transfers_start alone), design/prepared-machine.md (the M08 row) | 1, then 6 (the bridge files but mesh-dataflow.c) |
| mesh rdma/mesh-collective.h, rdma/mesh-collective.c | 2 |
| metal-microbench metal_recording.h, metal_recording.m | 3, then 4 |
| metal-microbench tools/native_bench.m, tools/primitive_constants.m | 5 |
| metal-microbench tools/e2b_replay.m | 7 |
| mesh rdma/mesh-link.c, mesh-ports.c, mesh-stat.c, mesh-events.c, mesh-verbs.h, mesh-holders.h (the count audit's fixes) | 6 |
| metal-microbench tools/prepare_e2b.py | no owner unless 7 reports a need |
| reviews (written findings only) | 8 (the mesh side), 9 (the GPU and program side), 10 (both repos: the four rules, D.2, the rulings, D.3) |
| this file, INDEX.md, design/disjoint-pages-revision-coordination.md (its open item on the provider's copy of a request, D.4) | coordinator |

## Names files share

### The plan call (mesh-collective.h, owner 2)

`struct mesh_value { struct mesh_section section; uint32_t kind,wire,state,begin,end,lag; };` (`lag` in place of
`padding`; the size and every other offset unchanged).

- `lag` 0: as before (one copy; a routed value's u).
- `lag` k >= 1: the value's instance of execution t is written by a producer the program orders by an event, not by
  its own order (a host producer, D.3.3: "The program declares the host's lag, and the allocator gives them copies by
  it, as modulo variable expansion does for any value"). The value is MESH_EXECUTION, wire 0, state 0, and named by no
  declared transfer (EINVAL otherwise); its `begin` and `end` are not read. The plan gives it k copies: `section.count`
  k, `stride` P, execution t's instance copy t mod k at mesh_section_address(section, t mod k). Each copy is an item
  of its own that interferes with every other item (its write is ordered by no edge of the template, so it shares no
  page with anything).
- The program's side of the order, which the plan neither knows nor checks: copy t mod k is written for execution t
  only after execution t - k has ended, and execution t reads it only after that write (R4).

### The bridge's cancellation of a plan whose map has moved (mesh-bridge.h and mesh-flow.c, owner 1)

`void plan_stale(void);` in mesh-flow.c, declared in mesh-bridge.h beside streams_cancel and streams_let_go. Where a
plan stands (hdr `hops` above 0) and its entries' `serial` is not hdr `map_serial` as the bridge parent reads it, every
receive of it not landed is cancelled: plan_cancel(UINT32_MAX, the generation it judged stale) (2026-11-04, 1's
accepted difference and review 8's finding 7: every call of plan_cancel names its generation), the plan read whole as plan_cancel reads it, each
cancellation an arrival on its count with no phase written. Called by mesh-records.c map_put and map_adopt after any
write that moved `map_serial` (after the write's closing notification), on the bridge parent's one event-loop thread
(R2).

### The start (mesh-dataflow.c mesh_transfers_start, owner 1)

No interface change. After it raises the plan's `executions`, a seq_cst fence, then hdr `map_serial` read: where it is
not the plan's entries' `serial`, the start returns ESTALE (R3).

### The recorder (metal_recording.h, owner 3)

```objc
@interface MetalRecorder : NSObject
@property(readonly) id<MTLDevice> device;
// every command this code commits outside the delegate's capture is committed here (D.3.4)
@property(readonly) id<MTL4CommandQueue> queue;
// the delegate's Metal 3 interface, at capture alone: nothing the run executes passes through it (D.3.4)
@property(readonly) id<MTLCommandQueue> recordingQueue;
- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (id)capture:(void (^)(void))encode;   // 2026-11-04 (review 9's finding 1): `repetitions` gone, the replay periodic
- (void)receive:(id<MTLBuffer>)operand;
- (void)publish:(id<MTLBuffer>)operand;
- (uint32_t)levels:(id)capture;
// its own values in one fixed order, their levels offset by `level`: its renamed intermediates, its passes' data, its
// counters, and every buffer of the delegate's a recorded command binds (D.3.2, R9); values NULL counts them
- (NSUInteger)values:(id)capture level:(uint32_t)level into:(struct MetalValue *)values;
// after the plan, per capture: execution t reads the operand at offsets[t % copies] of `buffer` and writes it at
// offsets[(t + ahead) % copies]
- (void)place:(id)capture operand:(id<MTLBuffer>)operand in:(id<MTLBuffer>)buffer offsets:(const NSUInteger *)offsets
    copies:(NSUInteger)copies ahead:(NSUInteger)ahead;
- (void)place:(id)capture value:(NSUInteger)index in:(id<MTLBuffer>)buffer offset:(NSUInteger)offset;
// the program's step word, a uint32 host input of `copies` copies: execution t's, t itself, at offsets[t % copies] of
// `buffer`, read by each later pass of repetition t (R8); a capture with a later pass and no step word raises at replay:
- (void)place:(id)capture step:(id<MTLBuffer>)buffer offsets:(const NSUInteger *)offsets copies:(NSUInteger)copies;
- (void)received:(id)capture operand:(id<MTLBuffer>)operand remote:(struct MetalRemote)remote;
- (void)published:(id)capture operand:(id<MTLBuffer>)operand supplied:(NSUInteger)offset;
- (void)state:(id)capture buffer:(id<MTLBuffer>)state status:(NSUInteger)offset;
// the capture's commands bound and its indirect command buffers built; each invariant of the delegate's written once
// into its placed pages, one copy from the delegate's buffer committed to `queue`, its completion waited for at most
// METAL_PLATFORM_NS (raises past it or on an error, R9)
- (struct MetalReplay)replay:(id)capture;
@end
void MetalReplayDestroy(struct MetalReplay *);
// a recording with no later pass; raises on a range that holds one
void MetalReplayExecute(id<MTL4ComputeCommandEncoder> encoder, struct MetalReplay *, NSRange range);
void MetalReplayOptimize(id<MTL4ComputeCommandEncoder> encoder, struct MetalReplay *, NSRange range);
NSIndexSet *MetalReplayPublishing(struct MetalReplay *);
// the waits of repetitions [range): pass p (from 0 in this call), repetition t, one a received operand it reads,
// {its count word of copy t mod u, (t / u + 1) x P, base + p + 1}, in p order, into `waits`; returns the passes,
// *written the waits; waits NULL counts (the room a second call takes); raises where `room` is short
NSUInteger MetalReplayWaits(struct MetalReplay *, NSRange range, uint64_t base, struct MetalWait *waits, NSUInteger room,
    NSUInteger *written);
// 2026-11-04 (review 10's finding 1): in place of the line above, one repetition's waits, the waiter computing
// repetition t's: struct MetalWait { uint64_t count; uint32_t pages, copies, pass, padding; } (mesh-metal.h struct
// mesh_metal_wait), and
// NSUInteger MetalReplayWaits(struct MetalReplay *, struct MetalWait *waits, NSUInteger room, NSUInteger *written);
// (2026-11-04, review 10's finding 9: MetalEncoder exported, R7's one encoder)
// id<MTL4ComputeCommandEncoder> MetalEncoder(id<MTL4CommandBuffer> command);
// repetitions [range) committed to `queue` (the replay's residency set the caller's, on the queue), every command
// buffer from `allocator` and committed with `options`: the commands through each first pass's compaction in one
// command buffer, committed; then [queue waitForEvent:event value:base + p + 1]; then that pass's later form and the
// commands after it in the next; returns the passes
NSUInteger MetalReplayEncode(id<MTL4CommandQueue> queue, id<MTL4CommandAllocator> allocator, MTL4CommitOptions *options,
    struct MetalReplay *, NSRange range, id<MTLSharedEvent> event, uint64_t base);
```

`struct MetalValue`, `struct MetalRemote`, `struct MetalReplay`, METAL_RECORDING_BUFFERS and METAL_RECORDING_COMMANDS
are unchanged (`struct MetalWait` changed with review 10's finding 1, above). The replay is periodic (review 9's
finding 1): replay: builds L repetitions, L the least common multiple of the copies a capture binds, 4u of each
received operand a first pass reads and the step word's copies; every range is in the run's numbering, repetition t
from slot t mod L. Gone: `place:operand:...stride:output:`, MetalReplayEncode's MTLCommandBuffer
form and its waits arguments, MetalReplayExecute's MTLComputeCommandEncoder form, MetalReplayOptimize's blit form.

### The program's copies and events (tools/e2b_replay.m, owner 7; for the reviewers)

- k = 2, the host's lag, declared by the program (D.3.3: "With the host writing step t + 1 while the GPU runs step t,
  the lag is 2"). Every per-step host input (the manifest's host inputs, the token, the step word; 2026-11-04, review
  10's finding 8: not the index commands' outputs, which the GPU writes in queue order, one copy as every other
  GPU-written per-step operand) and every per-chunk prefill input is a value of `lag` k, placed with k copies and bound with
  place:operand:...copies:k ahead: (1 for the token where it is the step's output, else 0).
- Three MTLSharedEvents, one writer each, values only rising: the input event (the host), the step event (the queue's
  signalEvent:value:), the landed event (the client's waiter, mesh-metal.h).

## Decisions this contract takes, and their sources

- **R1 (D.3.5): a received page is named once per landing.** link_receive_drain posts a receive record for its round
  only where the record's `landings` has reached that round (`f->receives[at.at].landings >= at.round`), and the
  posting stops at the first record that has not, in order: so a page is named by at most one posted receive at any
  moment, and enters its list again only once its last landing has landed (D.2, "How many pages are in flight at
  once"; D.3.5). link_prepare's note relying on the provider copying a request at its post, and the open item on it
  (design/disjoint-pages-revision-coordination.md, "Open at the end"), go (D.4). design/prepared-machine.md#M08 is
  rewritten to the record as it is (struct prepared_receive: its request and span, its phase word and field, its count,
  `landings`, `pages`, `copy`; one record a template entry, posted once for each execution of its copy, and again only
  once its last posting has landed).
- **R2 (D.3.6): a plan stands only while its map does.** plan_stale (above) on the bridge parent's event-loop thread,
  from map_put and map_adopt wherever `map_serial` moved, after the write; it reads `map_serial` itself (the parent is
  the map's one writer); a seq_cst fence after the serial's move and before plan_cancel reads the plan's `executions`
  (R3's pairing). plan_cancel's rules stay: its single thread with the release of retired claims
  (mesh_retired_release), and its whole-plan re-read. The cancellation of the lost peer's receives at the cable's two
  ends stays. Source: D.3.6; RFC 2328 section 16 (routes from one version of the database).
- **R3 (D.3.6, the start's window): a start after the cancel reads ESTALE.** plan_stale's cancellation is computed from
  the plan's `executions` as it reads then; a start that raises `executions` after it would leave the new executions'
  receives uncancelled, and their readers would end at their deadline (ETIMEDOUT), not by an error at once. So
  mesh_transfers_start, after its raise: a seq_cst fence, then `map_serial` read; where it differs from the plan's
  serial, ESTALE. With the bridge's move, fence and read of `executions`, at least one side sees the other: either the
  cancellation covers the raised executions, or the start returns ESTALE and the program runs none of them. It is the
  rule mesh_ports_loaded already applies before its wait (its ESTALE), applied after the raise; the pairing is D10's
  announcement, fence and re-read (K. Fraser, Practical lock-freedom, 2004), kept in force; nobody waits. It also closes
  the previous contract's open item that a cancellation's final value is computed from `executions` as it reads then.
- **R4 (D.3.3): host inputs in copies.** `lag` on struct mesh_value (above). Source: D.3.3; M. Lam, Software pipelining,
  PLDI 1988 (modulo variable expansion: a value live across k iterations gets k copies).
- **R5 (D.3.3, D.3.4): the host's loop, one step at a time.** For each step t, on the program's one thread: the step
  event waited for at t - k + 1 (step t - k done; none for t < k), bounded (waitUntilSignaledValue:timeoutMS:,
  MESH_PLATFORM_NS x k; 2026-11-04, review 9's finding 9: MESH_PLATFORM_NS x (the passes of the steps it covers + 1
  each), the waiter's own bound) and ending as ETIMEDOUT; allocator t mod k reset; step t - k's host-read outputs read (its
  token, from its copy); step t's inputs written into copy t mod k (the step word t among them); the input event set
  to t + 1; then [queue waitForEvent:input value:t + 1]; step t's commands committed (the index commands' repetition t,
  then MetalReplayEncode of repetition t, its base the passes before it); then [queue signalEvent:step value:t + 1].
  A step is committed only once its inputs are written and its input event set, so no queue wait the program commits
  outlives the host: on a failure or a stop (2026-11-04, review 9's finding 3: or a failure the status or the landed
  event shows before a step is committed) nothing more is committed, the waiter is stopped (mesh_metal_waits_stop),
  and every committed later pass is ended by the waiter's `final` as now. The run's end: the step event waited for at
  `steps`, bounded, then the last k steps' outputs read. The prefill has the same loop over its chunks, with no later
  pass. Source: D.3.3 (the host writes step t once the queue has signalled step t - k, then sets its input event; the
  readers committed after the queue's wait; the queue signals each step's end; one writer an event); D.3.4
  (MTL4CommandQueue.h:266 signalEvent:value:, :274 waitForEvent:value:); MTL4CommandAllocator.h:49-50 (an allocator
  reset once the command buffers from it are complete, which step t - k's signal says).
- **R6 (D.3.3): the index commands run in their step.** The index commands (e2b's phase 0, the hoisted preparations
  that read the step's host inputs) can no longer run every step ahead in the preparation, since each step's inputs
  exist only in their copy from step t - k's end; repetition t of them runs in step t, after the input wait and before
  step t's other commands, and their outputs are values of `lag` k like the inputs they are computed from. Source:
  D.3.3 ("Nothing the run binds grows with its steps"). (2026-11-04, review 10's finding 8: their outputs are one copy,
  not `lag` k: `lag` is for a producer ordered by an event, and these are written by GPU commands in queue order, which
  R7's barrier orders after every earlier read, as for every other GPU-written per-step operand.)
- **R7 (D.3.4): order between encoders.** MTL4 tracks no hazard between encoders. Every MTL4 compute encoder this code
  makes begins with barrierAfterQueueStages:(MTLStageDispatch | MTLStageBlit) beforeStages:(MTLStageDispatch |
  MTLStageBlit) visibilityOptions:MTL4VisibilityOptionDevice (MTL4CommandEncoder.h:50-72: the encoder's work waits for
  all work before it on the queue), in place of the Metal 3 path's fence between encoders; within an encoder the
  recorded commands' own barriers order them, as before. One form everywhere.
- **R8 (D.3.4, C3.1): the later pass's execution number.** MTL4 has no inline bytes (MTL4ArgumentTable.h: setAddress,
  setResource, setTexture, setSamplerState), so the later form's repetition t, which it writes as ~t into the status's
  `first`, comes from memory. A table of every repetition's number would grow with the steps (D.3.7); the number is a
  per-step host input instead, the step word (D.3.3's "the schedule's per-step values"): copy t mod k holds t, written
  by the host with step t's inputs (R5), bound to the later pass by place:step:. Every other binding of a later pass is
  an argument table's address (its buffer's gpuAddress plus its offset).
- **R9 (D.3.2): every buffer the capture binds is a value of the plan.** Each buffer of the delegate's that a recorded
  command binds (not a program operand, a value of the capture, the state buffer or a value word) becomes one capture
  value over its whole length, reported by values:level:into:. After the levels: METAL_INVARIANT where no command of a
  repetition writes it, or where a repetition's first command to bind it reads it; else METAL_EXECUTION at the levels
  that bind it. At replay:, each such invariant is written once into its placed pages by its producer: one copy from the
  delegate's buffer (the content the delegate loaded), committed to the recorder's MTL4 queue, its completion waited for
  at most METAL_PLATFORM_NS. After replay:, no recorded command, later pass or residency set names a delegate buffer;
  MetalNativeSubstitute binds the placed weights. Execution values of the delegate's are written by their producers in
  each execution and copied by nobody. Source: D.3.2 (C3.7 allows the copy: the producer writing its own value).
- **R10 (D.3.4): completions.** Every commit this code makes carries MTL4CommitOptions with a feedback handler; an
  error is MTL4CommitFeedback's `error`; GPU time is from its GPUStartTime and GPUEndTime. A completion waited for is
  either the queue's signalEvent:value: and a bounded waitUntilSignaledValue:timeoutMS:, or a semaphore the feedback
  handler signals, waited for with a timeout. Source: D.3.4.
- **R11 (D.3.4): the waits before the encoding.** mesh_metal_waits_start copies every wait at its start (mesh-metal.m),
  while the decode is committed a step at a time (R5), so the waits are written for the whole run before the waiter
  starts, by MetalReplayWaits, and MetalReplayEncode writes none. (2026-11-04, review 10's finding 1: one repetition's
  waits, not the run's, {the count word of copy 0, P, u, the pass}, from which the waiter computes repetition t's
  {count + 8 (t mod u), (t / u + 1) x P, t x passes + pass + 1}: a table of the run's waits was the trip-count
  unrolling C1 rejected; MetalReplayEncode's base for step t is t x passes.)
- **R12 (D.3.4): what stays Metal 3.** recordingQueue and the delegate's command buffers during a capture
  (MetalQueueRecorder, MetalCommandRecorder, their bounded completions, MetalCompletion and MetalCompleted at the
  delegate's commits). Everything else this code commits is MTL4: the decode, the prefill, the preparation (the
  indirect command buffers' optimization), the profile, the recorder's own copies (MetalSpecializeScalars' read of the
  delegate's scalars; R9's copies), tools/native_bench.m and tools/primitive_constants.m. mesh user/bench/mps.m, an MPS
  benchmark no Makefile builds, is outside D.4 and is reported, not changed.

## D.3 and D.4, item by item

| item | owners |
|---|---|
| D.3.2 every value placed, the delegate's included; R9 | 4, 7 |
| D.3.3 host inputs in copies ordered by events; R4, R5, R6, R8 | 2, 3, 7 |
| D.3.4 Metal 4 submission; R7, R8, R10, R11, R12 | 3, 5, 7 |
| D.3.5 a received page named once per landing; R1 | 1, coordinator (the open item) |
| D.3.6 a plan stands only while its map does; R2, R3 | 1 |
| D.3.7 what judges it (as it applies) | reviews 8, 9, 10; coordinator (the builds) |
| the rulings' count audit: counts assumed rather than reported, OS single-casing | 6 |
| D.4 mesh-flow.c: a receive record posted while its earlier posting is outstanding; link_prepare's note | 1 |
| D.4 the coordination file's open item on the provider's copy of a request | coordinator |
| D.4 e2b_replay.m commit_bounded and its Metal 3 command buffers; native_bench.m's | 7, 5 |
| D.4 metal_recording.m MetalCompletion and MetalCompleted outside the capture, line 492, the MTLCommandBuffer of MetalReplayRecorded and MetalReplayEncode | 3 |
| D.4 the per-step instances of the host inputs and their bindings at t x stride | 3, 7 |
| D.4 the delegate's buffers bound as captured, and MetalNativeSubstitute's reads of the weights from them | 4 |

## Record

### Sub-assignments

1. mesh-flow.c, mesh-records.c, mesh-bridge.h, mesh-dataflow.c's mesh_transfers_start, prepared-machine.md M08: R1,
   R2, R3.
2. mesh-collective.{h,c}: `lag` (R4); the files' comments in D.2's terms.
3. metal_recording.{h,m}, first: the header; MTL4 throughout the recorder (R7, R8, R10, R11, R12); place:operand:
   copies and ahead, place:step:.
4. metal_recording.m, second: the delegate's buffers as values and their copies at the bind (R9); the file read whole
   against the four rules and the rulings.
5. tools/native_bench.m, tools/primitive_constants.m: MTL4.
6. The count audit: the bridge's current files read for any count assumed rather than reported and any OS
   single-cased, fixed within this unit's bridge files, lifecycle findings reported; every name of or reasoning from the
   bridge's private interface outside its own files, in both repos, listed.
7. tools/e2b_replay.m: MTL4 throughout; host inputs, prefill inputs, the token, the index commands' outputs and the step
   word in k copies; the events and the host's loop (R5, R6); the bind of every value.
8, 9, 10. Reviews: written findings, routed.

### Commits and routing

- Contract: 79ed346a (this file, INDEX.md). 51523f66 (the coordinator, D.4): design/disjoint-pages-revision-coordination.md's
  open item on the provider's copy of a request and its stale-M08 sentence gone; the cancelled-field item kept.
- 2 (mesh-collective.{h,c}): 33a1ea03 (the header: `lag` in place of `padding`, the Values text's In and Out), bee8b2af
  (validation, EINVAL before anything is written: `lag` on a value not MESH_EXECUTION, wire, state, or named by a
  declared transfer; rows count x P counted before the colouring; plan_place's k items a lag value, each interfering
  with every item, its own copies included, in plan_interferes; copy c's rows bound to copy c's item; out `count` k,
  `stride` P; the files' text in D.2's terms). Its notes: mesh.h struct mesh_target's field `stream` (a supplied word's
  offset) and mesh-dataflow.h:89 ("a pair's cables summed") name transport outside the bridge, in files outside its
  set (routed below, with the count audit's list).
- 1 (mesh-flow.c, mesh-records.c, mesh-bridge.h, mesh-dataflow.c's start, mesh-call.h's text of it, prepared-machine.md
  M08): 1258766d (R1: link_receive_drain posts a record for its round only where `landings >= at.round`, stopping in
  order at the first that has not; the notes relying on the provider's copy of a request gone), 7968dc45 (M08 rewritten
  to struct prepared_receive as it is), 2f349ac4 (plan_stale declared), 35ce58a5 (plan_stale; map_put and map_adopt call
  it where `map_serial` moved, after the closing notify; every caller of both shown on the parent's event-loop thread,
  in plan_cancel's note, at plan_stale and in clients_turn's note), 1a8b10e8 (R3 in mesh_transfers_start, after the raise
  and its wake; mesh-call.h's returns line: every error with `executions` not raised but the ESTALE after the raise).
  mesh-call.h's comment was given to 1 by the coordinator. Its differences, accepted: (1) plan_cancel takes the
  generation plan_stale judged stale (0 from streams_cancel and streams_let_go: whichever plan is written), so a plan
  written between plan_stale's check and plan_cancel's first read, whose map stands, is not cancelled (a spurious
  ENOLINK); (2) plan_stale's seq_cst fence is its first statement, before it reads `hops` (still after the serial's move
  and before `executions` is read): a plan whose `hops` was stored after its last map read and just before the move is
  then seen by the bridge or its start sees the move; (3) plan_stale reads the plan whole (`hops`, `plan`, the serial,
  an acquire fence, both again), as plan_cancel and link_prepare do.
- 3 (metal_recording.{h,m}, first): 5e6512ba (the header as the contract writes it), bf5265b4 (the MTL4 queue of the
  real device, taken from the proxy's Metal 3 queue's device; MetalEncoder, each MTL4 compute encoder beginning with R7's
  barrier; MetalCommitted, a commit with a feedback handler waited for at most METAL_PLATFORM_NS, its error the
  feedback's, the handler holding what the GPU uses until it runs; MetalSpecializeScalars' copy in MTL4 with its own
  allocator and residency set; MetalReplayExecute and MetalReplayOptimize in the MTL4 form alone; MetalReplayEncode
  committing through each compaction, the queue's waitForEvent, then the later form and what follows; the later pass
  bound through one argument table, made at replay:, which the encoder snapshots at each dispatch
  (MTL4ComputeCommandEncoder.h:566-567), in place of the fence and setBytes; MetalReplayWaits; the step word
  (place:step:, a raise at replay: without one); place:operand: copies and ahead, the stride form gone). Its choices,
  accepted: a binding with write access takes the write copy (t + ahead) mod copies, a read-only one the read copy (the
  reflection cannot tell a writer that also reads), and place:operand: raises where ahead is a multiple of copies; the
  later form's kernel text unchanged (only its binding moved); an empty range commits nothing. Its notes for 7 relayed.
- 5 (tools/native_bench.m, tools/primitive_constants.m): 7f7d9243, c00f2020 (MTL4 queues, allocators reset after the
  feedback, argument tables by gpuAddress, residency sets, encoder barriers where one dispatch follows another,
  feedback-timed GPU time, waits bounded at 5 s, `setBytes` replaced by a buffer). Its departures, accepted with their
  evidence: primitive_constants.m's `encoded`, `resource_barrier` and `scope_barrier` probes removed, having no MTL4
  counterpart (MTL4CommandBuffer.h:130, no dispatch type; MTL4CommandEncoder.h:57-58, memory barriers are the Metal 3
  encoders'; its only barriers :70, :91, :113 and fences); R7's queue barrier left out of primitive_constants.m's
  encoders, whose probes read nothing an earlier command buffer wrote and whose barrier would fall inside the measured
  time (native_bench.m's encoders take it: a split kernel's next command buffer reads the counters the last restored);
  a timed-out wait ends either tool with exit status 1, since its allocator cannot be reset before its command buffer
  completes (MTL4CommandAllocator.h:48-50). Its notes: Makefile:99 and the tool's head comment cite a docs/kernels.md
  section that does not exist; the probes `packet` and `scoped` spin by design (they measure spins), unchanged.
- 2 again: e8ea7628 (struct mesh_target's field `stream` renamed `word`, the region offset of a source range's supplied
  word, in mesh.h, mesh-collective.c and mesh-metal.m; no other use in either repo), bdb1ce8f (mesh-graph.{h,c}'s text
  in the module's own generic terms: machines, units, rates, a cut's capacity).
- 7 (tools/e2b_replay.m): 9a196d2a (HOST_LAG k = 2; the R10 feedback records, commit_bounded in MTL4, step_wait bounded
  at MESH_PLATFORM_NS x k, encoder_of with R7's barrier, copy_at, host_loop (R5) shared by the prefill and the decode;
  every per-step and per-chunk host input, the token (ahead 1), the index commands' outputs and the step word lag-k
  values, stand-in memory one copy; the residency sets on the MTL4 queue; the preparation's optimization in an MTL4
  encoder; MetalReplayWaits for the whole run before the waiter; each decode step the index repetition t, then
  MetalReplayEncode of [t, t + 1), then signalEvent; E2B_PROFILE by an MTL4CounterHeap, still skipped where the decode
  holds a later pass (that pass would wait past the run's last landed value); `stride`, `instances`, every Metal 3
  command buffer outside the capture and the landed-event wait loop gone), 727b9c2e (tools/prepare_e2b.py, comments:
  "transfer" in place of the verbs word). Its differences, accepted: under RECORD_ONLY the preparation is not committed
  either (nothing committed); the stop's handler also sets a flag the loop reads before each step, and the stop
  sources exist wherever the loops run, a stop then ending ECANCELED; the prefill and the decode share the input and
  step events (chunk c at c + 1, step t at chunks + t + 1), the k allocators and the k command buffers; the run's error
  is its first failure (the loop, the waiter's cause, the feedback, the status); `invocation_gpu_seconds` the span of
  the decode's commits, `command_buffers` their count; a lag-k operand bound by copy in its own phase and copy 0 in
  another; without a decode the index commands run once, in step 0 of the same loop. Its notes: the final feedback
  wait counts passes + 1 commits a MetalReplayEncode call, as the header says; prepare_e2b.py:2384's argparse help
  says "SENDs" (runtime text); the C9 hook named older graph walks in prepare_e2b.py (kv_sections, prepare()), open.
- 4 (metal_recording.m, second: R9): bc906578 (MetalDelegateValues, after MetalMaskReaders and before the level walk:
  each distinct delegate buffer a recorded command or later pass binds one MetalStandIn over its length, its `delegate`
  the buffer, METAL_INVARIANT where nothing writes it or its first binder reads it, else METAL_EXECUTION at its levels;
  MetalDelegateCopies, last in replay:, every invariant copied in one MTL4 command buffer (R7's barrier, copyFromBuffer,
  a residency set of both buffers, MetalCommitted, raising on an error, the deadline or a value past its placed
  buffer); MetalBinding's "as captured" path gone, an unplaced binding raising; the replay's residency set no longer
  the capture's owners and resources (later passes' pipelines added explicitly); useResource:usage: raising),
  96edc20e (a stale comment in MetalMapShape). Its differences, accepted: MetalNativeSubstitute's `extent` reads the
  buffer's length at capture, equal to the value's bytes (the values are made after the passes that rename and inline
  by whether a binding is a buffer); the classification before the level walk (command order and access bits alone);
  useResource: raises (a kernel reading a buffer through an address in another, which the recorder cannot move to
  placed pages; Dawn calls it only with MetalUseArgumentBuffers, off by default:
  .build/Dawn/src/dawn/native/metal/PhysicalDeviceMTL.mm:416, UtilsMetal.mm:957-963); one commit for all copies. Its
  notes: MetalSpecializeScalars' host read of Dawn's uniforms and the zero points runs at capture, before the values;
  a delegate buffer bound by two captures becomes a value of each, placed and copied once per capture (to review 9).
- 6 (the count audit): 2a14d744 (mesh-bridge.h: MESH_LINK_LANES's note stated as a compile-time capacity, which
  link_plan meets with the smaller of the two ends' reported budgets, not as one OS's count; hello_budget(0) 0, not
  MESH_LINK_LANES). Its findings: (1) the note reasoned from macOS 26's report as a fact, the code being the tag's (a
  device reporting more gets the capacity, fewer its own count); (2) the tag's hello_budget(0) (mesh-bridge.h:133,
  76be3217) took a peer stating no budget as one holding MESH_LINK_LANES, an assumed count; only a bridge before
  76be3217 sends 0, every current one 1 + its budget (mesh-ports.c:320): ruled a fix, adapted from the tag and marked
  so (such a peer's LINK now fails EPROTO; no node runs one, the hourly deploying main; a one-line revert restores the
  tag's form if the operator prefers it verbatim). Judged and left as the tag's: mesh-flow.c's EPROTO for a cable past
  the capacity (unreachable: every record's count is plan_lanes(link_plan(..)), and mesh_cables takes the smaller);
  the budget path (mesh-verbs.h max_pairs, CHILD_OPENED, port_heard's minimum, pair_up's ENOSPC), each verbatim; the
  capacities that take the smaller of report and bound (QD, MESH_GID_ENTRIES, MESH_LANE_UNITS, MESH_PORTS); no branch
  on an OS or a device model. Lifecycle files, reported unchanged: mesh-children.c:52-56's note reasons from macOS 26's
  count (the tag's text verbatim; the code takes the report); mesh-lanes.c:19-22's note is stale against link_plan (not
  a count). Its list of names outside the bridge's own files, routed: mesh.h struct mesh_target `stream` (code) and its
  uses in mesh-collective.c and mesh-metal.m, and mesh-graph.{h,c}'s text, to 2; e2b_replay.m's two lines and
  prepare_e2b.py's two uses of the verbs word, to 7; the rest (mesh.h's text, mesh-map.h's records and joins,
  mesh-dataflow.{h,c} and mesh-call.{h,c}: the ports' `prepared`, the cables' join and capacity sum, the start's wait
  over ports) describe or are what D.3.1 replaces, open; the deployment scripts' (mesh_graph.py, mesh_preflight.py,
  mesh_hourly.py, peers.py) are not this unit's to change, open.

### Reviews and where their findings went

- 8 (the mesh side, at 854ad65c; every changed mesh file compiles alone; the lifecycle files and the deployment and
  launch scripts unchanged since 5c54134c; D.3.1's standing items untouched): no defect. R1 sound (each record at most
  one posting outstanding; no deadlock, a blocked record waiting only on its own earlier posting, ahead of it in the
  same order; the k-th SEND still fills the k-th receive; the cancellations still cover records never posted); R2/R3's
  store-buffering pairing holds as written (the serial's seq_cst add, plan_stale's seq_cst fence, then the plan and its
  `executions`; the start's release raise, its seq_cst fence, then `map_serial`), the generation check leaves a plan
  written between the check and the walk alone, and mesh_plan's step 6 rules out a false cancel; every caller of
  map_put and map_adopt on the parent's event-loop thread; the relay case reads ENOLINK; R4 and the count audit's
  fixes right. Findings and where they went: 1 (note, for the operator: D.3.6 cancels a standing plan wherever the
  serial moves, and the serial moves where the map grows (a cable pairing late, a node joining) as well as where it
  loses, ending a running decode at every node; the receives already posted for the round no peer will now send end
  their links at the plan's end, which re-pair, moving the map again, bounded and converging): kept as D.3.6 writes it
  ("Where a node's map serial moves away from a standing plan's serial ... cancels"; D.2: "When the map changes, each
  bridge cancels"), open for the operator (the reviewer's candidate: cancel only where the move removes a cable or a
  node), and the text "the raise left standing is safe" qualified, to 1; its side note (a node with no count of its
  own, a pure source or relay, reads 0, not ENOLINK; every e2b rank receives): open; 2 (mesh.h's region contract did
  not state D.3.6 or R3's ESTALE) to 1; 3 (the record of the count audit left out the bridge's fixed counts no device
  reports: MESH_PROVIDER_RESERVED_QPS 1, MESH_FRAME 4096, IBV_MTU_4096 on port 1, each verbatim from the tag and so by
  the failport's code): recorded here, judged and kept as the tag's; 4 (the previous contract's open item on the
  relay's timeout and the cancellation's final value, closed by R2 and R3, still undated) dated by the coordinator
  (a1d034b4); 5 (mesh_graph_list_schedule's parameters still named `links`, `link`) to 2; 6 (M08's assignment clause
  and the failed-status check; prepared-machine.md:105 and :161 still describe M08's old design) to 1; 7 (plan_cancel's
  generation 0, "whichever plan is written", a second form) to 1 (every caller names its generation); 8 (hello_budget(0)
  0 departs from the tag) recorded with 6's commit above.
  The fixes: 1 2ecb0f8c (plan_cancel acts only where the plan written is the generation it is given; streams_cancel and
  streams_let_go read hdr `plan` at the call and pass it; the 0 case gone), 6379bfc0 (mesh.h's Cancellation paragraph
  and its closing sentence state D.3.6 and R3's ESTALE after the raise), 8547afbd and d8ca39e2 (the start's text in
  mesh-call.h and mesh-dataflow.c: no execution read as landed that has not; the receives set for the raised
  executions end with the plan as any cancelled run's do), b12c49ec (M08's assignment clause as link_prepare does it,
  the failed status failing the link; prepared-machine.md:105 and :161 date-marked as M08 before the page lists); 2
  e2a5214d (mesh_graph_list_schedule's `machines` and `machine`; its one caller passes them by position, unchanged).
  The C9 hook's remark on 2ecb0f8c, that plan_cancel's walk over the hop list belongs in the graph module: ruled it
  stays (a linear pass over this node's entries raising their counts, no graph traversal; the walk predates this unit,
  which changed only its generation check), as plan_walk and mesh_ports_loaded were ruled before.
- 9 (the GPU and program side, at metal-microbench 96edc20e; every file compiles alone; the tree level with origin):
  every Metal 3 hit accounted for (the capture interface alone); allocators reset only after completion, one command
  buffer at a time; every resource resident; the argument table's snapshot; R7's barrier at every encoder (but
  primitive_constants.m's, as accepted); events and feedback placed as D.3.3 and D.3.4 say, one writer each, values
  rising and never repeating across the prefill and the decode; R5's order; MetalReplayWaits and MetalReplayEncode
  agreeing; no binding at t x stride; every value passed to mesh_plan independent of steps and chunks; R9's
  classification, single copies before anything runs, nothing naming a delegate buffer after replay:, the compiled
  models destroyed only after every replay:; found sound. Findings and where they went: 1 (gap: the replays are
  unrolled by the run's length, repetitions x count commands and the later passes' bindings by repetition, though
  every per-repetition binding repeats with the period of the copies: the trip-count unrolling C1 and C3.3 rejected)
  ruled a fix, to 4 (a replay of L repetitions, L the least common multiple of the copies, 4u and u; repetition t at
  slot t mod L) then 7; 2 (gap: a delegate buffer bound by two captures is two values, copied twice, silently diverging
  if one capture writes it; none such in e2b, each stage its own model) to 4 (one value across captures; a raise where
  either writes it); 3 (gap: the host commits steps after a failure it cannot see, against R5's "nothing more is
  committed") to 7 (the status and the landed event read before each step); 4 (gap: primitive_constants.m's `scoped`
  probe spins with no bound) to 5 (bounded, a miss refused); 5 (defect, minor: E2B_DUMP of the token wrote the last
  step's input) to 7; 6 (gap: a lag-k operand bound by copy 0 outside its own phases, a second form; an operand both
  per-step and per-chunk not refused) to 7 (refused); its third part (a binding with write access takes the write copy,
  so a reader declared read-write would read the copy ahead): the stride form's own rule (`output` where the command
  writes), which the runs relied on, kept, open; 7 (note: the host inputs' fill checks fail after the prefill and the
  start) to 7 (before the preparation); 8 (note: the reflection's read-write access classes most delegate buffers
  invariant, so scratch is copied once at replay: and never reused by liveness: sound, conservative) open; 9 (note: the
  step wait's bound shorter than the waiter's) to 7; 10 (note: MetalReplayEncode's command buffers held until main's
  autorelease pool ends) to 7; 11 (note: the program's own native stages encode into the capture interface's Metal 3
  command buffer, inside the capture, nothing executed) accepted, the exception being the capture interface; 12 (note:
  useResource:'s raise unwinds through the delegate's C++ frames) to 4 (raised after encode returns); 13 (notes): no
  MTL4 header says whether an indirect command's setBarrier orders the commands in an MTL4 encoder
  (MTLIndirectCommandEncoder.h:102, no MTL4 text; MTL4CommandEncoder.h:57-58 names encoder barriers within a pass); the
  MTL4 decode before ecb374e0 ran with them, and primitive_constants.m's `drain` measures them: open; R3's ESTALE ends
  e2b through OK() with status 1, an error value: accepted; the Swift engine's Metal 3 command buffers, outside D.4:
  open.
  The fixes: 5 bfc993cf (the `scoped` probe's poll bounded at 2^20 a wait, a miss written to a word the host reads and
  refuses with exit status 1; its kernel text read, not compiled: no offline Metal compiler here); 4 de92170a (the
  header: `- (id)capture:(void (^)(void))encode;` in place of capture:repetitions:), 5b55c443 (MetalReplayPeriod: L the
  least common multiple of every placement's copies, 4u of every received operand a first pass reads and the step
  word's copies; replay: builds L slots of commands, later-pass bindings and segments; the run's command c runs from
  slot command c mod (L x count), split at segment edges and the wrap; Optimize covers each slot once; Waits and Encode
  take any t; activeRepetitions and E2B_OPCUT_AT gone, the latter having keyed a recording by its repetitions, which the
  recorder no longer gets (tools/mesh/run_pair.py:47 still forwards it, harmless, a launcher this unit leaves); the
  period shown for the publishing and later index sets, the first command's barrier and E2B_PROFILE's repetition 1),
  0782af7e (one map across captures from a delegate buffer to its value and whether its capture writes it; a later
  capture binds the earlier's value, lists and copies it not, and raises where either writes it; the order this asks:
  an earlier capture's place:value: before a later's replay:, and its replay: before the later's run, which e2b's phase
  loop keeps), 310ea80b (MetalRefuse: every refusal reachable from a call the delegate makes during a capture (a
  binding index past 31, a recordDispatch exception, an unrecorded command, useResource:) recorded and raised by
  capture: after the delegate's preparation completes; the same commands refused as before, none newly); 7 ad64591e
  (before each step the run's failure read: mesh_metal_status, or the landed event at `final` while a wait of a step not
  yet committed is unmet (`final` is also reached early, legitimately, where every landing comes ahead); after such a
  stop the waiter is not stopped, so its cause, then the status, is the error (ENOLINK after a cancellation, not
  ECANCELED); E2B_DUMP of the token its output copy, steps mod k; a lag-k operand bound by its copies in every capture
  and refused where a capture outside its own phases binds it, and where it is both per-step and per-chunk; every host
  input's fill checked before anything runs; each step wait bounded at MESH_PLATFORM_NS x (the passes of the steps it
  covers + 1 each); an @autoreleasepool a step (MTL4CommandAllocator.h:27, the allocator holding the encoded commands'
  memory, kept until the step ends); capture: without repetitions, every capture released only after every replay:),
  63adb790 (prepare_e2b.py's --publication-ranges help: "ranges"). The C9 hook's remarks on prepare_e2b.py's older
  graph walks (kv_sections, prepare(), at the functions 727b9c2e's comments touched): outside this unit, open.
- 10 (both repos, at mesh e1060870 and metal-microbench 310ea80b; every changed file compiles alone; both trees clean
  and level; the lifecycle files, the deployment and launch scripts and everything D.3.1 leaves standing unchanged since
  the unit's start): the four traces (a host input from the manifest to the step event; a received range from mesh_plan
  through the bridge, the waiter and the later pass; a map change with a start racing it and a relay; a weight buffer
  of the delegate's from its capture to its placed copy) agree at every party; rulings 2 and 3 hold; every wait bounded;
  dated notes at every reversal. Findings and where they went: 1 (gap: the waiter's table is steps x passes x reads,
  each wait affine in t: the trip-count unrolling C1 rejected, moved from the replay to the waiter) ruled a fix: one
  repetition's waits, repetition t's computed by the waiter (struct mesh_metal_wait {count, pages, copies, pass}), to 2
  (mesh-metal.{h,m}), then 4 (MetalWait, MetalReplayWaits), then 7; 2 (gap: the recorder retains every delegate buffer
  until it is released, so the delegate's buffers stay allocated beside their placed copies for the whole run, against
  C3.4) to 4 (released once copied; no capture after the first replay:, the map keyed by address); 3 (defect, minor: an
  EPROTO run exits ETIMEDOUT after the waiter's deadline) to 7; 4 (note: R9's copy is a dead fill where the reflection's
  read-write access classes a buffer invariant that its kernel writes first) open, as review 9's note 8; 5 (note:
  mesh-call.h and mesh-dataflow.c name the bridge's re-pairing of links) to 1; 6 (note: this file's text behind the
  code: plan_cancel's generation, capture: without repetitions, R5's bound, R6, the empty sections; and D.5, added to the
  change request during this unit (4cdff8b3), not recorded) here; 7 (note: the replay's period counts placements a
  capture never binds) to 4; 8 (note: the index commands' outputs are lag k though their producer is the GPU in queue
  order, a second form of the kind) to 7 (one copy) and R6 amended; 9 (note: R7's encoder written twice, MetalEncoder
  and e2b's encoder_of, against C9) to 4 (exported) and 7 (used).
  The fixes so far: 1 0b27dd1a (mesh-call.h's and mesh-dataflow.c's text: the receives end with the plan as any
  cancelled run's do, by the bridge's own means); 2 f7b5f405 (mesh-metal.h: struct mesh_metal_wait {count, pages,
  copies, pass} and mesh_metal_waits_start(..., passes, repetitions, ...), `final` repetitions x passes), 7e5881e4
  (mesh-metal.m: one repetition's waits copied once, each repetition's computed as the waiter walks t; EINVAL beside
  those the coordinator listed where repetitions x passes or the last target would overflow 64 bits); 4 08083277
  (metal_recording.h: struct MetalWait in that layout, MetalReplayWaits of one repetition, MetalEncoder exported,
  capture: raising after the first replay:); 6 (this file): its names and R5, R6, R11 brought to the code with dated
  notes, D.5 recorded.

### Integration

(appended at the end)

### Open at the end

(appended at the end)
