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
receive of it not landed is cancelled: plan_cancel(UINT32_MAX), the plan read whole as plan_cancel reads it, each
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
- (id)capture:(void (^)(void))encode repetitions:(NSUInteger)repetitions;
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
// repetitions [range) committed to `queue` (the replay's residency set the caller's, on the queue), every command
// buffer from `allocator` and committed with `options`: the commands through each first pass's compaction in one
// command buffer, committed; then [queue waitForEvent:event value:base + p + 1]; then that pass's later form and the
// commands after it in the next; returns the passes
NSUInteger MetalReplayEncode(id<MTL4CommandQueue> queue, id<MTL4CommandAllocator> allocator, MTL4CommitOptions *options,
    struct MetalReplay *, NSRange range, id<MTLSharedEvent> event, uint64_t base);
```

`struct MetalValue`, `struct MetalRemote`, `struct MetalWait`, `struct MetalReplay`, METAL_RECORDING_BUFFERS and
METAL_RECORDING_COMMANDS are unchanged. Gone: `place:operand:...stride:output:`, MetalReplayEncode's MTLCommandBuffer
form and its waits arguments, MetalReplayExecute's MTLComputeCommandEncoder form, MetalReplayOptimize's blit form.

### The program's copies and events (tools/e2b_replay.m, owner 7; for the reviewers)

- k = 2, the host's lag, declared by the program (D.3.3: "With the host writing step t + 1 while the GPU runs step t,
  the lag is 2"). Every per-step host input (the manifest's host inputs, the token, the index commands' outputs, the
  step word) and every per-chunk prefill input is a value of `lag` k, placed with k copies and bound with
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
  MESH_PLATFORM_NS x k) and ending as ETIMEDOUT; allocator t mod k reset; step t - k's host-read outputs read (its
  token, from its copy); step t's inputs written into copy t mod k (the step word t among them); the input event set
  to t + 1; then [queue waitForEvent:input value:t + 1]; step t's commands committed (the index commands' repetition t,
  then MetalReplayEncode of repetition t, its base the passes before it); then [queue signalEvent:step value:t + 1].
  A step is committed only once its inputs are written and its input event set, so no queue wait the program commits
  outlives the host: on a failure or a stop nothing more is committed, the waiter is stopped (mesh_metal_waits_stop),
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
  D.3.3 ("Nothing the run binds grows with its steps").
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
  starts, by MetalReplayWaits, and MetalReplayEncode writes none.
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

(appended as the work proceeds)

### Reviews and where their findings went

(appended as the work proceeds)

### Integration

(appended at the end)

### Open at the end

(appended at the end)
