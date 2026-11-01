# Encoder lifetimes follow the page graph

Recorded 2026-11-01, 14:08 PST. Source review and implementation; no numerical
run or measured speedup is reported here.

`8fa0b005` introduced the shared submission dependency compiler.
`ebd8bd24` then gave each group of commands its own Metal encoder and fence.
That made an object boundary stand in for a dependency boundary. A chain with
many small steps paid for many encoders even when all its steps could share
one. All of this construction happened during preparation, but preparation
still consumed host and driver work.

## What the compiler now emits

The page access solve produces the command graph first. Commands with equal
incoming and outgoing edges can share a group. The next pass,
`MetalScheduleScopes` in `rdma/metal-scope-plan.inc`, assigns encoder scopes,
stage barrier masks and the fence checkpoints that leave those scopes.
`MetalScheduleEmit` consumes those decisions. It keeps the encoder open for
the whole scope and retains each needed checkpoint for its actual consumers.
The first scope uses the encoder already opened for declaration capture.

A simple chain with only its final value escaping uses one encoder. Its
internal edges use the required stage barriers; they do not each allocate a
fence. Independent commands can share that encoder when doing so adds no
ordering edge. A fence is created only for an edge crossing scopes or a live
page relation exported to the next prepared function. Repeated waits for the
same fence and stage are encoded once per scope.

The lifetime of an exported fence includes the queue's retained page frontier.
Finishing its local consumer is therefore not enough to recycle it. Producer
kind is now an explicit field: the absence of an optional GPU fence no longer
means that the producer is a host operation.

## Why the scope pass needs the graph

Metal's stage barrier covers a stage prefix and its later stage suffix. A
fence update also covers the preceding commands of its named stages. Neither
operation names just one page or command. These are the semantics stated by
the SDK's `MTL4CommandEncoder.h`.

The pass keeps the maximal preceding commands for each stage. It admits a
barrier only when the new consumer already depends on that entire prefix.
Later commands must also depend on the order imposed by that barrier or fence
wait. It checks future consumers before admitting a producer into a prefix.
For example, if D is independent of A and B reads only A, putting D before A
in one stage would make A's fence cover D too. Splitting only at B is too late.

These checks use the solved graph during preparation. They emit stage masks
and scope numbers, not an execution-time dependency search. Direct predecessor
checks handle ordinary chains without walking their whole history. The pass
does not claim to find the globally fewest possible encoder scopes.

Host event semantics remain a separate constraint. A queue signal covers all
prior queue work, and a queue wait holds subsequent queue work, as stated by
`MTL4CommandQueue.h`. Opening a fresh command buffer before a host producer
cannot make that signal specific to the producer. That redundant pre-producer
cut is removed. This change does not establish exact page-grained GPU-to-host
notification for the existing queue-event interface.

## What the observation budget means

The quoted claim that a GPU wait's deadline ages while preparation runs does
not match the code. Preparation stores a load budget. `mesh_wait_bounded` and
`mesh_step_observe` start their local counters when the numerical leaf first
observes its needed page. The platform's five-second estimate is not a stored
start time. An early rank can exhaust its executing budget while a peer is
still preparing; that is execution skew.

The review did find a separate defect introduced by `bc2f34de`: typed source
effects could receive zero before the old record binder resolved that default.
Zero could then allow a numerical consumer to proceed before its input arrived.
Lowering now resolves the configured budget once, before either consumer sees
it. Diagnostics report the exhausted load count rather than an inferred
elapsed duration. No new peer barrier or runtime timer is added.
