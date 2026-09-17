# E2B latency: the structure actually executed

This is an analysis of engine `fb3bab4` and Mesh runtime `e69761b` (documentation
revision `1766c7e` does not change that runtime). The controlling requirements are
[I25/E1d/E8 and H1–H8](deliverables.md). It follows the actual resident decode
producer/consumer chain, not the deleted executor or a hypothetical CPU consumer.
It adds no benchmark, runtime instrumentation, or new acceptance requirement.

The baseline account below remains tied to those revisions. The
[implementation follow-up](#follow-up-independent-native-queue-progress)
records the subsequent queue separation and the changed native paths.

**Verdict:** complete native recording removes the per-layer host encoding walk.
It does not establish the transport's dependency-depth/line budgets, eliminate
rearm allocation, or establish the E2B latency floor. Several remaining failures
are identifiable from source and generated machine code without execution. The
five-Metal-call count is one component, not the runtime cost of the whole step.

## What the measurements mean

Latency follows the longest causally dependent path. Total FLOPs, bytes, source
lines, function names and the number of small structs do not measure that path.
For every measurement, name both its trigger and first externally useful effect.

| Structural measurement | What it establishes | What must be included |
| --- | --- | --- |
| Dependent-load depth | How many address-dependent memory accesses cannot start together | Ring selection, record lookup, closure context, operand resolution and spills that supply an address |
| Metadata footprint | Which memory locations must become available before the effect | Both reads and writes; hot-field extent and alignment; no relabeling tags or addresses as payload |
| Executed instructions and call boundaries | Work on the selected path, including empty wrappers | Optimized native code, taken branches, ARC/Objective-C calls, stack spills and restores |
| Branches and loop bounds | Work that can postpone the first effect | Distinguish numerical dependencies, native refusal, queue scanning, optional diagnostics and lifecycle predicates |
| Serialization edges | Which events are prerequisites rather than concurrent work | Device barriers, host wakeups, queue order, collective hops and rearm before reuse |
| Runtime allocation and materialization | Work placed between invocations or before consumption | Command/encoder objects, blocks, arrays, payload copies and address reconstruction |
| Scaling with layers, peers, tiles and streams | Whether a fixed-looking helper hides growing work | CPU and GPU costs separately; a bulk API may still walk its resources |

A pass requires satisfying the relevant budget on the complete named path, not
just removing one load in a helper. Structural evidence can establish that work
is absent, or disprove a budget. It cannot convert an unspecified cache miss,
provider call or device fence into a nanosecond guarantee.

The goal's 32-byte record budget and a machine cache line are distinct measures.
`sysctl hw.cachelinesize` reports **128 bytes on both machines**. This is a CPU
property, not a GPU cache-line measurement. A 32-byte aligned record fits within
one such CPU line; several independently allocated records do not thereby share
one line. Likewise, six load instructions need not cause six cache misses.
Reports below name address regions and actual field extents rather than inventing
cache residency from byte counts.

## The execution and data flow

Setup selects the caller's placement, realizes weights and operand addresses,
binds native functions and registers transport storage. Per Mesh frame, the
engine records embedding, the PLE prefix, all numerical layers, direct partial
publication/sum, final projection and the root sampler in one native ICB.
Recording does not execute the tensor functions.

The metadata dependency initially makes the rank's resident command buffer
eligible to commit. Within each layer, every rank computes its full attention,
normalization and residual work, then its configured FFN-column slice. The
existing `mesh_add` kernel first makes that local down-projection partial visible
and publishes its prepared send targets. Dedicated TX threads post those sends.
The kernel then consumes available remote terms and finishes the local sum;
PLE/layer finishing produces the next layer's hidden input.

RX writes the actual received page's mapping/address values into the logical
page entry, then publishes its presence. **This ordinary E2B FFN receive does not
wake a CPU worker or submit another command buffer.** Its GPU consumer is already
resident. The older H audit's generic RX → worker → launch chain is not the
within-layer chain here. Worker launch still matters at step entry; worker rearm
still matters between steps.

Vocabulary projections are tiled and published to rank 0. The existing indexed
sampler consumes those operands. Replicated sampling and its changing policy are
not implemented. Kernel count, vocabulary communication and actual arithmetic
are not reduced merely by recording them together.

For a layer, a useful dependency model is

```
local term_i = FFN_slice_i(attention_i(hidden))
arrival_(i→j) = publication_i + TX_i + wire_(i→j) + RX_j + acquisition_j
next hidden_j = finish_j(sum of local term_j and arriving remote terms)
```

The next layer depends on that completed numerical sum. Sending attention or
normalization results to avoid cheap repeated arithmetic would add another
serial communication edge; I25 says not to do that when recomputation is cheaper.
Removing host encoding affects rearm between steps. Removing a TX/RX dependent
load affects a layer's communication edge. These are different contributions;
one cannot stand in for the other.

## Rearm: what five Metal calls left out

[Engine source](../../../metal-microbench/matrix_operations.swift),
[resident rearm](../swift/Mesh.swift), and optimized Swift SIL/ARM64 distinguish
three paths:

1. Mesh's launch thunk loads `call.index`, loads the prepared command-buffer
   object and tail-calls `commit`: **four instructions, two load instructions,
   no stack frame**. This is the thunk alone, not worker polling or Metal's commit.
2. Mesh rearm still calls `commandBufferWithUnretainedReferences`, reads the frame
   completion callback, bridges it through `_Block_copy`, registers the handler,
   reads `encoders[index]`, invokes it and replaces the stored command buffer.
   Its compiled frame is **144 bytes**. Array count checks, `swift_beginAccess`,
   retain/release calls and the block copy remain. There is still one new command
   buffer per frame reuse. H8's rearm-allocation deletion is not complete.
3. The engine's ordinary recorded-range invocation uses the five Metal methods
   previously reported. Its actual native path also calls the diagnostic-mode
   predicate, retains the autoreleased encoder and releases it afterward.

The native range invocation, with diagnostics false and successful encoder
creation, has the following **compiled caller-body** account:

| Item | Count |
| --- | ---: |
| ARM64 instructions, including prologue/epilogue but excluding callee bodies | 50 |
| Load instructions, including paired loads and restores | 13 |
| Store instructions, including spills | 7 |
| Metal method calls | 5 |
| Other call boundaries | 3: mode predicate, autorelease retain, object release |
| Conditional branches | 2: diagnostic selection, encoder-result nil trap |
| Stack frame | 112 bytes |
| Captured-context load instructions | 6, reading 11 64-bit fields over offsets 16–111 |
| Layer/dispatch-description traversal on this branch | 0 |

The input and output resource-array lengths are captured values. Their native
base addresses are formed directly; there is no Swift element traversal. But
`useResources` receives arrays of resources, so **five API calls is not evidence
of constant driver work independent of resource count**. Native ICB processing,
Objective-C dispatch and device scheduling remain outside the counted body.
The command/encoder objects still have runtime lifecycles. The assembly also
shows that the optional inspection target is loaded even when inspection is off;
recording has not magically removed all metadata from the invocation.

The [before/after count](../../../metal-microbench/docs/async_collectives.md#whole-step-native-recording)
remains useful: for the stated B=1 native E2B configuration, 251/250 compute
encoders become one per rank, and the 35-entry layer walk disappears. It is a
structural reduction in rearm. The count does not justify declaring rearm free,
the entire runtime constant-work, or H7/E8 passed.

## TX: successful post versus finding the event

In [`link_send_progress` / `link_publications`](../rdma/mesh-flow.c), the thread
holds the SEND-array base and the reader header in registers across iterations.
For a nonempty event and a successful first post, the compiled path from the
event's nonempty check through the native `post` call is:

```
clear event slot; reload/increment reader position
load indexed SEND extent and chunk count
load SEND span address and tag row; write the wire tag
load prepared queue-pair and native post target; call post
```

That selected in-library path is **30 instructions, six load instructions and
four store instructions**, including the invocation spill. It has two conditional
branches before posting: nonempty event and nonempty SEND range. There is no
pending-range enqueue/reload on this successful first-post path.

The SEND fields read before the native call occupy offsets **0–47** of its
128-byte-aligned record: **two 32-byte budget units, one reported CPU line**.
The complete native-containing allocation is 256 bytes; the provider's own WR,
SGE, queue-pair and doorbell accesses are not included in the 30 instructions.
After the event integer is available, this part has one indexed SEND-record
load stage; it no longer chases a queue object to discover its post function.

However, discovering that integer still follows

```
reader.inputs[cursor] → input.slots[position & mask] → SEND[event]
```

There are **three serial load stages starting at the selected input descriptor**.
The input descriptor, slot, SEND header, written tag and spill are separate
address regions. The reader also probes up to P streams. P is a realized-plan
quantity; this analysis does not invent P=1 from the fact there is one TX thread.
The whole TX path therefore has not met H1/H7's one-record/one-line requirement.

Before scanning new publications, the TX thread calls `mesh_send_progress`.
That polls each QP and drains pending native-refusal retries. Reference-count
retirement has been moved off this thread, but queue polling/retry work remains
before a newly noticed publication. The structural delay includes preceding CQ
polls, accepted retry posts and stream probes; it is not bounded by the short
first-post body alone. An empty wrapper count cannot establish immediate drainage.

## RX and the resident consumer

For the common final FFN chunk with no forwarding bindings and no host notices,
the compiled path from CQ poll return to the presence store is **33 instructions,
12 load instructions and four stores**. It still contains six conditional
branches: poll error, empty CQ, completion status, forwarding count, final-row
identity and notice count. Some are native-result checks; the others must not be
hidden by saying “the path has no guards.”

The essential dependency chain is

```
wc.wr_id → posted-buffer tag → records[tag] → logical page-entry stores
```

After `wr_id` is in a register, there are **two successive metadata-load stages**
before the logical entry is known; H7 allows one. The cached physical receive
values beside the tag remove affine reconstruction, but do not remove the
second logical-record lookup. Before the presence store, RX writes three fields:
`mapping`, host `address`, and `device` address. The path also reloads its page
array base from the stack. At least the CQ slot, posted-buffer header, logical
record, receive descriptor, destination page entry and stack are involved; these
are not one metadata line. Both 32-byte receive records have useful compactness,
but that does not collapse their serial dependency.

The resident shader loads its source-page addresses before polling, checks each
incoming presence word and selects an unused available peer. After selection,
`mesh::contiguous` still loads `pages[k].address` before reading the payload.
Thus there is **at least one remaining page-address load before the consumer's
first payload load**, including when the whole FFN partial fits one page. This
fails H6's zero-runtime-page-table-read target. That load is a memory-resolution
cost, not numerical work. Recording commands does not change it.

The full GPU consumer also retains visibility fences, peer selection and the
summation. The E1d resident presence dependency is explicit; it must not be
confused with a library-inserted host barrier. Nor does that permission make every
lookup around it free. The shader is unchanged by `fb3bab4`.

RX posts available replacement receives after processing a completion. It drains
`receive.pages[head]` through `requests[page]`, and availability is supplied by
retirement. It does not re-post the completed logical record directly. That is a
second unresolved H3/H6 difference. The post loop can precede polling the next
QP; preposting and queue-drain order must be counted as well as the presence store.

## GPU publication and numerical serialization

For B=1 E2B, a down-projection partial is 1536 FP16 values. Before publication the
supplied sum kernel touches **768 32-bit words with a coherent read and write**,
then executes a device-memory threadgroup barrier. Publication reads its
sequence, walks its declared targets and advances the target positions. Each
`push` has two system-scope fences; publication adds its presence store and
another fence. These are actual operations in the latency path, even though
there is no extra host command-buffer submission.

This analysis does not assume those coherence operations may be dropped without
changing producer stores. It identifies where the cost is, so compatibility with
the visibility contract can be addressed at the actual producer/transport
boundary rather than concealed in a claim of “zero copy.”

The native command barriers preserve the previous numerical ordering: projection
before normalization/use, KV write before attention, FFN down projection before
publication/sum, sum before finishing, and layer output before the next layer's
input. They replace boundaries between prior separate compute encoders. The
source still contains device barriers and resident polling; it does not have
“zero synchronization instructions.” A serial edge should be retained only for
a real data/visibility dependency or explicitly requested instrumentation.

## Queue work before the counted path

The short TX/RX instruction counts start after discovery. The user's latency
requirement starts when work becomes available. These boundaries differ in the
current implementation. The following loops are in
[`mesh-flow.c`](../rdma/mesh-flow.c); the stream reader is in
[`mesh.h`](../rdma/mesh.h).

| Newly available work | Work that can precede its inspection in the owning thread | Structural dependence |
| --- | --- | --- |
| Published SEND event | `mesh_send_progress` polls QPs and drains each QP's pending SEND ranges before `link_publications` | QP count and the number of accepted pending chunks; retries also read/write their range ring |
| Another published SEND event | `link_publications` finishes the selected event's ranges and chunks before taking another event | The selected event's native request count; one event can represent multiple requests |
| SEND CQ completion | `link_publications` keeps taking events until its reader returns empty, unless a native refusal invokes progress sooner | Accepted publication traffic can postpone the next regular CQ sweep |
| RX CQ completion on the next QP | `link_receive_post` drains returned pages for the current QP before the next poll | Returned-page count and successful native reposts |
| Event in a later publication stream | `mesh_event_take` visits preceding stream descriptors and their current slots | Up to P probes for one call to the reader |
| New worker arrival during rearm | `mesh_call_progress` drains its return loop, including native command-buffer creation and encoding, before inspecting arrivals again | Completed-call backlog and actual rearm work |

The finite configured storage bounds how much outstanding work can exist. That
does not establish a constant amount of work ahead of a particular event, and a
loop reading a concurrently advancing producer can receive more work while it
drains. None of the short-path counts is a bound on these preceding loops.
For example, a pending range is removed, advanced by one chunk and appended again
until exhausted or refused. The TX thread can therefore post every remaining
chunk of earlier pending ranges before inspecting a newly published value.

This is observable control flow without a timer: availability of event B does
not cause its inspection until the current loop over A finishes. A
`wait`/`synchronize` name search misses this serialization completely. Native
refusal handling is necessary; choosing to finish an entire software batch before
looking at independent work is a separate scheduling decision.

Dedicated threads remove competition with numerical work on those threads. They
do not remove competition among the thread's own queues. Moving retirement off
TX/RX removed its instructions there. It did not remove these loops. Conversely,
simply swapping the publication and completion sweeps would change who is delayed
without eliminating the batch-dependent delay. A replacement must account for
every queue's next useful operation, including receive replenishment.

## How the paths compose across 35 layers

The actual caller is [`bindMeshDecodeStep`](../../../metal-microbench/mesh_layer.swift).
Its setup loop declares one full-shaped FFN term per rank per layer and sends it
directly to every other rank. The supplied `mesh_add` consumes each remote term
as it becomes available. It does not route each term through a reduction owner
and then broadcast the result.

For layer l and rank r, let U[l,r] be the time its local FFN term finishes. Let
V[l,s,r] be the time rank r can first read rank s's term. The causal path for that
remote term is

```
U[l,s]
  -> coherent producer publication
  -> TX discovers the event and posts its native requests
  -> link transfers the term
  -> RX discovers completion and publishes its address/presence
  -> resident consumer acquires presence and resolves the address
  -> V[l,s,r]
```

Each arrow must include work already ahead of the event, as well as the event's
own instructions. The first-byte endpoint also does not imply that the whole
term has been consumed. For the current B=1 FFN term, the 3072-byte term fits the
configured transport block, so this example needs one native chunk.

The source's actual sum is sequential over selected peers, while arrivals are
concurrent. Let pi[k] be the peer selected for the k-th addition; it need not be
rank order. Let A[l,r,k] include the selection and addition work after both its
operands are usable. Then the dependency recurrence is

```
R[0] = time the local term has completed its publication path
R[k] = max(R[k-1], V[l,pi[k],r]) + A[l,r,k]    for k = 1 .. N-1
H[l+1,r] = R[N-1] + layer finishing work
U[l+1,r] = H[l+1,r] + next attention/normalization/FFN-slice work
```

This explains both the streaming benefit and the limit: earlier additions can
finish before the last term arrives, but the next layer still needs the
completed numerical sum. There is no host launch between these layer terms.
No overlap credit is assigned to unmeasured work; the recurrence states the
dependencies that would permit overlap. It is an analysis of the supplied
function and its operands, not another runtime component.

For 35 layers and N ranks, excluding vocabulary and step metadata:

| Work | Occurrences per rank per decode step |
| --- | ---: |
| Local FFN publication | 35 |
| Direct outgoing FFN deliveries | 35(N-1) |
| Incoming FFN terms and their address resolution | 35(N-1) |
| Resident sum kernels on the multi-rank path | 35 |
| Host launches triggered by these incoming FFN terms | 0 |
| Whole-step command-buffer creation on frame reuse | 1 |

The sum recurrence, rather than adding all ranks' elapsed work, determines the
critical path. If an extra delay delta is exposed on the last-arriving term of
each layer, it adds **35 delta** to the step. An illustrative 1 microsecond per
layer is 35 microseconds; 10 microseconds is 350 microseconds. These are arithmetic
consequences, not measured costs assigned to a load or a function call. A change
on an input that still arrives before its consumer needs it may have no effect
on that particular step's completion time; it still has to meet H's per-event
budget.

Using only [I25/E8's stated assumptions](deliverables.md), 35 crossings at
12 microseconds consume 420 microseconds of the 500-microsecond allowance.
That leaves 80 microseconds in that favorable accounting, about 2.29 microseconds
per layer if nothing else used it. Step entry, vocabulary, acquisition and rearm
cannot each spend that same remainder. This arithmetic is why a new per-layer
host callback or communication round cannot be dismissed as small. It does not
establish the actual link floor or a timing pass.

The whole-step endpoint also includes two paths absent from an FFN-only count:

- [`MeshDecodeGraph`](../../../metal-microbench/bootstrap.swift) broadcasts step
  metadata from rank 0. A follower derives its token/KV inputs on its CPU worker
  before committing its resident buffer. That is a startup communication and
  worker dependency on each step, separate from the 35 resident layer crossings.
- Vocabulary scopes are sent to rank 0. Its
  [`sampling_values`](../../../metal-microbench/kernels.swift) acquires each
  required scope and still resolves its page address. The ordinary sampler then
  combines its vocabulary-wide partial statistics. Followers do not sample the
  next token themselves. Thus the requested replicated-policy/replicated-sampling
  continuation is still absent, even though FFN exchange is direct all-to-all.

The supplied numerical algorithms determine the sum and sampling dependencies.
Mesh's representation and scheduling determine the extra work on their edges.
I25's instruction to repeat cheap computation avoids introducing another edge;
it does not justify interpreting all collective verbs as the same operation.

## What a subsequent reduction has to demonstrate

Use the same endpoints, caller configuration and native compilation settings
before and after a change. Follow each load's address in generated code: loads
whose addresses depend on an earlier result form serial stages; independent
fields in one known record do not. Include stack traffic and call targets. An
inlined helper can retain every dependent access; an uninlined wrapper can add
an actual call and frame. Neither source spelling establishes the result.

For a moved operation, record its new reader and when it executes. A lookup
removed from RX but required before the consumer's first payload read remains
on the arrival-to-use path. An allocation removed from launch but required
before frame reuse remains on the step recurrence. A lookup resolved only at
setup, with its terminal operand used directly thereafter, is a runtime deletion.
Runtime metadata bytes, payload traffic and serialization depth must be reported
separately, so a smaller allocation cannot substitute for deleting a stage.

The required evidence for this caller is therefore: actual load chains through
first payload use; native instructions and callee boundaries; preceding queue
work and its bounds; numerical versus added serialization edges; allocation and
resource lifetimes through reuse; and their multiplicities over the whole step.
Native provider/Metal method costs and GPU execution times remain unspecified
where their bodies have not been examined. An unspecified cost stays in the
account; it is never entered as zero. Source already establishes the failures
above, so resolving those failures does not depend on a runtime experiment.

## What is established and what remains to implement

| Requirement/property | Current structural verdict |
| --- | --- |
| Per-layer host command/descriptor walk in ordinary native E2B replay | Removed; zero traversal, one recorded-range execution |
| No backend numerical reimplementation for Mesh | Existing engine command descriptions are shared by ordinary and Mesh bindings |
| Prepared native post target and stackless resident launch thunk | Present; narrow properties, not full H passes |
| H1/H7 complete TX path | Unmet: stream selection/probing, extra metadata regions, pre-scan progress work |
| H3/H7 RX path | Unmet: tag → logical-record lookup and metadata writes before presence |
| H6 consumer operand access and receive placement | Unmet: runtime page-address reads and return-order reposting |
| H8 rearm allocation | Unmet: native command-buffer/encoder lifecycle and handler bridging remain |
| E1d communication and policy | Incomplete: rank-0 vocabulary consumption and non-replicated sampler policy |
| E8 nanosecond/step-latency thresholds | Not established by instruction counts or source structure |

The next implementation work is constrained by these findings: remove complete
resolution stages through prepared terminal targets and addresses; remove work
that precedes publication/CQ drainage without a data dependency; account for the
whole rearm lifecycle rather than only its encoder body; and finish E1d's actual
policy/dataflow. Moving a lookup behind a new descriptor or bulk function does
not satisfy any of those items. Any subsequent reduction must recount the same
trigger-to-effect path, including what moved into a callee.

## Follow-up: independent native queue progress

After the queue-backlog analysis, [`mesh-flow.c`](../rdma/mesh-flow.c) separates
native posting and completion consumption. SEND completion polling has its own
thread; receive reposting has its own thread. Their interfaces are the existing
native CQs and returned-page ring. There is no new inter-thread message on either
path. The [source and ownership account](algorithm-sources.md#independent-native-queues)
records all five thread roles, startup and teardown.

The changes to the earlier backlog table are specific:

| Preceding work in the baseline | Current source |
| --- | --- |
| SEND CQ sweep before publication discovery | Absent from the posting thread; CQ polling runs independently |
| Whole refusal queue drained before inspecting a new publication | Deleted; one native retry per QP is followed by another publication inspection |
| Publication stream drained before the next SEND CQ sweep | Deleted; CQ progress is independent of the publication reader |
| Whole RECV repost batch before the next RECV CQ poll | Absent from the CQ reader; reposting runs independently |
| Whole returned-page batch on QP A before reposting QP B | Deleted from runtime; each sweep attempts one prepared receive per QP |
| Stream-selection probes, work within a publication, worker rearm | Remain; not closed by this change |

The posting thread still feeds a selected publication's prepared requests to the
native interface until accepted or refused. A refusal records the unposted
range in the existing queue and leaves independent ranges eligible. Setup now
stores that range's terminal index directly. The successful path no longer loads
its chunk count or maintains a nested chunk cursor. Native-refusal retries remain
useful transport work; they are not an operation-completion wait or an additional
admission policy.

Recounting the same successful first-SEND and final-FFN-RECV paths as above:

| Compiled caller-body property | Baseline | Current |
| --- | ---: | ---: |
| Nonempty SEND event check through first native post: instructions | 30 | 23 |
| Same SEND path: load instructions | 6 | 5 |
| Same SEND path: stores | 4 | 3 |
| Same SEND path: conditional branches | 2 | 2 |
| Positive CQ result through no-forward/no-notice final-row presence: instructions | 33 | 32 |
| Same RECV path: load instructions | 12 | 11 |
| Same RECV path: stores / conditional branches | 4 / 6 | 4 / 6 |
| Posting thread's stack frame | 128 bytes | 128 bytes |
| RECV CQ thread's stack frame | 128 bytes | 144 bytes |

The SEND count includes an unconditional branch on the current path and excludes
provider execution, as did the baseline. Its invocation stays in a register;
the prior invocation spill and chunk-count load are gone. The RX canonical page
base stays in a register instead of reloading from a spill. Its larger frame
holds the local native completion record. The shared completion allocation and
provider pointer are removed, so SEND and RECV CQ readers no longer write
adjacent records in that allocation.

The emitted RX function, including all its branches, shrinks from 216 to 179
instructions. This is code-body size, not instructions executed per completion.
The SEND posting body now includes retry logic previously in a separate helper;
comparing its body size alone would omit that old helper. The new SEND CQ and
receive-posting bodies contain 61 and 68 instructions, with 112- and 80-byte
frames. These costs are present in separate running threads, not erased from the
program. Counts include instruction mnemonics and exclude labels, directives and
comments; paired loads/stores count as one instruction.

**Remaining limits:** the complete TX metadata path still has three dependent
stages; RX still has two stages after `wr_id`; GPU consumption still resolves
the page address. QP scans and publication-stream probes remain. All five
threads need CPU scheduling; this change does not prove the nanosecond budgets,
the E2B speedup or full H completion. It removes specific software ordering edges
and reduces instructions on the named first-effect paths. The public ABI,
collective choices, numerical functions and engine operation chain are unchanged.

The change is three net maintained C/header lines and two additional threads per
link; the documentation expansion is separate. `make -C rdma mesh-flow
native-audit` builds the bridge and emits strict-diagnostic native assembly.
The selected current paths are recorded in
`/tmp/mesh-independent-queue-paths.txt`; whole-function counts and frame sizes
are in `/tmp/mesh-drain-before.json` and `/tmp/mesh-drain-after.json`. No workload
or service restart is part of this source/build analysis.

## Evidence and reproducibility

The C and Mesh Swift evidence is emitted by the existing
`make -C rdma native-audit` target. The engine evidence uses:

```
swiftc -O -whole-module-optimization -emit-sil \
  matrix_operations.swift matrix_shaders.swift parameter_configuration.swift \
  -o /tmp/whole-step-matrix.sil
swiftc -O -whole-module-optimization -emit-assembly \
  matrix_operations.swift matrix_shaders.swift parameter_configuration.swift \
  -o /tmp/whole-step-matrix.s
```

The replay count follows `MetalProgram.bind`'s `cfU11_TA`, ordinary branch, and
includes its prologue/epilogue. The TX count follows `_link_send_progress` from
the nonempty-event branch through its first accepted `blr`; the RX count follows
`_link_receive_progress` from CQ return through the no-forward/no-notice final-row
presence store. Callee bodies and alternate paths are explicitly excluded from
those scoped instruction totals. Paired loads/stores count as one instruction;
dependent stages are counted separately by following their address operands.
The selected instruction lists are in `/tmp/whole-step-structural-paths.txt`.

All three engine targets built from `fb3bab4` locally and on the Mini. No model
workload, endpoint timing or runtime trace was run for this account.
