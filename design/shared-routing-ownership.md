# Shared routing ownership and reader groups

Source/literature design, September 13, 2026. Reader groups and the unique-owner
shared routing domain below are implemented. The initial source audit is preserved
to explain the replaced representation; implementation and remaining limits follow
at the end. Compilation is distinct from numerical and performance evidence.

## Two different multiplication problems in current source

`python/mesh/kernels.py:_lower_indexed_add` constructs a candidate input list of
all partial slots for each final destination region. `_indexed_range` avoids a
copied full selector vector, but `mesh_algebra_indexed_range` still allocates
candidate maps, static adjacency and 2C+2 logical metadata rows per consumer with
C possible candidates. D destination regions therefore retain O(D*C) setup and
reader metadata. Each candidate may also appear in each generated kernel's bound
pointer array. Sharing only the ordinal vector does not remove that multiplication.

Separately, `mesh_realize` allocates a source READ plane per reader map. A routing
metadata backing can acquire ordinary numerical readers, selector lifetime readers
and range readers for many segment functions. `MESH_READERS` is 64. A graph can
exhaust those planes long before its operand-byte budget, even for small tensors.
Increasing that constant or copying a routing table per consumer would preserve
the wrong ownership representation.

The numerical allocation is another cost: up to U segment slots each own a
publication allocation per feature stripe. Removing quadratic metadata does not
by itself remove U times that allocation quantum. These issues need separate
source and byte accounting.

## Published building blocks

Blelloch's [Prefix Sums and Their Applications](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/scandal/public/papers/CMU-CS-90-190.html)
provides scan-based compaction and grouping. Blelloch, Heroux and Zagha's
[Segmented Operations for Sparse Matrix Computation on Vector Multiprocessors](https://www.cs.cmu.edu/~scandal/papers/CMU-CS-93-173.html)
uses segmented operations for sparse matrix computation. The relevant stored
representation is a sparse incidence relation, not a dense candidate rectangle.

Google's [Pallas Megablox implementation](https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/main/src/maxtext/kernels/megablox/backend.py)
retains group offsets, group IDs, tile IDs and the numerical active tile count.
Its bounded group metadata maps grid programs to actual groups; unused capacity
need not imply executing every potential tile. Empty groups are visited when
necessary to initialize their required output. This is prior art for a bounded
numerical work domain, not permission to erase a mesh publication boundary.

[Monsoon](https://www.cs.cmu.edu/~18742/papers/Papadopoulos1990.pdf) supplies explicit
operand-store identity and presence matching. The exact reader-group and routing
ownership design below is a proposed mesh extension using those building blocks;
it is not attributed verbatim to these publications.

## 1. Represent the sparse relation once

For each invocation slot and feature stripe, retain one routing domain:

- One canonical source-candidate map table, including original binding input
  positions, physical views and logical source ranges. Numerical kernels bind this
  table once as immutable setup metadata rather than duplicating C arguments per
  destination. Entries address the original registered storage.
- A produced `owner[candidate]` vector containing the exact consuming output-region
  identity, or absent for no consumer in this domain. The compiler supplies the
  key-to-output-region map; the bridge does not interpret destination keys.
- A produced reverse CSR directory: `offsets[consumer]` and `ordinals[edge]`, naming
  the exact candidate ordinals read by each consumer. This is the same relation in
  reverse, retained because source notifications and numerical consumers traverse
  it in opposite directions. Do not infer either direction from arrival order.
- One candidate retirement result per domain occurrence, plus the existing
  produced directory and numerical consumer-completion result identities.
- One fixed consumer-id-to-configured-function-occurrence table in the existing
  executor. A consumer ID is an index, not an application task or future.

For scatter segment partials, each partial belongs to exactly one final output
region in its feature stripe. Thus E, the number of actual source-to-consumer
edges, is at most C. For genuine fanout retain CSR edges in both directions and
account for O(E); the compiler may use unique ownership only when the algebra
proves it. Duplicate numeric updates remain contributions even when their lifetime
hold collapses to one source/consumer edge.

All numerical storage is bounded at setup. When routing is dynamic, capacity is
bounded by update/segment capacity rather than destination-count times candidate
capacity. Index production fills ordinary canonical operands. No host graph
construction follows that production.

## 2. Give a source one routing-domain hold

Reserve one source READ obligation for the routing domain, not one possible
obligation for every destination. Before the route is known, source publication
leaves that hold unread. The owner vector identifies exactly which consumer can
satisfy it. Numerical readiness checks the consumer's reverse-directory range
and its selected source rows. No other source is a numerical dependency.

A source notification first checks its domain/candidate retirement identity. If
already retired for this domain occurrence, it belongs to a subsequent source
occurrence and the old directory must ignore it. Otherwise, an available owner
entry directly names the configured consumer to revisit. A directory publication
visits already-present sources once. Source arrival does not traverse D possible
consumer watches.

Consumer completion publishes its own logical result. Canonical metadata then
retires only the source edges named in that consumer's CSR range; it marks each
candidate's retirement result before releasing its source READ obligation. For
unique ownership, that is one bit per candidate. General fanout uses explicit edge
completion identities, releasing the source after all actual edges complete.
This adds no numerical kernel or per-candidate dispatch.

The directory retains a domain lifetime reader until every source occurrence in
its domain resolves. Ordinary consumers also retain their actual directory reads.
A source can publish its next occurrence before other sources resolve; its prior
retirement identity survives and prevents old-directory reuse. Directory
preparation resets its owned identities only after the preceding directory
lifetime ends. Multiple concurrent domains use distinct bounded value slots.

An absent owner means no consumer, not automatically no producer. If a producer
will publish the candidate occurrence, retire it when it arrives. If the compiler
proves an inactive numerical grid member produces no value for this occurrence,
retain that fact explicitly in the produced active-domain metadata and satisfy
its domain obligation without expecting a fictitious source publication. Mixing
these two cases would reintroduce the late-source/next-occurrence bug.

## 3. Replace fixed per-reader planes with canonical reader groups

For ordinary graph fanout, configuration constructs a source reader group with
one source READ plane and one logical completion-result identity per actual
reader occurrence. Each function input map retains its own member identity.
Function readiness requires source presence and an unsatisfied member, rather
than testing the group's aggregate READ bit as if it belonged to that function.
This prevents a completed function from reissuing while another reader is active.

Completion sets that member's result bit. The existing metadata execution owner
releases the source's group READ obligation once all member results are present.
Source producer preparation resets the group's member results before the next
source publication. The source cannot reset until the aggregate hold is released,
so ordinary member identities do not need occurrence counters. Constants bypass
mutable-source retirement as they already do.

Large groups use ordinary bitmap ranges; completion may first check whether its
bitmap word became full before checking the group. A hierarchy of bitmap summary
results is an optional setup specialization for large groups, not a reason to
introduce decrementing counters or numerical reduction launches.

Keep transport readers explicitly represented. One compute reader-group plane
plus the small transport-reader set replaces unbounded numerical fanout planes.
If transport fanout itself exceeds the remaining direct planes, put those actual
readers in the same completion-group representation instead of raising a magic
limit. The bridge must retain the exact completion member associated with each
posted request, just as it retains transfer indices today.

Shared routing metadata then has O(D+E) actual directory-reader/group membership,
not O(D*C) possible-source obligations. This mechanism also addresses ordinary
expression-DAG fanout; it is not a special case embedded in scatter code.

## 4. Treat physical partial storage as a separate lowering

Intern compiler-internal source bindings and suballocate registered storage with
explicit byte extents and lifetimes. Elide a partial only when every use is
subsumed and no independent publication boundary remains. Packing several values
must not make a ready value wait for unrelated absent data or the slowest sibling
kernel. A single packed slab is not evidence that its members publish independently.

The current canonical output binding is page/quantum based. Finer independent
values require logical byte-region identity separate from the backing page,
including offset/length in views, lifetime ownership and transfer tuples. Existing
MPS/Metal bindings already carry byte offsets, but allocator and publication
coverage must agree before subpage packing is sound. Store those offsets; do not
recover them from adjacent values or use an implicit packing cursor at invocation.

[Apple TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
remains authoritative for SEND/RECV frame constraints. The current implementation
rounds useful payloads to 4 KiB frames and allocates larger publication quanta.
Removing excess allocation can approach that constraint; it cannot justify
claiming arbitrarily small independently transmitted values have zero padding.
Do not share a transfer's writable bytes with an independently active producer
merely to save storage. Distinct concurrently live transfer destinations remain
required.

A compiled bounded active grid can avoid empty segment launches using the retained
active tile IDs/count, analogous to the cited Pallas group metadata. Its canonical
issue/completion semantics must retire inactive input obligations without fake
numerical work. It must also preserve each required output initialization; empty
final destinations still return their base. This is explicit generic grid-domain
lowering, not a second application scheduler.

## Source disposition and implementation sequence

1. `mesh-dataflow.h` input maps retain reader-member identities; `mesh_realize`
   constructs canonical reader groups, replacing plane-per-reader assignment.
   `mesh_issue_index`, completion and producer reset use those same member rows.
   Preserve exact identities in the existing trace API. Establish arbitrary
   numerical fanout before depending on large shared routing directories.
2. `mesh-algebra.m` owns one source-map/pointer binding table per routing domain.
   Kernel setup references it rather than emitting C distinct argument buffers for
   each consumer. `kernels.py` emits owner and reverse-directory operands from the
   retained key-to-consumer map. Generic indexed binding receives that domain and
   its consumer identity; no D-sized duplication of its C candidates.
3. `mesh_execution_add` registers one source-to-domain adjacency per source and
   fixed consumer lookup entries. Existing `mesh_events` performs indexed routing
   and lifetime completion. `mesh_index_ready` consumes CSR-selected maps;
   retirement uses domain/edge result identities. No new polling scheduler exists.
4. Add explicit active-grid ownership and byte-extent allocation/publication in
   their canonical owners. Remove per-empty-slot launches and excess physical
   quanta only when their replaced publication/lifetime contracts are preserved.
5. Extend existing streaming-algebra observations with >64 actual consumers,
   changing routing, empty domains, fanout, delayed unselected producers and early
   source-slot reuse. Inspect allocated logical rows, binding entries, native
   argument bytes, registered bytes, live operand bytes and launch counts. Matched
   numerical/timing evidence remains required; no separate evaluator is proposed.

Acceptance accounting is O(C+D+E) domain metadata plus O(actual graph edges) ordinary
reader completion identities, and operand storage tied to live numerical extents
rather than a dense destination/update rectangle. Those bounds and source
invariants can be audited before running a workload. They do not establish
latency, wire utilization or a no-regression claim without measurements.

## Implemented reader-group specialization

The native realization now surveys numerical inputs, selector/range lifetime
readers, dynamic candidates, exports and outgoing transport obligations before
assigning compute readers. A map retains the original direct-plane path when its
projected source fanout fits MESH_READERS and a common free plane exists. Only
maps touching overflowing rows or without an available common plane use groups.
This is a setup specialization; no invocation chooses a backend or rebuilds
ownership. A small existing gold graph therefore need not allocate group metadata.

A grouped map retains member IDs and per-occurrence offsets. There is one member
per actual source logical row, not one member for a whole multirow operand. Each
source group owns one READ plane, a group-completed logical result, and contiguous
member-result ranges from setup. Constants have absent member identities and
retain constant-read behavior. Normal numerical readers, dynamic selector/range
lifetime readers and export consumers all use the same map operations. Hardware
SEND readers still use their existing explicit direct planes.

Each function requires its own member unsatisfied, so releasing one consumer's
output cannot reissue it while another source reader remains active. Completion
sets the member results and notifies the source rows. The existing metadata owner
releases a source only when all that source's group members are present. Normal
selector numerical membership remains distinct from selector lifetime membership;
empty selection cannot release selector storage before the numerical call ends.

Local producer issue and remote receive preparation reset the source's aggregate
READ bit, while member reset belongs only to the existing metadata event owner.
The bridge has no process-local member list. The group-completed result retains the preceding
completion: when the source publication arrives with this result present and its
aggregate READ bit clear, the existing metadata owner clears the old members and
then the group-completed result before firing new consumers. The group-completed
result was published before the preceding aggregate READ, so independent source
reuse cannot lose this distinction. Export polling returns unavailable during
this pending metadata transition; it does not clear or guess group state itself.

Group storage is configured once and reused. Source-row release removes its group
and logical result ranges; teardown synchronizes with the existing metadata queue
to avoid freeing an in-use group. Program teardown unbinds its normal, dynamic and export map memberships before
freeing their maps. Each storage allocation retains exact source/member pairs;
unbinding removes its pending setup assignments or satisfies its live members.
A member-result allocation is released when its final owning map is unbound.
These ownership counts apply only to setup/teardown memory resources, never to
numerical readiness. Member-index and offset arrays are freed at Program.close,
even when other Programs keep the context attached. No allocation, teardown synchronization
or numerical no-op is added to the invocation path.

`mesh_reader_trace`, `mesh_algebra_trace_input_reader` and
`mesh_algebra_trace_indexed_reader` expose exact source/member/group-completed rows
and aggregate plane. Snapshot flags distinguish source presence, own membership
completion, aggregate source READ and group-completed presence. Direct readers
have absent member/group IDs. These are current identities and state, not timing
reconstructions.

The source ABI audit finds mesh_row_map/mesh_ctx consumers only in
mesh-dataflow.c/.h and mesh-algebra.m/.h. Python holds their opaque context and uses
RowRange rather than a ctypes mesh_row_map. Both native libraries require a
coherent rebuild; shared-header layout is unchanged. Compilation passed. Runtime
fanout/reuse and matched performance evidence remain necessary, particularly for
new grouped domains. Reader groups alone remove the 64-plane numerical ceiling but do not remove
O(D*C) candidate metadata. The shared-domain increment below supplies that
separate lowering; active-grid and finer physical partial storage remain open.


### One owner for member reset

Member reset is performed only by the existing metadata event owner, for local and
remote sources alike. A source producer clears ordinary PRESENT/READ state; the
preceding group-completed result stays present until the event owner clears every
member and then that result last. Grouped numerical readiness and export polling
reject the intervening closed group. An event that began before source reuse can
therefore perform the reset for the newly published source without racing a second
host reset or clearing a newly consumed member. No blocking invocation dispatch or
new epoch field implements this ordering.

Program.close already removes its watches and waits for its numerical executions.
Map unbinding runs under the existing metadata queue before freeing membership
arrays, and no released array remains in a pending setup assignment. Group member
ranges shared by several maps remain until their last owning map is removed;
removed members are marked satisfied while that storage remains. Numerical source
lifetimes continue to use presence bits, not the teardown resource-owner count.


## Implemented shared unique-owner domain

`mesh_algebra_route_create` retains one domain for a feature stripe: the exact
candidate maps, produced owner/ordinal/offset views and strides, configured
consumer function indices, and one constant C-by-3 U64 address/stride table.
`mesh_algebra_route_table` returns that canonical tensor. The CPU address is the
registered extent address plus the view's scalar offset; Metal uses its buffer's
GPU address plus that same offset. Each table entry retains both row and column
strides. No candidate payload is copied.

`mesh_algebra_route_attach` attaches the domain and exact consumer ID to an
existing numerical function. The consumer binds the table once. It retains actual
ordinary metadata reads, including when another indexed descriptor is added to
the function later. Every configured consumer must attach exactly once. Consumer
output aliases with the domain's source or metadata maps are rejected
at setup. This API currently requires one function occurrence per consumer; it
does not infer a consumer from the function's output buffer.

The produced relation has a precise numerical contract: `owners[c]` is the unique
consumer ID, or UINT32_MAX for an unconsumed but still-produced candidate;
`offsets[d]:offsets[d+1]` names exactly that consumer's candidate ordinals. Each
owned candidate occurs exactly once in this inverse relation. Ordinals outside
these intervals are unused capacity. Native preparation checks interval bounds,
ordinal bounds and owner agreement. The compiler supplies the unique partition
and complete inverse relation; this API does not claim arbitrary sparse fanout or
inactive producers. Different domains can retain separate actual reads of the
same source, using canonical reader groups when needed.

`mesh_realize` surveys and binds candidate and domain-metadata lifetime maps once,
through consumer zero's retained domain reference. It does not clone those maps
into other consumers. Per-consumer ordinary metadata reads remain real reader
identities. A domain allocates C retired result rows, D numerical-completed result
rows and one prepared result row, without operand payload allocation for these
logical results. `mesh_execution_route` installs one candidate-to-domain edge per
source row, metadata edges, and one completion-result edge per consumer.
`mesh_execution_add` stores each exact consumer watch only inside its successful
serial installation, so failed allocations cannot leave a dangling watch.

Metadata publication prepares the relation once and makes configured consumers
eligible for normal canonical readiness checks. Already-present unowned sources
retire then; owned consumers read only their CSR-selected maps. A later candidate
publication first checks its retained retirement row, then indexes its actual
owner and revisits that configured consumer. It never walks D possible readers.
Consumer completion publishes its own result, whose event retires only that
consumer's CSR interval. Retirement precedes source READ release. Completion
notices are only wake-ups: the current completed result must be present before
retiring anything. A stale notice after metadata reset therefore cannot retire a
new candidate occurrence. Registration also replays current canonical state so
already-published metadata need not generate another notification.

The shared domain lifetime reader releases only after every candidate has retired
and every consumer has completed, including consumers with empty intervals. An
unowned late source remains an unresolved occurrence until it arrives. An already
retired source may publish a new occurrence while another candidate remains late;
the old retirement fact makes that new arrival irrelevant to the old directory.
Local metadata producer issue resets the C+D+1 result rows, and all domain
metadata maps must be available for the new occurrence before preparation.
Metadata storage consequently cannot be recycled while any old candidate or
numerical consumer still depends on it. Ordinary grouped/direct reader planes
supply this ordering; no phase number or remaining-work counter is introduced.

Metal setup places candidate allocations in the algebra's single queue residency
set. Realization commits that set once. Indirect GPU table loads therefore do not
require walking C resources on each consumer encoder. Numerical source readiness
and lifetime still belong to canonical mesh maps; residency alone is not a
producer/consumer synchronization mechanism. Program teardown removes execution
edges, waits for numerical completion, unbinds each domain map, and frees domain
metadata before releasing its source tensors. The queue residency set is removed
before its allocations are released.

`mesh_algebra_trace_route_count` and `mesh_algebra_trace_route` expose each domain
once, using its creation-order numeric ID. Role 0 is a metadata map, role 1 a
candidate map with its exact ordinal and retirement row, role 2 a configured
consumer with its function index and completed row, and role 3 the shared table's
logical rows. Candidate entries expose the current produced owner/function when
the prepared bit is present. Flags 1, 2 and 4 indicate prepared, candidate retired,
and named consumer completed. `mesh_algebra_trace_route_reader` exposes ordinary
source/group member identity for roles 0 and 1, and absent identities otherwise.
The trace contains actual candidates rather than making their apparent count
shrink by omitting the shared table.

The retained domain storage and fixed adjacency are O(C+D+E), with E<=C for this
partition. Initial relation processing is O(C+D+E); a source notification selects
one consumer, and consumer retirement visits its own interval. Ordinary canonical
readiness still checks that interval when revisited: successive arrivals can
repeat readiness checks for a long interval. This commit does not claim linear
total readiness work or measured latency parity with Pallas. Source-row count and
ordinary graph edges also remain in the accounting. Setup overlap validation can
compare candidate maps quadratically; it is outside invocation and does not
allocate a dense candidate/consumer structure.

Both native libraries compile together. The `mesh_row_function` source ABI gains
route uses; the shared bridge header remains unchanged. Existing bridge processes
need no shared-header restart, while Python/native processes must load the rebuilt
pair. Operational CPU/Metal, repeated-occurrence and delayed-source evidence is
owned by the existing streaming-algebra workflow. Remaining independent work
includes active-grid ownership, repeated readiness traversal and independently
published byte extents; sparse domains do not eliminate per-capacity partial
launches or publication allocation padding.
