# Shared routing ownership and reader groups

Source/literature proposal, September 13, 2026. This is the next storage and
lifetime lowering after [segmented scatter](scatter-lowering.md), not an
implementation or performance claim. No native source changes accompany it.

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
