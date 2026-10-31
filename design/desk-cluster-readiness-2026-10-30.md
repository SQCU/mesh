# Desk-cluster deployment review

Recorded 2026-10-31 00:51 UTC (October 30 on the desk). Reviewed mesh
`8b5c7371` and metal-microbench `55c1fad1`, plus the current uncommitted
access-export edits in `native.py`, `operand.py`, and `_stream.mm`.
Those edits are incomplete and are not deployment evidence. This review reads
source and history; it starts no cluster job or numerical execution.

**The ordinary distributed compiler is not ready for the requested deployment.**
The target is several Studios running whole models with sharded sparse table
reads and fine-grained sharded feed-forward work. The missing connection is
between capabilities already present in the library. A separate executor or a
special implementation for a model name would not close it.

## What the two WRITE interfaces actually do

`ncclMeshWritePagesPrepare` was introduced in `ac1ea093` on October 26.
`ncclMeshWriteLayoutPrepare` followed in `bf4f1454` on October 28. Both reach
the same `write_prepare` and return the same plan, executed by
`ncclMeshWriteExecute`. They are two public preparation contracts, not two
independent transport engines. See `rdma/nccl-mesh.c:6289`, `:6309`, `:6449`,
`:6498`, and `:6607`.

The later interface accepts a tensor layout over page storage. Its lowering
supports strided/mapped views, framed publications, local binding periods,
sub-byte encoded storage and library-owned relays. Both variants also persist
through `write_declared.kind` and separate template restoration branches.
Current tensor frontends call the layout variant: `_stream.mm:1371`, `:1958`,
`mesh_mpi.py:273`, and microbench `mesh_rank.m:940`.

The consolidation target is one public page-and-mask WRITE contract, with tensor
layout conversion performed by the shared compiler before that boundary.
Remove the second preparation contract and the duplicate declaration/restore
branches after moving its callers and representational features into that
lowering. Do not lose framing, local periods, relay placement or compact indexed
views by reverting callers to an older, narrower binding. No second execution
mechanism is needed.

One historical duplication is already removed: `f48ded9b` made declared groups
and standalone WRITE plans use the same `definition_prepare`. The current
code must not be reported as still having two independent definition builders.
The larger collective compiler also calls the same physical
`mesh_write_compile`; its remaining divergence is scheduling, below.

## Current blockers and removal targets

### Changing indices do not yet cross the ordinary distributed pipeline

`ownership.project_numerical` still marks live index/predicate accesses pending
(`ownership.py:204–225`). `BoundGraph._bind_owned` then stops preparation
(`declarations.py:1168`). Native IR retains the index producer; the new
`operand.access_domain` exposes its possible byte reads, but that helper is not
yet connected to the owned graph and the required selected-value crossing.

There is already an owner-local `index_select` expression that preserves value
bits, including masks and collective reduction (`partition.py:2068–2117`).
It is selected through actual DTensor dispatch, which the new global source
capture does not invoke. Its expression must join the retained compiler path.
Local-view scopes and typed Partial/complete transitions likewise remain
pending (`partition.py:1665`, `:1749`, `:1803`).

Compile supplied work assignments forward. Use local access analysis to derive
input pages and retain the actual index-producing edges. A possible read domain
is not a request to transfer every possibly selected table value. The execution
must move the selected values through the same fixed-width page operations,
with changing index contents as ordinary data.

### Encoded collective calls still have blanket barriers

`metal_begin` emits a dispatch-to-dispatch barrier for every collective group
(`nccl-mesh-metal.m:233`). `dispatch4` adds another after every dispatch outside
the presentation phase (`:297`). These commands are encoded during preparation
but run on every replay. Source-level preparation does not make their runtime
cost disappear.

`ncclMeshGroupAccesses` also describes shared channel `taken`/`presented` words
as accesses (`nccl-mesh.c:3222–3248`). `MetalScheduleCrossWork` adds them to
the same dependency analysis as payload pages (`metal-recording.m:1470`).
Repeated crossings can therefore acquire order through shared descriptor-list
state even when their payload ranges do not overlap.

Replace blanket ordering with the compiled page, channel and publication
relations. Preserve actual producer/consumer dependencies and hardware credit
limits. The current list protocol depends on its ordering; simply deleting its
edges without replacing that dependency representation is not the refactor.

### Exact access records are still lost at consumer boundaries

The native emitter's new per-command read/write relations have not yet reached
all consumers. The `_stream.Commands` constructor edit is not yet connected to
`native.bind` or rebuilt. LiteRT's `native_shader` takes only the argument access
bitmask (`tools/prepare_e2b.py:4244`); its dispatch recorder declares the whole
argument layout (`litert_program.m:2657`). Disjoint sparse uses of a shared
binding can thus look dependent.

Carry separate read and write selections through the existing command record
and recorder. Binding a buffer does not mean every command accesses all of it.
The actual LiteRT indexed-source importer connection is also unfinished.

### Collectives and WRITE-only groups get different scheduling quality

Only all-WRITE groups receive `order_writes` and its capacity-aware ordering
(`nccl-mesh.c:3118`). General collective dependencies use a lowest-numbered-ready
event order (`mesh-routed.c:384–394`), preserving incidental transaction order
where another ready transfer could use a cable sooner.

`mesh_fabric_refine` recognizes COPY producers but not REDUCE producers
(`mesh-fabric.c:73`). Combined relay output can therefore retain a larger
publication boundary even though the runtime supports per-unit sum forwarding.
The current `mesh_solve` still uses shortest-path trees for all-leaf demands
(`mesh-collective.c:858–877`); the prior review's tree-packing fix is not landed.
These are remaining causes of link underuse, not evidence that extra links or
relays are intrinsically counterproductive.

### Synchronous alternative entry points remain

The main `Executions.run` only enqueues the prepared schedule
(`_stream.mm:1940`), as does `MeshRankRun`. `Drained` is an autorelease pool
(`_stream.mm:2060`), not a stream drain.

The separate `_stream.encode` helper instead creates a schedule and semaphore,
marks every operand read/write, submits and CPU-waits (`:907–925`). Its current
callers are input-import/repartition copies and an explicit layout probe, not
the main Function replay. `schedule_after_inputs` also commits prior PyTorch
work and installs a broad entry event during preparation (`:410–417`).

The host `mesh_mpi.WritePlan.Execute` uses a null stream (`mesh_mpi.py:129`),
which reaches `host_finish` and waits for GPU completion (`nccl-mesh.c:4942`).
These alternatives need migration to the shared prepared-function interface;
they must not become hidden execution paths for model operations.

Separately, microbench `Rank.take` serializes whole server requests until the
previous request and its output handling finish (`rank.swift:379`, `:451`).
That is service admission, not a dependency between tensor pages. Removing it
requires giving independent requests their declared state and output operands,
including replacing process-wide output redirection.

### The current MoE demo does not establish the target workload

`examples/ep.py` fixes expert count, top-k, token count and bucket capacity
(`:19–20`). It masks overflowing assignments and writes them to a replacement
slot (`:43–45`), then transfers and computes the bucket slots (`:46–52`). It
also uses the still-unconnected local scopes (`:36–37`). This is not evidence
of arbitrary, fine-grained selected work across the cluster.

The radix selector and static sparse-product schedule already exist. Static
sparse layouts explicitly remain fixed during execution; that does not
establish changing sparse table lookups or changing work selection. The demo
must consume the same completed shared compiler interfaces with explicit
configuration, without a separate expert executor or an overflow workaround.

## What is already established, and what establishes readiness

The source has a shared physical WRITE compiler, multiple physical cable paths,
relay placement, retained numerical bodies, page aliases, native selection
kernels, static sparse products and an asynchronous recorded execution entry.
Recent work removed the older classic executor, duplicate definition builders,
several bridge per-unit checks, and barriers between individual presentations.
Those fixes do not close the remaining paths above.

Readiness requires an ordinary configured whole model to compose changing
indices, sharded table reads, selected feed-forward work and reductions through
one compiler and recorder, with no unresolved transitions or substitute
executors. Emitted dependencies must follow actual pages and resources.
After that structural connection, use ordinary demos/hourlies to record complete
function latency and useful throughput together. This review supplies no new
performance estimate and makes no claim about installed peer builds.
