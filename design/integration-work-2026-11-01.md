# Shared compiler integration work

Started 2026-11-01, after mesh `250e58dc`. This is the current implementation
handoff for the seven remaining integration items. It records work in progress;
a listed interface is not evidence that its consumers are complete.

## Required result

CPU, Metal and ANE are ordinary implementations of declared tensor work. The
compiler can assign a complete section or selected work domains to them using
explicit types, shapes, resource capabilities and configuration operands. Index
and activity pages remain numerical inputs. They can select masked work within
a compiled domain without choosing an executor during execution.

All frontends retain source and exact page relations until transfer effects have
been composed into the numerical work. Backend binding follows that solve.
Execution consumes the prepared pages and commands. Backend choice, layout
conversion, transport and source-fold choices are operands of the same
whole-function optimization; none introduces an execution controller.

## Work and file handoffs

| Item | Required connection | Current implementation scope |
|---|---|---|
| 1 | Every collective and frontend consumes the same typed page effects before binding | Native collective effects in `nccl-mesh.c`/`nccl.h`; Python adapter in `_stream.mm`/`page_effects.py`; common source/effect pass and native adapters described below |
| 2 | Submission and DMA posting consume exact compiled dependencies and flat operands | Recorder queue-event scope, host visibility operands and bridge descriptor traversal remain open |
| 3 | Ordinary source sections choose CPU, Metal or ANE through declared capabilities | `mesh_compute.py`, `compilation.py`, `native.py`, `algebra.py`, `engines.py`, ownership handoff |
| 4 | Existing whole-function feedback consumes backend and source-fold choices too | Extend the existing assignment operands; no second optimizer or runtime wrapper |
| 5 | Indexed contractions and partial-result fusion share numerical declarations | Generalize row grouping and finish/reduction composition through existing relations |
| 6 | LiteRT ownership and complete serving requests use the shared compiler | Remove outer model-role assumptions; propagate activity into crossings; compose prompt/generation and request state |
| 7 | Ordinary demos and hourlies establish whole-function behavior and performance | After structural integration, committed-source LSTM, sparse/indexed, native/LiteRT and prefill/decode programs; latency and useful throughput together |

The current parallel work owns disjoint boundaries:

- Native collective lowering owns the typed effect ABI and its exact source and
  result coordinates. It also removes the lower-and-record compatibility entry.
- The Python binding consumes those effects and removes its standalone
  collective execution bypass. WRITE and other collective kinds have one
  occurrence-lowering contract.
- The common native source-effect compiler owns projection and source effect
  emission. Swift/LiteRT adapters retain source until this pass; Python also
  consumes this pass. It replaces the Python-only solve, rather than adding
  another implementation alongside it.
- Backend assignment owns implementation candidates and source-region/provider
  integration. It preserves the differentiated source and page identities.

These are coordination notes for overlapping sessions, not exclusive ownership
of a repository. Commit notes should identify changed shared operands and which
consumers were migrated. Recheck current source before editing the same boundary.

## Item 1: the complete handoff

The native effect record names its collective call, source or result operand,
frame-local logical byte selection, publication/observation word and threshold.
The collective compiler resolves offsets for each collective kind. Frontends
must not reconstruct a source/result role from a kernel name or address.

The shared source pass consumes retained numerical declarations: source recipe,
backend, typed argument pages, and exact dispatch-to-page relations. It produces
observation and contribution tables, publication prefixes, source-input
dependencies and terminal result words. The same source emitter specializes
Metal, CPU and ANE entry/completion recipes. Only then are pipelines, provider
requests and replay records bound.

Swift now retains kernel source and carries it through the same solve before
binding pipelines. Its command recipes still need complete access declarations.
LiteRT's prepared-provider export needs the same retained source and exact
access relations. Existing manually supplied tile protocols are being removed;
they are not substitutes for the general relation.

## Evidence required before completion

Source inspection must show every named frontend using the shared contract,
with no lower-and-record bypass or second source-effect solver. Build the actual
libraries and applications. Then use the ordinary committed programs to verify
numerical results and whole-function latency/useful throughput across declared
shapes, backend assignments and collective configurations. Source compilation
alone does not establish the end state.

## Current closure boundary

The encoder lifecycle change is committed. Compatible dependency scopes reuse
an encoder; a command group no longer creates its own encoder and fence by
default. Queue-wide host event ordering and bridge descriptor traversal remain
separate unfinished work under item 2.

The bridge now consumes absolute readiness bases prepared for each occurrence.
The relative-base lookup and the old repeated-call table rewrite are removed;
shared-region layout 161 records that ABI change. Linked descriptor traversal
and the actual page prerequisites still remain. This is not completion of
item 2. The bridge, collective library and CoreML library build with this
change; there is no new numerical or performance result.

The remaining host/device ordering has a specific missing handoff.
`MetalPageEffectsSolve` currently emits observations and publications for
transfer effects. Local numerical dependencies also reach the recorder, but
`MetalScheduleBindHost` turns GPU predecessors into `MetalSchedulePublication`
queue events. Submission waits for host events through the same queue-wide
interface. Exact input page relations therefore still produce a wider hardware
ordering scope. Removing those events alone would lose required visibility.
The shared source solve must also carry local cross-backend page dependencies
through emission and provider binding, using the same page effects; the
recorder must consume that result. Source-local sites are part of this handoff,
but their presence alone does not complete it. Actual provider resource and
visibility constraints remain explicit compilation operands.

Native reduction source now supplies incoming arrivals, outgoing-transfer
overwrite prerequisites and result publications to the common solver. Read
observations constrain numerical inputs; Written observations constrain writes
which reuse pages with outstanding readers. Distinct masked prerequisites retain
their own predicates. Their mask-page dependencies precede the predicate reads.
CPU and Metal fold bodies retain arithmetic and numerical cell operands; their
separate arrival arrays, overwrite checks and prefix publication code are gone.
This source change has passed C/Objective-C and generated CPU/Metal compilation;
it has no new numerical or performance result.

The next source pass has replaced the Python-only effect solver with a binding
to `metal-page-effects.h`. The common library, CoreML library, CFFI binding and
PyTorch extension build with that connection. This is source/build evidence;
the integration changes are still in progress and have no new numerical or
performance result.

Native collective reductions and source-preservation copies now expose their
retained numerical declarations and internal transfer effects through
`ncclMeshCommSources`. They join frontend commands in the shared solve.
`ncclMeshCommSpecialize` installs the resulting source before final encoding;
encoding no longer runs a separate collective effect solve. The direct eager
collective/group entry points and the standalone Python collective executor
have been deleted.

The Chain/DNN frontend migration is written and built. The old `PreparedChain`,
`FusedForm`, fused registry and native fused-record sink are removed. Chain keeps
its numerical body while declaring grouped frames and their actual physical
page generations. DNN reuses retained numerical declarations by tile shape.
The common source adapter interleaves native and frontend leaves by explicit
frame order before recorder dependencies are built. `CommEncode` consumes an
explicit selection of prepared command ordinals; it has no separate whole-list
execution substitute. These changes join this source checkpoint. Integrated numerical/performance
evidence remains required.

Swift's enclosing preparation now gathers native and frontend declarations,
solves them together, then binds commands. Its recipes and the LiteRT provider
export still need complete group-to-page relations, including host and ANE
contracts. Whole-buffer access lists repeated for every group do not supply
that relation. LiteRT's scalar serializer was dropping dispatch metadata; it
is being connected to the shared logical access stage. That stage preserves
source coordinates before target launch geometry or physical pages are bound.

The GPU provider also retains source, but its existing selector/codegen drops
the access equations and final-write coverage. The final pipeline and tensor
shapes cannot supply those facts. The provider integration must retain the
actual indexed Read/Write sites and their coverage through code generation,
then pass those ordinary source cells to the common solve. Repeated or masked
writes need their declared contribution relation; a possible-address union is
not that relation. This missing producer remains item 6 work.

The retained relations now share the Function's explicit compilation context.
Command construction, native source queries and the shared solve consume it.
A source-only capture confirmed that the retained relations use that context.
This establishes preparation dataflow, not numerical or performance evidence.

The common access contract still needs dynamic index and activity operands.
Named source-local sites now retain fixed cell domains in that same contract.
The solver places a site's observations and publications at its enter/leave
points rather than the enclosing kernel's boundary. Shared operand loads and
stores retain the corresponding coherency qualifiers, including vector uses.
The CPU declaration now supplies its source before resource binding; binding
consumes the chosen recipe without choosing another numerical body.
CPU source-local cells now use the same phase and contribution indices as
Metal cells. Uniform observation thresholds lower to constants for either
backend. GPU cells declare lane, SIMD-group or threadgroup participants;
source-local effects do not add a whole-threadgroup barrier to a smaller scope.
Generated CPU and Metal site source compiles without numerical execution.
The bridge, collective/CoreML libraries, CFFI binding, PyTorch extension and
generated Metal kernels build with these shared changes. Native recipe and
provider consumers remain in progress; this build did not execute payloads.

Dynamic index and activity coverage now has shared source machinery, committed
as `7c4ed19c`. `ncclMeshPageCoverage` contains ordinary active/expected UInt32
page operands and their fixed domain. Each transfer retains its explicit
coverage-cut projection; selected contribution rows are ordinary index pages.
The common source solve attaches count/mask announcements to their actual
writers or to an explicitly declared imported-value continuation. The count
and contribution order may vary; zero-contribution cuts still complete from
their metadata producer. Exact index/count input reads enter the same dependency
solve. Source compilation established this implementation's syntax, not complete
frontend or transport integration.

Native transport and Python/Swift/provider bindings are being connected to
that contract. The native routed compiler already retains matching packet
domains and positions for source, relay and destination messages. Coverage must
project from those identities before the common source solve; it must not
reconstruct them from kernel names or source/destination offsets after binding.
A union of all possible pages would impose false dependencies and count writers
which never contribute. It is not a substitute for these ordinary index inputs.

The bridge side now builds with shared-region layout 162. Each entry supplies a
descriptor-production prefix independent of payload readiness. Payload readiness
retains its affine unit stride: a packet prefix uses stride 1; a scalar tile
publication uses stride 0. This removes the old `1 - packet_count` underflow
which made early packets of a multi-packet copied tile wait forever. A masked unit
advances logical progress without a SEND or receive. Actual DMA counters and
frame charges exclude those units. Completions name their retained span directly;
masked gaps extend the preceding pending completion's logical coverage, or
complete directly on a lane with no request pending. This preserves a prefix
through completed/masked units without waiting for a later active transfer.
There is no second completion lookup list, ring, or payload copy. Existing
holding/table lifetimes retain these descriptors through completion reads.
Coverage now enters `ncclMeshCall` before packetization. Each interval names
an original logical SEND, a frame and relative bytes, with active/expected page
slots. The routing compiler retains that transaction and byte offset through
every fragment and relay. The complete original SEND schema retains its endpoint
ranks, endpoint step identities, byte origins and extent. Preparation resolves
coverage references against that schema even on ranks outside the chosen route;
an unresolved reference cannot silently select unmasked execution. Packet-cut
domains alone cannot distinguish different partial results which traverse the
same bytes. Semantic interval ends join the
shared backing cuts before either endpoint's span list is built. `CommSources`
then projects those intervals onto the final spans and resolves the occurrence's
page slots; the public late `CommCoverage` interface and Python occurrence method
are removed. WRITE and other collectives use the same final packet cursor.
Python Call declarations and native MeshRank declarations consume this early
contract. WRITE cuts follow the retained payload selection through fragmentation;
source call boundaries are mapped to the resulting native declaration boundaries.
Native general-collective records keep their exact source owner and SEND-local
coordinates. Shared schemas are independent of local bindings: off-route ranks
need no mask buffer, while a rank consuming a fragment must supply its pages.
Automatic generalized-collective transaction synthesis through Python kind/tree
rewrites remains open. This does not remove the linked occurrence walk or
queue-wide events.

Automatic coverage generation from indexing/scatter/provider source remains
open. The shared emitter now accepts the page index a source-local site already
computes. Its exact retained relation determines contributions under dynamic
counts too; identity and affine selections emit arithmetic rather than a second
index lookup. Separate index tensors remain ordinary inputs where the numerical
source actually produces them. Mixed static/dynamic cuts and different coverage
arrays may share one publication word. Native CPU/Metal fold equations now consume contributor activity, omit the
inactive payload observation/combine, and use the declared identity for an
inactive preceding COPY. The higher-order Reduction source accepts an ordinary
activity operand and selects the identity before its combine graph and AD.
Transport omission and arithmetic identity are distinct consequences of that
same input. Automatic source-activity/count generation and projection through
all frontend rewrites remain open. A native collective's separate partial-result
generations require their own retained arithmetic provenance; a common packet
domain or directed cable cannot supply it.

Item 3 has CPU/ANE source assignment and provider connections written. CPU
products retain partitionable product equations until work domains and
placement are chosen. A hardware invocation that is truly indivisible declares
that granularity as a provider constraint. Complete frontend coverage and live
provider/page preparation remain unfinished.

Activity now reaches retained source through `MetalPageCommandActivity`.
Preparation declares its page read in the common solve. Emission puts the
predicate's own input observations before that predicate and keeps the selected
numerical inputs, work and contributions inside it. The same composition emits
CPU, Metal and ANE source. `MeshHostTask.activity` and the recorder's separate
activity loop are removed; native Swift callers declare the source operand.
The ANE entry submits its already-bound request only for selected work and
reports completion for an empty invocation through the existing completion ABI.
No numerical implementation is selected during submission.

An inactive numerical source contributes no value. A zero-contribution transfer
cut completes from its declared active/expected metadata producer. Connecting
every source mask to that coverage remains required; skipping numerical work
cannot stand in for that declaration. Per-site conditional writers still need
their actual masks and selected-index provenance in the shared solve. Their
existing numerical predicates must not acquire a second runtime mask evaluator.

The activity handoff builds the collective/CoreML libraries and the native
Swift Engine. Masked and unmasked CPU, Metal and ANE emitted source compiles.
These checks do not establish the complete conditional-writer or local
cross-backend dependency handoff, and no payload run was launched for them.

Items 4 through 7 remain open. Backend and fold preferences now enter the
existing assignment vector and observations. The regression fit keeps source
choices as causal features when their lowered parameter domains differ;
whole-function cohorts carry latency and useful bytes over the same interval.
These source changes still need complete application evidence. General indexed
contraction batching and finish fusion, complete LiteRT request graphs, and
ordinary committed application evidence remain required. The immediate
priority is completing the shared indexed-page/source/dependency/binding
contract in items 1 through 3.

The combined source checkpoint builds the bridge, collective/CoreML libraries,
CFFI binding, PyTorch extension and generated Metal kernels. Subsequent source
checks cover direct relation-driven contributions and source-local aliases.
Coverage refinement computes packet cuts once per transport and refines the
retained transfers in one pass over sorted bindings. Source preparation reads
semantic intervals by transaction/frame and resolves each occurrence's active
and expected slots; the declaration owns those intervals. It no longer rescans all
span cuts and rebuilds the transfer array for every mask. These are preparation
and build results; no numerical payload run or latency/throughput result is
claimed by this checkpoint.

## Item 7: the LSTM's preparation, measured (the LSTM demonstration session)

lstm-service now declares and lowers on every rank (mesh ecc21b26 to b26a5fd9). It does not reach execution,
because preparation outlasts every bound. These measurements are from one rank on m5ult2 (gloo, CPU tensors in
its region, the stepper's capture alone), 2026-11-01.

- Declaration is about 3.5 s a step and grows with the steps composed: 12-15 s for four. Most of it is the run
  search on sets with divisions (MeshIndexSetRuns' subtraction). Placement makes those sets from byte relations
  built with `floor(x / width)` and `x mod width` (placement.py field_storage's `bytes_` preimage, and selected's
  `use['positions']`). The positions are typed elements expanded to bytes, so their runs are the elements' runs
  scaled by the width.
- Realization is 130-170 ms a step after 2c647fc1 and 29b4c255 (56 s before). It is paid at every occurrence, as
  replay-templates-2026-10-29.md requires. At 4,096 steps that is about 9 minutes, whatever steps_per_replay is.
  transfer_effects solves page effects again for every command of each occurrence, but only the effect words
  differ between occurrences: the accesses, groups, relations and reaches repeat.
  The merged numerical-source change `e8d20c52` now emits a declaration once
  per preparation and copies its emitted source for each occurrence; the reuse
  identity includes all source operands, including configuration. This removes
  repeated numerical selection/emission. The subsequent integration below also
  removes repeated page-effect solving. This checkpoint has not established the
  required preparation bound for the 4,096-step program.

`Execution` now owns its retained numerical declarations and a page-effect plan.
`CommPlan` supplies each native definition's identity and complete payload and
coverage binding-phase vector. That declaration authority selects the solved
relations. `MetalPageEffectsRebind` attaches a later occurrence's words, counters
and pages to the same command/transfer ordinals. It gives that occurrence fresh
prefix state; it does not reuse completed counters or an executable snapshot.
Predicate prerequisites and observation order are solved once too. Source
binding consumes their retained ordinals without further relation intersections.

Absolute observation thresholds and diagnostic entry values now occupy metadata
pages, so changing those values does not rewrite the numerical source. Metadata
field order, aliasing and page-relative offsets belong to the retained binding
phase. Source bindings do not choose a declaration by a hash or guessed shape.
The bridge, collective/CoreML libraries, CFFI binding, PyTorch extension and
generated Metal kernels build with this connection. No preparation speedup or
whole-function performance result has been measured for it yet.

## Active handoff — 2026-11-02 03:02:55 UTC

This is the next pass after mesh `d1d503e9` (merged as `b951ab72`) and
microbench `2ecca438`. Record a step's purpose and intended handoff here before
starting it. Update its written/build/commit status as evidence arrives. These
entries describe the current work; they do not replace items 1–7 above.

### C1. Consume conditional writer relations in the shared solve

Owner: `fallback_inventory`. Goal items: 1 and 2.

Why: the solver still subtracts every later possible write range as if that
writer always ran. That can hide an earlier conditional producer. Counting
possible writers also counts work that the numerical source never performs.

Scope: `rdma/metal-page-effects.h` and `.m`. Consume the original source-cell
selection, its cell-to-activity-page relation, and its value generation. A final
write demand must distinguish an earlier writer from a later active overwrite;
an earlier reader still consumes its earlier generation. Repeated targets need
the source's actual assignment/reduction semantics. Do not guess those semantics
from the address union.

The numerical source already chooses whether a site runs. Its activity typing
must not add a second runtime predicate around that site. Count and active
outputs must be ordinary source equations, with their actual producer boundary
available to transport declaration before packetization. An earlier writer must
not wait for a later-produced mask merely to decide whether it contributes.

Status: contract/design work; no shared-source changes in this pass yet.
Completion evidence: source masks and generations affect solved dependencies
and contributor counts, both frontend adapters consume that contract, and the
old unconditional subtraction/counting behavior has no authority over masked
writers. Source compilation precedes any payload run.

### C2. Bind every native invocation before solving

Owner: `fallback_inventory`; implementation handed back to the parent.
Goal item: 2.

Why: standalone `MetalProgram.bind(invocations:)` solved base declarations and
then relocated numerical operands during replay. Activity/effect pages could
therefore keep the first invocation's binding. Inspection corrected an earlier
inventory claim: `MeshRankPrepare` already solves its complete invocation
sequence once. Splitting that solve would lose cross-invocation dependencies.

Scope: microbench `matrix_operations.swift`, `mesh_rank.h`, `docs/kernels.md`.
The written change binds each declared invocation, including indirect-grid
phase, before one complete graph solve. It preserves MeshRank's existing
whole-function solve and adds no cache identity or execution wrapper.

Status at 2026-11-02 03:22 UTC: committed and pushed as microbench `3c09ab3a`.
Review matched the stated intent. Its `invocations == 0` return is the source's
empty round sequence (`rounds = max(0, …)` in `lm_engine.swift`), as the old
empty replay range was. The `-O` Engine module and native `libmetal_native`/
`libmesh_rank` built against mesh `d51ddcec`, and again in clean worktrees at
microbench `89f12f47` against mesh `8cccb59d` (Xcode 26.4.1). Rank, goal-a and
forward_graph sources typecheck against that module. Numerical execution has
not been run.

### C3. Retain the provider's original writer-domain program

Owner: `barrier_inventory`. Goal items: 1, 3 and 6.

Why: the retained convolution fields do not describe every final store. MPP
uses tile-rounded guards, and fused epilogues can change the destination view.
A buffer's capacity or a guessed formula cannot provide the writer count.

Scope: microbench's pinned LiteRT source patches, `prepared_export.mm`,
`litert_prepared.h`, `litert_program.m`, and its provider closure note. Factor
the original guard/iteration/coordinate generation once and retain it through
fusion. Numerical stores and ordinary selected-index/activity outputs must use
that same source program. C1 consumes the resulting relations; C5 makes them
available before transport planning. Do not create a separate provider-only
expression evaluator or numerical-source parser.

Status: source seams identified; the next writer-domain implementation has not
been written. The earlier clipped MPP read-site work is committed as microbench
`69a469b5`. Complete this step when exact writer domains replace the independent
`provider_dispatch` description for their consumers; do not claim all provider
recipes are covered merely because one recipe exports the new contract.

### C4. Make direct cache writers consume the shared selector

Owner: `schedule_equivalence`. Goal items: 1 and 5.

Why: direct fused heads still repeated the page-table lookup already computed
by `cache_select` for standalone writers. Their source-site metadata also lost
activity/join information during fused-slot remapping.

Scope: microbench native recipes and `streamed_weights.swift`; the shared
`rdma/torch-mesh/torch_mesh/streamed.metal` head's output operand. Reuse the same
`CacheSelection` pages. Keep the existing arithmetic once. Publish the exact
head-sized output run at its existing participant boundary, rather than adding
an atomic publication for every scalar store.

Status at 2026-11-02 03:22 UTC: committed and pushed as microbench `89f12f47`
and mesh `8cccb59d`. Composed sources carrying the page sites' bodies and the
solve's observed/published binding masks compile under Metal 4 (Xcode 26.4.1):
`cache_select`, `kv_write`, standalone `streamed_heads`, and fused heads, cached,
with and without stored rows; head widths 7, 65 and 128 also compile.
microbench `docs/kernels.md` records the provenance of injective active cache
targets in ordinary session construction. The output run uses the existing
page-site mechanism: `begin`/`end` bracket each head's D-value run at threadgroup
scope, so the shared solve's site body runs once per run. No transport decision
was added. Arbitrary repeated descriptor targets still have no declared
overwrite policy. That missing semantic input must remain explicit for C1. The
rank link waits on a LiteRT prepared-library rebuild. No payload run.

### C5. Declare numerical sources before transport planning

Owner: parent (`root`). Goal items: 1 and 6.

Why: MeshRank currently obtains numerical declarations in `MeshRankPrepare`,
after `MeshRankPlan` has packetized transfers. At that point it is too late for
the source's conditional output domains to supply early coverage operands.

Scope: microbench `mesh_rank.m` and its interface documentation. Retain the
actual invocation declarations before `ncclMeshCommPlan`, in one enclosing
relation context. Preparation must consume those same declarations, rather
than calling the source producer again. Preserve the complete function order,
including its declared epilogue, and release preparation relations after the
joined solve/binding.

Status at 2026-11-02 03:29 UTC: the ordering substep is committed and pushed as
microbench `42564104`. `MeshRankPlan` retains `sourceRuns` and each
invocation's `sourceCommands` in one `sourceContext`, once, before it waits on
its communicator and before the declaration exchange and `CommPlan`.
`MeshRankPrepare` consumes those arrays; its second source callback is gone.
The context and source relations are released after binding, or at detach if
preparation did not reach that stage. The interface documents that the complete
run and epilogue are declared before Plan. The native rank library and `-O`
Engine module build against mesh `862ece24`; rank, goal-a and forward_graph
sources typecheck. No payload run.

Completion evidence met: source/selector facts exist before packetization;
their original declaration identities reach the joined solve; no second
declaration traversal reconstructs them. This completes the ordering substep,
not C1/C3 coverage generation: the projection consumer still has to install the
original source's active/count operands into the early transfer declarations.
The C1 adapter hunks (`MeshRankCoverage` retaining the demand, native
`DispatchSite.activity` through `MetalPageAccessLayout`) depend on C1's shared
ABI and are not in this commit.

### Next handoff and validation

Keep C1 and C5 coordinated at the early coverage boundary. C3 and C4 supply
original source facts to that same boundary. C2 can be built and committed
independently once reviewed. The broader host-event/visibility replacement,
flat bridge posting, complete serving graphs and whole-function performance
evidence remain open under the original seven items.

Use the existing builds: mesh `make -C rdma .build/mesh-flow
.build/libnccl-mesh.dylib .build/libmesh-coreml.dylib modules kernels`, then the
microbench native libraries and Swift Engine with `MESH_RDMA` and `MESH_LIBDIR`
pointing at that build. Use Xcode 26.4.1 and `SDKROOT=macosx`. Do not substitute
synthetic timing, private executors or payload runs for the unfinished source
handoffs. Commit only owned changes; microbench's scheduled-task lock and
unrelated `serving_state.swift` edits remain outside this work.
