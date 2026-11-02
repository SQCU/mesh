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

Native reduction source has another precise missing handoff. Its incoming
scratch arrivals, outgoing-transfer overwrite prerequisites and summed-result
publications still live in the handwritten CPU/Metal fold body. The reduction
branch does not supply their transfer effects to the common solver. They are
not duplicate waits which can simply be deleted. Complete this declaration
first: observations must select Read or Written accesses as appropriate, retain
contributor activity, and avoid merging distinct masked prerequisites into an
unconditional maximum. Then the shared emitter can replace the handwritten
arrival/overwrite/publication code. The numerical combine equation stays.

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

Activity also has to cross these interfaces. Skipping a host numerical entry
must not skip the publications needed to complete a declared transfer domain.
Index and activity pages therefore belong to the shared numerical and transfer
declarations. They cannot be separate runtime execution policies.

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
  differ between occurrences: the accesses, groups, relations and reaches repeat. One solve per declaration,
  with each occurrence's words as an operand of that solve (absolute, prepared before execution, as 9ba0e43c
  keeps them), is what lets a 4,096-step program prepare within the operator's 8 s bound.
