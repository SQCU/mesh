# Progress by composition — 13 November 2026

Style: [Google developer documentation style guide](https://developers.google.com/style/highlights).

This records the contracts reviewed in the current workgroup, through mesh `264227fb` and metal-microbench `f132ccf2`.
It is evidence about those revisions, not a new specification or an indefinitely current description of code.
The method is the operator's [local compositional review](../../design/compiler-workgroup-2026-11-13.md#local-compositional-review).
The requirement roles below come from the existing [composition request](../../design/compilation-composition-change-request.md),
[AOT request](../../design/aot-compilation-change-request.md),
[source-lowering request](../../design/torch-source-lowering-change-request.md),
[unequal-shards request](../../design/unequal-shards-change-request.md), and
[plan-to-bridge request](../../design/plan-to-bridge-change-request.md).

## Accounting

The current wave has **eleven established local contracts**, composing into three longer sections of the required
data flow. Establishment here is by source review against the assigned inputs, transformation and output, with
the consumer connection inspected. The numerical and build evidence has its own scope below.

Each row counts once even where several requests require it. A row covers the stated boundary, not every function
in its file. Clause references identify the local obligation being discharged, not completion of every occurrence
of a cross-cutting clause. This is not a percentage based on document, function or line counts.

| # | Established local contract | Requirements covered locally | Source / checkpoint |
|---|---|---|---|
| 1 | Two finite shard progressions → intersection and both local index maps → slice/scatter construction | Unequal-shards 2/8/9/12; composition 3/6 | `rdma/mesh_graph.py::intersection`, `placement.py::reshard`; mesh `1e82f8e6` |
| 2 | Placement's exact transfer endpoints → device-copy correspondence → endpoints belonging to the transformed graph, consumed by scheduling | Composition 2/3/6; AOT A2/A8 | `placement.py`, `devices.py`, `schedule.py`; mesh `3a2aa8c1` |
| 3 | Solved dependence endpoints and sender layouts → assembled body metadata → lowering's existing node/region correspondence | AOT A6/A8; source-lowering 3; composition 3/6 | `schedule.py::assemble`, `lower_inductor.py::lower`; mesh `ca1e4785` |
| 4 | Processor/target operands and scheduled bodies → one explicit lowering selection → artifact mapping | Composition 1's selection obligation, 2/3/6 | `driver.py::lowered`; mesh `ca1e4785` |
| 5 | Scheduled numerical region, constants and typed values → upstream AOT compilation → numerical package and operand correspondence | Composition 2/6; source-lowering 1; AOT A8 | `lower_inductor.py::body/lower`; mesh `ca1e4785` |
| 6 | Completed numerical package and interface counts → public native loader/ownership helpers → bound Model invocation | Composition 4/6; AOT A8/A9 | `rdma/mesh-aoti.cpp`, `rdma/mesh_aoti.py`; mesh `76d6c351` |
| 7 | Numerical artifacts, state, bindings and solved records → archive → restored bound numerical models | AOT A8/A9; composition 2/6 | `archive.py`, `lower_inductor.py::saved/restore`; mesh `ca1e4785` |
| 8 | LiteRT compiled model/signature → retained typed Program and ordered interface → phase binding/composition | Composition 2/5/6 and typed-source correction | `provider/litert_mesh.h/.cc`; metal `76d81c90` |
| 9 | Retained typed body and backing correspondence → canonical typed clone/remapping → composed phase Program with views and ownership retained | Composition 2/5/6, values/effects contract | `metal_ir_append`, `metal_clone`; metal `76d81c90` |
| 10 | Source-declared alias identities → canonical storage owner/access policy → planner value, host address and final binding, with separate transfer identities/geometry | Composition 2/6.4; AOT A5/A8; bridge R2/R4 | `tools/e2b_replay.m`; metal `12d5303b` |
| 11 | Exact dependence witnesses → combined endpoint-side obligation → selected native wait, without another overlap query | Composition 6/6.4 | `mesh-graph.{h,c}`, `mesh_graph.py`, `metal_recording.m`, IR/archive propagation; mesh `e4b2a96e`, metal `f61a2ff3` |

These results compose as follows:

1. **Graph placement and scheduling:** shard correspondence → explicit transfers → transformed endpoint identity
   → scheduled body with retained dependencies/layouts (1–3).
2. **Numerical artifact ownership:** scheduled body → selected lowering → numerical artifact → native Model →
   saved/restored Model (3–7). The effect recipe's native consumer is a separate open obligation.
3. **Typed program and physical storage:** LiteRT export → retained body composition → source-declared storage
   ownership → planning → host/target binding → witnessed native wait selection (8–11).

An unchanged numerical producer can remain the correct owner behind an established boundary. Untouched runtime
or target work is counted as remaining work; it does not revoke these local results.

## What this has accomplished along the requirements

- **Typed data reaches its actual consumer:** transfer endpoints, dependence edges, physical strides, source
  ValueIDs, views, alias classes, constants and lifetime ownership now cross the inspected boundaries explicitly.
- **Analysis has one owner at these sites:** consumers use the intersection's index maps, the scheduler's relation,
  the canonical clone, the existing graph-components algorithm and mesh planner. They do not reconstruct those
  results at each boundary.
- **Compilation produces retained artifacts:** native Model and archive consume completed numerical packages;
  restoration does not rerun capture, scheduling or numerical compilation.
- **Storage identity is a whole local connection:** the E2B correction reaches the actual allocation/address
  consumers. Joining aliases only inside an IR would not have discharged row 10.

Additional source inspection found two obligations already implemented before this wave: synchronization consumes
the supplied dependence arrays (composition 6.1), and saved native programs restore their solved waits without
another solve. They should not be assigned again or credited as new work from this wave.

## Next assignments, in priority order

### 1. Dependence causes through native wait selection — completed locally

The paired unit retains actual view endpoints and the caller's CPU/GPU side labels through the composed relation,
then consumes them during wait selection. `MetalHostSide` and its second overlap query are deleted. The existing
graph algorithm remains the only intersection owner.

The agreed fields are `size_t from_view,to_view` on the existing cause/conflict records. They refer to the exact
input view array; GIVEN edges have SIZE_MAX witnesses and retain their given identity. Distinct witness causes
must survive until the consumer has combined their obligations. Serialization/remapping must preserve the
association or first discharge the relevant property; naked indices are not independent provenance.

Codex reviewed both sides as one connection, including IR/archive propagation. The canonical E2B build and fresh
shared-library ABI checks pass; see [the bounded review](dependence-witness-review.md). This does not remove repeated
intra-invocation conflict derivation or implement endpoint-aware transitive reduction. Those remain distinct
obligations; neither is counted as completed here.

### 2. Solved local storage and effects into native whole-body lowering

Producer/consumer inspection refined this assignment: when source processors select the same actual machine,
target and device, their bodies can compose while scheduling still owns the global graph and transfer identities.
Their internal SEND/RECV pairs become ordinary value edges before AOT compilation. Such an erased channel does
not require a native queue, readiness counter or fabricated RDMA declaration. Genuine device copies and external
effects remain explicit. Original task/resource correspondence and required dependence ordering survive composition.

The running assignments connect that composed body to one upstream AOT package/model entry, displacing the
region/run/send/recv recipe. Remaining native effects consume actual prepared views, plan and invocation operands
through installed Torch's direct C-shim interface. The existing Model and archive remain their owners. CPU bodies
stay lowerable until composition, as composition requirement 7 states. `Plan.completion` remains a timing estimate;
it is not a runtime completion counter. These assignments are **in progress**, not additional completed rows.

### 3. The resulting native program into invocation, archive and signature

Bind the actual native entry and its remaining inputs through the existing callers. Archive and signatures take
the emitted program's records and rebinding operands. Remove the displaced Python queue/callback execution and
the signature's reconstruction of those old handoffs as their actual replacement becomes the shared consumer.
This can be co-designed with item 2; the ordering here is a dependency, not a ban on concurrent consumer work.

### 4. Extend the same contracts to further target and optimization obligations

Further CPU/MPS/ANE lowering, fusion, publication and reuse under changed placement/routing must extend the same
retained body/effect/storage contracts. Assign a concrete transformation with its actual consumers and deletion
target, rather than reopening already established boundaries or introducing another execution regime. These
unreviewed units are not included in the eleven-row count.

## Measurement for subsequent workgroups

Use the same accounting: name the required local contract, its clauses and actual consumer; record its completed
connection; then record which longer composition now follows. Give priority to open connections that consume
already established results and eliminate a concrete reconstruction. Keep uncovered responsibility/coverage
obligations visible. Source reduction, extra helper APIs, worker self-evaluations and unrelated whole-program
appearance are not completion measures.

Each subsequent review can use established contracts as premises, revisiting them when the relevant producer,
representation or consumer changes. No new scoring program, test suite or review gate is required by this ledger.

## Evidence

The latest E2B source connection and canonical build are recorded in metal-microbench
`output_data/compiler-workgroup-20261113/integration-storage-review.md` (commit `12d5303b`).
Numerical package invocation and archive restoration are recorded in `executor/archive-region-check.log`:
projection max absolute difference 0; FP16 FFN 0.00048828125; restored outputs bit-identical and restored signature
recomposition identical. These numerical observations have their stated input scope. They are recorded separately
from the local structural acceptance above; an additional end-to-end run is not its completion criterion.
