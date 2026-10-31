# Desk-cluster deployment review

Reviewed October 30, 2026. The initial review used mesh `8b5c7371` and
metal-microbench `55c1fad1`. The source corrections below are part of the same
integration pass. They are not measurements or evidence of installed peers.

The target is several Studios running whole models with sharded table reads
and fine-grained sharded numerical work. The remaining task is to connect the
ordinary compiler components. A second executor for a model name does not
close that task.

## Correction: one declaration owns WRITE operands

A declaration is compiler input: an operation, its page operands, layouts and
production relations. `ncclMeshCall` holds that input; `ncclMeshCommPlan` compiles
it. Neither is a reason to give WRITE a separate lifecycle.

The initial review compared `ncclMeshWritePagesPrepare` with
`ncclMeshWriteLayoutPrepare` and proposed retaining the former. That identified
the wrong boundary. Both belonged to the competing WRITE-plan lifecycle.
The required interface is `ncclMeshCommPlan` for every operation.

The history is:

- `4bb01da0` (October 24) introduced a separate WRITE plan and execute call.
- `ac1ea093` (October 26) added the page-mask preparer.
- `fb024020` (October 28) added actual bindings to `ncclMeshCall`, but explicitly
  skipped WRITE. Its branch excluded counted transfers too; `6393626e` removed
  that exclusion while leaving WRITE excluded.
- `bf4f1454` (October 28) added the layout preparer over the separate plan.
- `f48ded9b` shared a definition-building helper between the two lifecycles.
  Sharing that helper did not remove their competing contracts.

Thus the separate lifecycle predates the binding exclusion. The later binding
work preserved the duplication instead of completing the common interface.

The correction removes both preparers, `ncclMeshWriteExecute`, the public plan
handle, its readiness/landing setters, its standalone definitions, reference
lifecycle and serializer. `ncclMeshCall` now contains source/result page slots,
local layout geometry, frames, packet bounds, source-ready words and landing
records. The sole declaration serializer carries those same fields.

`ncclMeshCommEncode` records the next declared group. It accepts a communicator
and recording context, not buffers or a per-WRITE handle. Every group's slots
are prepared before recording. WRITE groups no longer rebuild placement at the
call site; mixed WRITE/collective groups use the same prepared issue path.
The Python, fused chain/DNN and native rank callers supply their bindings before
`CommPlan` and use that common recording entry.

## Shared scheduling and dependencies

WRITE and general collective groups now use the same capacity/dependency
ordering engine. The routed compiler's separate ready-event loop is removed.
The engine represents publication, streamed consumption, completion and
multi-parent combination. Relay refinement accepts combined producers as well
as copies.

Transport dispatches use the recorder's existing access frontier. Blanket
barriers at group entry, dispatch, pool and presentation are removed. A
monotone progress observation depends on its preceding publication; it does
not stop a later publication from advancing the counter. Ordinary payload
read/write dependencies remain.

Native emitted commands export separate read and write selections. The Python
binder and LiteRT importer carry them to the same recorder. An argument's
residency no longer substitutes for its actual access range.

## Removed blocking alternatives

`_stream.encode` and its private submit-and-wait path are removed. Migration
copies use the ordinary layout-copy compiler in the same schedule as their
transfers. Imported source state becomes an explicit initializer recorded once
before a function's first invocation; original attributes are not mutated into
not-yet-filled pages.

The transport's private host submission/commit/wait path and unused GPU polling
calibration are removed. The common recorder owns submissions. No null-stream
adapter is silently upgraded into a second executor.

The old `mesh_mpi` imperative methods and the Xonotic NumPy/Torch world callers
still need whole-function integration. Their numerical code consumes results
between immediate calls. Replacing one world wrapper with another does not
make that code a prepared function. They are not a supported blocking fallback.
Their standalone WRITE helpers are removed.

## Remaining deployment work

The source has physical multipath writes, relay placement, retained numerical
bodies, page aliases, selection kernels, sparse products and recorded execution.
The ordinary distributed pipeline still needs these connections:

- Changing index/predicate accesses must join the selected-value crossing and
  continuation of the retained numerical body. A possible read domain must not
  become a transfer of every possibly selected value.
- Local scope and Partial/complete transitions must consume the supplied work
  assignments through the common forward compiler.
- Indexed update/gather composition must eliminate unnecessary full intermediate
  materialization through shared fusion.
- Landing metadata reused across several recorded tiled invocations still needs
  immutable occurrence-specific records. Current DNN declarations use a distinct
  record table per group and chain's repeated groups are untiled; they do not
  establish the general repeated tiled case.
- The imperative Xonotic solver callers must declare their numerical work and
  crossings together.
- Ordinary model demos must consume explicit capacities and the same completed
  compiler interfaces. The existing fixed-capacity MoE demo is not deployment
  evidence for arbitrary changing sparse work.

Readiness means an ordinary whole model composes changing indices, sharded
reads, selected work and reductions through this compiler and recorder. After
those structural connections, ordinary demos/hourlies record whole-function
latency and useful throughput together. This review makes no new performance
claim.
