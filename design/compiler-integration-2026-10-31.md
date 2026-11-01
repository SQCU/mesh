# Compiler integration after the runtime inventory

Recorded 2026-10-31. This records source changes after
[the dated inventory](compiler-runtime-inventory-2026-10-31.md); it does not
replace that inventory's historical findings.

## Shared numerical and collective lowering

Supplied reductions now enter the ordinary numerical compiler before capture
and differentiation. A declaration holds the combine, identity, value schema
and permitted changes in fold order. The selected binary association is source
visible to automatic differentiation. Its retained tree columns and shares are
placement operands, including rank-preserving alternatives when order matters.
Transfer cuts compose with command-to-page relations. Required arrival
observations and final output publications are emitted inside the commands
which read and write those pages. See
[the interface and example](function-reductions-2026-10-31.md).

Collective COPY and REDUCE outputs use the same landing refinement. Differing
packet cuts, relay inputs and repeated frames no longer select a readiness
pool. Prepared value cells select their work directly. Built-in reductions
select their arithmetic during preparation. Float64 host arithmetic uses the
same page effects and shared publication primitives.

The recorder consumes exact GPU and host accesses. Preparation composes their
dependencies, including explicitly named later producers and commuting atomic
contributions. Independent commands do not acquire a whole-group barrier.
Monotone publication uses the shared maximum operation. Each physical bridge
lane has its own walk cursor; prerequisite lists have explicit lengths.

Physical overwrite relations are solved after the actual pages are bound.
A COPY constrains the transfer which fills its landing. A reduction constrains
its output writes while its incoming values can arrive independently. The same
graph chooses a distinct original source generation only where retaining the
alias would create a cycle. Ragged binding periods share a prepared phase;
later recorded uses retain that choice and order.

The Python frontend has one captured numerical source contract. The eager
ragged dispatcher and host MPI numerical adapter were removed. Repartitioning,
selection and scalar fusion consume retained page relations. Row products and
matrix tiles share readers, output relations and capability operands. Dynamic
index values remain ordinary data in pages.

## Connected native consumers

The sibling metal-microbench change `d4afaa23` binds LiteRT host leaves, GPU
leaves and invocation pages through the shared recorder. It compiles bounded
generation/speculation recurrences and observes output prefixes without a
per-step completion callback choosing the next numerical program. Vision owns
its invocation operands and declares individual consumers. Its
[source contract](../../../metal-microbench/docs/serving-page-operands.md)
records the remaining service boundaries.

Those serving boundaries are still material: masked serving collectives need an
activity operand at the crossing, multi-chunk prompt admission remains staged,
and `Rank.take` still serializes a process-global engine/output lifecycle.
They are separate from the compiler ordering corrected in this pass.

## One lowering order

[The compiler data flow](compiler-data-flow-2026-10-31.md) states the stage
boundaries. Host instructions, layout copies and native shaders remain unbound
declarations until transfer lowering has supplied their page effects. Every
native shader declaration carries its group-to-value relation. CPU and ANE
entries retain source recipes with exact page accesses. The effect pass emits
all channels together, then the provider binds that completed source.

`ncclMeshCommLower` returns an occurrence and typed page effects before any
numerical binding. `ncclMeshCommEncode` consumes that occurrence later. Channel
counters and predecessor links belong to the occurrence, so lowering another
one cannot change what the first will record. Microbench lowers all its
crossings before recording the function.

The custom reduction callback API and its record-time compiler have been
deleted. Stable partial-summary merges use the same supplied-function source
interface as other reductions. The separate executable `ViewWrites` class,
early host binding and early layout-copy binding are also gone. These are
interface removals and changes in stage order, not annotations attached to
already-bound executable objects.

Input-only source spans feed the existing transfer presentation. Returned pages
without a numerical reader feed the shared nonblocking completion observer.
Neither emits an empty GPU command. Result thresholds already observed by
numerical work are removed; remaining thresholds share one entry per physical
word. The frontend has one `completion_observe` operation. Metrics accept late
results without a drain, and the collective demo no longer stops between
independent cases just to align their timing.

## Evidence and limits

Validation in this pass is source compilation and graph metadata capture.
The native library, bridge, extension, Metal source, and sibling Swift/ObjC
consumers are built against the shared contracts. No numerical payload or
cluster performance measurement was run. Full Python reducer preparation
needs a live region attachment; this session did not establish one. No heap
substitute was used to bypass that requirement. Source compilation establishes
neither numerical correctness nor recovered latency and useful bandwidth.
The pure source-rewriting stage was also compiled with observation and
publication fragments in the shared row and matrix kernels. This checks the
emitted Metal syntax without standing in for region binding or execution.
