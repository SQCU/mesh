# Compiler integration after the runtime inventory

Recorded 2026-10-31. This records source changes after
[the dated inventory](compiler-runtime-inventory-2026-10-31.md); it does not
replace that inventory's historical findings.

## Shared numerical and collective lowering

Supplied reductions now enter the ordinary numerical compiler. A declaration
holds the combine, identity, value schema and permitted changes in fold order.
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

Those boundaries are still material: masked serving collectives need an
activity operand at the crossing, multi-chunk prompt admission remains staged,
and `Rank.take` still serializes a process-global engine/output lifecycle.
The supplied reduction interface also does not yet declare distributed
derivatives or optimize among rank-preserving association trees. Opaque native
and host-boundary leaves still need dispatch-to-page metadata for transfer
effect fusion. These are not closed by removing the older competing executors.

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
