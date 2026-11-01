# Compiler data flow

Recorded 2026-10-31. This is the order of the compiler's work. Backend
binding consumes its result; binding is not how the compiler discovers the
dependencies it needs to solve.

```text
function + types + capabilities + ownership/routing operands
  → captured graph, including derivatives and chosen reduction associations
  → selected producer/consumer domains and page placement
  → unbound numerical commands and transfer declarations, with exact accesses
  → transfer cuts, page versions, required observations and publications
  → numerical source with those effects, and a complete dependency graph
  → backend binding and recorded commands
  → execution over the assigned pages
```

The source follows these boundaries:

| Stage | Main code | Output consumed by the next stage |
|---|---|---|
| Capture | `Function.capture`, supplied reduction source | Typed graph and chosen fold relations |
| Ownership and placement | `partition`, `ownership`, `placement` | Selected work domains and page views |
| Command lowering | `LoweredGraph`, `host.lower_views`, `kernel`, `layout.copy_declarations` | Unbound commands with exact accesses |
| Transfer lowering | `ncclMeshCommLower`, `ncclMeshCommSources` | Occurrence index, native numerical source, exact transfer selections and page effects |
| Effect solve and source emission | `MetalPageEffectsSolve`, `MetalPageEffectsSource` | Numerical source containing all of its required effects |
| Binding and recording | `Program.prepare`, provider binders, `ncclMeshCommEncode` | Native commands and their dependency graph |
| Submission | The prepared execution | Page values and completion observations |

## 1. Capture the work

`Function.capture` retains the source graph and result types. Functions supplied
as reductions are source too. Their types, identities and allowed changes of
order are operands of compilation. Automatic differentiation must see the
association actually chosen for the forward function. It cannot infer a
distributed derivative from an associative-forward flag.

Backend requirements enter here as explicit capability and operand contracts.
A provider may need contiguous pages or an indivisible invocation. Those facts
must be available before placement; they are not discovered by trying to run
an operation and selecting a substitute.

## 2. Assign work and pages

Ownership and partitioning select the domains computed on each peer. The
relation algebra composes those domains with the source's reads and writes.
Placement chooses the pages used by each value and each API use. Necessary
layout copies become ordinary declarations with source and destination page
relations. Merely changing a view emits no copy.

This stage may reserve region pages: physical aliasing is an input to the
dependency solve. It does not create executable pipelines, host tasks, ANE
requests or recorded command sequences.

## 3. Lower declarations

Each numerical command retains its source, geometry, typed page operands and
exact read/write relations. A CPU recipe expands into its constituent
instructions. A native shader retains its dispatch-to-element relation. An
opaque hardware invocation states the accesses of that indivisible request.
None of these becomes an executable object yet.

All participating relations use the function's explicit ISL context. Shared
page mapping converts local buffer selections to bridge-region coordinates.
Two views of the same page therefore describe the same storage in this solve.

Transfer declarations retain their source/destination selections and the chosen
route and fold. The transport planner determines packet cuts, landing pages,
publication words and page versions from these declarations. Numerical source
has not been bound, so those transfer facts can still take part in lowering.

## 4. Solve the dependencies and emit effects

Compose command accesses with the transfer cuts. Put arrival observations at
the commands that need the arriving pages. Put publications at the commands
that finish writing those pages. Resolve overwrite relationships from the
actual selected pages. A layout copy participates in this solve just like any
other producer or consumer.

Numerical loop source is still retained here. For CPU work, each cut maps back
through the assigned pages to the source's value domains. Forward relations
find the affected outputs; backward demand includes each complete computation
needed for those outputs. Subtract previously selected work and keep the
remaining work once. Only then emit the loop source. A cut through a reduction
input therefore cannot create a partial result or duplicate its calculation.

These are relations between specific operations and pages. The presence of a
collective, host operation or API boundary does not itself create an ordering
edge. Source and index tables emitted here contain the decisions made by the
compiler; execution does not rediscover them by scanning accesses.

Solve all channels before emitting a command's source. A command can read from
several links and publish to several links. Those effects share one entry and
one set of required visibility fences; they do not cause repeated source
wrapping or one numerical dispatch per link. Independent publication prefixes
retain their own completion domains.

The native declaration has two preparation steps. `ncclMeshCommLower` selects
one occurrence, lays its transfers, fixes channel counters and predecessor
links, and returns typed page effects. The enclosing compiler uses those
effects before it binds numerical programs. `ncclMeshCommEncode` later records
that same occurrence by index. It does not select a new issue or reconstruct
its dependencies from backend addresses. Several occurrences may be lowered
before any is recorded; each retains its own counter and link snapshots.

Every DMA entry stores its absolute readiness base. Preparation resolves the
channel position into that field once. The bridge does not look up a second
per-call base table or add its contents during posting. Layout 161 removes
that obsolete shared-region representation. The wire payload format stays the
same; the local bridge and its clients must use the same region layout.

The source prefixes emitted by this lowering belong to an occurrence and a
physical lane. Each new occurrence has fresh region words, including repeated
uses of a declaration
whose page-binding period is one. Frames within that occurrence use increasing
prefix thresholds. The effect pass publishes a prefix only after all its
earlier source cuts are complete. A separate Function cannot advance that
word. Relay sends instead consume their declared incoming landing or reduction
word. Destination observations use the exact incoming channel threshold.
Source preservation copies split at both physical selection cuts and source
SEND cuts, so each part observes the word for the values it actually copies.
These words and thresholds are compiler operands; execution neither allocates
nor resets them.

`ncclMeshCommSources` exposes the occurrence's retained numerical declarations
and complete transfer selections. Native copies and reductions join the same
solve as frontend numerical work. `ncclMeshCommSpecialize` installs their emitted
source. Encoding consumes that result; it does not run a separate dependency
solve for native numerical work.

`ncclMeshCommSourceInputs` adds exact initial source spans to that lowered
occurrence. Earlier arrivals are prerequisites of the corresponding DMA send.
Preparation fixes the leading initial prefix. The command that exposes its
transfer descriptors names the entry pages it reads; prior arrivals remain
exact SEND prerequisites. Later producers share the same solved publication
domain. Source-preservation copies consume those incoming observations too.
This step creates no numerical command to stand in for an input.

## 5. Bind and record

Only now do `Program.prepare` and the provider binders create pipelines,
argument records and hardware requests from the finished declarations. The
common recorder lowers the complete dependency graph to backend commands.
Its resource constraints come from the selected capability operands.

The recorder assigns encoder scopes and fence lifetimes from that graph before
encoding commands. Compatible steps keep one encoder. Stage barriers and
cross-scope checkpoints preserve the solved edges; a command group does not
itself require a new encoder or fence. See
[encoder lifetimes](encoder-lifetimes-2026-11-01.md) for the backend scope rules
and the remaining queue-event limitation.

CPU numerical bodies are shared headers. ANE submit and completion entries use
one shared source recipe. The effect pass specializes these sources before
binding. Host access records retain the same observation and atomic modes as
device records, including the pages holding native workspace. They are not
reduced to a coarse pair of input/output lists.

Compilation may use several passes within a stage. A later pass consumes the
earlier pass's result. It must not require a hidden execution, an already-bound
opaque object or a second copy of the access model to reconstruct that result.

## 6. Execute

The resulting function consumes its page operands and recorded schedule. Page
index values can be outputs of other functions. That is an ordinary data
relationship already represented by the graph, not a request to plan again.
Runtime does not choose an implementation, allocate scratch, infer layout,
search for dependencies or build a task schedule.

An input already supplied by the caller belongs to the transfer's initial
source relation. A returned remote page with no local reader belongs to the
function's completion relation. Neither requires an empty numerical command.
The existing nonblocking completion observer reads the returned pages' words
and thresholds. It retains the result and its observation time once complete;
late statistics do not reclassify a finished function when a link later closes.

## Reordering in this integration

The host branch of `BoundGraph` had called `host.prepare_views` before
`page_effects.lower`. Its CPU programs and ANE bindings were already opaque
objects when the effect pass ran. `layout.resolve` also bound required copy
pipelines before that pass. Generic collective bindings hid transfer effects
behind a similar early executable boundary.

The correction is to move these bindings to stage 5. Their declarations belong
to stage 3 and their dependency effects belong to stage 4. Keeping annotations
beside prematurely bound objects would leave the incorrect ordering in place.
This file specifies the integration contract; source compilation alone does
not establish runtime correctness or performance.
