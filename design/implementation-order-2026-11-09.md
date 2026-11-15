# Compiler composition and remaining work — revised 14 November 2026

The change requests assign the mechanisms. This memorandum locates their producer/consumer dependencies for work
assignment. The earlier implementation-order text is retained in git at `cdd44406` as history.

## Composition

| Producer | Retained result | Consumer and cited request |
|---|---|---|
| Torch export / FX capture | Function bodies, calls, shapes, state, alias and public binding information | Placement and call analysis: AOT A1/A2/A10 |
| Placement and numerical decomposition | Placed operations, views, accesses, source correspondence, collectives | Shared graph analysis: composition 2/3/6; source lowering 5/6 |
| Graph analysis, routing and allocation | Dependences, routes, copy counts, lifetimes, page bindings | Joint scheduler: AOT A4–A6; plan-to-bridge R5 |
| Scheduler | Order, source relation, synchronization and allocated operands | Target emission, bridge, signature and archive: composition 4–7; capture 9 |
| Target emission | Specialized CPU/GPU/ANE bodies and effect calls | Native entry: source lowering 1/3/5/6 |
| Archive | Stored results and their references | Load-time binding and invocation: AOT A9 |

These are dependencies between values. Producers and consumers can be implemented concurrently against their
agreed operands. The unit of completion is a producer's result actually consumed by its assigned downstream
operation. `~/schedule-emit` demonstrates this composition; source literature supplies each algorithm's vocabulary.

The compiler retains derived values and their references in compact storage. Lowering specializes their use into
native operations. Conditions and iterations within a numerical algorithm retain the cited algorithm's semantics.
The operator's [14 November clarification](compilation-composition-change-request.md#clarification-2026-11-14-one-compilation-over-retained-results)
provides this distinction.

## Assignable remainder

[The dated work account](compiler-uncovered-work-2026-11-14.md#recorded-state-2026-11-14) records completed portions
and concrete remaining consumers. Work orders select those remaining units and the applicable request clauses.
Completed work stays completed; a new consumer uses the existing result. A later concrete defect is recorded
against its affected producer/consumer connection.
