# The rubric, actionable

The short form of [design/rubrics.md](rubrics.md). Each line below is that file's own first sentence of a dimension;
the full dimension, its poles, the operator's pairs and quotes are at the line given. For a case these lines do not
settle, read the full rubric: its Quantities (line 105), Inversions (651), "Two channels" (672) and "Failure across
interfaces" (714).

## The procedure (rubrics.md:69-77)

1. Name the kind of every file a change touches, from the table under Quantities.
2. Rank the change against the form it replaces, using that kind's dimensions only, each dimension separately. Give the
   ranking as a pair, `old ≺ new` or `new ≺ old`, with the code that decides it.
3. Read a change that ranks higher in one kind only by taking in another kind's concern as misplaced. It is split along
   the interfaces above.
4. When two requirements conflict, decide by the rubric of the kind the code belongs to. In a conflict between kinds,
   decide by the interface: what crosses it, and nothing else.
5. Rank a rule, a memory note or a pass/fail check written in place of a ranking below the ranking.

## The interfaces (rubrics.md:14-18)

| Interface | What crosses it |
|---|---|
| M and C | rows a client owns, positions in them, supplied and landed words |
| C and D | the NCCL API, the device-function header, memory through `ncclMemAlloc` |
| M and D | nothing |

## The kinds (rubrics.md:155-158)

- **Mesh (M):** `rdma/mesh-flow` and what it publishes; the node services. Lives as long as the node.
- **Collective (C):** libnccl-mesh, the schedule compiler (mesh-collective.c, mesh-plan.h), the device functions in
  mesh.metal, the bindings (torch-mesh backend.py, mesh_mpi.py). Lives as long as a communicator, in its client's process.
- **Data (D):** everything that computes values (metal-microbench's engine, mesh_rank.m, rank.swift, torch programs,
  examples/, torch-mesh's kernels) and the configuration it loads. Lives one call.

## Mesh

| Dimension | Name | Ranks higher | rubrics.md line |
|---|---|---|---|
| M1 | The bridge's scope | the form whose bridge does fewer kinds of work | 180 |
| M2 | Messages about messages | the form in which less of what crosses a link describes other traffic | 197 |
| M3 | States and their exits | the form in which more of the reachable states have a transition to off, taken on a signal from outside, and which has fewer states that exist only to wait | 211 |
| M4 | Containment | the form in which a failure's effects stay with the party that failed, and in which no mechanism is added to survive someone else's failure | 247 |
| M5 | Mediation | the form in which clients can name less of the device and the bridge decides more of what goes where | 272 |
| M6 | Recovery without people | the form that gets back to serving after a loss with fewer parties involved and less downtime | 290 |
| M7 | Memory commitment | the form that commits registrations earlier and changes address translation less while DMA is live | 315 |
| M8 | The hot path | the form with fewer steps between a client's supplied buffer and the link | 325 |
| M9 | Addressing | the form that uses what the bridge already knows about where each buffer goes, instead of reconstructing it at run time | 338 |
| M10 | Attribution | the form whose accounts assign each lost frame to its real cause and owner | 347 |

## Collective

| Dimension | Name | Ranks higher | rubrics.md line |
|---|---|---|---|
| C1 | Provenance | the form that keeps NCCL's API and semantics as its interface and whose implementation is Pareto dominant to NCCL's: fewer transactions per collective operation, more sparsity, lower latency | 357 |
| C2 | Invariance | the form whose one implementation serves more cablings, member counts, uneven counts and dtypes without a branch for any of them | 374 |
| C3 | Routing as data | the form in which more of the route and the collective's form is a compiled object passed as an operand, and less is inferred where it runs | 388 |
| C4 | Asynchrony | the form that issues earlier and waits later, and adds fewer waits between them | 398 |
| C5 | When time may decide | the form whose time-dependent decisions are diachronic (made between calls, on known buffer depth) rather than synchronic (made during a call, from a timing estimate) | 410 |
| C6 | Failure's reach | the form in which a failure ends less (the call alone) and leaves less for the next call | 423 |
| C7 | Symmetric knowledge | the form in which less is learned from peers at run time, because more is known identically at every member from the compiled schedule | 447 |
| C8 | Numerical candor | the form that declares its precision and order, and checks payload less | 456 |
| C9 | One primitive, offered as a header | the form with fewer copies of each primitive, offered for others to call rather than spliced into their code | 463 |

## Data

| Dimension | Name | Ranks higher | rubrics.md line |
|---|---|---|---|
| D1 | The universal before the particular | the form whose definition serves more of linear algebra with one interface, with a model as the demonstration | 479 |
| D2 | When choices are made | the form that makes its choices earlier: when the configuration is loaded or synthesized, rather than at the call, and at the call rather than inside a tensor function | 492 |
| D3 | Control flow | the form with fewer branches | 507 |
| D4 | One implementation of each computation | the form with fewer implementations of each computation, and with foreign code called rather than edited | 513 |
| D5 | Shares | the form that divides work across more axes by evidence | 527 |
| D6 | How allocations adapt | the form whose allocations are revised between calls by an estimator that beats the update-free baseline, rather than priced by a model or looked up in a table | 539 |
| D7 | Mediated use of the mesh | the form that knows less about links, peers, routes and the bridge, and reaches the mesh only through the collectives API and its own node's bridge | 556 |
| D8 | Memory the program knows it needs | the form whose buffers are sized by the program itself from its configuration, held from load and reused, caches included | 568 |
| D9 | Evidence | the form whose evidence is rankable statistics from end-to-end calls on two or more members, sampled at temperature 1, rather than pass/fail checks | 576 |
| D10 | Pace | the form at or past the best measured frontier | 590 |
| D11 | Kernel units | the form whose kernels work in matmul tiles, fitted parametrically, and never run slower than the dense reference | 604 |
| D12 | Correlation of error | the form whose errors are decorrelated with one another, across positions, layers, chunks of a sum, members' shares, forms of the same computation, and runs | 615 |
