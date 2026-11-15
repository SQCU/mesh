# Change request: compile captured functions and their calls ahead of time

### A1. Capture produces function graphs

`torch.export.export` and FX capture the assigned known-shape tensor functions, parameters, state and public
input/output bindings (PyTorch `torch.export`; Reed et al., *torch.fx*, MLSys 2022). The compiler passes these
values to placement. Its domain is all tensor and linear-algebra programs expressible in Torch and Torch-like
languages with the assigned AOT properties: known shapes and a call-graph structure independent of tensor values.
Indexes and values within those shapes remain invocation inputs. The compiler lowers the domain by composition
of source numerical operations and their complete lowering rules (MLIR full conversion; source-lowering 1/5/6).
The operator's [program-domain instruction](documentary-review-2026-11-13.md#program-domain-correction-14-november)
assigns that coverage; numerical algorithms retain their own internal conditions as source-lowering 2 specifies.

### A2. Placement supplies sections and processors

The caller supplies placement and sharding operands. Placement propagates sharding and inserts collective nodes
using GSPMD's sharding propagation and resharding (Xu et al., 2021, §§3–4). The resulting graph carries each
operation's processor, sections, views and collective operands to scheduling. Unequal-shards specifies counts.

### A3. Costs supply task and transfer weights

The caller supplies task execution costs, transfer costs and link capacities as the weighting operand of Sinnen's
weighted task graph (*Task Scheduling for Parallel Systems*, 2007). Capture's execution-time model produces costs
from operation work, bytes, mapping and machine rates using XLA's GPU performance model and Roofline (Williams,
Waterman and Patterson, 2009). Routing and scheduling consume these costs; unequal-shards consumes their size dependence.
The edge-cost producer can supply Hockney's `alpha + bytes * beta` model (*The communication challenge for MPP*,
1994; MSCCLang §5.1; HEFT, Topcuoglu et al. 2002), fitting startup and per-byte time from measurements of at least
two transfer sizes. Each routed hop receives its cable's resulting cost.

### A4. Collective lowering produces page transfers

Collective lowering expands ring reduce-scatter/all-gather and tree reductions/broadcasts through Patarasuk and
Yuan 2009, Rabenseifner 2004, and Thakur, Rabenseifner and Gropp 2005. Garg–Könemann's concurrent-flow/packing
method (FOCS 1998) and the declared link capacities supply packed routes for the complete transfer set.
The graph carries local computation, receive-reduce-copy, forwarding and page-transfer operations together
(MSCCLang, Cowan et al., ASPLOS 2023, §§4–5; GSPMD §4.2).

### A5. Allocation produces page bindings

The allocator consumes value lifetimes, dependency order and Lam's expanded copies (PLDI 1988, §2.3).
It derives legal sharing from happens-before (Lamport 1978) and assigns pages through chain decomposition
(Dilworth 1950; Fulkerson 1956), cyclic-interval allocation (collective-values item 14), and contiguous-offset
allocation (capture item 6). The page table binds names to frames (Kilburn et al., *One-level storage system*,
1962; Denning, *Virtual memory*, 1970). Target lowering and transport consume the resulting bindings.

### A6. Scheduling produces one ordered program

The scheduler consumes the placed task graph and costs, solves the completion-time program and emits heap-based
topological order through plan-to-bridge R5 (Chen 2022; Hall et al. 1997; MSCCLang §5.2).
The scheduled value carries that order and its derivation to every target consumer.

### A7. The island sequencer orders function executions

The bridge's fixed sequencer assigns FIFO sequence numbers to submitted function executions through the
fixed-sequencer total-order broadcast of Défago, Schiper and Urbán, ACM Computing Surveys 36(4), 2004, §7.1,
and Pathways' asynchronous dispatch (Barham et al., MLSys 2022, §4.4–4.5). The chosen sequencer is the island's
highest-numbered live node, as the operator's approved configuration records at `cdd44406` specify.
Destinations implement §7.1 Fig. 5's `pending` and `nextdeliver`: receipt inserts a numbered message and delivery
advances through available consecutive numbers. A message names a compiled function and a range of executions;
Pathways §4.5 submits a captured subgraph in one message. Delivery releases the calls; task dependencies determine
when their numerical work and transfers run, including overlap across delivered calls.

Each static schedule has its own FIFO of delivered messages at each node (Lee and Messerschmitt 1987). A10's call
bounds and the program's IPC graph determine its capacity through BSL §4.3's self-timed buffer bound. Feedforward
edges use Convert-to-SC-graph and DetermineDelays (BSL §9, Figs. 6/8); the common graph module computes these results.
The bridge consumes the FIFOs by a multiway merge of sequence numbers (Knuth, vol. 3, §5.4.1). Function-call delivery
uses sequencer order; page transfers use each cable's compiled order and data availability.

### A8. The IR operand carries solved results

Placement, dependency analysis and allocation return explicit values and retained references. The task graph
(Sinnen 2007), its views, weights, source correspondence, call analysis and derived schedules pass to their next
consumers through those references (MSCCLang §§4–5; composition items 2/6/7).
The same graph and allocation algorithms consume parameters, numerical operands, calls and transport operands.

### A9. Serialization stores the compiled operand

The archive stores the scheduled program, target programs, page bindings, state sections, source relations and
derived results as compact sections with retained references (MSCCL-IR, Cowan et al. §5; ExecuTorch memory-planned
serialization; the operator's paged-result instruction in the documentary review of 14 November).
Loading exposes their views and rebinds live addresses. The bridge executes its ordered transfer lists and each
target executes its compiled program using Lee and Ha's self-timed strategy (Bhattacharyya, Sriram and Lee 1997, §2).

### A10. Call analysis produces ordering and concurrency bounds

FX captures calls of compiled functions as leaves in a program graph, following Pathways' program tracer
(Barham et al. §3). Bernstein's read/write conditions (1966) and Rinard and Diniz's commutativity analysis
(TOPLAS 19(6), 1997) produce call dependence and commuting relations. The graph module computes transitive
closure and chain decomposition through Fulkerson matching in Hopcroft–Karp phases. The output contains each
function's outstanding-call bound and the call ordering consumed by scheduling and submission.

## Repeated execution and coexistence

Lam's modulo variable expansion and Bhattacharyya, Sriram and Lee's self-timed buffer bounds determine copy
counts and the ordering between their uses. Each invocation uses its compiled residue. The island sequencer
retains per-function execution order while the compiled dependency relation determines overlap.

Independent flow sets contain one demand's pages on one route, consumed in any order under R4. Each execution
applies one Fisher–Yates permutation to the set's positions across every hop, parallel cable and consumer task
(Fisher and Yates; Durstenfeld 1964; MSCCLang's channel subsequences). Threefry-4×64-20 supplies the draws, keyed
by the compiled function's name and counter `(execution, set)` (Salmon et al., *Parallel Random Numbers: As Easy
as 1, 2, 3*, SC 2011, §4.2). Lemire's unbiased bounded-integer method (2019) supplies each shuffle index.
The producer declares which consumers take pages independently and which take a complete operand (R4).

A compiled function's name is `(program group, compilation ordinal)`, derived alike at every rank; the launcher
draws a fresh group for each launch (operator's independent-flow/coexistence instructions, this file at `cdd44406`).
The same permutation renames positions throughout the route. A relay posts a page for round `j` using the range's
monotonic count threshold `j * pages_per_copy`, as the operator-approved rule in appendix item 5 specifies.
The phase mechanism in disjoint-pages C3.7 supplies readers ordered within its phase window.
Distinct compiled functions retain their published storage and schedule identities; deferred reclamation releases
each retired function after its users finish (McKenney and Slingwine, *Read-copy update*, PDCS 1998; Fraser 2004).

Map/view changes select the next compiled realization through the membership and routing mechanisms in
plan-to-bridge R7/R8 and collective-values items 11–13. Numerical kernels, transport and retained compiler
operands share the ordinary page-allocation and lifetime mechanisms above.
