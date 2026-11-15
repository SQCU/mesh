# Change request: connect compiled programs to execution and observation

## 1. Execution of the program domain

The compiler transforms every program in AOT A1's domain through the shared placement, sharding, dependency,
allocation, transport and scheduling mechanisms (AOT A2–A10; GSPMD; plan-to-bridge R5). Torch-export programs and
typed LiteRT programs supply their source numerical bodies and explicit operands to those functions. Each assigned
target consumes the completed lowering through the native entry (composition 1–7).

A work or measurement assignment names the properties `P` of its program domain and the transformation under
review. Compositional review relates each lowering rule's output guarantees to its consumer's input assumptions
(Hoare 1969, rule of composition; Abadi and Lamport, *Composing Specifications*, 1993). Existing clients supply
instances of `P`, and their records identify the numerical semantics, storage/alias/mutation, dependence order,
communication, cost or overhead property observed (operator's
[program-domain instruction](documentary-review-2026-11-13.md#program-domain-correction-14-november)).

The installed bridge configuration supplies the topology, region and block size used by each rank, through
`bootstrap.sh`, `install.sh` and `bin/mesh-bridge.sh`. The program consumes its compiled result on the configured
collective (operator's integration instruction, 8 November, this file at `cdd44406`).

## 2. Command order

The compilation scheduler produces a node's command order from R5's completion times and dependence edges by
`mesh_graph_heap_order` (Hall, Schulz, Shmoys and Wein, completion-time list scheduling;
[list-scheduling sources, LS](list-scheduling-sources-2026-11-08.md)). Native execution consumes that order using
Lee and Ha's self-timed execution ([source lowering, item 3](torch-source-lowering-change-request.md)).

## 3. Task weights

The compilation caller supplies a weight for each task and transfer, together with link capacities. Routing and
scheduling consume those costs through [AOT A3](aot-compilation-change-request.md); measured observations supply
their source values through [unequal-shards items 6/7](unequal-shards-change-request.md).

## 4. Shared graph operations

The graph module consumes graph operands and masks and produces the masked walk used by native preparation.
Its callers consume that result through the common graph interface (operator instruction, 8 November 2026,
item 4 at `cdd44406`; [composition item 3](compilation-composition-change-request.md)).

## 7. Commit feedback

Each native submission constructs its own `MTL4CommitOptions` and feedback handler, which writes its error and GPU
times to that commit's record. The completion consumer receives that record through the handler's semaphore.
The mechanism follows `addFeedbackHandler:` and the per-commit options observation in
[commit-feedback sources, CF §§1/3](commit-feedback-sources-2026-11-08.md).
