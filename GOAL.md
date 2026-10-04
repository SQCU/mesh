# Goal

The mesh's collective task: the lowest attainable overhead collective operations for arbitrary programs on Apple
GPUs, libnccl-mesh and torch.distributed's `mesh` backend ([rdma/NCCL.md](rdma/NCCL.md)), on any number of Macs in
any cabling.

- Operator (2026-10-01): "MAKING K-MANY COMPUTERS INTO A SHARED ADDRESS SPACE WHERE EXTREMELY HIGH NUMBERS OF CPU
  CORES CAN ASYNCHRONOUSLY START WRITING TO STRIPES OF THE REALLY BIG SHARED ADDRESS SPACE."
- Operator (2026-10-03): "i cannot and will not precommit to a specific ordering or routing!" Every algorithm
  holds for every membership and topology, read from an explicit link map (metal-microbench docs/principles.md §4).

Goals A and B ([design/collective-goals.md](design/collective-goals.md)) were the two-node targets of September
2026: history, not the task.

The separate game/policy release goal is [design/RELEASE-CLOSURE-GOAL.md](design/RELEASE-CLOSURE-GOAL.md).
