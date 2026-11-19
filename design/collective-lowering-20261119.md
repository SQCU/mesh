# Collective lowering and declaration lifetime, 19 November 2026

## A. What the two producers returned

The scatter solver selected a directed peer hop and an exact byte interval for each piece. The collective compiler converted those intervals to tree weights, reconstructed the cuts, then routed every hop again. A generated tree also sent each selected arc back through the default logical-write router. These two routes to physical lowering changed the byte traffic after the solver chose it. On a branching broadcast, treating each discontinuity of logical byte offsets as another CPU publication added thousands of operations to the declared program.

The bridge recomputed a segment's checksum from the current occupants of its program slots whenever it compared declarations. A client could complete its last transfer, close its program, and reuse the slot before the bridge prepared the closing message. A two-node execution of the native order code produced two different checksums for that same completed segment.

## B. What each consumer reads

The default broadcast and all-gather use the scatter solver's source ranges, recipients, and communicator topology, including ranks that only forward. A generated tree supplies its selected physical arcs to the lowerer. A logical WRITE supplies its routing operand. The scatter lowerer builds each peer's ordered DMA span view from the selected byte offsets. Within that view, each forwarded slice names the incoming completion it consumes. A branch can read the same landed bytes for both local use and forwarding. The bridge stripes the peer's spans over its paired cables. [Scatter and tree interfaces](../rdma/mesh-plan.h) · [Collective compiler](../rdma/nccl-mesh.c)

The declaring client computes the canonical segment checksum while it owns the program slots and crossing tables. It stores the checksum alongside the ranges and round size before publishing the segment. The bridge reads that value for both the opening and closing messages. Region layout 159 carries the additional word. Both endpoints use the existing wire digest format. [Declaration](../rdma/mesh-call.c) · [Order and segment agreement](../rdma/mesh-order.h) · [Slot reuse exercise](../rdma/tests/order.c)
